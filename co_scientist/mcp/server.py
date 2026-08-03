"""The co-scientist MCP server: structured-output capture + research tools.

Launched by a CLI backend for the duration of one LLM call:

    python -m co_scientist.mcp.server

Everything is configured through the environment so the command line stays
identical across calls (and so nothing sensitive lands in a process listing):

    COSCI_CAPTURE_DIR   required — where records and provenance are written
    COSCI_RECORD_TOOLS  comma-separated record_* tools to expose
    COSCI_AGENT         agent name; selects research tools via AGENT_TOOLS
    COSCI_SESSION_ID    session id, threaded into ToolCtx
    COSCI_TASK_ID       task id, threaded into ToolCtx
    COSCI_TOOL_TIMEOUT  per-tool timeout in seconds (default 30)

Two families of tools are served:

**record_\\*** — the schemas in `agents/schemas.py`, verbatim. This is how a
CLI agent returns structured output: it calls the tool, we validate the
payload against the same JSON Schema the Anthropic API used to enforce, and
write it to `<capture>/record-NNNN-<name>.json` for the backend to read back.

**research tools** — the existing `ToolRegistry` (PubMed, arXiv, Europe PMC,
web_fetch, web_search). Every call is appended to `<capture>/tools.jsonl`
along with the URLs it surfaced, which is what keeps the citation verifier's
"you may only cite what you actually saw" rule enforceable.

This process never touches SQLite. The parent backend ingests `tools.jsonl`
and performs all database writes, preserving the single-writer discipline.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..agents.schemas import (
    RECORD_HYPOTHESIS_TOOL,
    RECORD_RESEARCH_PLAN_TOOL,
    RECORD_REVIEW_TOOL,
    RECORD_RUBRIC_SCORE_TOOL,
    RECORD_SAFETY_ASSESSMENT_TOOL,
    RECORD_SYSTEM_FEEDBACK_TOOL,
)
from ..config import load_config
from ..ids import tool_run_id
from ..tools.base import ToolCtx
from ..tools.registry import ToolRegistry
from ..tools.urls import extract_urls
from .protocol import McpStdioServer, log, text_result

SERVER_NAME = "cosci"
SERVER_VERSION = "0.1.0"

RECORD_TOOLS: dict[str, dict[str, Any]] = {
    "record_hypothesis": RECORD_HYPOTHESIS_TOOL,
    "record_review": RECORD_REVIEW_TOOL,
    "record_system_feedback": RECORD_SYSTEM_FEEDBACK_TOOL,
    "record_research_plan": RECORD_RESEARCH_PLAN_TOOL,
    "record_rubric_score": RECORD_RUBRIC_SCORE_TOOL,
    "record_safety_assessment": RECORD_SAFETY_ASSESSMENT_TOOL,
}

# Returned after a successful record. The CLI agents keep going until they
# believe the task is done, so we tell them explicitly that it is.
_RECORD_ACK = (
    "Recorded successfully. This is the final step — do not call this tool "
    "again and do not run further searches. End your turn now."
)


class CoScientistMcpServer:
    def __init__(self) -> None:
        self.capture_dir = Path(_require_env("COSCI_CAPTURE_DIR"))
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.agent = os.environ.get("COSCI_AGENT", "")
        self.session_id = os.environ.get("COSCI_SESSION_ID", "")
        self.task_id = os.environ.get("COSCI_TASK_ID") or None
        self.tool_timeout = float(os.environ.get("COSCI_TOOL_TIMEOUT", "30"))

        wanted = [
            n.strip()
            for n in os.environ.get("COSCI_RECORD_TOOLS", "").split(",")
            if n.strip()
        ]
        self.record_tools = {n: RECORD_TOOLS[n] for n in wanted if n in RECORD_TOOLS}

        self.cfg = load_config()
        self.registry = ToolRegistry(self.cfg).discover()
        self.research_tools = (
            {t.name: t for t in self.registry.tools_for(self.agent)} if self.agent else {}
        )

        self._seq = 0
        self._lock = asyncio.Lock()

    # ----------------------------- tool list ----------------------------- #

    def tool_list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, schema in self.record_tools.items():
            out.append({
                "name": name,
                "description": schema["description"],
                "inputSchema": schema["input_schema"],
            })
        for name, tool in self.research_tools.items():
            out.append({
                "name": name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            })
        return out

    # ----------------------------- dispatch ----------------------------- #

    async def handle(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in self.record_tools:
            return await self._handle_record(name, args)
        if name in self.research_tools:
            return await self._handle_research(name, args)
        return text_result(
            {"error": f"unknown tool: {name}", "available": sorted(
                [*self.record_tools, *self.research_tools]
            )},
            is_error=True,
        )

    async def _handle_record(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.record_tools[name]["input_schema"]
        errors = validate_against_schema(args, schema)
        if errors:
            # Hand the model the specific problem so its retry is informed.
            return text_result(
                {"error": "schema validation failed", "problems": errors},
                is_error=True,
            )
        async with self._lock:
            self._seq += 1
            seq = self._seq
        path = self.capture_dir / f"record-{seq:04d}-{name}.json"
        _write_json(path, {"tool": name, "seq": seq, "input": args})
        return text_result(_RECORD_ACK)

    async def _handle_research(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self.research_tools[name]
        ctx = ToolCtx(
            cfg=self.cfg,
            db=None,                 # this process never writes SQLite
            session_id=self.session_id or None,
            task_id=self.task_id,
            run_id=tool_run_id(),
        )
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                tool.call(args, ctx), timeout=self.tool_timeout
            )
        except TimeoutError:
            body: Any = {"error": f"tool {name!r} timed out after {self.tool_timeout}s"}
            await self._log_tool_call(name, args, body, is_error=True, t0=t0)
            return text_result(body, is_error=True)
        except Exception as e:
            body = {"error": f"tool {name!r} failed: {e!r}"}
            await self._log_tool_call(name, args, body, is_error=True, t0=t0)
            return text_result(body, is_error=True)

        if result.is_error:
            body = {"error": result.error_message or "unknown error"}
            await self._log_tool_call(name, args, body, is_error=True, t0=t0)
            return text_result(body, is_error=True)

        body = result.content if result.content is not None else {"ok": True}
        await self._log_tool_call(name, args, body, is_error=False, t0=t0)
        return text_result(body)

    async def _log_tool_call(
        self, name: str, args: dict[str, Any], body: Any, *, is_error: bool, t0: float
    ) -> None:
        """Append one provenance record. The backend replays this into the DB."""
        entry = {
            "name": name,
            "args": args,
            "is_error": is_error,
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "urls": extract_urls(body),
        }
        async with self._lock:
            with (self.capture_dir / "tools.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# --------------------------------------------------------------------------- #
# schema validation


def validate_against_schema(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems; empty means valid.

    Uses `jsonschema` when available. The fallback checks the constraints our
    record schemas actually rely on (required keys, object/array/scalar types,
    and enums) so validation still bites in a minimal install.
    """
    try:
        import jsonschema
    except ImportError:
        return _fallback_validate(payload, schema, path="$")

    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}"
        for e in sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    ][:20]


_TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def _fallback_validate(node: Any, schema: dict[str, Any], *, path: str) -> list[str]:
    problems: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str):
        py = _TYPE_CHECKS.get(expected)
        # bool is a subclass of int in Python; don't let True pass as integer.
        if py is not None and (
            not isinstance(node, py) or (expected in ("number", "integer") and isinstance(node, bool))
        ):
            return [f"{path}: expected {expected}, got {type(node).__name__}"]

    enum = schema.get("enum")
    if isinstance(enum, list) and node not in enum:
        problems.append(f"{path}: {node!r} not one of {enum}")

    if expected == "object" and isinstance(node, dict):
        for key in schema.get("required", []):
            if key not in node:
                problems.append(f"{path}: missing required property {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in node:
                problems.extend(_fallback_validate(node[key], sub, path=f"{path}/{key}"))
    elif expected == "array" and isinstance(node, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(node):
                problems.extend(_fallback_validate(item, item_schema, path=f"{path}[{i}]"))
    return problems[:20]


# --------------------------------------------------------------------------- #
# entry point


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        log(f"[mcp] {name} is required")
        raise SystemExit(2)
    return value


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)           # atomic — the backend may be polling this dir


async def _amain() -> None:
    server = CoScientistMcpServer()
    log(
        f"[mcp] serving records={sorted(server.record_tools)} "
        f"tools={sorted(server.research_tools)} agent={server.agent!r}"
    )
    transport = McpStdioServer(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        tools=server.tool_list(),
        handler=server.handle,
    )
    await transport.serve()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:                        # pragma: no cover
        pass
    except BrokenPipeError:                          # pragma: no cover
        # Client went away mid-write; nothing useful left to do.
        sys.exit(0)


if __name__ == "__main__":
    main()
