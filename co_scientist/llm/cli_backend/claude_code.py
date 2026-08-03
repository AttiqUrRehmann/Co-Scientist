"""Claude Code backend — `claude -p` on subscription auth.

Translation notes, each of which is a real constraint of the headless CLI:

- **No forced tool choice.** `tool_choice={"type":"tool","name":X}` becomes a
  hard instruction in the prompt plus a post-hoc check that the record landed
  in the capture directory.
- **No `--max-turns`** in Claude Code 2.1.x, so a runaway loop is bounded by
  the subprocess timeout rather than a turn count. `num_turns` comes back in
  the result JSON for accounting.
- **`--system-prompt` replaces the default system prompt.** That matters a
  lot: a trivial call under the default prompt costs ~19k cache-creation +
  ~24k cache-read tokens of Claude Code scaffolding, versus ~1-2k with ours.
  (`--bare` would also strip it, but it forces `ANTHROPIC_API_KEY` auth and
  never reads OAuth, which defeats the purpose of this backend.)
- **`--tools ""`** disables every built-in tool, so all tool use flows through
  our MCP server and stays in the provenance log.
- Prompts go over **stdin**, not argv: meta-review prompts routinely carry
  tens of reviews and would otherwise risk the argument-length limit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...config import Config
from ..retry import CliBackendError, classify_failure
from ..types import AgentCallSpec, CallContext
from .base import AgentCliProvider, CliInvocation, CliOutcome

# Thinking-token budgets predate the CLI's discrete effort tiers; map them onto
# the closest tier rather than dropping the signal.
_EFFORT_TIERS: tuple[tuple[int, str], ...] = (
    (1, "low"),
    (4_000, "medium"),
    (8_000, "high"),
    (16_000, "xhigh"),
)
_MAX_EFFORT = "max"


def effort_for_thinking_tokens(thinking_tokens: int) -> str:
    for ceiling, tier in _EFFORT_TIERS:
        if thinking_tokens < ceiling:
            return tier
        if thinking_tokens == ceiling:
            return tier
    return _MAX_EFFORT


class ClaudeCliProvider(AgentCliProvider):
    backend_name = "claude_cli"

    def _resolve_backend_cfg(self, cfg: Config) -> Any:
        return cfg.llm.claude_cli

    # ----------------------------- command ----------------------------- #

    def build_invocation(
        self, spec: AgentCallSpec, ctx: CallContext, capture_dir: Path
    ) -> CliInvocation:
        record_tools = [t["name"] for t in spec.tools if _is_record_tool(t)]
        research_tools = [t["name"] for t in spec.tools if not _is_record_tool(t)]
        # The MCP server resolves research tools from the agent's allowlist;
        # leave the agent blank when this call is not supposed to have any.
        mcp_agent = ctx.agent if research_tools else ""

        mcp = self.mcp_config(
            capture_dir=capture_dir,
            record_tools=record_tools,
            agent=mcp_agent,
            ctx=ctx,
        )
        allowed = [
            f"mcp__{self.server_label}__{name}"
            for name in [*record_tools, *research_tools]
        ]

        argv = [
            self._binary,
            "-p",
            "--output-format", "json",
            "--model", spec.route.model,
            "--permission-mode", self._backend_cfg.permission_mode,
            # Built-ins off: every tool call must go through our server so it
            # lands in the provenance log.
            "--tools", "",
        ]
        if self._backend_cfg.replace_system_prompt:
            argv += ["--system-prompt", build_system_prompt(spec)]
        else:
            argv += ["--append-system-prompt", build_system_prompt(spec)]
        if allowed:
            argv += [
                "--strict-mcp-config",
                "--mcp-config", json.dumps(mcp),
                "--allowedTools", ",".join(allowed),
            ]
        effort = effort_for_thinking_tokens(spec.route.thinking_tokens)
        argv += ["--effort", effort]

        return CliInvocation(argv=argv, stdin=build_prompt(spec))

    # ----------------------------- parsing ----------------------------- #

    def parse_outcome(
        self, *, stdout: str, stderr: str, returncode: int, capture_dir: Path
    ) -> CliOutcome:
        payload = _extract_json_object(stdout)
        if payload is None:
            detail = (stderr or stdout or "no output").strip()[:800]
            raise classify_failure(
                f"claude did not emit parseable JSON (exit {returncode}): {detail}"
            )

        if payload.get("is_error") or returncode != 0:
            message = str(payload.get("result") or stderr or "unknown error")
            subtype = str(payload.get("subtype") or "")
            raise classify_failure(f"claude error [{subtype}]: {message[:800]}")

        usage = payload.get("usage") or {}
        return CliOutcome(
            text=str(payload.get("result") or ""),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_read=int(usage.get("cache_read_input_tokens") or 0),
            cache_write=int(usage.get("cache_creation_input_tokens") or 0),
            num_turns=int(payload.get("num_turns") or 1),
            reported_cost_usd=_as_float(payload.get("total_cost_usd")),
            stop_reason=str(payload.get("stop_reason") or "end_turn"),
            raw=payload,
        )


# --------------------------------------------------------------------------- #
# prompt assembly


def build_system_prompt(spec: AgentCallSpec) -> str:
    return "\n\n".join(b.text for b in spec.system_blocks if b.text).strip()


def build_prompt(spec: AgentCallSpec) -> str:
    """Flatten user blocks, any tool_use/tool_result thread, and tool rules."""
    parts = [b.text for b in spec.user_blocks if b.text]

    thread = _render_thread(spec.extra_messages)
    if thread:
        parts.append(thread)

    instructions = _tool_instructions(spec)
    if instructions:
        parts.append(instructions)

    return "\n\n".join(p for p in parts if p).strip()


def _tool_instructions(spec: AgentCallSpec) -> str:
    """Stand in for the API's `tool_choice`, which headless mode lacks."""
    record_tools = [t["name"] for t in spec.tools if _is_record_tool(t)]
    if not record_tools:
        return ""

    forced = None
    if spec.tool_choice and spec.tool_choice.get("type") == "tool":
        forced = spec.tool_choice.get("name")

    target = forced or record_tools[0]
    lines = [
        "## Required output",
        f"You MUST finish this task by calling the `{target}` tool exactly once.",
        "That tool call IS your answer — prose alone does not count and will be "
        "discarded. Call it even if your findings are partial or negative.",
    ]
    if len(record_tools) > 1 and forced is None:
        lines.append(f"Available recording tools: {', '.join(record_tools)}.")
    return "\n".join(lines)


def _render_thread(messages: list[dict[str, Any]]) -> str:
    """Flatten an Anthropic-style message thread into readable transcript text.

    Only Ranking's debate path uses `extra_messages`; the CLI has no notion of
    a partial assistant turn, so we present it as prior context.
    """
    out: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).capitalize()
        content = msg.get("content")
        if isinstance(content, str):
            out.append(f"### {role}\n{content}")
            continue
        chunks: list[str] = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                chunks.append(str(block.get("text", "")))
            elif btype == "tool_use":
                chunks.append(
                    f"[called {block.get('name')} with "
                    f"{json.dumps(block.get('input', {}), default=str)[:2000]}]"
                )
            elif btype == "tool_result":
                body = block.get("content")
                text = body if isinstance(body, str) else json.dumps(body, default=str)
                chunks.append(f"[tool result]\n{text[:8000]}")
        if chunks:
            out.append(f"### {role}\n" + "\n\n".join(chunks))
    return "\n\n".join(out)


def _is_record_tool(tool: dict[str, Any]) -> bool:
    return str(tool.get("name", "")).startswith("record_")


# --------------------------------------------------------------------------- #
# output parsing helpers


def _extract_json_object(stdout: str) -> dict[str, Any] | None:
    """Pull the result object out of stdout.

    The CLI prints a single JSON object under `--output-format json`, but a
    stray warning line ahead of it should not lose us the result.
    """
    text = stdout.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        try:
            parsed = json.loads(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ClaudeCliProvider",
    "CliBackendError",
    "build_prompt",
    "build_system_prompt",
    "effort_for_thinking_tokens",
]
