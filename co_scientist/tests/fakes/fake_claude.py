"""A stand-in for `claude -p` that is deterministic and free.

It plays the part faithfully enough to exercise the real backend: it accepts
the same flags, reads the prompt from stdin, writes MCP-style records into the
capture directory named in `--mcp-config`, and prints a result object in
`--output-format json` shape.

Two environment variables drive it:

    FAKE_CLI_BEHAVIOR   JSON describing what to do (see BEHAVIOR keys below)
    FAKE_CLI_DUMP       path to write {argv, stdin, env} for assertions

BEHAVIOR keys:
    mode      "success" (default) | "error" | "rate_limit" | "garbage"
    text      the assistant's final text
    records   [[tool_name, payload], ...] written to the capture dir
    tools     [{name, args, urls, is_error}, ...] appended to tools.jsonl
    usage     override for the reported token counts
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_USAGE = {
    "input_tokens": 818,
    "output_tokens": 514,
    "cache_creation_input_tokens": 1278,
    "cache_read_input_tokens": 0,
}


def _behavior() -> dict[str, Any]:
    raw = os.environ.get("FAKE_CLI_BEHAVIOR")
    return json.loads(raw) if raw else {}


def _dump(argv: list[str], stdin: str) -> None:
    path = os.environ.get("FAKE_CLI_DUMP")
    if not path:
        return
    Path(path).write_text(
        json.dumps({"argv": argv, "stdin": stdin, "env": dict(os.environ)}),
        encoding="utf-8",
    )


def _capture_dir(argv: list[str]) -> Path | None:
    """Recover the capture directory the backend wired into --mcp-config."""
    if "--mcp-config" not in argv:
        return None
    raw = argv[argv.index("--mcp-config") + 1]
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        return None
    for server in (config.get("mcpServers") or {}).values():
        target = (server.get("env") or {}).get("COSCI_CAPTURE_DIR")
        if target:
            return Path(target)
    return None


def _write_capture(capture: Path, behavior: dict[str, Any]) -> None:
    capture.mkdir(parents=True, exist_ok=True)
    for i, (tool, payload) in enumerate(behavior.get("records") or [], start=1):
        (capture / f"record-{i:04d}-{tool}.json").write_text(
            json.dumps({"tool": tool, "seq": i, "input": payload}), encoding="utf-8"
        )
    calls = behavior.get("tools") or []
    if calls:
        with (capture / "tools.jsonl").open("a", encoding="utf-8") as f:
            for call in calls:
                f.write(json.dumps({
                    "name": call.get("name", "pubmed_search"),
                    "args": call.get("args", {}),
                    "is_error": bool(call.get("is_error")),
                    "duration_ms": int(call.get("duration_ms", 5)),
                    "urls": call.get("urls", []),
                }) + "\n")


def main() -> int:
    argv = sys.argv[1:]
    stdin = "" if sys.stdin.isatty() else sys.stdin.read()
    _dump(argv, stdin)

    behavior = _behavior()
    mode = behavior.get("mode", "success")

    if mode == "garbage":
        sys.stdout.write("not json at all\n")
        return 1

    capture = _capture_dir(argv)
    if capture is not None and mode == "success":
        _write_capture(capture, behavior)

    if mode == "rate_limit":
        payload = {
            "is_error": True, "subtype": "error_during_execution",
            "result": "Claude usage limit reached. Try again later.",
            "num_turns": 1, "usage": DEFAULT_USAGE, "total_cost_usd": 0.0,
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0

    if mode == "error":
        payload = {
            "is_error": True, "subtype": "error_during_execution",
            "result": behavior.get("text", "invalid model name"),
            "num_turns": 1, "usage": DEFAULT_USAGE, "total_cost_usd": 0.0,
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0

    usage = {**DEFAULT_USAGE, **(behavior.get("usage") or {})}
    payload = {
        "is_error": False,
        "subtype": "success",
        "stop_reason": "end_turn",
        "session_id": "fake-session",
        "num_turns": behavior.get("num_turns", 2),
        "total_cost_usd": behavior.get("total_cost_usd", 0.0178),
        "usage": usage,
        "result": behavior.get("text", "done"),
    }
    sys.stdout.write(json.dumps(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
