"""Minimal MCP stdio transport (JSON-RPC 2.0, newline-delimited).

We implement the handful of methods the CLI agents actually call rather than
depending on the `mcp` SDK: this server ships inside the co-scientist process
tree, is launched once per LLM call, and only ever needs `initialize`,
`tools/list`, and `tools/call`.

Transport rules that matter:
- One JSON object per line on stdin/stdout.
- stdout carries protocol traffic ONLY. Anything else (logs, warnings,
  tracebacks) must go to stderr or the client will fail to parse the stream.
- Notifications (no `id`) never get a response.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from collections.abc import Awaitable, Callable
from typing import Any

# How often the orphan watchdog checks whether our launcher is still alive.
ORPHAN_CHECK_SECONDS = 5.0

# Protocol revisions we know how to speak. We echo back the client's version
# when we recognize it, else fall back to the newest we support.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def log(msg: str) -> None:
    """Diagnostics go to stderr — stdout is reserved for protocol frames."""
    print(msg, file=sys.stderr, flush=True)


class McpStdioServer:
    """Serves a fixed tool list over MCP stdio.

    `tools` is the `tools/list` payload (each entry needs `name`,
    `description`, `inputSchema`). `handler` runs one `tools/call` and returns
    an MCP result dict: `{"content": [...], "isError": bool}`.
    """

    def __init__(
        self,
        *,
        name: str,
        version: str,
        tools: list[dict[str, Any]],
        handler: ToolHandler,
    ) -> None:
        self._name = name
        self._version = version
        self._tools = tools
        self._handler = handler
        self._out_lock = asyncio.Lock()

    # ----------------------------- run loop ----------------------------- #

    async def serve(self) -> None:
        pending: set[asyncio.Task[None]] = set()
        watchdog = asyncio.create_task(_exit_when_orphaned())
        try:
            await self._read_loop(pending)
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _read_loop(self, pending: set[asyncio.Task[None]]) -> None:
        while True:
            # A thread hop rather than connect_read_pipe: the latter rejects a
            # regular file, which is exactly what tests and `< input.jsonl`
            # hand us. Threads read pipes and files alike.
            line = await asyncio.to_thread(sys.stdin.buffer.readline)
            if not line:
                return                     # client closed stdin → shut down
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            # Each message is handled as its own task so a slow tool call
            # (a 30s literature search) doesn't stall protocol traffic.
            t = asyncio.create_task(self._handle_line(text))
            pending.add(t)
            t.add_done_callback(pending.discard)

    async def _handle_line(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError as e:
            await self._send({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": PARSE_ERROR, "message": f"invalid JSON: {e}"},
            })
            return

        if not isinstance(msg, dict):
            return
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        # Notification: no id → never respond (this includes
        # notifications/initialized and notifications/cancelled).
        if msg_id is None:
            return

        try:
            result = await self._dispatch(str(method), params)
        except _RpcError as e:
            await self._send({
                "jsonrpc": "2.0", "id": msg_id,
                "error": {"code": e.code, "message": e.message},
            })
            return
        except Exception as e:
            log(f"[mcp] internal error in {method}: {e!r}")
            await self._send({
                "jsonrpc": "2.0", "id": msg_id,
                "error": {"code": INTERNAL_ERROR, "message": repr(e)},
            })
            return

        await self._send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    async def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = (
                requested
                if requested in SUPPORTED_PROTOCOL_VERSIONS
                else DEFAULT_PROTOCOL_VERSION
            )
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self._name, "version": self._version},
            }

        if method == "ping":
            return {}

        if method == "tools/list":
            return {"tools": self._tools}

        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                raise _RpcError(INVALID_PARAMS, "tools/call requires a string `name`")
            args = params.get("arguments")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise _RpcError(INVALID_PARAMS, "`arguments` must be an object")
            return await self._handler(name, args)

        raise _RpcError(METHOD_NOT_FOUND, f"unsupported method: {method}")

    async def _send(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str)
        async with self._out_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def _exit_when_orphaned() -> None:
    """Exit if the process that launched us goes away.

    We are a grandchild: the co-scientist backend spawns the agent CLI, which
    spawns us. If the CLI dies without closing our stdin, the blocking
    `readline` never returns and we linger — holding the inherited stdout
    write end open and wedging whoever is reading it. Being reparented to
    init (ppid 1) is the reliable signal that has happened.
    """
    original_ppid = os.getppid()
    while True:
        await asyncio.sleep(ORPHAN_CHECK_SECONDS)
        current = os.getppid()
        if current != original_ppid or current == 1:
            log("[mcp] launcher exited; shutting down")
            os._exit(0)          # blocked in a reader thread — cannot unwind


def text_result(body: Any, *, is_error: bool = False) -> dict[str, Any]:
    """Wrap a Python value as an MCP `tools/call` result."""
    text = body if isinstance(body, str) else json.dumps(
        body, ensure_ascii=False, default=str
    )
    return {"content": [{"type": "text", "text": text}], "isError": is_error}
