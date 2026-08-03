"""Tests for the MCP capture + tools server.

The server is the load-bearing piece of the CLI backends: it is how a
subscription-driven agent returns structured output and how URL provenance
survives. These tests exercise it in-process — no `claude` subprocess, no
subscription quota.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from co_scientist.mcp.protocol import (
    METHOD_NOT_FOUND,
    McpStdioServer,
    text_result,
)
from co_scientist.mcp.server import (
    CoScientistMcpServer,
    validate_against_schema,
)
from co_scientist.tools.base import ToolResult
from co_scientist.tools.urls import extract_urls


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CoScientistMcpServer:
    monkeypatch.setenv("COSCI_CAPTURE_DIR", str(tmp_path / "cap"))
    monkeypatch.setenv("COSCI_RECORD_TOOLS", "record_hypothesis,record_research_plan")
    monkeypatch.setenv("COSCI_AGENT", "generation")
    monkeypatch.setenv("COSCI_SESSION_ID", "ses_test")
    return CoScientistMcpServer()


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# tool listing


def test_lists_requested_record_tools_and_agent_research_tools(
    server: CoScientistMcpServer,
) -> None:
    names = {t["name"] for t in server.tool_list()}
    assert {"record_hypothesis", "record_research_plan"} <= names
    # generation's allowlist in tools/registry.AGENT_TOOLS
    assert {"pubmed_search", "arxiv_search", "europe_pmc_search", "web_fetch"} <= names
    # record_review was not requested for this agent
    assert "record_review" not in names


def test_tool_list_uses_mcp_camelcase_schema_key(server: CoScientistMcpServer) -> None:
    for tool in server.tool_list():
        assert "inputSchema" in tool, tool["name"]
        assert "input_schema" not in tool


# --------------------------------------------------------------------------- #
# record capture


async def test_valid_record_is_written_to_capture_dir(
    server: CoScientistMcpServer,
) -> None:
    payload = {
        "title": "SCFA depletion drives barrier loss",
        "statement": "Loss of butyrate producers weakens the gut barrier.",
        "mechanism": "Reduced butyrate starves colonocytes, loosening tight junctions.",
        "entities": ["butyrate", "colonocyte"],
        "anticipated_outcomes": "Elevated serum LPS in depleted cohorts.",
        "novelty_argument": "Prior work reports correlation, not causation.",
        "citations": [{"url": "https://example.org/a", "title": "A"}],
    }
    result = await server.handle("record_hypothesis", payload)

    assert result["isError"] is False
    files = list(server.capture_dir.glob("record-*-record_hypothesis.json"))
    assert len(files) == 1
    written = json.loads(files[0].read_text())
    assert written["tool"] == "record_hypothesis"
    assert written["input"] == payload


async def test_invalid_record_is_rejected_with_specific_problems(
    server: CoScientistMcpServer,
) -> None:
    result = await server.handle("record_hypothesis", {"title": "only a title"})

    assert result["isError"] is True
    body = json.loads(_text(result))
    assert body["error"] == "schema validation failed"
    # The model needs to know *what* was wrong to fix it on retry.
    assert any("statement" in p for p in body["problems"])
    assert not list(server.capture_dir.glob("record-*.json"))


async def test_records_are_numbered_in_call_order(server: CoScientistMcpServer) -> None:
    plan = {"objective": "o", "preferences": [], "idea_attributes": []}
    await server.handle("record_research_plan", plan)
    await server.handle("record_research_plan", plan)

    names = sorted(p.name for p in server.capture_dir.glob("record-*.json"))
    assert names == [
        "record-0001-record_research_plan.json",
        "record-0002-record_research_plan.json",
    ]


async def test_unknown_tool_reports_what_is_available(
    server: CoScientistMcpServer,
) -> None:
    result = await server.handle("record_nonsense", {})

    assert result["isError"] is True
    body = json.loads(_text(result))
    assert "record_hypothesis" in body["available"]


# --------------------------------------------------------------------------- #
# research tools + URL provenance


async def test_research_tool_call_logs_urls_for_citation_verification(
    server: CoScientistMcpServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_call(args: dict[str, Any], ctx: Any) -> ToolResult:
        return ToolResult(
            content={"results": [
                {"title": "Paper A", "url": "https://pubmed.example/1"},
                {"title": "Paper B", "url": "https://pubmed.example/2"},
            ]},
            duration_ms=7,
        )

    monkeypatch.setattr(server.research_tools["pubmed_search"], "call", fake_call)
    result = await server.handle("pubmed_search", {"query": "microbiome"})

    assert result["isError"] is False
    entries = [
        json.loads(line)
        for line in (server.capture_dir / "tools.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 1
    assert entries[0]["name"] == "pubmed_search"
    assert set(entries[0]["urls"]) == {
        "https://pubmed.example/1", "https://pubmed.example/2",
    }


async def test_failing_research_tool_is_logged_and_reported_not_raised(
    server: CoScientistMcpServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(args: dict[str, Any], ctx: Any) -> ToolResult:
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(server.research_tools["arxiv_search"], "call", boom)
    result = await server.handle("arxiv_search", {"query": "x"})

    assert result["isError"] is True
    assert "upstream 503" in _text(result)
    entries = (server.capture_dir / "tools.jsonl").read_text().splitlines()
    assert json.loads(entries[0])["is_error"] is True


async def test_research_tool_timeout_is_bounded(
    server: CoScientistMcpServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    async def hang(args: dict[str, Any], ctx: Any) -> ToolResult:
        await asyncio.sleep(10)
        return ToolResult()

    server.tool_timeout = 0.05
    monkeypatch.setattr(server.research_tools["web_fetch"], "call", hang)
    result = await server.handle("web_fetch", {"url": "https://example.org"})

    assert result["isError"] is True
    assert "timed out" in _text(result)


# --------------------------------------------------------------------------- #
# JSON-RPC transport


async def test_dispatch_handles_initialize_list_and_call() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def handler(name: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, args))
        return text_result({"ok": True})

    transport = McpStdioServer(
        name="cosci", version="0.1.0",
        tools=[{"name": "t", "description": "d", "inputSchema": {"type": "object"}}],
        handler=handler,
    )

    init = await transport._dispatch("initialize", {"protocolVersion": "2025-06-18"})
    assert init["protocolVersion"] == "2025-06-18"
    assert init["serverInfo"]["name"] == "cosci"

    listed = await transport._dispatch("tools/list", {})
    assert [t["name"] for t in listed["tools"]] == ["t"]

    await transport._dispatch("tools/call", {"name": "t", "arguments": {"a": 1}})
    assert calls == [("t", {"a": 1})]


async def test_dispatch_falls_back_on_unknown_protocol_version() -> None:
    transport = McpStdioServer(
        name="cosci", version="0.1.0", tools=[], handler=_never_called,
    )
    init = await transport._dispatch("initialize", {"protocolVersion": "1999-01-01"})
    assert init["protocolVersion"] == "2025-06-18"


async def test_dispatch_rejects_unknown_method() -> None:
    from co_scientist.mcp.protocol import _RpcError

    transport = McpStdioServer(
        name="cosci", version="0.1.0", tools=[], handler=_never_called,
    )
    with pytest.raises(_RpcError) as e:
        await transport._dispatch("resources/list", {})
    assert e.value.code == METHOD_NOT_FOUND


async def _never_called(name: str, args: dict[str, Any]) -> dict[str, Any]:
    raise AssertionError("handler should not be called")


# --------------------------------------------------------------------------- #
# validation helper


def test_fallback_validator_matches_jsonschema_on_our_record_schemas() -> None:
    """The no-jsonschema path must still catch missing required fields."""
    from co_scientist.mcp.server import _fallback_validate

    schema = {
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "preferences": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["objective", "preferences"],
    }
    assert _fallback_validate({"objective": "x", "preferences": []}, schema, path="$") == []
    assert _fallback_validate({"objective": "x"}, schema, path="$")
    assert _fallback_validate({"objective": 1, "preferences": []}, schema, path="$")
    assert _fallback_validate({"objective": "x", "preferences": [1]}, schema, path="$")


def test_enum_violation_is_caught() -> None:
    schema = {"type": "object", "properties": {"v": {"type": "string", "enum": ["a", "b"]}}}
    assert validate_against_schema({"v": "a"}, schema) == []
    assert validate_against_schema({"v": "z"}, schema)


def test_extract_urls_walks_nested_tool_results() -> None:
    body = {
        "results": [
            {"url": "https://a.example", "meta": {"pdf_url": "https://b.example"}},
            {"title": "no url here"},
        ],
        "next": {"abs_url": "https://c.example"},
        "ignored": "https://not-a-url-key.example",
    }
    assert set(extract_urls(body)) == {
        "https://a.example", "https://b.example", "https://c.example",
    }
