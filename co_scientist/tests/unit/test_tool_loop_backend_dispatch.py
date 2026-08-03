"""`run_tool_loop` must pick its mode from the real backends, not from a fake.

The two suites either side of this one (`test_tool_loop_driven`,
`test_tool_loop_captured`) each exercise one mode through a stand-in client that
declares the mode under test. That leaves the dispatch itself unverified: if
`ClaudeCliProvider` lost `runs_own_loop`, the CLI backend would silently be
driven — one call, no captured records read, an empty result rather than an
error. And if the flag were ever read with a plain truth test, a `MagicMock`
client (which answers every getattr truthily) would route an API provider into
captured mode.

So this drives the genuine providers: a `ClaudeCliProvider` over a fake
`claude` binary, and an `OpenAIClient` over a stubbed SDK.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from co_scientist.config import Config
from co_scientist.llm.budgets import TokenBudget
from co_scientist.llm.provider import CLI_BACKENDS, get_provider
from co_scientist.llm.routing import ModelRoute
from co_scientist.llm.tool_loop import run_tool_loop
from co_scientist.llm.types import AgentCallSpec, CachedBlock, CallContext

RECORD_TOOL = {
    "name": "record_hypothesis",
    "description": "record it",
    "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
}

HYPOTHESIS = {"title": "terazosin activates PGK1"}


@pytest.fixture
def fake_claude(tmp_path: Path) -> Path:
    """A `claude -p` stand-in that records a hypothesis and reports usage."""
    from co_scientist.tests.fakes import fake_claude as module

    launcher = tmp_path / "fake-claude"
    launcher.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{Path(module.__file__)}" "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


async def _session(conn, cfg: Config, sid: str) -> TokenBudget:
    from co_scientist.models import ResearchPlan, Session
    from co_scientist.storage.repos import sessions as sess_repo

    now = datetime.now(UTC)
    await sess_repo.insert(conn, Session(
        id=sid, created_at=now, updated_at=now, status="running",
        research_goal="g", research_plan=ResearchPlan(objective="g"),
        config_snapshot={}, budget_tokens=1_000_000, budget_usd=10.0,
        wall_deadline=now + timedelta(hours=1),
    ))
    return TokenBudget(cfg=cfg, budget_tokens=1_000_000, budget_usd=10.0)


def _spec() -> AgentCallSpec:
    return AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="claude-opus-4-7"),
        system_blocks=[CachedBlock("sys")],
        user_blocks=[CachedBlock("go")],
        tools=[RECORD_TOOL],
        tool_choice={"type": "tool", "name": "record_hypothesis"},
        max_output_tokens=512,
    )


def _ctx(sid: str) -> CallContext:
    return CallContext(session_id=sid, task_id=None, agent="generation", action="t")


@pytest.mark.asyncio
async def test_a_real_cli_backend_takes_the_captured_path(
    tmp_cfg: Config, conn, fake_claude: Path, monkeypatch
) -> None:
    tmp_cfg.llm.provider = "claude_cli"
    tmp_cfg.llm.claude_cli.binary = str(fake_claude)
    tmp_cfg.llm.claude_cli.timeout_seconds = 30
    monkeypatch.setenv("FAKE_CLI_BEHAVIOR", json.dumps({
        "records": [["record_hypothesis", HYPOTHESIS]],
        "tools": [{"name": "pubmed_search", "urls": ["https://pubmed.gov/1"]}],
    }))

    budget = await _session(conn, tmp_cfg, "ses_cli")
    client = get_provider(tmp_cfg, db=conn, budget=budget)
    assert client.runs_own_loop is True
    assert tmp_cfg.llm.provider in CLI_BACKENDS

    result = await run_tool_loop(
        client, spec=_spec(), ctx=_ctx("ses_cli"),
        registry=MagicMock(), max_iters=8,
        force_terminal_tool="record_hypothesis",
    )

    # Captured mode reads the MCP server's log rather than dispatching tools.
    assert [t["name"] for t in result.tool_calls] == [
        "pubmed_search", "record_hypothesis",
    ]
    assert result.seen_urls == {"https://pubmed.gov/1"}
    assert result.response.capture is not None
    record = next(
        b for b in result.response.raw.content
        if getattr(b, "type", None) == "tool_use"
    )
    assert record.name == "record_hypothesis"
    assert record.input == HYPOTHESIS


@pytest.mark.asyncio
async def test_a_real_api_provider_takes_the_driven_path(tmp_cfg: Config, conn) -> None:
    """The OpenAI client must be driven — and its `capture` must stay None."""
    tmp_cfg.llm.provider = "openai"
    tmp_cfg.secrets.OPENAI_API_KEY = "sk-fake"
    budget = await _session(conn, tmp_cfg, "ses_api")

    # One response: a tool_calls turn naming the recording tool. Driven mode
    # short-circuits on a terminal tool without dispatching it.
    tc = SimpleNamespace(
        id="call_1", type="function",
        function=SimpleNamespace(
            name="record_hypothesis", arguments=json.dumps(HYPOTHESIS),
        ),
    )
    raw = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=[tc]),
            finish_reason="tool_calls",
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model="gpt-5",
    )

    with patch("openai.AsyncOpenAI") as mock_sdk:
        inst = mock_sdk.return_value
        inst.chat = MagicMock()
        inst.chat.completions = MagicMock()
        create = AsyncMock(return_value=raw)
        inst.chat.completions.create = create
        client = get_provider(tmp_cfg, db=conn, budget=budget)
        assert getattr(client, "runs_own_loop", False) is False

        result = await run_tool_loop(
            client, spec=_spec(), ctx=_ctx("ses_api"),
            registry=MagicMock(), max_iters=8,
            force_terminal_tool="record_hypothesis",
        )

    assert create.await_count == 1, "driven mode short-circuits on a terminal tool"
    assert result.response.capture is None, "API responses carry no capture bundle"
    assert [t["name"] for t in result.tool_calls] == ["record_hypothesis"]
    assert result.tool_calls[0]["args"] == HYPOTHESIS


@pytest.mark.asyncio
async def test_a_duck_typed_client_is_driven_not_captured(tmp_cfg: Config) -> None:
    """A Mock answers every getattr truthily; that must not select captured mode.

    Captured mode on a client with no capture bundle returns an empty result
    instead of raising, so a truth test here would fail silently.
    """
    client = MagicMock()
    client.call = AsyncMock(return_value=SimpleNamespace(
        raw=SimpleNamespace(stop_reason="end_turn", content=[]),
        transcript_id="trn", cost_usd=0.0, input_tokens=0, output_tokens=0,
        cache_read=0, cache_write=0, num_turns=1, capture=None,
    ))
    assert bool(getattr(client, "runs_own_loop", False)) is True   # the trap

    result = await run_tool_loop(
        client, spec=_spec(), ctx=_ctx("ses_mock"),
        registry=MagicMock(), max_iters=3,
    )
    # Driven mode returns on stop_reason != "tool_use" after one real call.
    assert result.iterations == 1
    assert client.call.await_count == 1
