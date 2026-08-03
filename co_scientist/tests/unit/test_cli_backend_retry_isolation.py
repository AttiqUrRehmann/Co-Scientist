"""A retry must not inherit the abandoned attempt's captured records.

`AgentCliProvider.call` used to make one capture directory per *call* and let
`_run_with_retry` loop inside it. Because the MCP server numbers records per
process from 0001, a retry overwrote the low numbers and anything the failed
attempt had written beyond that survived into the result — so a rate-limited
attempt that had already recorded two hypotheses leaked one into the answer.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from co_scientist.config import Config
from co_scientist.llm.budgets import TokenBudget
from co_scientist.llm.cli_backend.claude_code import ClaudeCliProvider
from co_scientist.llm.routing import ModelRoute
from co_scientist.llm.types import AgentCallSpec, CachedBlock, CallContext

RECORD_TOOL = {
    "name": "record_hypothesis",
    "description": "record it",
    "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
}

# Attempt 1 writes two records, then reports a usage limit (retryable).
# Attempt 2 writes one record and succeeds. The call must yield exactly that one.
FAKE_CLI = r'''
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]
if not sys.stdin.isatty():
    sys.stdin.read()

state = Path(os.environ["FAKE_STATE"])
n = int(state.read_text()) if state.exists() else 0
state.write_text(str(n + 1))

cap = None
raw = argv[argv.index("--mcp-config") + 1]
for server in (json.loads(raw).get("mcpServers") or {}).values():
    cap = Path(server["env"]["COSCI_CAPTURE_DIR"])

usage = {"input_tokens": 10, "output_tokens": 5,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

if n == 0:
    for i, title in enumerate(["abandoned-A", "abandoned-B"], start=1):
        (cap / f"record-{i:04d}-record_hypothesis.json").write_text(json.dumps(
            {"tool": "record_hypothesis", "seq": i, "input": {"title": title}}))
    print(json.dumps({"is_error": True, "subtype": "error_during_execution",
                      "result": "Claude usage limit reached. Try again later.",
                      "num_turns": 1, "usage": usage, "total_cost_usd": 0.0}))
    raise SystemExit(0)

(cap / "record-0001-record_hypothesis.json").write_text(json.dumps(
    {"tool": "record_hypothesis", "seq": 1, "input": {"title": "the-real-answer"}}))
print(json.dumps({"is_error": False, "subtype": "success", "stop_reason": "end_turn",
                  "num_turns": 1, "usage": usage, "total_cost_usd": 0.0,
                  "result": "done"}))
'''


@pytest.fixture
def flaky_claude(tmp_path: Path) -> Path:
    script = tmp_path / "flaky.py"
    script.write_text(FAKE_CLI, encoding="utf-8")
    launcher = tmp_path / "flaky-claude"
    launcher.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
    )
    launcher.chmod(0o755)
    return launcher


@pytest.mark.asyncio
async def test_retry_does_not_inherit_records_from_the_failed_attempt(
    tmp_cfg: Config, conn, flaky_claude: Path, tmp_path: Path, monkeypatch
) -> None:
    from co_scientist.models import ResearchPlan, Session
    from co_scientist.storage.repos import sessions as sess_repo

    monkeypatch.setenv("FAKE_STATE", str(tmp_path / "attempts.txt"))
    tmp_cfg.llm.provider = "claude_cli"
    tmp_cfg.llm.claude_cli.binary = str(flaky_claude)
    tmp_cfg.llm.claude_cli.timeout_seconds = 30

    now = datetime.now(UTC)
    await sess_repo.insert(conn, Session(
        id="ses_r", created_at=now, updated_at=now, status="running",
        research_goal="g", research_plan=ResearchPlan(objective="g"),
        config_snapshot={}, budget_tokens=1_000_000, budget_usd=10.0,
        wall_deadline=now + timedelta(hours=1),
    ))
    budget = TokenBudget(cfg=tmp_cfg, budget_tokens=1_000_000, budget_usd=10.0)
    provider = ClaudeCliProvider(tmp_cfg, db=conn, budget=budget)

    # Rate-limit backoff has a 30s floor of its own; don't sleep through it.
    import co_scientist.llm.cli_backend.base as base_mod
    monkeypatch.setattr(base_mod, "backoff_seconds", lambda *a, **k: 0.0)

    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="opus"),
        system_blocks=[CachedBlock("sys")],
        user_blocks=[CachedBlock("go")],
        tools=[RECORD_TOOL],
        tool_choice={"type": "tool", "name": "record_hypothesis"},
        max_output_tokens=512,
    )
    ctx = CallContext(
        session_id="ses_r", task_id=None, agent="generation", action="t",
    )

    resp = await provider.call(spec, ctx)

    titles = [p.get("title") for _, p in resp.capture.records]
    assert titles == ["the-real-answer"], (
        f"retry inherited records from the abandoned attempt: {titles}"
    )
