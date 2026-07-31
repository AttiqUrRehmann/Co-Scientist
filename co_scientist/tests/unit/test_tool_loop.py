"""Tests for the tool-loop adapter over a CLI backend's own agentic loop.

The adapter no longer drives turns — the CLI does that internally. What it
must still guarantee is the contract the agents depend on: a structured
record comes back, tool provenance is preserved, and a call that produces no
record is retried once and then fails loudly rather than silently returning
prose.
"""

from __future__ import annotations

from typing import Any

import pytest

from co_scientist.llm.routing import ModelRoute
from co_scientist.llm.tool_loop import ToolLoopExhausted, run_tool_loop
from co_scientist.llm.types import (
    AgentCallSpec,
    CachedBlock,
    CallContext,
    CaptureBundle,
    LLMResponse,
    SynthMessage,
    TextBlock,
    ToolUseBlock,
)

RECORD_TOOL = {"name": "record_hypothesis", "description": "", "input_schema": {}}
SEARCH_TOOL = {"name": "pubmed_search", "description": "", "input_schema": {}}


class FakeProvider:
    """Returns a scripted response per call and records the specs it saw."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.specs: list[AgentCallSpec] = []

    async def call(
        self, spec: AgentCallSpec, ctx: CallContext, *, est_input_tokens: int | None = None
    ) -> LLMResponse:
        self.specs.append(spec)
        return self._responses[min(len(self.specs) - 1, len(self._responses) - 1)]


def _response(
    *,
    records: list[tuple[str, dict[str, Any]]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    urls: set[str] | None = None,
    text: str = "done",
    num_turns: int = 3,
) -> LLMResponse:
    records = records or []
    capture = CaptureBundle(
        records=records,
        tool_calls=tool_calls or [],
        seen_urls=urls or set(),
    )
    content: list[Any] = [TextBlock(text=text)]
    content += [ToolUseBlock(name=n, input=p, id=f"c{i}") for i, (n, p) in enumerate(records)]
    return LLMResponse(
        raw=SynthMessage(content=content),
        transcript_id="trn_x",
        cost_usd=0.0, input_tokens=10, output_tokens=5,
        cache_read=0, cache_write=0,
        num_turns=num_turns,
        capture=capture,
    )


def _spec(tools: list[dict[str, Any]] | None = None) -> AgentCallSpec:
    return AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="opus"),
        user_blocks=[CachedBlock("go")],
        tools=tools if tools is not None else [SEARCH_TOOL, RECORD_TOOL],
        max_output_tokens=512,
    )


def _ctx() -> CallContext:
    return CallContext(session_id="s", task_id="t", agent="generation", action="a")


async def _run(provider: FakeProvider, spec: AgentCallSpec, **kw: Any):
    return await run_tool_loop(
        provider, spec=spec, ctx=_ctx(), registry=None, max_iters=8, **kw,
    )


# --------------------------------------------------------------------------- #


async def test_returns_immediately_when_a_record_is_captured() -> None:
    provider = FakeProvider([_response(records=[("record_hypothesis", {"title": "T"})])])

    result = await _run(provider, _spec())

    assert len(provider.specs) == 1
    assert result.response.num_turns == 3
    assert [c["name"] for c in result.tool_calls] == ["record_hypothesis"]


async def test_iterations_reflect_the_backends_own_turn_count() -> None:
    """The CLI runs the loop, so `iterations` reports its turns, not ours."""
    provider = FakeProvider([
        _response(records=[("record_hypothesis", {})], num_turns=7)
    ])

    result = await _run(provider, _spec())

    assert result.iterations == 7


async def test_tool_provenance_and_urls_survive() -> None:
    provider = FakeProvider([_response(
        records=[("record_hypothesis", {"title": "T"})],
        tool_calls=[{"name": "pubmed_search", "args": {"q": "x"},
                     "is_error": False, "duration_ms": 12}],
        urls={"https://pubmed.example/1"},
    )])

    result = await _run(provider, _spec())

    assert result.seen_urls == {"https://pubmed.example/1"}
    names = [c["name"] for c in result.tool_calls]
    assert names == ["pubmed_search", "record_hypothesis"]


async def test_missing_record_triggers_exactly_one_escalated_retry() -> None:
    provider = FakeProvider([
        _response(records=[], text="I searched a lot but did not commit."),
        _response(records=[("record_hypothesis", {"title": "finally"})]),
    ])

    result = await _run(provider, _spec())

    assert len(provider.specs) == 2
    assert result.tool_calls[-1]["name"] == "record_hypothesis"


async def test_escalation_demands_the_record_and_drops_search_tools() -> None:
    """The failure being recovered from is 'kept searching, never committed'."""
    provider = FakeProvider([
        _response(records=[]),
        _response(records=[("record_hypothesis", {})]),
    ])

    await _run(provider, _spec())

    retry_spec = provider.specs[1]
    assert [t["name"] for t in retry_spec.tools] == ["record_hypothesis"]
    assert retry_spec.tool_choice == {"type": "tool", "name": "record_hypothesis"}
    assert "did not commit" not in retry_spec.user_blocks[-1].text
    assert "call" in retry_spec.user_blocks[-1].text.lower()
    assert "record_hypothesis" in retry_spec.user_blocks[-1].text


async def test_persistent_failure_to_record_raises_exhausted() -> None:
    provider = FakeProvider([_response(records=[]), _response(records=[])])

    with pytest.raises(ToolLoopExhausted) as e:
        await _run(provider, _spec())

    assert e.value.agent == "generation"


async def test_calls_that_expect_no_record_return_prose_without_retrying() -> None:
    """Ranking's debate turns have no record_* tool and must not be retried."""
    provider = FakeProvider([_response(records=[], text="better idea: 1")])

    result = await _run(provider, _spec(tools=[]))

    assert len(provider.specs) == 1
    assert "better idea" in result.response.raw.content[0].text


async def test_force_terminal_tool_selects_the_escalation_target() -> None:
    provider = FakeProvider([
        _response(records=[]),
        _response(records=[("record_review", {})]),
    ])
    spec = _spec(tools=[
        SEARCH_TOOL,
        {"name": "record_review", "description": "", "input_schema": {}},
    ])

    await _run(provider, spec, force_terminal_tool="record_review")

    assert provider.specs[1].tool_choice == {"type": "tool", "name": "record_review"}


async def test_response_without_capture_is_tolerated() -> None:
    """A backend that reports no capture must not crash the adapter."""
    resp = LLMResponse(
        raw=SynthMessage(content=[TextBlock(text="hi")]),
        transcript_id="t", cost_usd=0.0, input_tokens=0, output_tokens=0,
        cache_read=0, cache_write=0, capture=None,
    )
    provider = FakeProvider([resp])

    result = await _run(provider, _spec(tools=[]))

    assert result.tool_calls == []
    assert result.seen_urls == set()
