"""Adapter over the backend's own agentic loop.

This module used to drive the assistant↔tool_use↔tool_result cycle by hand:
call the API, dispatch any tool_use blocks, feed tool_result back, repeat.
The CLI backends run that loop internally — they *are* agent harnesses — so
there is nothing left to drive. What the agents still need is what the loop
used to give them:

- the final assistant message (`response`)
- which tools ran (`tool_calls`)
- every URL seen in a tool result (`seen_urls`), the allowlist the citation
  verifier checks against

All three come back in `LLMResponse.capture`, populated by the MCP tool server.
The signature and result shape are unchanged, so `generation.py`,
`reflection.py`, and `evolution.py` call this exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger
from ..tools.registry import ToolRegistry
from .provider import LLMProvider
from .types import AgentCallSpec, CallContext, LLMResponse

log = get_logger("llm.tool_loop")

# Structured-output tools. A response is only complete once one of these has
# been captured — prose alone is not an answer.
DEFAULT_TERMINAL_TOOLS: tuple[str, ...] = (
    "record_hypothesis",
    "record_review",
    "record_system_feedback",
    "record_rubric_score",
    "record_research_plan",
    "record_safety_assessment",
)


class ToolLoopExhausted(RuntimeError):
    """The backend finished without producing a required structured record."""

    def __init__(self, agent: str, iters: int):
        super().__init__(
            f"tool loop for agent {agent!r} produced no record after {iters} attempts"
        )
        self.agent = agent
        self.iters = iters


@dataclass
class ToolLoopResult:
    response: LLMResponse                        # final assistant message
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    """Union of URLs that appeared in any tool result over the call.

    Used by structured-output validation to reject hallucinated citations:
    Generation's record_hypothesis.citations[].url must be in this set.
    """


async def run_tool_loop(
    client: LLMProvider,
    *,
    spec: AgentCallSpec,
    ctx: CallContext,
    registry: ToolRegistry,
    max_iters: int,
    parallel_cap: int = 4,
    tool_timeout_s: float = 30.0,
    force_terminal_tool: str | None = None,
    terminal_tool_names: tuple[str, ...] = DEFAULT_TERMINAL_TOOLS,
) -> ToolLoopResult:
    """Run one backend call and return its captured work.

    `registry`, `parallel_cap`, and `tool_timeout_s` are now enforced inside
    the MCP tool server (which is the process that actually runs the tools);
    they stay in the signature so agent call sites are untouched.

    Retry policy: if the backend returns without any terminal record, we make
    one more attempt with an escalated instruction before giving up. That
    replaces the old `force_terminal_tool` trick of forcing `tool_choice` on
    the final iteration — headless CLIs have no such flag.
    """
    terminal = set(terminal_tool_names)
    attempts = max(1, min(2, max_iters))
    last: LLMResponse | None = None

    for attempt in range(1, attempts + 1):
        call_spec = spec if attempt == 1 else _escalate(spec, force_terminal_tool, terminal)
        resp = await client.call(call_spec, ctx)
        last = resp

        capture = resp.capture
        tool_calls = list(capture.tool_calls) if capture else []
        seen_urls = set(capture.seen_urls) if capture else set()
        records = capture.records if capture else []

        for name, payload in records:
            tool_calls.append(
                {"name": name, "args": payload, "is_error": False, "duration_ms": 0}
            )

        got_record = any(name in terminal for name, _ in records)
        if got_record or not _requires_record(spec, terminal):
            return ToolLoopResult(
                response=resp,
                iterations=resp.num_turns or attempt,
                tool_calls=tool_calls,
                seen_urls=seen_urls,
            )

        log.warning(
            "no_record_captured",
            agent=ctx.agent, action=ctx.action, attempt=attempt,
            turns=resp.num_turns, tool_calls=len(tool_calls),
        )

    assert last is not None
    raise ToolLoopExhausted(ctx.agent, attempts)


# --------------------------------------------------------------------------- #
# helpers


def _requires_record(spec: AgentCallSpec, terminal: set[str]) -> bool:
    """Does this call expect a structured record at all?"""
    return any(str(t.get("name", "")) in terminal for t in spec.tools)


def _escalate(
    spec: AgentCallSpec, force_terminal_tool: str | None, terminal: set[str]
) -> AgentCallSpec:
    """Re-issue the call, demanding the record and nothing else.

    Dropping the research tools is deliberate: the failure mode this recovers
    from is a model that keeps searching and never commits, so the retry
    removes the option to search again.
    """
    target = force_terminal_tool or next(
        (str(t["name"]) for t in spec.tools if str(t.get("name", "")) in terminal),
        None,
    )
    record_tools = [t for t in spec.tools if str(t.get("name", "")) in terminal]
    demand = (
        "\n\nYour previous attempt ended without recording a result. Do not run "
        "any further searches. Using what you already know, call "
        f"`{target}` now with your best answer."
    )
    user_blocks = list(spec.user_blocks)
    if user_blocks:
        from .types import CachedBlock

        user_blocks[-1] = CachedBlock(
            text=user_blocks[-1].text + demand, cache=user_blocks[-1].cache
        )

    return AgentCallSpec(
        route=spec.route,
        system_blocks=spec.system_blocks,
        user_blocks=user_blocks,
        tools=record_tools or spec.tools,
        tool_choice={"type": "tool", "name": target} if target else spec.tool_choice,
        max_output_tokens=spec.max_output_tokens,
        stop_sequences=spec.stop_sequences,
        extra_messages=spec.extra_messages,
    )
