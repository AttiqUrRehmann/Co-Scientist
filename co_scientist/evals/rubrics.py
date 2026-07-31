"""LLM-as-judge rubric scoring.

The judge is a separate call (defaults to Sonnet) that takes:
- a candidate artifact (hypothesis record / review record / final overview)
- a rubric: list of criteria with name, weight, scoring guidance

and returns per-criterion 1-5 scores + a total. We do NOT use the same model
as the agent under test, to reduce echo-judge bias.

Like everything else, it runs through the configured agent CLI on a
subscription — the judge is deliberately cheap to run so evals stay routine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents.schemas import RECORD_RUBRIC_SCORE_TOOL
from ..config import Config


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    weight: float = 1.0
    guidance: str = ""


# Historical alias; the schema itself lives with the other record_* tools so
# the MCP server can serve it.
JUDGE_TOOL: dict[str, Any] = RECORD_RUBRIC_SCORE_TOOL


def weighted_total(rubric: list[RubricCriterion], scores: list[dict[str, Any]]) -> float:
    by_name = {s["name"]: int(s["score"]) for s in scores}
    num = 0.0
    den = 0.0
    for c in rubric:
        if c.name in by_name:
            num += c.weight * by_name[c.name]
            den += c.weight * 5.0
    return (num / den) if den > 0 else 0.0


async def judge(
    cfg: Config,
    *,
    rubric: list[RubricCriterion],
    candidate: str,
    label: str,
) -> dict[str, Any]:
    """Issue one judge call. Returns {scores: [...], weighted: float}.

    Routes through whichever LLM provider is configured in `cfg.llm.provider`.
    No retries here — the eval runner aggregates over many fixtures, so a
    single flaky judgment is noise we accept.
    """
    rubric_text = "\n".join(
        f"- {c.name} (weight {c.weight}): {c.guidance}" for c in rubric
    )
    system = (
        "You are a calibrated evaluator. Score the candidate against each "
        "criterion on a 1-5 integer scale with 1 = poor and 5 = excellent. "
        "Be parsimonious; reserve 5 for exemplary work. Always call "
        "record_rubric_score."
    )
    user = (
        f"Candidate to evaluate (label={label}):\n\n"
        f"<CANDIDATE>\n{candidate[:12_000]}\n</CANDIDATE>\n\n"
        f"Rubric:\n{rubric_text}"
    )

    return await _judge_via_backend(cfg, system=system, user=user, rubric=rubric)


async def _judge_via_backend(
    cfg: Config, *, system: str, user: str, rubric: list[RubricCriterion]
) -> dict[str, Any]:
    """Score via the configured agent CLI.

    Runs with `db=None`: a judgement is a one-off with no session to attach a
    transcript to, and the eval runner aggregates over many fixtures anyway.
    """
    from ..llm.budgets import TokenBudget
    from ..llm.provider import get_provider
    from ..llm.routing import route
    from ..llm.types import AgentCallSpec, CachedBlock, CallContext

    budget = TokenBudget(
        cfg=cfg, budget_tokens=cfg.run.budget_tokens, budget_usd=cfg.run.budget_usd,
    )
    try:
        provider = get_provider(cfg, db=None, budget=budget)
    except Exception as e:  # backend missing / not signed in
        return {"scores": [], "weighted": 0.0, "notes": f"judge unavailable: {e}"}

    spec = AgentCallSpec(
        route=route(cfg, "judge"),
        system_blocks=[CachedBlock(system, cache=True)],
        user_blocks=[CachedBlock(user)],
        tools=[RECORD_RUBRIC_SCORE_TOOL],
        tool_choice={"type": "tool", "name": "record_rubric_score"},
        max_output_tokens=1024,
    )
    ctx = CallContext(
        session_id="eval", task_id=None, agent="judge", action="RubricScore",
    )

    try:
        resp = await provider.call(spec, ctx)
    except Exception as e:
        return {"scores": [], "weighted": 0.0, "notes": f"judge call failed: {e}"}

    for name, payload in (resp.capture.records if resp.capture else []):
        if name == "record_rubric_score":
            scores = payload.get("scores", [])
            return {
                "scores": scores,
                "weighted": weighted_total(rubric, scores),
                "notes": payload.get("overall_notes", ""),
            }
    return {"scores": [], "weighted": 0.0, "notes": "no rubric score recorded"}


# Pre-built rubrics for the four agents that have measurable outputs.

GENERATION_RUBRIC = [
    RubricCriterion("novelty", 1.0,
                    "Differs meaningfully from established literature."),
    RubricCriterion("specificity", 1.0,
                    "Names concrete entities, mechanisms, expected outcomes."),
    RubricCriterion("citation_grounding", 1.0,
                    "Citations support the claims; URLs look real and relevant."),
    RubricCriterion("testability", 1.0,
                    "Proposes a measurable, near-term experiment."),
]

REFLECTION_RUBRIC = [
    RubricCriterion("assumption_decomposition", 1.0,
                    "Breaks the hypothesis into testable assumptions."),
    RubricCriterion("evidence_quality", 1.0,
                    "Cites URLs with verbatim excerpts for each factual claim."),
    RubricCriterion("verdict_consistency", 1.0,
                    "The verdict matches the body of the review."),
]

RANKING_RUBRIC = [
    RubricCriterion("verdict_clarity", 1.0,
                    "Ends with 'better idea: 1' or 'better idea: 2'."),
    RubricCriterion("reasoning_quality", 1.0,
                    "Rationale references concrete differences, not vibes."),
    RubricCriterion("order_independence", 0.5,
                    "Verdict would not depend on which hypothesis was listed first."),
]

OVERVIEW_RUBRIC = [
    RubricCriterion("novelty", 1.0,
                    "Lead directions differ from boilerplate research summaries."),
    RubricCriterion("plausibility", 1.0,
                    "Mechanisms are physically / biologically reasonable."),
    RubricCriterion("testability", 1.0,
                    "Proposes concrete experiments for each direction."),
    RubricCriterion("specificity", 1.0,
                    "Entities, doses, timeframes are named."),
    RubricCriterion("diversity", 0.5,
                    "Top directions are meaningfully distinct."),
    RubricCriterion("citation_honesty", 1.0,
                    "URLs cited actually exist and are relevant to the claim."),
]
