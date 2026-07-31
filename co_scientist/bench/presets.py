"""Built-in bench candidate presets.

Curated comparison setups so users can reproduce known benchmarks with one
flag instead of typing N `--candidate` lines.

What changed with the move to subscription CLIs
-----------------------------------------------
The historical `paper` presets compared the system against the Co-Scientist
paper's baselines (Gemini 2.0 Flash Thinking Experimental, Gemini 2.0 Pro
Experimental, OpenAI o1) by routing every candidate through OpenRouter. That
is no longer possible: candidates now run through an agent CLI on a
subscription, and no CLI serves third-party models. Those presets are gone
rather than silently broken.

The archived cross-vendor numbers in `docs/BENCH_RESULTS.md` were produced
under the previous API-key architecture and are kept for reference. They are
not reproducible with this code, and are not comparable to new runs: the
prompts still match, but the harness now delegates the tool loop to the CLI.

What survives is the science: the paper's AML drug-repurposing goal and its
top-3 gold set, which are the genuinely reusable part. Candidates are now
models the local CLIs can actually serve.

Judge model is configurable via --judge as "<backend>:<model>". The
recommended default is `claude_cli:sonnet`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .goldset import AML_REPURPOSING_PAPER_TOP3, GoldSet
from .runner import BenchCandidate


@dataclass(frozen=True)
class BenchPreset:
    name: str
    description: str
    candidates: tuple[BenchCandidate, ...]
    suggested_judge: str        # "provider:model"
    # Optional preset defaults — let the CLI invoke a preset without forcing
    # the user to retype the goal or attach a gold set.
    default_goal: str | None = None
    goldset: GoldSet | None = None


# Claude Code tiers. Spans the capability range so the bench still surfaces
# a quality gradient, even though nothing is billed per token any more.
_CLAUDE_CANDIDATES: tuple[BenchCandidate, ...] = (
    BenchCandidate(label="opus", provider="claude_cli", model="opus"),
    BenchCandidate(label="sonnet", provider="claude_cli", model="sonnet"),
    BenchCandidate(label="haiku", provider="claude_cli", model="haiku"),
)


# The AML drug-repurposing goal mirrors the exact methodology in the
# Co-Scientist paper:
#   - Ranked list of repurposing candidates for AML.
#   - Candidates must NOT have been previously repurposed for AML.
#   - There must be NO prior preclinical evidence supporting the candidate
#     in AML.
#   - The system uses only its internal knowledge — no DepMap dependency
#     scores, no genomic priors, no human expert feedback.
#
# We additionally tell models to NAME the specific drug (INN, brand, or
# research code) rather than a drug class so the gold-set matcher can
# score recall against named entities.
_PAPER_AML_GOAL = (
    "Produce a ranked list of drug repurposing candidates for acute "
    "myeloid leukemia (AML), strictly under the following constraints:\n\n"
    "(1) Each candidate must NOT have prior published evidence of being "
    "repurposed for AML, and there must be no preclinical studies in AML "
    "for the proposed compound at the time of writing.\n"
    "(2) Use only your internal knowledge. Do NOT assume access to DepMap "
    "dependency scores, gene-essentiality datasets, transcriptomic "
    "screens, or human expert curation. No external inputs.\n"
    "(3) Name the specific compound (INN, brand name, or research-code "
    "alias) — do not propose generic drug classes (e.g. \"MEK inhibitors\")."
    " Cover diverse mechanisms across the ranked list (avoid 5 ideas all "
    "hitting the same pathway).\n\n"
    "For each candidate hypothesis, give: the named compound, the "
    "molecular mechanism by which it would act against AML blasts or "
    "leukemic stem cells (with the specific target/pathway), the "
    "scientific reasoning that licenses the AML hypothesis even though "
    "no AML-specific evidence currently exists, and one concrete in vitro "
    "or in vivo experiment that would falsify the hypothesis."
)


def _vs_raw(candidates: tuple[BenchCandidate, ...]) -> tuple[BenchCandidate, ...]:
    """Double every candidate: once in pipeline mode, once in direct mode.

    This is the apples-to-apples comparison of co-scientist's multi-agent
    Generation harness against a raw single-shot LM call on the same goal.
    The label gets a `[pipe]` / `[raw]` suffix so the result table makes
    the distinction obvious.
    """
    out: list[BenchCandidate] = []
    for c in candidates:
        out.append(BenchCandidate(
            label=f"{c.label}[pipe]", provider=c.provider, model=c.model,
            mode="pipeline",
        ))
        out.append(BenchCandidate(
            label=f"{c.label}[raw]", provider=c.provider, model=c.model,
            mode="direct",
        ))
    return tuple(out)


# Cross-backend set. Codex requires a ChatGPT account entitled to the model —
# check with `codex exec -m <model>` before running this preset, since an
# unentitled model fails the run rather than degrading.
_CROSS_BACKEND_CANDIDATES: tuple[BenchCandidate, ...] = (
    BenchCandidate(label="claude-opus", provider="claude_cli", model="opus"),
    BenchCandidate(label="claude-sonnet", provider="claude_cli", model="sonnet"),
    BenchCandidate(label="codex", provider="codex_cli", model="gpt-5.6-codex"),
)


PRESETS: dict[str, BenchPreset] = {
    "claude-aml": BenchPreset(
        name="claude-aml",
        description=(
            "The paper's AML repurposing benchmark under its strict "
            "methodology: candidates with NO prior repurposing evidence and "
            "NO preclinical evidence in AML, no external inputs (DepMap, "
            "expert feedback). Recall is scored against the top-3 candidates "
            "the paper surfaced: Nanvuranlat, KIRA6, and Leflunomide. Runs "
            "the three Claude Code tiers."
        ),
        candidates=_CLAUDE_CANDIDATES,
        suggested_judge="claude_cli:sonnet",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
    "claude-aml-vs-raw": BenchPreset(
        name="claude-aml-vs-raw",
        description=(
            "AML repurposing benchmark where each Claude tier runs TWICE: "
            "once through the full co-scientist Generation pipeline "
            "(literature tools + dedup) and once as a single raw call. "
            "Isolates how much performance comes from the multi-agent "
            "harness versus the model itself. Same gold set as `claude-aml`."
        ),
        candidates=_vs_raw(_CLAUDE_CANDIDATES),
        suggested_judge="claude_cli:sonnet",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
    "cross-backend-aml": BenchPreset(
        name="cross-backend-aml",
        description=(
            "Claude Code versus Codex on the AML repurposing benchmark. "
            "Requires both CLIs installed and signed in, and a ChatGPT "
            "account entitled to the Codex model — verify with "
            "`co-scientist doctor` after setting [llm] provider = codex_cli."
        ),
        candidates=_CROSS_BACKEND_CANDIDATES,
        suggested_judge="claude_cli:sonnet",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
}


def get_preset(name: str) -> BenchPreset:
    try:
        return PRESETS[name]
    except KeyError as e:
        names = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown bench preset {name!r}; available: {names}") from e
