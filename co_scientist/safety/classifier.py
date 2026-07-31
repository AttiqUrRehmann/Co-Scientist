"""Abuse / dual-use classifier.

A small Haiku-backed classifier with a structured-output tool. Placement:
- goal-parse time (mandatory): block / warn / allow
- hypothesis-save time (mandatory): ok / quarantine / block
- final-report time (optional): ok / redact_quarantined / block_publish

Defensive use only. When the backend is unavailable we return `ok` and log a
warning rather than crashing the session — agents stay functional in dev.
"""

from __future__ import annotations

from typing import Any, Literal

from ..agents.schemas import RECORD_SAFETY_ASSESSMENT_TOOL
from ..config import Config
from ..logging import get_logger

log = get_logger("safety.classifier")


Action = Literal["allow", "warn", "block", "quarantine"]

# The schema lives with the other record_* tools so the MCP server can serve
# it to the CLI backends.
CLASSIFY_TOOL: dict[str, Any] = RECORD_SAFETY_ASSESSMENT_TOOL


_SYSTEM_PROMPT = (
    "You are a content-safety classifier for an AI research-hypothesis system. "
    "You will be shown a passage of text (a research goal, a hypothesis, or a "
    "synthesized report). Classify the content into one or more categories and "
    "report your confidence.\n\n"
    "Categories:\n"
    " - none: benign scientific content.\n"
    " - dual_use_bio: research that could plausibly be misused for biological "
    "harm (gain-of-function, enhancement of pathogen transmissibility, etc.).\n"
    " - cbrn: chemical, biological, radiological, or nuclear weapons.\n"
    " - weapons: conventional weapons synthesis or improvement.\n"
    " - illicit_synthesis: drug or precursor synthesis routes for unlawful use.\n"
    " - csam: child sexual abuse material.\n\n"
    "Be precise. Benign biology research is NOT dual_use_bio; only flag when the "
    "stated outcome could plausibly increase harm capability. When in doubt, "
    "use 'none' with low confidence. Always call the record_safety_assessment tool."
)


class ClassifierResult:
    def __init__(self, categories: list[str], confidence: float, rationale: str) -> None:
        self.categories = categories or ["none"]
        self.confidence = confidence
        self.rationale = rationale

    @property
    def is_benign(self) -> bool:
        return self.categories == ["none"] or ("none" in self.categories and len(self.categories) == 1)

    def action(self, cfg: Config) -> Action:
        if self.is_benign:
            return "allow"
        block = set(cfg.safety.classifier_block_categories)
        warn = set(cfg.safety.classifier_warn_categories)
        flagged = set(self.categories) - {"none"}
        if flagged & block:
            return "block"
        if flagged & warn and self.confidence >= 0.6:
            return "quarantine"
        if flagged & warn:
            return "warn"
        return "allow"


class SafetyClassifier:
    """One classifier per Config; lazily builds the backend on first use.

    Each classification is its own CLI call, so this runs on the cheapest
    configured model (`[models] classifier`, default "haiku") and is kept to a
    single turn with no research tools.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._provider: Any = None
        self._unavailable = False

    def _get_provider(self) -> Any:
        if self._provider is None and not self._unavailable:
            from ..llm.budgets import TokenBudget
            from ..llm.provider import get_provider

            try:
                self._provider = get_provider(
                    self._cfg,
                    db=None,
                    budget=TokenBudget(
                        cfg=self._cfg,
                        budget_tokens=self._cfg.run.budget_tokens,
                        budget_usd=self._cfg.run.budget_usd,
                    ),
                )
            except Exception as e:            # backend missing / not signed in
                log.warning("classifier_backend_unavailable", err=str(e))
                self._unavailable = True
        return self._provider

    async def classify(self, text: str, *, label: str = "input") -> ClassifierResult:
        """Always returns a result; degrades to benign + warning log on failure."""
        if not self._cfg.safety.enable_classifier:
            return ClassifierResult(categories=["none"], confidence=0.0,
                                    rationale="classifier disabled")
        text = text.strip()
        if not text:
            return ClassifierResult(categories=["none"], confidence=1.0,
                                    rationale="empty input")

        provider = self._get_provider()
        if provider is None:
            return ClassifierResult(categories=["none"], confidence=0.0,
                                    rationale="classifier backend unavailable")

        from ..llm.routing import route
        from ..llm.types import AgentCallSpec, CachedBlock, CallContext

        spec = AgentCallSpec(
            route=route(self._cfg, "classifier"),
            system_blocks=[CachedBlock(_SYSTEM_PROMPT, cache=True)],
            user_blocks=[CachedBlock(
                f'<TEXT label="{label}">\n{text[:8000]}\n</TEXT>'
            )],
            tools=[CLASSIFY_TOOL],
            tool_choice={"type": "tool", "name": "record_safety_assessment"},
            max_output_tokens=512,
        )
        ctx = CallContext(
            session_id="safety", task_id=None, agent="classifier", action="Classify",
        )

        try:
            resp = await provider.call(spec, ctx)
        except Exception as e:
            log.warning("classifier_call_failed", err=str(e))
            return ClassifierResult(categories=["none"], confidence=0.0,
                                    rationale=f"classifier_error: {e!s}")

        for name, payload in (resp.capture.records if resp.capture else []):
            if name == "record_safety_assessment":
                return ClassifierResult(
                    categories=list(payload.get("categories", ["none"])),
                    confidence=float(payload.get("confidence", 0.0)),
                    rationale=str(payload.get("rationale", "")),
                )
        return ClassifierResult(categories=["none"], confidence=0.0,
                                rationale="no safety assessment recorded")
