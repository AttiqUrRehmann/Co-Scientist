"""Meta-review citation provenance.

Meta-review runs with no tools, so it can only cite what the agent layer hands
it. Before this was wired up it received hypotheses and reviews stripped of
their URLs and — correctly — refused to invent any, producing a research
overview with no references at all. These tests pin the plumbing that fixes
that, and the rule that nothing unverified can sneak in.
"""

from __future__ import annotations

from datetime import UTC, datetime

from co_scientist.agents.metareview import MetaReviewAgent
from co_scientist.models import CitedPaper, Hypothesis, Review
from co_scientist.models.review import Evidence, ReviewScores


def _hyp(hid: str, citations: list[CitedPaper]) -> Hypothesis:
    return Hypothesis(
        id=hid, session_id="s", created_at=datetime.now(UTC),
        created_by="generation",
        title="t", summary="sum", full_text="body",
        strategy="literature", citations=citations,
        artifact_path=f"artifacts/{hid}.json",
    )


def _review(hid: str, evidence: list[Evidence]) -> Review:
    return Review(
        id=f"rev_{hid}", hypothesis_id=hid, session_id="s",
        created_at=datetime.now(UTC), kind="full", verdict="neutral",
        scores=ReviewScores(), evidence=evidence, body="b",
        artifact_path=f"artifacts/rev_{hid}.json",
    )


def test_citations_and_review_evidence_both_reach_the_source_block() -> None:
    hyps = [_hyp("hyp_a", [CitedPaper(title="Paper A", url="https://a.example")])]
    reviews = [_review("hyp_a", [
        Evidence(claim="claim B", url="https://b.example", excerpt="quote B"),
    ])]

    block = MetaReviewAgent._source_block(hyps, reviews)

    assert "https://a.example" in block
    assert "https://b.example" in block
    assert "Paper A" in block
    assert "quote B" in block


def test_sources_are_deduped_by_url_across_hypotheses_and_reviews() -> None:
    url = "https://shared.example"
    hyps = [
        _hyp("hyp_a", [CitedPaper(title="Shared", url=url)]),
        _hyp("hyp_b", [CitedPaper(title="Shared", url=url)]),
    ]
    reviews = [_review("hyp_a", [Evidence(claim="c", url=url, excerpt="e")])]

    block = MetaReviewAgent._source_block(hyps, reviews)

    assert block.count(url) == 1
    # …but the block still records every hypothesis the source supports.
    assert "hyp_a" in block and "hyp_b" in block


def test_sources_are_tagged_so_the_overview_can_reference_them() -> None:
    hyps = [
        _hyp("hyp_a", [CitedPaper(title="One", url="https://one.example")]),
        _hyp("hyp_b", [CitedPaper(title="Two", url="https://two.example")]),
    ]

    block = MetaReviewAgent._source_block(hyps, [])

    assert "[S1]" in block
    assert "[S2]" in block


def test_no_sources_yields_an_empty_block_not_a_fabricated_one() -> None:
    """An empty block makes the prompt's default fire, which tells the model
    to say so rather than invent references."""
    assert MetaReviewAgent._source_block([], []) == ""
    assert MetaReviewAgent._source_block([_hyp("hyp_a", [])], []) == ""


def test_long_source_lists_are_truncated_with_an_explicit_count() -> None:
    hyps = [_hyp("hyp_a", [
        CitedPaper(title=f"P{i}", url=f"https://e{i}.example") for i in range(50)
    ])]

    block = MetaReviewAgent._source_block(hyps, [], limit=10)

    assert "[S10]" in block
    assert "[S11]" not in block
    # Silent truncation would read as "these are all the sources".
    assert "40 more sources" in block


def test_entries_without_a_url_are_dropped() -> None:
    reviews = [_review("hyp_a", [Evidence(claim="c", url="", excerpt="e")])]
    assert MetaReviewAgent._source_block([], reviews) == ""


async def test_sources_are_hydrated_from_artifacts_not_db_rows(tmp_cfg) -> None:
    """The DB rows carry no citations — both repos note they "live in the JSON
    artifact, not the row". A first attempt at this feature read the model
    fields directly and silently produced an empty source list on every real
    session.
    """
    import json

    from co_scientist.agents.base import AgentDeps

    hyp_rel = "artifacts/s/hypotheses/hyp_a.json"
    rev_rel = "artifacts/s/reviews/rev_a.json"
    for rel, record in (
        (hyp_rel, {"citations": [
            {"url": "https://paper.example", "title": "Paper A",
             "excerpt": "quote", "doi": "10.1/x", "year": 2025},
        ]}),
        (rev_rel, {"evidence": [
            {"claim": "c", "url": "https://evidence.example", "excerpt": "e"},
        ]}),
    ):
        path = tmp_cfg.data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"record": record}), encoding="utf-8")

    agent = MetaReviewAgent(AgentDeps(cfg=tmp_cfg, db=None, llm=None, tools=None))

    hyp = _hyp("hyp_a", [])
    hyp = hyp.model_copy(update={"artifact_path": hyp_rel})
    review = _review("hyp_a", [])
    review = review.model_copy(update={"artifact_path": rev_rel})

    hyps, reviews = await agent._hydrate_sources([hyp], [review])

    assert hyps[0].citations[0].url == "https://paper.example"
    assert hyps[0].citations[0].year == 2025
    assert reviews[0].evidence[0].url == "https://evidence.example"

    block = agent._source_block(hyps, reviews)
    assert "https://paper.example" in block
    assert "https://evidence.example" in block


async def test_hydration_refuses_paths_outside_the_data_dir(tmp_cfg) -> None:
    from co_scientist.agents.base import AgentDeps

    agent = MetaReviewAgent(AgentDeps(cfg=tmp_cfg, db=None, llm=None, tools=None))

    assert await agent._read_record("../../etc/passwd") == {}
    assert await agent._read_record(None) == {}
    assert await agent._read_record("artifacts/does-not-exist.json") == {}


def test_final_prompt_renders_sources_and_forbids_inventing_urls() -> None:
    from co_scientist.llm.prompts import render

    prompt = render(
        "metareview.final",
        goal="g", preferences="", system_feedback="",
        top_hypotheses_block="(hyps)",
        sources="- [S1] Paper A\n  https://a.example",
    )

    assert "https://a.example" in prompt
    assert "never invent" in prompt.lower()


def test_final_prompt_without_sources_tells_the_model_to_say_so() -> None:
    from co_scientist.llm.prompts import render

    prompt = render(
        "metareview.final",
        goal="g", preferences="", system_feedback="",
        top_hypotheses_block="(hyps)", sources="",
    )

    assert "no literature sources were recorded" in prompt
