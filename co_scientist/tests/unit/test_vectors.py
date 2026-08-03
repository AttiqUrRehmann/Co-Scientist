"""Tests for the FAISS store. Embedder is network-bound; we feed fake vectors."""

from __future__ import annotations

import numpy as np
import pytest

from co_scientist.vectors.store import FaissStore


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype("float32")
    return v / np.linalg.norm(v)


@pytest.mark.asyncio
async def test_faiss_store_add_search_persist(tmp_cfg) -> None:
    store = FaissStore(tmp_cfg, "ses_v", dim=8)
    await store.load_or_create()
    assert store.n == 0

    o1 = await store.add("hyp_1", _vec(1))
    o2 = await store.add("hyp_2", _vec(2))
    assert (o1, o2) == (0, 1)
    assert store.n == 2

    # k-NN should find itself first
    results = await store.search(_vec(1), k=2)
    assert results[0][0] == "hyp_1"
    assert results[0][1] == pytest.approx(1.0, abs=1e-3)

    # cosine matrix is 2x2 with 1s on diagonal
    m = await store.cosine_matrix()
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(1.0, abs=1e-3)

    # Persist, then re-open
    await store.save()

    store2 = FaissStore(tmp_cfg, "ses_v", dim=8)
    await store2.load_or_create()
    assert store2.n == 2
    assert store2.hypothesis_at(0) == "hyp_1"
    assert store2.hypothesis_at(1) == "hyp_2"


@pytest.mark.asyncio
async def test_faiss_offset_lookup(tmp_cfg) -> None:
    store = FaissStore(tmp_cfg, "ses_v2", dim=4)
    await store.load_or_create()
    await store.add("a", _vec(1, 4))
    await store.add("b", _vec(2, 4))
    assert store.offset_of("a") == 0
    assert store.offset_of("b") == 1
    assert store.offset_of("missing") is None


@pytest.mark.asyncio
async def test_index_of_a_different_dim_is_rebuilt_not_reused(tmp_cfg) -> None:
    """Changing the embedding model invalidates the vectors already stored.

    Without the width check FAISS accepts the stale index and then fails deep
    inside `add` with an assertion that names nothing useful.
    """
    store = FaissStore(tmp_cfg, "ses_dim", dim=8)
    await store.load_or_create()
    await store.add("hyp_1", _vec(1, 8))
    await store.save()

    # Same session, wider embeddings — as after switching embedding model.
    reopened = FaissStore(tmp_cfg, "ses_dim", dim=16)
    await reopened.load_or_create()

    assert reopened.n == 0, "stale 8-d vectors must not survive into a 16-d index"
    assert reopened.index.d == 16
    assert reopened.offset_of("hyp_1") is None
    # And the fresh index is usable at the new width.
    await reopened.add("hyp_1", _vec(1, 16))
    assert reopened.n == 1


# ----------------------------- embedder fallback ----------------------------- #


@pytest.mark.asyncio
async def test_make_embedder_falls_back_to_hash_when_no_keys() -> None:
    """Without OPENAI_API_KEY, make_embedder should return HashEmbedder so
    dedup / proximity degrade rather than crash."""
    import os

    from co_scientist.config import Config
    from co_scientist.vectors.embedder import HashEmbedder, make_embedder

    cfg = Config()
    cfg.embeddings.provider = "openai"
    cfg.secrets.OPENAI_API_KEY = ""
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        emb = make_embedder(cfg)
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved
    assert isinstance(emb, HashEmbedder)


@pytest.mark.asyncio
async def test_hash_embedder_produces_normalized_unit_vectors() -> None:
    from co_scientist.config import Config
    from co_scientist.vectors.embedder import HashEmbedder

    cfg = Config()
    cfg.embeddings.dim = 128
    emb = HashEmbedder(cfg)
    vecs = await emb.embed(["microbiome inflammation hypothesis",
                            "tournament ranking hypothesis"])
    assert vecs.shape == (2, 128)
    # L2-normalized → ||v|| ≈ 1
    norms = np.linalg.norm(vecs, axis=1)
    assert all(abs(n - 1.0) < 1e-5 for n in norms)


@pytest.mark.asyncio
async def test_hash_embedder_similar_texts_have_higher_cosine() -> None:
    """The hash embedder is a bag-of-features stub, but near-duplicates of
    a text should still produce a higher cosine than unrelated text."""
    from co_scientist.config import Config
    from co_scientist.vectors.embedder import HashEmbedder

    cfg = Config()
    cfg.embeddings.dim = 1024
    emb = HashEmbedder(cfg)
    vecs = await emb.embed([
        "the gut microbiome drives chronic systemic inflammation",
        "the gut microbiome drives chronic systemic inflammation in humans",
        "quantum computing for solving prime factorization problems",
    ])
    sim_near = float(vecs[0] @ vecs[1])
    sim_far  = float(vecs[0] @ vecs[2])
    assert sim_near > sim_far


@pytest.mark.asyncio
async def test_openai_fallback_uses_large_at_the_configured_dim() -> None:
    """Falling back from another provider must not also downgrade the model.

    `embeddings.model` names a Voyage model here, so it cannot be forwarded to
    OpenAI. The fallback picks `-large` shortened to the configured dim, rather
    than `-small` at that dim — same vector width, better vectors, and existing
    FAISS indices stay valid because the dim is unchanged.
    """
    from co_scientist.config import Config
    from co_scientist.vectors.embedder import OpenAIEmbedder, make_embedder

    cfg = Config()                       # provider="voyage", dim=1024
    cfg.secrets.OPENAI_API_KEY = "sk-fake"
    emb = make_embedder(cfg)
    assert isinstance(emb, OpenAIEmbedder)
    assert emb.model == "text-embedding-3-large"
    assert emb.dim == cfg.embeddings.dim == 1024


@pytest.mark.asyncio
async def test_openai_as_the_configured_provider_honors_model_and_dim() -> None:
    """Choosing OpenAI outright forwards `embeddings.model` verbatim."""
    from co_scientist.config import Config
    from co_scientist.vectors.embedder import OpenAIEmbedder, make_embedder

    cfg = Config()
    cfg.embeddings.provider = "openai"
    cfg.embeddings.model = "text-embedding-3-large"
    cfg.embeddings.dim = 3072
    cfg.secrets.OPENAI_API_KEY = "sk-fake"
    emb = make_embedder(cfg)
    assert isinstance(emb, OpenAIEmbedder)
    assert emb.model == "text-embedding-3-large"
    assert emb.dim == 3072


def test_fallback_warning_emits_once_per_process() -> None:
    """Regression: ranking calls make_embedder() inside the pair-selection
    loop (potentially hundreds of times per session). The fallback warning
    must emit exactly once per process, not once per call.

    We probe the internal `_FALLBACK_WARNED` set rather than caplog because
    the project uses structlog, which doesn't always route through pytest's
    logging capture. The set is the source of truth for the once-per-process
    contract.
    """
    import os

    from co_scientist.config import Config
    from co_scientist.vectors import embedder as emb_mod

    emb_mod._reset_fallback_warned_for_tests()
    cfg = Config()
    cfg.embeddings.provider = "openai"
    cfg.secrets.OPENAI_API_KEY = ""
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        for _ in range(50):
            emb_mod.make_embedder(cfg)
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved

    # Exactly one warning marker recorded; subsequent calls hit the cache.
    assert {"openai_key_missing_using_hash_fallback"} == emb_mod._FALLBACK_WARNED
