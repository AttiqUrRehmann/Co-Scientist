"""Configuration loader.

Layered: config/default.toml → ~/.co-scientist/config.toml → ./co-scientist.toml → env.
Secrets come from environment variables only.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"


class RunCfg(BaseModel):
    concurrency: int = 4
    max_ideas: int = 60
    max_matches_per_idea: int = 12
    wall_clock_seconds: int = 7200
    budget_tokens: int = 5_000_000
    budget_usd: float = 25.0


class StorageCfg(BaseModel):
    data_dir: str = "./data"


class ScienceSkillsCfg(BaseModel):
    path: str = "./vendor/science-skills"
    pinned_commit: str = ""


class EmbeddingsCfg(BaseModel):
    """Embeddings are the one remaining hosted-model dependency.

    No agent CLI exposes an embedding endpoint, and Proximity/dedup needs real
    semantic vectors, so this stays on OpenAI's best embedding model. With no
    key set it degrades to the local hash embedder rather than failing.
    """

    provider: str = "openai"
    model: str = "text-embedding-3-large"
    dim: int = 3072


class VectorsCfg(BaseModel):
    backend: str = "faiss"
    dedup_cosine_threshold: float = 0.92
    cluster_threshold: float = 0.15
    full_recluster_every_matches: int = 20


class RankingCfg(BaseModel):
    k_factor_new: int = 32
    k_factor_warm: int = 16
    elo_initial: int = 1200
    debate_when_matches_lt: int = 2
    debate_when_elo_delta_lt: int = 50
    batch_below_decile: bool = True
    batch_submit_every_seconds: int = 1800
    p_new: float = 0.4
    p_close: float = 0.4
    p_random: float = 0.2


class TerminationCfg(BaseModel):
    elo_stability_k: int = 5
    elo_stability_n: int = 3
    elo_stability_eps: float = 25.0
    match_snapshot_every: int = 10
    # Guards that prevent elo_stable from firing on a small pool.
    # Defaults of 0 preserve the original behaviour.
    min_ideas_before_stable: int = 0
    min_matches_before_stable: int = 0


class EvolutionCfg(BaseModel):
    """Controls when the idle-refinement loop triggers evolution."""
    # Minimum number of *mature* hypotheses (>= 3 matches played) required
    # before the supervisor schedules an EvolveTopHypotheses task.
    # Default matches the original hardcoded value.
    min_mature: int = 20
    # How many top hypotheses to evolve per idle pass.
    top_k: int = 5


class BudgetSharesCfg(BaseModel):
    generation: float = 0.20
    reflection: float = 0.20
    ranking: float = 0.25
    evolution: float = 0.15
    metareview: float = 0.10
    proximity: float = 0.02
    reserve: float = 0.08


class ModelsCfg(BaseModel):
    """Per-agent model selection.

    Values are passed verbatim to the configured backend's `--model` flag, so
    they are CLI aliases ("opus", "sonnet", "haiku") under `claude_cli` and
    Codex model ids under `codex_cli`.
    """

    parse_goal: str = "sonnet"
    generation: str = "opus"
    reflection: str = "opus"
    evolution: str = "opus"
    ranking_pairwise: str = "sonnet"
    ranking_debate: str = "sonnet"
    ranking_priority: str = "opus"
    metareview_feedback: str = "sonnet"
    metareview_final: str = "opus"
    classifier: str = "haiku"
    judge: str = "sonnet"


class ThinkingCfg(BaseModel):
    generation_literature: int = 4000
    generation_debate: int = 8000
    reflection_full: int = 0
    reflection_verification: int = 12000
    reflection_observation: int = 6000
    ranking_pairwise: int = 4000
    ranking_debate: int = 8000
    evolution_combine: int = 6000
    evolution_out_of_box: int = 6000
    evolution_feasibility: int = 0
    evolution_simplify: int = 0
    metareview_feedback: int = 8000
    metareview_final: int = 16000


class ToolLoopCfg(BaseModel):
    generation_max_iters: int = 8
    reflection_max_iters: int = 8
    ranking_max_iters: int = 3
    evolution_max_iters: int = 6
    metareview_max_iters: int = 12
    parallel_cap: int = 4
    tool_timeout_seconds: int = 30


class RetryCfg(BaseModel):
    max_attempts_429: int = 6
    max_attempts_529: int = 8
    max_attempts_5xx: int = 5
    max_attempts_timeout: int = 3
    base_ms: int = 1000
    cap_ms: int = 60_000
    per_call_timeout_seconds: int = 120
    per_call_timeout_thinking_seconds: int = 300


class LeaseCfg(BaseModel):
    default_seconds: int = 300
    reflection_seconds: int = 600
    metareview_final_seconds: int = 1800
    heartbeat_seconds: int = 60
    max_attempts: int = 3


class WebSearchCfg(BaseModel):
    provider: str = "tavily"
    max_results: int = 8


class WebFetchCfg(BaseModel):
    max_bytes: int = 5_000_000
    timeout_seconds: int = 30
    user_agent: str = "co-scientist/0.1"


class CodeExecCfg(BaseModel):
    provider: str = "local"
    local_cpu_seconds: int = 30
    local_mem_mb: int = 512


class SafetyCfg(BaseModel):
    enable_classifier: bool = True
    enable_citation_verifier: bool = True
    classifier_block_categories: list[str] = Field(
        default_factory=lambda: ["cbrn", "csam", "weapons", "illicit_synthesis"]
    )
    classifier_warn_categories: list[str] = Field(default_factory=lambda: ["dual_use_bio"])


class ClaudeCliCfg(BaseModel):
    """Settings for the `claude -p` backend (Claude Code, subscription auth)."""

    binary: str = "claude"
    permission_mode: str = "dontAsk"
    timeout_seconds: int = 900
    max_parallel: int = 3
    """Concurrent CLI processes. Bounded by the subscription's rate limit
    rather than by CPU, so this is deliberately lower than `run.concurrency`."""
    replace_system_prompt: bool = True
    """Pass `--system-prompt` (replace) rather than `--append-system-prompt`.

    Replacing drops Claude Code's own scaffolding, which otherwise costs about
    19k cache-creation plus 24k cache-read tokens on every single call."""


class CodexCliCfg(BaseModel):
    """Settings for the `codex exec` backend (Codex, ChatGPT subscription)."""

    binary: str = "codex"
    sandbox: str = "read-only"
    timeout_seconds: int = 900
    max_parallel: int = 3


class LLMCfg(BaseModel):
    """Choose which agent CLI backs the agents.

    Both backends drive a locally-installed agent harness over its subscription
    login. No LLM API key is involved anywhere in this project.

    - "claude_cli" — `claude -p` (Claude Code). Models are CLI aliases:
      "opus", "sonnet", "haiku".
    - "codex_cli" — `codex exec` (Codex). Models are Codex model ids; note
      that a ChatGPT account must be entitled to the model you pick.
    """

    provider: str = "claude_cli"
    claude_cli: ClaudeCliCfg = Field(default_factory=ClaudeCliCfg)
    codex_cli: CodexCliCfg = Field(default_factory=CodexCliCfg)


class WebUICfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7878


class Secrets(BaseSettings):
    """Secrets pulled from env only. Empty string means 'not configured'.

    Note what is *absent*: there is no LLM API key here. Reasoning runs
    through the Claude Code / Codex CLIs on their subscription login, so the
    only remaining keys are for embeddings and non-LLM data sources.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENAI_API_KEY: str = ""
    """Embeddings only (`text-embedding-3-large`). Never used for generation,
    and explicitly stripped from every CLI subprocess environment."""
    TAVILY_API_KEY: str = ""
    BRAVE_API_KEY: str = ""
    NCBI_API_KEY: str = ""
    OPENALEX_API_KEY: str = ""


class Config(BaseModel):
    run: RunCfg = Field(default_factory=RunCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    science_skills: ScienceSkillsCfg = Field(default_factory=ScienceSkillsCfg)
    embeddings: EmbeddingsCfg = Field(default_factory=EmbeddingsCfg)
    vectors: VectorsCfg = Field(default_factory=VectorsCfg)
    ranking: RankingCfg = Field(default_factory=RankingCfg)
    termination: TerminationCfg = Field(default_factory=TerminationCfg)
    evolution: EvolutionCfg = Field(default_factory=EvolutionCfg)
    budget_shares: BudgetSharesCfg = Field(default_factory=BudgetSharesCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    thinking: ThinkingCfg = Field(default_factory=ThinkingCfg)
    tool_loop: ToolLoopCfg = Field(default_factory=ToolLoopCfg)
    retry: RetryCfg = Field(default_factory=RetryCfg)
    lease: LeaseCfg = Field(default_factory=LeaseCfg)
    web_search: WebSearchCfg = Field(default_factory=WebSearchCfg)
    web_fetch: WebFetchCfg = Field(default_factory=WebFetchCfg)
    code_exec: CodeExecCfg = Field(default_factory=CodeExecCfg)
    safety: SafetyCfg = Field(default_factory=SafetyCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    web_ui: WebUICfg = Field(default_factory=WebUICfg)
    secrets: Secrets = Field(default_factory=Secrets)

    @property
    def data_dir(self) -> Path:
        p = Path(self.storage.data_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "co_scientist.db"

    def session_artifact_dir(self, session_id: str) -> Path:
        return self.data_dir / "artifacts" / session_id

    def session_vector_dir(self, session_id: str) -> Path:
        return self.data_dir / "vectors" / session_id

    def session_log_path(self, session_id: str) -> Path:
        return self.data_dir / "logs" / f"session-{session_id}.jsonl"


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(extra_path: Path | None = None) -> Config:
    """Layered load: default.toml → ~/.co-scientist/config.toml → ./co-scientist.toml → extra_path → env."""
    merged: dict[str, Any] = _read_toml(DEFAULT_CONFIG)

    for p in (
        Path.home() / ".co-scientist" / "config.toml",
        Path.cwd() / "co-scientist.toml",
        extra_path,
    ):
        if p is not None:
            merged = _deep_merge(merged, _read_toml(p))

    cfg = Config.model_validate(merged)
    # secrets pulled from env via Secrets() — already wired by default_factory above
    return cfg


def backend_binary(cfg: Config) -> str:
    """Name of the agent CLI the configured backend drives."""
    name = (getattr(cfg.llm, "provider", "claude_cli") or "claude_cli").strip().lower()
    if name == "codex_cli":
        return cfg.llm.codex_cli.binary
    return cfg.llm.claude_cli.binary


def has_embeddings_key(cfg: Config) -> bool:
    """True if real embeddings are available; otherwise dedup falls back to
    the hash embedder (see vectors/embedder.py)."""
    return bool(cfg.secrets.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"))
