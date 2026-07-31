# AI co-scientist

An open source re-implementation of Google's **AI co-scientist** ([Gottweis et al., *Nature*, 2026](https://www.nature.com/articles/s41586-026-10644-y); [research blog, 2025](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)) — a multi-agent system that takes a natural-language research goal and produces a tournament-ranked **research overview** of novel hypotheses.

**It runs on your Claude Code or Codex subscription, not on an LLM API key.** Every agent call is executed by the local `claude -p` / `codex exec` CLI over its own OAuth login. Structured output and citation provenance come back through a bundled MCP server — see [LLM backend](#llm-backend).

The agent roster, prompts, and control flow follow the paper. Source materials that were used to instruct the coding agent (mainly Claude Code) include:

- [`reference/8 Pseudocode of Co-Scientist agents`](reference/) — the supplementary pseudocode for Supervisor, Generation, Reflection, Ranking, Evolution, Proximity, Meta-review.
- [`reference/9 Prompts for the specialized agents in .md`](reference/) — the per-agent prompts from the paper's supplement, used verbatim (modulo Jinja interpolation) in [`config/prompts/`](config/prompts/).
- [`reference/AICoScientist-*.png`](reference/) — the architecture and component diagrams from the paper.

The agents:

- **Generation** — proposes hypotheses via literature review and simulated scientific debate.
- **Reflection** — reviews hypotheses for novelty, correctness, and testability; deep-verifies the underlying assumptions.
- **Ranking** — runs an Elo tournament with simulated debates between hypotheses.
- **Evolution** — combines, simplifies, makes more feasible, or out-of-box-reimagines top-ranked hypotheses.
- **Proximity** — embeds and clusters hypotheses to drive dedup and informative tournament pairings.
- **Meta-review** — synthesizes system-wide feedback and the final research overview.

A **Supervisor** parses the goal into a research plan and schedules agent tasks through a durable SQLite-backed queue with bounded concurrency.

This is an independent re-implementation in Python on top of pluggable LLM provider SDKs — not affiliated with Google or the paper's authors.

> [`docs/BENCH_RESULTS.md`](docs/BENCH_RESULTS.md) — every cross-model bench ever run on this code, with per-candidate Elo, every hypothesis produced, gold-set hits, and direct file pointers. Auto-generated from the bench DB.

## Contents

- [Architecture](#architecture)
- [Install](#install)
- [Initialize](#initialize)
- [Run a research session](#run-a-research-session)
- [LLM backend](#llm-backend)
- [Configuration](#configuration)
- [Bench: compare models head-to-head](#bench-compare-models-head-to-head)
- [Repository layout](#repository-layout)

## Architecture

```
                       co-scientist run "<goal>"
                                  │
                                  ▼
            ┌──────────────────────────────────────┐
            │            Supervisor                │  durable task queue (SQLite)
            │  • parse_goal → ResearchPlan         │  bounded concurrency
            │  • enqueue initial Generation tasks  │  lease + dead-letter + resume
            │  • main loop: claim → run → follow-up│  termination: BUDGET / WALL_CLOCK
            │  • decide_next_steps when idle       │              / ELO_STABLE / IDLE / EXTERNAL
            │  • finalize: meta-review overview    │
            └──────────────────────────────────────┘
                                  │  tasks
            ┌─────────────────────┼─────────────────────────────┐
            ▼                     ▼                             ▼
   ┌──────────────┐      ┌──────────────┐              ┌──────────────┐
   │  Generation  │ hyp  │  Reflection  │ review       │   Ranking    │
   │  literature  │─────►│  full +      │─────────────►│ pairwise vs  │──► Elo
   │  + debate    │      │  verification│              │   debate     │
   └──────────────┘      └──────────────┘              └──────────────┘
            ▲                     ▲                             │
            │                     │ informative pairings        ▼
   ┌──────────────┐      ┌──────────────┐              ┌──────────────┐
   │  Evolution   │◄─────│ Meta-review  │              │  Proximity   │
   │ combine /    │ feed │ system fdbk  │              │ FAISS embed  │
   │ simplify /   │ back │ + final      │              │ + cluster /  │
   │ feasibility /│      │ overview     │              │ dedup        │
   │ out_of_box   │      └──────────────┘              └──────────────┘
   └──────────────┘
            │
            ▼
       new hypotheses re-enter the cycle


  Shared infrastructure
  ─────────────────────
  • LLMProvider  ─ claude_cli (`claude -p`) / codex_cli (`codex exec`),
                   driven over subscription auth — no API key anywhere
  • ToolRegistry ─ web_fetch + pubmed_search / arxiv_search / europe_pmc_search;
                   web_search auto-registered iff TAVILY/BRAVE key set;
                   science-skills discovered via SKILL.md frontmatter
  • MCP server   ─ record_* schemas + research tools served to the CLI;
                   captures structured output and URL provenance
  • TokenBudget  ─ per-agent shares + global cap; reservation released on retry
  • EventBus     ─ in-memory fan-out to SSE for the live web UI
  • FaissStore   ─ IndexFlatIP per session, asyncio-locked, atomic save/load;
                   OpenAI text-embedding-3-large → hash fallback
  • SQLite       ─ sessions / hypotheses / reviews / tournament_matches /
                   elo_journal / tasks / transcripts / system_feedback /
                   embeddings_meta / spans / events / bench_* (15 tables;
                   WAL, busy_timeout, idempotent migration runner)
```

## Install

```bash
# Recommended: Python 3.11–3.13 (FAISS wheel availability)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# No LLM key goes in here — see "LLM backend" below.
```

You also need one of the agent CLIs installed and signed in:

```bash
claude          # Claude Code — complete the subscription login once, then quit
# or
codex login     # Codex — sign in with your ChatGPT account
```

## Initialize

```bash
co-scientist init
co-scientist list
```

`init` creates `data/` (artifacts, vectors, logs) and applies migrations to `data/co_scientist.db`. The output prints which backend it found, its version, and whether you are signed in.

```bash
co-scientist doctor    # binary + subscription + MCP handshake, no model call
```

## Run a research session

```bash
co-scientist run "Identify hypotheses about microbiome-driven inflammation" \
  --n 3 --budget-usd 2.0 --wall-clock 600
```

This kicks off Generation → Reflection → Ranking → Evolution → Meta-review under the configured backend. The Supervisor schedules tasks, the Elo tournament refines a leaderboard, and the final research overview is written to `data/artifacts/<session_id>/final/overview.md`.

```bash
co-scientist serve            # FastAPI + htmx + SSE dashboard at localhost:7878
co-scientist report <id>      # print the final overview
co-scientist status <id>      # session metadata + counts
co-scientist pause <id> | resume <id> | abort <id>
co-scientist feedback <id> --kind directive --text "focus on metabolic pathways"
co-scientist doctor           # verify CLI, subscription login, and MCP server
co-scientist estimate         # pre-flight equivalent-cost estimate
co-scientist eval [agent]     # run the rubric eval bundle (offline mode optional)
co-scientist tools list       # show every registered tool the agents can call
```

## LLM backend

**There is no LLM API key in this project.** Every agent call is executed by a
local agent CLI running on a subscription you already pay for:

| `[llm] provider` | Drives | Auth | `[models]` values |
| --- | --- | --- | --- |
| `claude_cli` *(default)* | `claude -p` (Claude Code) | OAuth login — run `claude` once | `opus`, `sonnet`, `haiku` |
| `codex_cli` | `codex exec` (Codex) | `codex login` (ChatGPT account) | Codex model ids your account is entitled to |

```toml
[llm]
provider = "claude_cli"

[llm.claude_cli]
binary = "claude"
timeout_seconds = 900
max_parallel = 3          # bounded by the subscription rate limit, not CPU
replace_system_prompt = true

[models]
generation = "opus"       # reasoning-heavy agents
ranking_pairwise = "sonnet"
classifier = "haiku"
```

Any `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` present in your environment is
**stripped from the CLI subprocess** ([`cli_backend/base.py`](co_scientist/llm/cli_backend/base.py)
`BILLING_ENV_VARS`), so a call can never silently fall back to metered billing.
A test asserts this.

### How structured output survives without an API

The agents depend on forced tool use (`record_hypothesis`, `record_review`, …),
and headless CLIs have no `tool_choice` flag. The project ships its own MCP
server ([`co_scientist/mcp/`](co_scientist/mcp/)) that the backend launches per
call and wires in via `--mcp-config`:

- **`record_*` tools** — the schemas from [`agents/schemas.py`](co_scientist/agents/schemas.py)
  verbatim, validated at the tool boundary exactly as the API used to validate
  them, then written to a capture directory the backend reads back.
- **research tools** — the existing PubMed / arXiv / Europe PMC / web_fetch
  registry, with every result's URLs appended to a provenance log. That is what
  keeps the citation verifier's "you may only cite what you actually saw" rule
  enforceable.

`--tools ""` disables the CLI's built-in tools so every tool call flows through
that server and lands in your transcripts.

### Things worth knowing

- **The CLI runs its own agentic loop.** `run_tool_loop` no longer drives
  turns; it issues one call and reads the capture. If a call ends without a
  structured record, it retries once with the search tools removed and the
  record demanded, then raises.
- **`--system-prompt` replaces Claude Code's default prompt.** Measured on this
  machine: a trivial call under the default prompt costs ~19k cache-creation +
  ~24k cache-read tokens of harness scaffolding; with ours it is ~1–2k. Set
  `replace_system_prompt = false` if you want the harness context back.
- **`--bare` is unusable here** — it forces `ANTHROPIC_API_KEY` auth and never
  reads OAuth, which defeats the point.
- **Rate limits are hours, not minutes.** Subscription limits clear over a
  multi-hour window, so rate-limit backoff starts at 30s and grows to 10min
  ([`llm/retry.py`](co_scientist/llm/retry.py)) instead of the seconds an API
  429 deserves.
- **`budget_usd` is no longer a spend gate.** Nothing is billed per token. The
  CLI reports an equivalent API cost per call and `budget_usd` caps that gauge,
  which still catches a runaway session. The real limits are
  `wall_clock_seconds` and `max_ideas`.
- **Codex model entitlement is a real gate.** A ChatGPT account that is not
  entitled to the model you name fails the run with
  `"model is not supported when using Codex with a ChatGPT account"`. Check
  with `co-scientist doctor` before a long run.

### Embeddings — the one remaining hosted model

No agent CLI exposes an embedding endpoint, and Proximity/dedup needs real
semantic vectors, so embeddings still call OpenAI directly:

```toml
[embeddings]
provider = "openai"                # openai | hash
model = "text-embedding-3-large"
dim = 3072
```

Without `OPENAI_API_KEY` this degrades to a local hash embedder that catches
literal token overlap but not paraphrase — sessions still run, dedup is just
weaker. Changing `dim` invalidates existing FAISS indices under `data/vectors/`.

## Configuration

Layered: [`config/default.toml`](config/default.toml) → `~/.co-scientist/config.toml` → `./co-scientist.toml` → `--config <path>`. Secrets come from environment only (see [`.env.example`](.env.example)).

## Bench: compare models head-to-head

`co-scientist bench` runs the same goal under N different `(provider, model)` configurations and ranks them via a single shared Elo tournament. Each candidate independently generates hypotheses; then every candidate-pair plays `--matches` head-to-head debates, judged by ONE fixed judge model (picked separately so no candidate scores its own work).

> **For live numbers** — per-candidate Elo, the actual hypotheses each model proposed, gold-set hits, and what the data showed — see [`docs/BENCH_RESULTS.md`](docs/BENCH_RESULTS.md). It includes a headline-findings section at the top so you don't have to scroll through every bench.

### Presets

| `--preset`             | What it does |
| ---                    | --- |
| `claude-aml`           | The paper's AML drug-repurposing goal + gold-set recall scoring (strict top-3: Nanvuranlat / KIRA6 / Leflunomide), run across the three Claude Code tiers |
| `claude-aml-vs-raw`    | Same, but each tier runs **both** through the full pipeline AND as a single raw call — isolates the multi-agent harness's value-add |
| `cross-backend-aml`    | Claude Code vs Codex on the same goal. Needs both CLIs signed in **and** a ChatGPT account entitled to the Codex model |

> The historical `paper*` and `frontier*` presets are gone. They compared
> against the paper's baselines (Gemini 2.0 Flash Thinking, Gemini 2.0 Pro,
> OpenAI o1) by routing every candidate through OpenRouter, which is
> impossible now that candidates run on subscription CLIs. The archived
> numbers in [`docs/BENCH_RESULTS.md`](docs/BENCH_RESULTS.md) were produced
> under the previous API-key architecture: they are kept for reference, are
> **not** reproducible with this code, and are not directly comparable to new
> runs (same prompts, but the tool loop now belongs to the CLI).

```bash
# Score the Claude tiers against the paper's AML drug picks:
co-scientist bench --preset claude-aml --n 3 --matches 2

# Isolate how much the multi-agent harness actually adds:
co-scientist bench --preset claude-aml-vs-raw --n 3 --matches 2

# Compare multi-agent pipeline vs raw model call on the same goal
# (--budget-per-candidate defaults to 3.0; frontier models need it):
co-scientist bench --preset claude-aml-vs-raw --n 1

# Current frontier models, pipeline vs raw:
co-scientist bench --preset cross-backend-aml --n 1
```

### Pipeline vs raw LM (one model, isolated)

The `--preset *-vs-raw` presets pit each model's **full co-scientist Generation pipeline** (literature tools + tool loop + dedup + `record_hypothesis`) against a **single raw LM call** with the same model + a forced `record_hypothesis` function call (no tools). Lets you measure how much of the system's output quality comes from the multi-agent harness vs the underlying model. → live numbers in [`docs/BENCH_RESULTS.md`](docs/BENCH_RESULTS.md#headline-findings).

### Gold-set scoring (AML drug repurposing)

`claude-aml*` presets score **recall** against a curated answer key from the Co-Scientist paper. Two gold sets ship; both stay registered so historical bench artifacts remain interpretable.

| label                                                   | size | what it is |
| ---                                                     | --- | --- |
| `aml-repurposing-paper-top3` *(default for `claude-aml*`)* | 3 | Top-3 of the original paper's list: candidates with no prior published AML repurposing, no prior preclinical evidence in AML, and no external inputs (no DepMap scores, no expert curation). → **Nanvuranlat (JPH-203 / KYT-0353), KIRA6, Leflunomide (Arava / HWA-486 / Teriflunomide / Aubagio)** |
| `aml-repurposing-paper-5`                               | 5 | Broader 5-drug list referenced in the paper's main text: **Binimetinib (MEK162), Pacritinib (SB1518 / Vonjo), Cerivastatin (Baycol), Pravastatin (Pravachol), Dimethyl fumarate (DMF / BG-12 / Tecfidera)** |

Swap with `--goldset`:

```bash
co-scientist bench --preset claude-aml --goldset aml-repurposing-paper-5   # broader list
co-scientist bench --preset claude-aml --goldset none                       # head-to-head only
```

The matcher is whole-token, case-insensitive, and looks at every searched field of every hypothesis (title / summary / full_text / `entities` / citation excerpts). Drug **class** mentions (e.g. "DHODH inhibitor") do **not** count — the candidate has to name the actual compound (or one of its registered aliases).

### Custom candidates

`label=provider:model[@mode]`. `mode` is `pipeline` (default) or `direct`. Pipeline goes through the full Generation agent stack; direct is a single forced-tool LM call with no literature tools.

```bash
co-scientist bench "Identify hypotheses about X" \
  -c opus=anthropic:claude-opus-5 \
  -c opus-raw=anthropic:claude-opus-5@direct \
  -c sol=openai:gpt-5.6-sol \
  -c luna=openai:gpt-5.6-luna \
  --judge anthropic:claude-sonnet-5
```

### Where results live

Every bench writes to SQLite + JSON on disk:

```
data/co_scientist.db                          ← SQLite, all metadata
  bench_runs                                  one row per bench
  bench_candidates                            one row per (bench × candidate × mode)
  bench_matches                               one row per head-to-head

data/artifacts/<session_id>/                  ← JSON on disk
  bench/<bench_id>.json                       run summary + per-entity gold_hit_detail
  hypotheses/<hyp_id>.json                    every hypothesis the bench produced
  transcripts/generation/<trn_id>.json        every LLM call
```

The auto-generated [`docs/BENCH_RESULTS.md`](docs/BENCH_RESULTS.md) (rebuild with `python scripts/build_bench_report.py`) walks every recorded bench and renders the per-candidate result table, every hypothesis attributed to the model that produced it, and a post-hoc rescore against every registered gold set.

### Mechanics

- **Generation runs in parallel** per candidate under a deep-copied Config (`cfg.llm.provider`, `cfg.models.*`, thinking budgets zeroed for non-Anthropic).
- **Round-robin pairings**: every pair plays `--matches` head-to-heads (one random hypothesis from each side per match).
- **Structured verdict** via a forced `record_verdict` function call — no fragile `better idea: <N>` text parsing across providers.
- Bench runs are **isolated from regular sessions** — they don't write to `tournament_matches` or affect any session's leaderboard.

## Repository layout

```
co_scientist/
  agents/       # supervisor + 6 specialized agents (base, generation, reflection,
                # ranking, evolution, proximity, metareview)
  bench/        # cross-model bench runner (Elo tournament + gold-set scoring)
  llm/          # provider abstraction (anthropic / openai / openai_compatible),
                # tool loop, token budgets, model routing, retry, batch, estimator
  storage/      # SQLite schema + migrations, db connection, 10 repos
  tools/        # tool registry; web_fetch, web_search, pubmed/arxiv/europe_pmc,
                # science-skills bridge
  vectors/      # embeddings (OpenAI text-embedding-3-large / hash) + FAISS
  orchestrator/ # task scheduling, Elo updates, termination, event bus
  safety/       # injection quoting, classifier, citation verifier
  obs/          # metrics (tokens, cost, cache hit ratio, latency)
  web/          # FastAPI + htmx + SSE UI + sanitized markdown renderer
  evals/        # per-agent + e2e + regression evals
  tests/        # 213 unit tests + fixtures + smoke
config/
  default.toml
  prompts/      # 14 Jinja2 templates (one per agent.mode), derived from
                # the paper's supplementary prompts
docs/
  BENCH_RESULTS.md   # every bench ever run (auto-generated)
scripts/
  build_bench_report.py
reference/      # paper source materials (pseudocode, prompts, diagrams)
data/           # gitignored; runtime artifacts (SQLite, FAISS, transcripts)
vendor/         # gitignored; pinned clone of google-deepmind/science-skills
```

## License

Apache-2.0.
