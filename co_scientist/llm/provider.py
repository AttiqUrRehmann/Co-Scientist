"""LLMProvider — the interface every backend implements, and its factory.

There is exactly one kind of backend now: a local agent CLI driven over its
subscription login. No API keys, no metered per-token billing, no vendor SDKs
in the request path.

Backends take a normalized `AgentCallSpec` and return an `LLMResponse` whose
`.raw` is Anthropic-Message-shaped (`.content` blocks, `.stop_reason`,
`.usage`) — see `llm/types.py` for why that shape is the contract.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .types import AgentCallSpec, CallContext, LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    """Common interface every backend implements."""

    async def call(
        self,
        spec: AgentCallSpec,
        ctx: CallContext,
        *,
        est_input_tokens: int | None = None,
    ) -> LLMResponse:
        ...


# Backend names accepted in `[llm] provider`.
KNOWN_BACKENDS = frozenset({"claude_cli", "codex_cli"})

# Names that used to select a metered API provider. Kept only to give a
# pointed error instead of a confusing fallback when an old config is loaded.
RETIRED_API_PROVIDERS = frozenset({
    "anthropic", "openai", "openai_compatible", "openrouter",
    "gemini", "google", "groq", "together", "mistral", "ollama",
})


class BackendUnavailable(RuntimeError):
    """The configured backend's CLI is missing or not usable."""


def get_provider(cfg, *, db, budget, retry_policy=None) -> LLMProvider:
    """Construct the backend named in `cfg.llm.provider`."""
    from ..logging import get_logger

    log = get_logger("llm.provider")
    name = (getattr(cfg.llm, "provider", "claude_cli") or "claude_cli").strip().lower()

    if name in RETIRED_API_PROVIDERS:
        raise BackendUnavailable(
            f"`[llm] provider = {name!r}` selects an API-key provider, which this "
            "project no longer supports. Use \"claude_cli\" (Claude Code) or "
            "\"codex_cli\" (Codex) — both run on your existing subscription."
        )

    if name not in KNOWN_BACKENDS:
        log.warning("unknown_llm_backend", configured=name, fallback="claude_cli")
        name = "claude_cli"

    if name == "codex_cli":
        from .cli_backend.codex import CodexCliProvider

        return CodexCliProvider(cfg, db=db, budget=budget, retry_policy=retry_policy)

    from .cli_backend.claude_code import ClaudeCliProvider

    return ClaudeCliProvider(cfg, db=db, budget=budget, retry_policy=retry_policy)


# --------------------------------------------------------------------------- #
# preflight


@dataclass
class BackendStatus:
    """What `co-scientist doctor` and `init` report."""

    backend: str
    binary: str
    binary_path: str | None
    version: str | None
    authenticated: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.binary_path is not None and self.authenticated


def check_backend(cfg) -> BackendStatus:
    """Verify the configured CLI exists and is signed in.

    Deliberately cheap — it runs a version/status probe, never a model call,
    so `doctor` costs nothing against the subscription.
    """
    import subprocess

    name = (getattr(cfg.llm, "provider", "claude_cli") or "claude_cli").strip().lower()
    if name == "codex_cli":
        binary = cfg.llm.codex_cli.binary
        auth_cmd = [binary, "login", "status"]
        auth_marker = "logged in"
    else:
        name = "claude_cli"
        binary = cfg.llm.claude_cli.binary
        auth_cmd = None
        auth_marker = ""

    path = shutil.which(binary)
    if path is None:
        return BackendStatus(
            backend=name, binary=binary, binary_path=None, version=None,
            authenticated=False,
            detail=f"{binary!r} not found on PATH — install it and sign in.",
        )

    version = _probe(subprocess, [path, "--version"])

    if auth_cmd is not None:
        status_text = _probe(subprocess, [path, *auth_cmd[1:]]) or ""
        authenticated = auth_marker in status_text.lower()
        detail = status_text or "no auth status reported"
    else:
        authenticated = _claude_signed_in()
        detail = (
            "signed in with a Claude subscription"
            if authenticated
            else "not signed in — run `claude` once and complete the login."
        )

    return BackendStatus(
        backend=name, binary=binary, binary_path=path, version=version,
        authenticated=authenticated, detail=detail,
    )


def _claude_signed_in() -> bool:
    """Detect a Claude Code subscription session without spending quota.

    Claude Code has no `login status` subcommand, and where it keeps the OAuth
    token is platform-dependent: the macOS keychain, or a credentials file
    elsewhere. `~/.claude.json` carries an `oauthAccount` record on every
    platform once login completes, so that is the portable signal; the file
    check covers installs that store credentials on disk.
    """
    import json
    from pathlib import Path

    config = Path.home() / ".claude.json"
    if config.exists():
        try:
            if json.loads(config.read_text(encoding="utf-8")).get("oauthAccount"):
                return True
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return (Path.home() / ".claude" / ".credentials.json").exists()


def _probe(subprocess_mod, argv: list[str]) -> str | None:
    try:
        out = subprocess_mod.run(
            argv, capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, Exception):
        return None
    return (out.stdout or out.stderr or "").strip() or None
