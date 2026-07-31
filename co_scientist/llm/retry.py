"""Retry semantics for CLI-driven backends.

The previous version of this module classified `anthropic` SDK exceptions by
HTTP status. There is no HTTP client in the request path any more — a call is
a subprocess — so failures are classified from exit codes and message text
instead.

The important behavioural difference from an API key: a metered API
rate-limits per minute, so seconds of backoff suffice. A subscription
rate-limits over a multi-hour window, so retrying in seconds just burns
attempts. Rate-limit backoff here starts in the tens of seconds and is allowed
to grow into minutes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Message fragments meaning "back off and try again", not "this was malformed".
RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "usage limit",
    "quota exceeded",
    "too many requests",
    "overloaded",
    "429",
    "529",
)

TRANSIENT_MARKERS: tuple[str, ...] = (
    "connection error",
    "network error",
    "socket hang up",
    "econnreset",
    "etimedout",
    "fetch failed",
    "internal server error",
    "timed out",
    "503",
    "502",
)

# Floor/ceiling applied to rate-limit backoff, overriding the ordinary knobs.
RATE_LIMIT_BASE_MS = 30_000
RATE_LIMIT_CAP_MS = 600_000


class CliBackendError(RuntimeError):
    """The CLI ran but could not produce a usable answer."""


class CliRetryableError(CliBackendError):
    """Transient failure — worth retrying after a backoff."""


@dataclass
class RetryPolicy:
    """Retry knobs, sourced from `[retry]` in config."""

    max_attempts: int = 6
    base_ms: int = 1_000
    cap_ms: int = 60_000

    @classmethod
    def from_config(cls, cfg) -> RetryPolicy:
        return cls(
            max_attempts=max(1, cfg.retry.max_attempts_429),
            base_ms=cfg.retry.base_ms,
            cap_ms=cfg.retry.cap_ms,
        )


def is_rate_limit(message: str) -> bool:
    return _matches(message, RATE_LIMIT_MARKERS)


def is_transient(message: str) -> bool:
    return _matches(message, RATE_LIMIT_MARKERS) or _matches(message, TRANSIENT_MARKERS)


def classify_failure(message: str) -> CliBackendError:
    """Map an error string onto retryable vs terminal."""
    if is_transient(message):
        return CliRetryableError(message)
    return CliBackendError(message)


def backoff_seconds(
    attempt: int, message: str, policy: RetryPolicy, *, jitter: float | None = None
) -> float:
    """Exponential backoff with jitter, widened for subscription rate limits.

    `attempt` is 1-based. `jitter` is injectable so tests can be deterministic.
    """
    base_ms, cap_ms = policy.base_ms, policy.cap_ms
    if is_rate_limit(message):
        base_ms = max(base_ms, RATE_LIMIT_BASE_MS)
        cap_ms = max(cap_ms, RATE_LIMIT_CAP_MS)
    delay_ms = min(cap_ms, base_ms * (2 ** max(0, attempt - 1)))
    factor = jitter if jitter is not None else (0.5 + random.random())
    return (delay_ms / 1000.0) * factor


def _matches(text: str, markers: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(m in low for m in markers)
