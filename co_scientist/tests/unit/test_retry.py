"""Retry classification and backoff for CLI backends.

The distinction that matters: a metered API rate-limits per minute, so a few
seconds of backoff is right. A subscription rate-limits over hours, so short
retries just burn the attempt budget. These tests pin that behaviour.
"""

from __future__ import annotations

import pytest

from co_scientist.llm.retry import (
    RATE_LIMIT_BASE_MS,
    CliBackendError,
    CliRetryableError,
    CliRetryPolicy,
    backoff_seconds,
    classify_failure,
    is_rate_limit,
    is_transient,
)


@pytest.mark.parametrize("message", [
    "Error: rate limit exceeded, try again later",
    "429 Too Many Requests",
    "You have hit your usage limit for this 5-hour window",
    "API Error: 529 overloaded",
    "quota exceeded",
])
def test_rate_limit_messages_are_recognized(message: str) -> None:
    assert is_rate_limit(message)
    assert isinstance(classify_failure(message), CliRetryableError)


@pytest.mark.parametrize("message", [
    "fetch failed",
    "socket hang up",
    "claude timed out after 900s",
    "502 Bad Gateway",
])
def test_transient_messages_retry_but_are_not_rate_limits(message: str) -> None:
    assert is_transient(message)
    assert not is_rate_limit(message)
    assert isinstance(classify_failure(message), CliRetryableError)


@pytest.mark.parametrize("message", [
    "claude error [error_during_execution]: invalid model name 'nope'",
    "schema validation failed",
    "permission denied",
])
def test_terminal_messages_are_not_retried(message: str) -> None:
    err = classify_failure(message)
    assert isinstance(err, CliBackendError)
    assert not isinstance(err, CliRetryableError)


def test_backoff_grows_exponentially() -> None:
    policy = CliRetryPolicy(max_attempts=5, base_ms=1_000, cap_ms=60_000)
    delays = [
        backoff_seconds(a, "fetch failed", policy, jitter=1.0) for a in (1, 2, 3)
    ]
    assert delays == [1.0, 2.0, 4.0]


def test_backoff_is_capped() -> None:
    policy = CliRetryPolicy(max_attempts=20, base_ms=1_000, cap_ms=5_000)
    assert backoff_seconds(10, "fetch failed", policy, jitter=1.0) == 5.0


def test_rate_limit_backoff_starts_far_higher_than_ordinary_backoff() -> None:
    """A 1s retry against a multi-hour subscription limit is wasted."""
    policy = CliRetryPolicy(max_attempts=5, base_ms=1_000, cap_ms=60_000)

    ordinary = backoff_seconds(1, "fetch failed", policy, jitter=1.0)
    limited = backoff_seconds(1, "usage limit reached", policy, jitter=1.0)

    assert ordinary == 1.0
    assert limited == RATE_LIMIT_BASE_MS / 1000.0
    assert limited > ordinary * 25


def test_rate_limit_backoff_may_exceed_the_ordinary_cap() -> None:
    policy = CliRetryPolicy(max_attempts=8, base_ms=1_000, cap_ms=60_000)
    delay = backoff_seconds(6, "rate limit", policy, jitter=1.0)
    assert delay > 60.0


def test_jitter_keeps_delays_inside_half_to_one_and_a_half() -> None:
    policy = CliRetryPolicy(max_attempts=5, base_ms=2_000, cap_ms=60_000)
    for _ in range(50):
        delay = backoff_seconds(1, "fetch failed", policy)
        assert 1.0 <= delay <= 3.0


def test_policy_reads_the_config_section() -> None:
    from co_scientist.config import Config

    policy = CliRetryPolicy.from_config(Config())
    assert policy.max_attempts >= 1
    assert policy.base_ms > 0
