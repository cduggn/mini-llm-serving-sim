"""Admission policy cases for gateway.admit.should_shed(req, snap).

Each case assumes the other admission gates pass. The result is
(shed, code, retry_after_seconds); acceptance is (False, None, 0.0).
Queue tests check length times p50 TTFT; live queue aggregation is pending.
KV tests supply replica usage and prefix residency; they do not test lookups.
Block requirements are supplied estimates; estimator implementation/tests are pending.
TTFT tests supply p50/p99 directly; histogram percentile calculation is pending.
Queue growth is supplied; live trend calculation is pending.
Quota tests supply accounting values; reservation and settlement are pending.
"""

import pytest

from admit import AdmissionSnapshot, should_shed
from state import PendingRequest


def _req(
    deadline_s: float = 2.0,
    n_in: int = 200,
    request_class: str = "interactive",
    priority: int | None = None,
) -> PendingRequest:
    return PendingRequest(
        n_in=n_in,
        n_out=100,
        deadline_s=deadline_s,
        tenant="chat",
        token_ids=[1] * n_in,
        messages=[],
        stream=False,
        body={},
        deadline_at=10.0 + deadline_s,
        deadline_ms=int(deadline_s * 1000),
        request_class=request_class,
        priority=priority,
    )


def _snap(**overrides) -> AdmissionSnapshot:
    """Supply healthy inputs for gates outside the case under test."""
    values = dict(
        queue_length=0,
        ttft_p50_s=0.5,
        kv_usage=0.2,
        cached_prefix_tokens=0,
        free_kv_blocks=100,
        required_kv_blocks=20,
        ttft_p99_s=1.0,
        queue_growing=False,
        tenant_tokens_used=0,
        tenant_tokens_reserved=0,
        tenant_token_limit=100_000,
        tenant_requests_used=0,
        tenant_requests_reserved=0,
        tenant_request_limit=1_000,
    )
    values.update(overrides)
    return AdmissionSnapshot(**values)


@pytest.mark.parametrize(
    "used, reserved, limit, expected",
    [
        pytest.param(94, 0, 100, (False, None, 0.0), id="below-threshold"),
        pytest.param(95, 0, 100, (True, 429, 1.0), id="at-threshold"),
        pytest.param(96, 0, 100, (True, 429, 1.0), id="above-threshold"),
        pytest.param(94, 1, 100, (True, 429, 1.0), id="reserved-reaches-threshold"),
        pytest.param(90, 6, 100, (True, 429, 1.0), id="reserved-exceeds-threshold"),
        pytest.param(0, 0, 0, (True, 429, 1.0), id="no-allowance"),
    ],
)
def test_token_allowance_admission(used, reserved, limit, expected):
    """Reject when used plus reserved tokens reach 95% of the allowance."""
    snap = _snap(
        tenant_tokens_used=used,
        tenant_tokens_reserved=reserved,
        tenant_token_limit=limit,
    )

    assert should_shed(_req(), snap) == expected


@pytest.mark.parametrize(
    "field",
    ["tenant_tokens_used", "tenant_tokens_reserved", "tenant_token_limit"],
)
@pytest.mark.parametrize(
    "invalid_count",
    [None, -1, 1.5, True],
    ids=["missing", "negative", "fractional", "boolean"],
)
def test_sheds_invalid_token_accounting(field, invalid_count):
    """Unusable accounting returns 503, not a quota-exceeded 429."""
    snap = _snap(**{field: invalid_count})

    assert should_shed(_req(), snap) == (True, 503, 1.0)


@pytest.mark.parametrize(
    "used, reserved, limit, expected",
    [
        pytest.param(94, 0, 100, (False, None, 0.0), id="below-threshold"),
        pytest.param(95, 0, 100, (True, 429, 1.0), id="at-threshold"),
        pytest.param(96, 0, 100, (True, 429, 1.0), id="above-threshold"),
        pytest.param(94, 1, 100, (True, 429, 1.0), id="reserved-reaches-threshold"),
        pytest.param(90, 6, 100, (True, 429, 1.0), id="reserved-exceeds-threshold"),
        pytest.param(0, 0, 0, (True, 429, 1.0), id="no-allowance"),
    ],
)
def test_request_allowance_admission(used, reserved, limit, expected):
    """Reject when dispatched plus reserved requests reach 95% of allowance."""
    snap = _snap(
        tenant_requests_used=used,
        tenant_requests_reserved=reserved,
        tenant_request_limit=limit,
    )

    assert should_shed(_req(), snap) == expected


@pytest.mark.parametrize(
    "field",
    ["tenant_requests_used", "tenant_requests_reserved", "tenant_request_limit"],
)
@pytest.mark.parametrize(
    "invalid_count",
    [None, -1, 1.5, True],
    ids=["missing", "negative", "fractional", "boolean"],
)
def test_sheds_invalid_request_accounting(field, invalid_count):
    """Unusable request accounting returns 503, not a quota-exceeded 429."""
    snap = _snap(**{field: invalid_count})

    assert should_shed(_req(), snap) == (True, 503, 1.0)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        pytest.param(
            {"tenant_tokens_used": 96_000, "tenant_requests_used": None},
            (True, 429, 1.0),
            id="token-limit-before-invalid-request-count",
        ),
        pytest.param(
            {"tenant_tokens_used": None, "tenant_requests_used": 960},
            (True, 503, 1.0),
            id="invalid-token-count-before-request-limit",
        ),
        pytest.param(
            {"tenant_requests_used": 960, "queue_length": None},
            (True, 429, 1.0),
            id="request-limit-before-invalid-queue",
        ),
    ],
)
def test_quota_checks_take_precedence(overrides, expected):
    """The first refusal wins: token accounting, request accounting, then load."""
    assert should_shed(_req(), _snap(**overrides)) == expected


@pytest.mark.parametrize(
    "depth, p50, timeout, expected",
    [
        pytest.param(0, 0.5, 2.0, (False, None, 0.0), id="empty-queue"),
        pytest.param(1, 0.5, 2.0, (False, None, 0.0), id="below-half"),
        pytest.param(2, 0.5, 2.0, (False, None, 0.0), id="exactly-half"),
        pytest.param(3, 0.5, 2.0, (True, 503, 1.0), id="above-half"),
        pytest.param(3, 0.25, 2.0, (False, None, 0.0), id="lower-p50"),
        pytest.param(3, 0.5, 4.0, (False, None, 0.0), id="longer-timeout"),
    ],
)
def test_queue_wait_admission(depth, p50, timeout, expected):
    """Compare queue length times median TTFT with half the request timeout."""
    req = _req(deadline_s=timeout)
    snap = _snap(queue_length=depth, ttft_p50_s=p50)

    assert should_shed(req, snap) == expected


@pytest.mark.parametrize(
    "depth",
    [None, -1, 1.5, True],
    ids=["missing", "negative", "fractional", "boolean"],
)
def test_sheds_invalid_queue_length(depth):
    """The waiting count must be a nonnegative integer."""
    assert should_shed(_req(), _snap(queue_length=depth)) == (True, 503, 1.0)


@pytest.mark.parametrize(
    "p50",
    [None, 0.0, -0.1, float("nan"), float("inf")],
    ids=["missing", "zero", "negative", "nan", "infinite"],
)
def test_sheds_invalid_p50_even_with_empty_queue(p50):
    """No cold-start fallback: unusable p50 rejects even when no work waits."""
    snap = _snap(queue_length=0, ttft_p50_s=p50)

    assert should_shed(_req(), snap) == (True, 503, 1.0)


@pytest.mark.parametrize(
    "usage, cached_tokens, expected",
    [
        pytest.param(0.91, 0, (False, None, 0.0), id="more-than-8-percent-free"),
        pytest.param(0.92, 0, (False, None, 0.0), id="exactly-8-percent-free"),
        pytest.param(0.93, 0, (True, 503, 1.0), id="less-than-8-percent-new-prefix"),
        pytest.param(0.93, 160, (False, None, 0.0), id="less-than-8-percent-cached"),
        pytest.param(0.95, 0, (True, 503, 1.0), id="5-percent-new-prefix"),
        pytest.param(0.95, 160, (False, None, 0.0), id="5-percent-cached"),
        pytest.param(1.00, 0, (True, 503, 1.0), id="full-new-prefix"),
    ],
)
def test_kv_pressure_admission(usage, cached_tokens, expected):
    """Below 8% KV free, require a cached prefix and room for new tokens."""
    snap = _snap(
        kv_usage=usage,
        cached_prefix_tokens=cached_tokens,
        free_kv_blocks=25,
        required_kv_blocks=20,
    )

    assert should_shed(_req(), snap) == expected


@pytest.mark.parametrize(
    "kv_usage",
    [None, -0.1, 1.1, float("nan"), float("inf")],
    ids=["missing", "negative", "above-one", "nan", "infinite"],
)
def test_sheds_invalid_kv_usage(kv_usage):
    """Unknown or invalid KV fractions cannot establish available capacity."""
    snap = _snap(
        queue_length=0,
        ttft_p50_s=0.5,
        kv_usage=kv_usage,
        cached_prefix_tokens=0,
        free_kv_blocks=100,
        required_kv_blocks=20,
    )

    assert should_shed(_req(), snap) == (True, 503, 1.0)


@pytest.mark.parametrize(
    "cached_tokens",
    [None, -1, 201, 1.5, True],
    ids=["missing", "negative", "exceeds-prompt", "fractional", "boolean"],
)
def test_sheds_invalid_cached_prefix_tokens(cached_tokens):
    """Cached tokens must be an integer between zero and the prompt length."""
    snap = _snap(
        queue_length=0,
        ttft_p50_s=0.5,
        kv_usage=0.2,
        cached_prefix_tokens=cached_tokens,
        free_kv_blocks=100,
        required_kv_blocks=20,
    )

    assert should_shed(_req(), snap) == (True, 503, 1.0)


@pytest.mark.parametrize(
    "free_blocks, required_blocks, expected",
    [
        pytest.param(10, 9, (False, None, 0.0), id="sufficient-capacity"),
        pytest.param(9, 9, (False, None, 0.0), id="exact-fit"),
        pytest.param(8, 9, (True, 503, 1.0), id="growth-exceeds-capacity"),
        pytest.param(0, 1, (True, 503, 1.0), id="no-free-blocks"),
    ],
)
def test_existing_prefix_under_kv_pressure(free_blocks, required_blocks, expected):
    """At 95% KV usage, a resident prefix still needs room for new tokens."""
    snap = _snap(
        queue_length=0,
        ttft_p50_s=0.5,
        kv_usage=0.95,
        cached_prefix_tokens=160,
        free_kv_blocks=free_blocks,
        required_kv_blocks=required_blocks,
    )

    assert should_shed(_req(), snap) == expected


def test_sheds_long_request_below_kv_ceiling():
    """A large new prompt must fit even when KV usage is only 20%."""
    req = _req(n_in=10_000)
    snap = _snap(
        queue_length=0,
        ttft_p50_s=0.5,
        kv_usage=0.2,
        cached_prefix_tokens=0,
        free_kv_blocks=100,
        required_kv_blocks=632,
    )

    assert should_shed(req, snap) == (True, 503, 1.0)


@pytest.mark.parametrize("field", ["free_kv_blocks", "required_kv_blocks"])
@pytest.mark.parametrize(
    "invalid_count",
    [None, -1, 1.5, True, float("nan")],
    ids=["missing", "negative", "fractional", "boolean", "nan"],
)
def test_sheds_invalid_kv_block_counts(field, invalid_count):
    """Both block counts must be nonnegative integers."""
    values = dict(
        queue_length=0,
        ttft_p50_s=0.5,
        kv_usage=0.2,
        cached_prefix_tokens=0,
        free_kv_blocks=100,
        required_kv_blocks=20,
    )
    values[field] = invalid_count

    assert should_shed(_req(), _snap(**values)) == (True, 503, 1.0)


@pytest.mark.parametrize(
    "priority, p50, p99, growing, expected",
    [
        pytest.param(9, 0.5, 3.0, True, (False, None, 0.0), id="protected-priority"),
        pytest.param(10, 0.5, 3.0, True, (True, 503, 1.0), id="priority-cutoff"),
        pytest.param(11, 0.5, 3.0, True, (True, 503, 1.0), id="above-cutoff"),
        pytest.param(10, 0.5, 2.0, True, (False, None, 0.0), id="exactly-four-times"),
        pytest.param(10, 1.0, 3.0, True, (False, None, 0.0), id="below-four-times"),
        pytest.param(10, 0.5, 3.0, False, (False, None, 0.0), id="queue-not-growing"),
        pytest.param(None, 0.5, 3.0, True, (True, 503, 1.0), id="missing-priority"),
    ],
)
def test_tail_latency_admission(priority, p50, p99, growing, expected):
    """Shed priority >=10 when p99 exceeds 4*p50 and the queue grows."""
    req = _req(priority=priority)
    snap = _snap(
        queue_length=1,
        ttft_p50_s=p50,
        ttft_p99_s=p99,
        queue_growing=growing,
    )

    assert should_shed(req, snap) == expected


@pytest.mark.parametrize(
    "priority, growing, expected",
    [
        (9, True, (False, None, 0.0)),
        (10, True, (True, 503, 1.0)),
        (10, False, (False, None, 0.0)),
    ],
)
@pytest.mark.parametrize(
    "p99",
    [None, -0.1, float("nan"), float("inf")],
    ids=["missing", "negative", "nan", "infinite"],
)
def test_invalid_p99_only_sheds_unprotected_growing_queue(priority, growing, expected, p99):
    """p99 is needed only for unprotected requests when the queue grows."""
    snap = _snap(queue_length=1, ttft_p99_s=p99, queue_growing=growing)

    assert should_shed(_req(priority=priority), snap) == expected


@pytest.mark.parametrize(
    "priority, expected",
    [(9, (False, None, 0.0)), (10, (True, 503, 1.0))],
)
@pytest.mark.parametrize("growing", [None, 0, "true"], ids=["missing", "integer", "string"])
def test_invalid_growth_signal(priority, expected, growing):
    """An unknown/nonboolean trend cannot establish safety for priority >=10."""
    snap = _snap(queue_length=1, ttft_p99_s=3.0, queue_growing=growing)

    assert should_shed(_req(priority=priority), snap) == expected


@pytest.mark.parametrize("priority", [True, 1.5, "10"], ids=["boolean", "fractional", "string"])
def test_sheds_invalid_priority(priority):
    """Explicit priorities must be integers; None is handled as priority 10."""
    assert should_shed(_req(priority=priority), _snap()) == (True, 503, 1.0)


@pytest.mark.parametrize(
    "request_class, priority, expected",
    [("batch", 9, (False, None, 0.0)), ("interactive", 10, (True, 503, 1.0))],
)
def test_numeric_priority_controls_latency_exemption(request_class, priority, expected):
    """The old request-class label no longer decides this gate."""
    snap = _snap(queue_length=1, ttft_p99_s=3.0, queue_growing=True)

    assert should_shed(_req(request_class=request_class, priority=priority), snap) == expected


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"queue_length": 3, "ttft_p50_s": 0.5}, 503),
        ({"kv_usage": 0.93, "cached_prefix_tokens": 0}, 503),
        ({"free_kv_blocks": 10, "required_kv_blocks": 20}, 503),
        ({"tenant_tokens_used": 96_000}, 429),
        ({"tenant_requests_used": 960}, 429),
    ],
    ids=["excessive-wait", "new-prefix-under-pressure", "insufficient-blocks", "tokens", "requests"],
)
def test_protected_priority_still_checks_other_gates(overrides, code):
    """The priority exemption does not override quotas, queue wait, or KV limits."""
    values = dict(queue_length=1, ttft_p99_s=3.0, queue_growing=True)
    values.update(overrides)
    snap = _snap(**values)

    assert should_shed(_req(priority=9), snap) == (True, code, 1.0)
