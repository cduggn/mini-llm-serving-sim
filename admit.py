"""Admission helpers return a refusal tuple, or None to continue checking."""

from dataclasses import dataclass
from math import isfinite

from state import PendingRequest


AdmissionDecision = tuple[bool, int | None, float]


@dataclass(frozen=True)
class AdmissionSnapshot:
    """Supplied replica signals and accounting for this tenant's quota window."""

    # Waiting ahead: gateway/slot waiters + candidate backend queue.
    # Excludes running requests, this arrival, and unrelated replica queues.
    queue_length: int | None
    ttft_p50_s: float | None  # Recent median backend TTFT, in seconds.
    kv_usage: float | None  # Fraction: 0.95 means 95%.
    cached_prefix_tokens: int | None  # Resident prefix tokens on this replica.
    free_kv_blocks: int | None  # Available after safety margin and reservations.
    required_kv_blocks: int | None  # Additional blocks for uncached input/output.
    ttft_p99_s: float | None  # Recent p99 time to first token, in seconds.
    # Current queue exceeds the previous fresh sample with the same scope.
    queue_growing: bool | None
    tenant_tokens_used: int | None
    tenant_tokens_reserved: int | None  # Accepted work not yet settled.
    tenant_token_limit: int | None  # Same allowance/window as used and reserved.
    tenant_requests_used: int | None  # Dispatched requests in the quota window.
    tenant_requests_reserved: int | None  # Accepted requests awaiting dispatch.
    tenant_request_limit: int | None  # Same allowance/window as request counts.


def should_shed(
    req: PendingRequest,
    snap: AdmissionSnapshot,
) -> AdmissionDecision:
    """Run admission checks in order, returning the first refusal.

    Return (shed, HTTP refusal code, retry-after seconds).
    This function does not reserve capacity, retry, or route requests.
    req.deadline_s must be positive and finite.
    """
    if decision := _check_token_allowance(snap):
        return decision

    if decision := _check_request_allowance(snap):
        return decision

    if decision := _check_queue_wait(req, snap):
        return decision

    if decision := _check_kv_pressure(req, snap):
        return decision

    if decision := _check_kv_capacity(snap):
        return decision

    if decision := _check_tail_latency(req, snap):
        return decision

    return False, None, 0.0


def _check_token_allowance(snap: AdmissionSnapshot) -> AdmissionDecision | None:
    """Check committed token usage before considering the request allowance."""
    return _check_allowance(
        snap.tenant_tokens_used,
        snap.tenant_tokens_reserved,
        snap.tenant_token_limit,
    )


def _check_request_allowance(snap: AdmissionSnapshot) -> AdmissionDecision | None:
    """Check dispatched requests plus reservations against their allowance."""
    return _check_allowance(
        snap.tenant_requests_used,
        snap.tenant_requests_reserved,
        snap.tenant_request_limit,
    )


def _check_allowance(
    used: int | None,
    reserved: int | None,
    limit: int | None,
) -> AdmissionDecision | None:
    """Return 503 for invalid counts or 429 when committed usage reaches 95%."""
    counts = (used, reserved, limit)
    if any(
        not isinstance(n, int) or isinstance(n, bool) or n < 0
        for n in counts
    ):
        return True, 503, 1.0

    committed = used + reserved
    # Integer arithmetic avoids rounding at the 95% boundary.
    # Retry delay is provisional until accounting supplies a suitable delay.
    if limit == 0 or committed * 100 >= limit * 95:
        return True, 429, 1.0

    return None


def _check_queue_wait(
    req: PendingRequest,
    snap: AdmissionSnapshot,
) -> AdmissionDecision | None:
    """Reject unusable queue data or estimated wait above half the timeout."""
    queue_length = snap.queue_length
    p50 = snap.ttft_p50_s
    if (
        not isinstance(queue_length, int)
        or isinstance(queue_length, bool)
        or queue_length < 0
    ):
        return True, 503, 1.0

    # Unknown/invalid p50 rejects even with an empty queue.
    if p50 is None or not isfinite(p50) or p50 <= 0:
        return True, 503, 1.0

    expected_wait_s = queue_length * p50
    if expected_wait_s > 0.5 * req.deadline_s:
        return True, 503, 1.0

    return None


def _check_kv_pressure(
    req: PendingRequest,
    snap: AdmissionSnapshot,
) -> AdmissionDecision | None:
    """Validate KV usage/prefix data and reject uncached work below 8% free."""
    kv_usage = snap.kv_usage
    cached_tokens = snap.cached_prefix_tokens
    if kv_usage is None or not 0.0 <= kv_usage <= 1.0:
        return True, 503, 1.0

    if (
        cached_tokens is None
        or not isinstance(cached_tokens, int)
        or isinstance(cached_tokens, bool)
        or not 0 <= cached_tokens <= req.n_in
    ):
        return True, 503, 1.0

    # Less than 8% free means strictly more than 92% used.
    if kv_usage > 0.92 and cached_tokens == 0:
        return True, 503, 1.0

    return None


def _check_kv_capacity(snap: AdmissionSnapshot) -> AdmissionDecision | None:
    """Require valid block counts and enough capacity for additional tokens."""
    blocks = (snap.free_kv_blocks, snap.required_kv_blocks)
    if any(
        not isinstance(n, int) or isinstance(n, bool) or n < 0
        for n in blocks
    ):
        return True, 503, 1.0

    # Large requests can exceed capacity even below the KV usage ceiling.
    if snap.required_kv_blocks > snap.free_kv_blocks:
        return True, 503, 1.0

    return None


def _check_tail_latency(
    req: PendingRequest,
    snap: AdmissionSnapshot,
) -> AdmissionDecision | None:
    """Shed priority >=10 when the queue grows and p99 TTFT exceeds 4*p50.

    Requires p50 to have passed _check_queue_wait. Unknown growth or invalid
    p99 rejects only when that signal is needed for the priority decision.
    """
    # Priority <10 is protected only from this gate; None gets no exemption.
    priority = 10 if req.priority is None else req.priority
    if not isinstance(priority, int) or isinstance(priority, bool):
        return True, 503, 1.0

    if priority >= 10:
        growing = snap.queue_growing
        if not isinstance(growing, bool):
            return True, 503, 1.0

        if growing:
            p99 = snap.ttft_p99_s
            if p99 is None or not isfinite(p99) or p99 < 0:
                return True, 503, 1.0

            # Both percentiles must cover the same metric and observation window.
            # p50 was already validated by the queue-wait gate.
            if p99 > 4 * snap.ttft_p50_s:
                return True, 503, 1.0

    return None
