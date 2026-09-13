#!/usr/bin/env python3
"""Routing: which worker gets this request, and when no worker should.

`pick(req, workers)` reads one telemetry view per worker and returns the
worker to send the request to, or a `Shed` when the answer is nobody. Four
strategies decide *which* healthy worker wins; two rules decide *whether*
there is one to win at all, and those two outrank every strategy:

    H6  An unknown worker is never treated as an idle one.
    H4  A request no worker would accept is refused here, not bounced.

Both are applied in `pick` before a strategy runs, so the candidate list a
strategy sees is already known-good and already willing. A strategy cannot
forget a safety rule it never had the chance to break.

This is the routing half of the simulation: no GPU, no network, no model.
It was extracted from a live vLLM gateway, whose own router is a different
file with a different job: this one decides, that one also scrapes, tracks
freshness and maintains the prefix index.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

from admit import AdmissionSnapshot, should_shed
from state import PendingRequest


RoutingStrategy = Literal["random", "least_loaded", "p2c", "prefix_then_load"]

STRATEGIES: tuple[RoutingStrategy, ...] = (
    "random",
    "least_loaded",
    "p2c",
    "prefix_then_load",
)

# Telemetry older than this counts as unknown rather than current. The scrape
# interval sets the floor; this leaves a few intervals of grace before a
# worker goes dark.
STALE_AFTER_S = 5.0

BLOCK_SIZE = 16  # Matches sched.py's KvPool: blocks are what admission counts.

# Seeded at import, so replaying one trace routes identically on every run and
# two strategies can be compared on the same draws. Pass rng= to pick() to
# vary it deliberately; nothing here reads the global random module.
ROUTER_SEED = 7
_RNG = random.Random(ROUTER_SEED)

# Tenant allowance belongs to the tenant, not to a worker: a 429 follows the
# client to every worker in the fleet, so the gateway settles it once before
# routing. The router asks admission only the capacity question, and supplies
# quota inputs that pass so gates 1 and 2 cannot stand in for a capacity
# answer. Zero committed against an allowance of one is 0% used.
_QUOTA_UNUSED = 1


@dataclass(frozen=True)
class Shed:
    """A refusal to route: no worker in the fleet would accept this request.

    Returned in place of a worker so a caller never has to wonder whether it
    was handed a real placement. The code is admission's 503 — server
    capacity, not tenant budget — so retrying later is worthwhile and
    retrying immediately elsewhere is not.
    """

    code: int = 503
    retry_after: float = 2.0  # Seconds.
    reason: str = "no_worker_would_accept"


@dataclass(frozen=True)
class WorkerView:
    """What the router knows about one worker, as of `age_s` ago.

    Every signal but the name and health is `| None`, because telemetry goes
    missing as easily as it goes wrong, and the distance between "empty" and
    "not reporting" is the whole of H6.

    cached_prefixes maps a prefix hash to the tokens of that prefix resident
    on this worker; a hash that is absent is not cached there. Health and
    freshness are reported about the *link* to the worker, so a view can be
    unhealthy or stale while still carrying the last numbers that arrived.
    """

    name: str
    healthy: bool = True
    age_s: float = 0.0  # Seconds since this view was sampled.
    running: int | None = None
    waiting: int | None = None  # Queued ahead of this arrival; excludes running.
    free_kv_blocks: int | None = None
    total_kv_blocks: int | None = None
    cached_prefixes: dict[str, int] = field(default_factory=dict)
    ttft_p50_s: float | None = None
    ttft_p99_s: float | None = None
    # Whether this worker's queue grew since the previous sample. Admission
    # reads it when deciding whether to shed batch traffic.
    queue_growing: bool = False


@dataclass(frozen=True)
class RouteRequest:
    """One arrival, paired with the prefix identity the router routes on.

    `pending` is handed to should_shed unchanged, so the router and the
    gateway ask admission the identical question about the identical request.
    `prefix_hash` is the workload's identity for a shared prompt; None means
    there is nothing to reuse anywhere.
    """

    pending: PendingRequest
    prefix_hash: str | None = None


def pick(
    req: RouteRequest,
    workers: list[WorkerView],
    *,
    strategy: RoutingStrategy = "prefix_then_load",
    rng: random.Random | None = None,
) -> WorkerView | Shed:
    """Choose the worker for one request, or refuse on the fleet's behalf.

    Args:
        req: The arrival, carrying the request admission reads and the prefix
            hash the cache-aware strategy routes on.
        workers: Telemetry views, one per worker. Order matters only for
            exact ties, which keep the first worker listed.
        strategy: Which of STRATEGIES decides between acceptable workers.
            The default degrades to least_loaded when no prefix is resident,
            so it is never worse than picking by load alone.
        rng: Draw source for the sampling strategies, or None for the
            module's seeded one.

    Returns:
        The chosen view — its `.name` identifies the worker — or a Shed when
        every worker would refuse the request.

    Raises:
        ValueError: Unknown strategy.

    Reads state only: nothing here reserves capacity, records a placement, or
    mutates a view.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")

    # No fleet is the degenerate case of every worker refusing.
    if not workers:
        return Shed(503, retry_after=2.0, reason="no_workers")

    candidates, fleet_is_dark = _eligible(workers)  # H6.
    # H4, and only where it can be answered: admission needs telemetry, so a
    # dark fleet is routed on hope rather than refused wholesale. See
    # _admissible for why that trade is the right way round.
    if not fleet_is_dark:
        candidates = _admissible(req, candidates)
        if not candidates:
            # H4: nobody can take it, so say so once, here. The alternative is
            # the bounce: A refuses, the client retries, B refuses, the client
            # retries A, and the request burns the fleet's capacity being
            # rejected instead of being told to come back later.
            return Shed(503, retry_after=2.0)

    return _choose(req, candidates, strategy, _RNG if rng is None else rng)


def _eligible(workers: list[WorkerView]) -> tuple[list[WorkerView], bool]:
    """H6: unknown is not idle. Keep only the workers we can vouch for.

    A worker whose telemetry stopped arriving keeps reporting whatever it last
    said, and a worker that never reported reports nothing at all — either way
    it looks like the emptiest, most attractive target in the fleet, so every
    request piles onto the one machine we have the least reason to trust. The
    quietest worker is usually the broken one.

    Returns:
        The candidate set, and whether the entire fleet is unknown — the one
        case where unknown workers are all there is, so excluding them would
        turn a monitoring outage into a total outage.
    """
    known = [view for view in workers if not _is_unknown(view)]
    if known:
        return known, False

    return list(workers), True


def _is_unknown(view: WorkerView) -> bool:
    """Report whether this view can answer "how loaded is that worker?".

    Three ways to fail, treated identically because they have identical
    consequences: the worker is failing its health check, its numbers are too
    old to describe the present, or a signal the decision needs is missing.
    """
    if not view.healthy:
        return True

    if view.age_s > STALE_AFTER_S:
        return True

    return any(
        signal is None
        for signal in (
            view.running,
            view.waiting,
            view.free_kv_blocks,
            view.total_kv_blocks,
            view.ttft_p50_s,
            view.ttft_p99_s,
        )
    )


def _admissible(
    req: RouteRequest,
    candidates: list[WorkerView],
) -> list[WorkerView]:
    """H4: keep the workers that would accept this request if it arrived.

    Asked before choosing rather than after being refused. Skipped entirely
    when the whole fleet is unknown: admission fails closed, so asking it a
    question the telemetry cannot answer would refuse every request in the
    fleet for as long as the scrape is down. That is the one place this module
    deliberately fails open, and the gateway's own admission pass has already
    screened the request before it reached the router.
    """
    return [view for view in candidates if not _would_shed(req, view)]


def _would_shed(req: RouteRequest, view: WorkerView) -> bool:
    """Run Part 1's admission gates against one worker's telemetry.

    should_shed is called rather than re-expressed, so the rule the router
    predicts and the rule the worker applies cannot drift apart. Its refusal
    code is discarded here: the router only needs to know whether this worker
    is a dead end.
    """
    shed, _code, _retry_after = should_shed(req.pending, _snapshot(req, view))
    return shed


def _snapshot(req: RouteRequest, view: WorkerView) -> AdmissionSnapshot:
    """Assemble what admission reads about this request on this worker.

    Missing signals are passed through as None rather than filled in, so
    should_shed keeps failing closed on them. Normally _eligible has already
    excluded such a worker; this keeps the two rules independent instead of
    making one depend on the other having run.
    """
    cached_tokens = _cached_tokens(req, view)
    kv_usage = None
    if view.free_kv_blocks is not None and view.total_kv_blocks:
        kv_usage = 1.0 - view.free_kv_blocks / view.total_kv_blocks

    return AdmissionSnapshot(
        queue_length=view.waiting,
        ttft_p50_s=view.ttft_p50_s,
        kv_usage=kv_usage,
        cached_prefix_tokens=cached_tokens,
        free_kv_blocks=view.free_kv_blocks,
        required_kv_blocks=_required_blocks(req, cached_tokens),
        ttft_p99_s=view.ttft_p99_s,
        queue_growing=view.queue_growing,
        tenant_tokens_used=0,
        tenant_tokens_reserved=0,
        tenant_token_limit=_QUOTA_UNUSED,
        tenant_requests_used=0,
        tenant_requests_reserved=0,
        tenant_request_limit=_QUOTA_UNUSED,
    )


def _cached_tokens(req: RouteRequest, view: WorkerView) -> int:
    """Tokens of this request's prompt already resident on this worker.

    Capped at the prompt length: admission treats a cached prefix as a
    discount on this request's own input, so it can never exceed it.
    """
    if req.prefix_hash is None:
        return 0

    return min(view.cached_prefixes.get(req.prefix_hash, 0), req.pending.n_in)


def _required_blocks(req: RouteRequest, cached_tokens: int) -> int:
    """Blocks this request still has to fill; a resident prefix costs nothing.

    Matches the estimate serve.py hands admission, and the ceil(tokens / 16)
    that sched.py charges as the work actually runs. If the three disagree,
    the router predicts refusals that never happen, or misses ones that do.
    """
    uncached = req.pending.n_in - cached_tokens + req.pending.n_out
    return math.ceil(max(uncached, 0) / BLOCK_SIZE)


def _choose(
    req: RouteRequest,
    candidates: list[WorkerView],
    strategy: RoutingStrategy,
    rng: random.Random,
) -> WorkerView:
    """Apply the selected strategy to workers H6 and H4 have already cleared.

    Every candidate here is known and would accept the request, so a strategy
    answers only the cheap question: which of several acceptable workers is
    the best of them. None of them can shed, and none of them can pick a dark
    worker, because neither is theirs to decide.
    """
    if strategy == "random":
        return rng.choice(candidates)

    if strategy == "p2c":
        return _power_of_two(candidates, rng)

    if strategy == "prefix_then_load":
        return _prefix_then_load(req, candidates)

    return _least_loaded(candidates)


def _load_key(view: WorkerView) -> tuple[float, float]:
    """Sort key for load, lowest first: work in front, then memory to spare.

    Requests already on the worker are what this arrival waits behind, and
    free blocks break the tie because the request has to fit as well as queue.

    H6 again, in depth: a view missing a count sorts as maximally loaded, so
    even if an unknown worker reaches a strategy it can never win by looking
    empty. The guard in _eligible is the rule; this is the rule holding when
    the guard is bypassed.
    """
    if view.running is None or view.waiting is None:
        return math.inf, 0.0

    free_blocks = 0 if view.free_kv_blocks is None else view.free_kv_blocks
    return float(view.running + view.waiting), -float(free_blocks)


def _least_loaded(candidates: list[WorkerView]) -> WorkerView:
    """Choose the shortest queue; an exact tie keeps the first worker listed."""
    return min(candidates, key=_load_key)


def _power_of_two(candidates: list[WorkerView], rng: random.Random) -> WorkerView:
    """Score two workers drawn at random and take the better of them.

    Sampling beats scanning the whole fleet under load: everyone reading the
    same telemetry agrees on the same best worker and sends it everything,
    and the herd lands a scrape interval before the numbers that would have
    warned them off. Two choices remove nearly all of that pile-up for the
    cost of one comparison.
    """
    sample = rng.sample(candidates, 2) if len(candidates) > 1 else candidates
    return _least_loaded(sample)


def _prefix_then_load(req: RouteRequest, candidates: list[WorkerView]) -> WorkerView:
    """Choose the largest resident prefix, breaking ties by load.

    A warm worker skips the prefill for those tokens, which is the most
    expensive thing this request would otherwise ask for. Load still decides
    between equally warm workers, so cache affinity concentrates traffic only
    as far as the queues allow.

    When no worker holds the prefix every candidate ties at zero tokens and
    this is exactly _least_loaded, so a cache miss needs no special case.
    Nor does a warm worker that is out of memory: H4 removed it already.
    """
    warmest = max(_cached_tokens(req, view) for view in candidates)
    return _least_loaded(
        [view for view in candidates if _cached_tokens(req, view) == warmest]
    )
