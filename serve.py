#!/usr/bin/env python3
"""Simulated serving system: one clock, admission control, scheduling, routing.

Per arrival the path is route -> admit -> a worker queue, and every worker
advances through step(). Part 2 runs a single worker so the scheduling policy
is the only variable in the comparison; Part 3 adds a second worker and makes
`router.pick` the only variable instead.

Each worker keeps its own clock. The run loop is a next-event loop: the worker
whose next step is due earliest takes it, and the global clock follows that
worker. With one worker this is exactly the Part 2 loop -- release arrivals,
step, advance -- which is why Part 2's numbers are unchanged by Part 3.

There is no model and no GPU. A step advances integer counters, and the clock
advances by a cost model rather than by measuring anything real.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from admit import AdmissionSnapshot, should_shed
from sched import DrrState, KvPool, ScheduledRequest, step
from state import PendingRequest
from router import STRATEGIES, RouteRequest, Shed, WorkerView, pick


# --- Simulated hardware ----------------------------------------------------
# One step is one forward pass. A decode-only pass costs the same whatever the
# batch size, because every sequence advances together; prefill adds time in
# proportion to the tokens it processes. This difference is the whole reason a
# long prompt hurts everyone else's latency.
STEP_BASE_S = 0.006
PREFILL_TOKENS_PER_S = 20_000.0
TOKEN_BUDGET = 2_048
DECODE_SLOTS = 32
BLOCK_SIZE = 16
# Sized against the decode concurrency it has to support: DECODE_SLOTS
# sequences averaging ~4k tokens each. A pool smaller than that cannot hold one
# full batch, so it preempts continuously and no policy gets to show its shape.
KV_TOTAL_BLOCKS = 8_192  # 131,072 tokens of KV cache.

HORIZON_S = 60.0

# --- Admission inputs ------------------------------------------------------
# Percentiles need history. Until enough exists these estimates stand in, so a
# cold start does not shed every arrival through the fail-closed queue gate.
COLD_START_P50_S = 0.15
COLD_START_P99_S = 0.30
TTFT_WINDOW = 64  # Recent samples the percentiles are computed over.

SHARED_PREFIX_TOKENS = 3_500  # Resident tokens once an agent prefix is seen.

# Deliberately generous: quota refusals would otherwise mask the capacity
# rules, and Part 2 is about scheduling, not billing.
TENANT_TOKEN_LIMIT = 50_000_000
TENANT_REQUEST_LIMIT = 1_000_000

POLICIES = ("fcfs", "priority", "drr")

# Resident prefixes are charged to the pool under their own key, so a shared
# prefix costs KV blocks exactly once per worker that holds it rather than once
# per request that uses it. No ScheduledRequest carries this ID, so sched.py
# never claims or releases it; serve.py owns its whole lifetime.
PREFIX_HOLD_KEY = "prefix:"

DEFAULT_SEED = 7  # Routing draws. Recorded with every result for replay.

BUCKET_S = 1.0  # Timeline resolution for the soak plot.


@dataclass(frozen=True)
class TraceRequest:
    """One line of the workload trace."""

    id: str
    arrival_t: float
    priority: int
    prompt_tokens: int
    max_new_tokens: int
    prefix_hash: str | None
    timeout_s: float
    tenant: str

    @classmethod
    def from_json(cls, line: str) -> TraceRequest:
        return cls(**json.loads(line))


@dataclass(frozen=True)
class Telemetry:
    """One scrape of a worker's load signals.

    Kept as a value so a lagged copy can be handed to the router in place of
    the live one. Prefix residency is deliberately not here: the routing index
    and the load scrape are different data paths, and only the load path is
    made stale in the T3 experiment.
    """

    running: int
    waiting: int
    free_kv_blocks: int
    total_kv_blocks: int
    ttft_p50_s: float
    ttft_p99_s: float
    queue_growing: bool


@dataclass(frozen=True)
class StaleSpec:
    """Make one worker's telemetry arrive `lag_s` late.

    `hidden` is the difference between a router that can see the lag and one
    that cannot: with it set, the view is reported as fresh (age_s = 0) even
    though its numbers are `lag_s` old, which is the only way to observe what
    router.pick would do without H6 -- the rule is unconditional inside
    `pick`, so the experiment lies to it rather than disabling it.
    """

    worker: str
    lag_s: float = 15.0
    hidden: bool = False


@dataclass
class Worker:
    """One simulated GPU: its queues, its memory, its credit, its telemetry."""

    name: str = "w0"
    waiting: list[ScheduledRequest] = field(default_factory=list)
    running: list[ScheduledRequest] = field(default_factory=list)
    kv: KvPool = field(
        default_factory=lambda: KvPool(
            total_blocks=KV_TOTAL_BLOCKS, block_size=BLOCK_SIZE
        )
    )
    drr: DrrState = field(default_factory=DrrState)
    # Prefixes this worker has served. The routing affinity index: it survives
    # the requests that created it, which is what makes a second request with
    # the same prefix worth sending here.
    seen_prefixes: set[str] = field(default_factory=set)
    ttft_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=TTFT_WINDOW)
    )
    previous_queue: int = 0
    next_step_at: float = 0.0  # This worker's own clock.
    # request_id -> the prefix it holds a claim on, while it is in flight here.
    hits: dict[str, str] = field(default_factory=dict)
    # hash -> in-flight requests carrying it. Nonzero means the blocks stand.
    resident: dict[str, int] = field(default_factory=dict)
    history: list[tuple[float, Telemetry]] = field(default_factory=list)
    routed: int = 0
    peak_blocks: int = 0


@dataclass
class Bucket:
    """One second of the run, for the soak plot."""

    arrived: int = 0
    admitted: int = 0
    completed: int = 0
    rejected: int = 0
    ttfts: list[float] = field(default_factory=list)
    kv_blocks: int = 0  # Peak fleet blocks held during this second.


@dataclass
class RunMetrics:
    """Totals for one policy and one routing strategy over one trace."""

    policy: str
    strategy: str = "least_loaded"
    workers: int = 1
    seed: int = DEFAULT_SEED
    prefix_cache: bool = False
    stale: StaleSpec | None = None
    seconds: float = 0.0
    arrived: int = 0
    admitted: int = 0
    completed: int = 0
    steps: int = 0
    preempts: int = 0
    wasted_decode_tokens: int = 0
    rejected_by_code: dict[int, int] = field(default_factory=dict)
    shed_by_router: int = 0  # Refusals raised by pick() rather than by admit().
    ttft_by_tenant: dict[str, list[float]] = field(default_factory=dict)
    routed_by_worker: dict[str, int] = field(default_factory=dict)
    peak_blocks_by_worker: dict[str, int] = field(default_factory=dict)
    kv_block_seconds: dict[str, float] = field(default_factory=dict)
    prefix_block_seconds: dict[str, float] = field(default_factory=dict)
    prefill_tokens: int = 0
    cached_prefill_tokens: int = 0  # Prefill skipped because a prefix was warm.
    timeline: dict[int, Bucket] = field(default_factory=dict)

    @property
    def rejected(self) -> int:
        return sum(self.rejected_by_code.values())

    @property
    def in_flight(self) -> int:
        """Admitted work that never finished inside the horizon."""
        return self.admitted - self.completed

    @property
    def ttfts(self) -> list[float]:
        return [t for samples in self.ttft_by_tenant.values() for t in samples]

    @property
    def kv_total_block_seconds(self) -> float:
        """Fleet KV occupancy over the run, in block-seconds.

        Time-weighted rather than peak: a strategy that holds the same blocks
        for twice as long has used twice the memory, and a peak reading cannot
        see that.
        """
        return sum(self.kv_block_seconds.values())

    @property
    def prefix_total_block_seconds(self) -> float:
        return sum(self.prefix_block_seconds.values())

    def bucket(self, clock: float) -> Bucket:
        return self.timeline.setdefault(int(clock / BUCKET_S), Bucket())


def step_seconds(prefill_tokens: int) -> float:
    """Return how long one step takes, given the prefill work it carried."""
    return STEP_BASE_S + prefill_tokens / PREFILL_TOKENS_PER_S


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; NaN when there is nothing to report."""
    if not values:
        return float("nan")

    ordered = sorted(values)
    rank = math.ceil(q / 100 * len(ordered))
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def load_trace(path: str | Path) -> list[TraceRequest]:
    """Read a JSONL trace, ordered by arrival."""
    with Path(path).open() as handle:
        trace = [TraceRequest.from_json(line) for line in handle if line.strip()]
    return sorted(trace, key=lambda req: req.arrival_t)


def _to_pending(req: TraceRequest, clock: float) -> PendingRequest:
    """Adapt a trace line to the gateway request shape admission expects."""
    return PendingRequest(
        n_in=req.prompt_tokens,
        n_out=req.max_new_tokens,
        deadline_s=req.timeout_s,
        tenant=req.tenant,
        token_ids=[],
        messages=[],
        stream=False,
        body={},
        deadline_at=clock + req.timeout_s,
        priority=req.priority,
        request_class=req.tenant,
    )


def _snapshot(
    req: TraceRequest,
    worker: Worker,
    cached_prefix_tokens: int,
    ttft_samples,
    queue_growing: bool,
    accounts: dict[str, list[int]],
) -> AdmissionSnapshot:
    """Assemble the live signals admission reads for this arrival."""
    samples = list(ttft_samples)
    p50 = percentile(samples, 50) if samples else COLD_START_P50_S
    p99 = percentile(samples, 99) if samples else COLD_START_P99_S

    # Blocks this request still has to fill: a resident prefix costs nothing.
    uncached = req.prompt_tokens - cached_prefix_tokens + req.max_new_tokens
    required = math.ceil(uncached / BLOCK_SIZE)

    tokens_reserved, requests_reserved = accounts.get(req.tenant, [0, 0])
    return AdmissionSnapshot(
        queue_length=len(worker.waiting),
        ttft_p50_s=p50,
        kv_usage=1.0 - worker.kv.free_blocks / worker.kv.total_blocks,
        cached_prefix_tokens=cached_prefix_tokens,
        free_kv_blocks=worker.kv.free_blocks,
        required_kv_blocks=required,
        ttft_p99_s=p99,
        queue_growing=queue_growing,
        # Settled usage is not modeled separately: everything admitted and
        # unfinished counts as reserved until it completes.
        tenant_tokens_used=0,
        tenant_tokens_reserved=tokens_reserved,
        tenant_token_limit=TENANT_TOKEN_LIMIT,
        tenant_requests_used=0,
        tenant_requests_reserved=requests_reserved,
        tenant_request_limit=TENANT_REQUEST_LIMIT,
    )


# --- Telemetry, and the router seam ----------------------------------------


def _telemetry(worker: Worker) -> Telemetry:
    """Scrape one worker's current load signals."""
    samples = list(worker.ttft_samples)
    return Telemetry(
        running=len(worker.running),
        waiting=len(worker.waiting),
        free_kv_blocks=worker.kv.free_blocks,
        total_kv_blocks=worker.kv.total_blocks,
        ttft_p50_s=percentile(samples, 50) if samples else COLD_START_P50_S,
        ttft_p99_s=percentile(samples, 99) if samples else COLD_START_P99_S,
        queue_growing=len(worker.waiting) > worker.previous_queue,
    )


def _lagged(worker: Worker, clock: float, lag_s: float) -> Telemetry:
    """Return the newest scrape at least `lag_s` old, or an empty worker.

    Before the first sample ages out there is nothing to report but the state
    the worker started in -- idle, with all of its memory free. That is the
    whole failure mode: a worker that has not reported yet looks like the most
    attractive target in the fleet.
    """
    cutoff = clock - lag_s
    for sampled_at, telemetry in reversed(worker.history):
        if sampled_at <= cutoff:
            return telemetry

    return Telemetry(
        running=0,
        waiting=0,
        free_kv_blocks=worker.kv.total_blocks,
        total_kv_blocks=worker.kv.total_blocks,
        ttft_p50_s=COLD_START_P50_S,
        ttft_p99_s=COLD_START_P99_S,
        queue_growing=False,
    )


def warm_tokens(worker: Worker, req: TraceRequest, prefix_cache: bool) -> int:
    """Prefix tokens this worker already holds for this request.

    Two definitions, one per mode, and the router and admission always read
    the same one so they cannot disagree about how much memory a request
    still needs.

    With the prefix cache modelled, a worker advertises a prefix only while it
    is actually holding the blocks -- the last request using it drains, the
    prefix goes with it. Without it, Part 2's rule stands: a prefix the worker
    has ever served is credited forever, because there it is only an admission
    discount and no blocks are ever charged for it.
    """
    if req.prefix_hash is None:
        return 0

    if prefix_cache:
        return SHARED_PREFIX_TOKENS if worker.resident.get(req.prefix_hash, 0) else 0

    return SHARED_PREFIX_TOKENS if req.prefix_hash in worker.seen_prefixes else 0


def _view(
    worker: Worker,
    req: TraceRequest,
    clock: float,
    stale: StaleSpec | None,
    prefix_cache: bool = False,
) -> WorkerView:
    """Adapt a Worker into the telemetry view the router reads.

    This is where a worker's numbers become a router input, so it is also
    where staleness is injected: a stale worker reports the scrape from
    `lag_s` ago, with an age the router either can or cannot see.

    Only this request's own prefix is reported as cached. The router looks up
    exactly one hash, and building the whole index per arrival would be
    quadratic in a trace where every request has its own prefix.
    """
    worker.history.append((clock, _telemetry(worker)))

    reported = worker.history[-1][1]
    age_s = 0.0
    if stale is not None and stale.worker == worker.name:
        reported = _lagged(worker, clock, stale.lag_s)
        age_s = 0.0 if stale.hidden else stale.lag_s

    cached: dict[str, int] = {}
    warm = warm_tokens(worker, req, prefix_cache)
    if warm and req.prefix_hash is not None:
        cached = {req.prefix_hash: warm}

    return WorkerView(
        name=worker.name,
        healthy=True,
        age_s=age_s,
        running=reported.running,
        waiting=reported.waiting,
        free_kv_blocks=reported.free_kv_blocks,
        total_kv_blocks=reported.total_kv_blocks,
        cached_prefixes=cached,
        ttft_p50_s=reported.ttft_p50_s,
        ttft_p99_s=reported.ttft_p99_s,
        queue_growing=reported.queue_growing,
    )


def route(
    req: TraceRequest,
    pending: PendingRequest,
    fleet: list[Worker],
    *,
    strategy: str,
    rng: random.Random,
    clock: float,
    stale: StaleSpec | None = None,
    prefix_cache: bool = False,
) -> Worker | None:
    """Choose the worker for one arrival, or None when the fleet refuses it.

    The seam Part 2 filled with `workers[0]`. Everything about the decision
    lives in router.pick; this function only builds the views it reads and
    maps the chosen name back to the worker that owns the queues.
    """
    views = [_view(worker, req, clock, stale, prefix_cache) for worker in fleet]
    decision = pick(
        RouteRequest(pending=pending, prefix_hash=req.prefix_hash),
        views,
        strategy=strategy,
        rng=rng,
    )
    if isinstance(decision, Shed):
        return None

    return next(worker for worker in fleet if worker.name == decision.name)


# --- Prefix residency ------------------------------------------------------


def _claim_prefix(worker: Worker, request_id: str, prefix_hash: str, tokens: int) -> None:
    """Charge this worker for holding `prefix_hash` resident, once.

    The first request carrying a prefix materialises it; every later one that
    overlaps with it joins the same blocks and pays only for its own tail.
    When the last of them leaves, the prefix leaves with it -- which is what
    makes affinity worth anything: a worker that stops receiving a prefix goes
    cold on it, rather than advertising it forever.

    The first request also prefills the whole prompt, so while it runs the
    prefix is counted twice: once inside its own holding and once here. That
    overcharges prefix-aware routing, which is the safe direction for the
    experiment this number is read by.
    """
    count = worker.resident.get(prefix_hash, 0)
    worker.resident[prefix_hash] = count + 1
    worker.hits[request_id] = prefix_hash
    if count == 0:
        worker.kv.held[PREFIX_HOLD_KEY + prefix_hash] = math.ceil(tokens / BLOCK_SIZE)


def _release_prefix(worker: Worker, request_id: str) -> None:
    """Drop this request's claim, freeing the prefix when it was the last."""
    prefix_hash = worker.hits.pop(request_id, None)
    if prefix_hash is None:
        return

    count = worker.resident.get(prefix_hash, 1) - 1
    if count > 0:
        worker.resident[prefix_hash] = count
        return

    worker.resident.pop(prefix_hash, None)
    worker.kv.held.pop(PREFIX_HOLD_KEY + prefix_hash, None)


def _prefix_blocks(worker: Worker) -> int:
    """Blocks this worker holds for resident prefixes rather than requests."""
    return sum(
        blocks
        for request_id, blocks in worker.kv.held.items()
        if request_id.startswith(PREFIX_HOLD_KEY)
    )


def _integrate(metrics: RunMetrics, fleet: list[Worker], clock: float, dt: float) -> None:
    """Add the memory held over an interval to the time-weighted KV totals."""
    if dt <= 0:
        return

    bucket = metrics.bucket(clock)
    total = 0
    for worker in fleet:
        blocks = sum(worker.kv.held.values())
        total += blocks
        metrics.kv_block_seconds[worker.name] += blocks * dt
        metrics.prefix_block_seconds[worker.name] += _prefix_blocks(worker) * dt
        worker.peak_blocks = max(worker.peak_blocks, blocks)
    bucket.kv_blocks = max(bucket.kv_blocks, total)


def run(
    trace: list[TraceRequest],
    policy: str,
    seconds: float = HORIZON_S,
    *,
    workers: int = 1,
    strategy: str = "least_loaded",
    prefix_cache: bool = False,
    stale: StaleSpec | None = None,
    seed: int = DEFAULT_SEED,
) -> RunMetrics:
    """Run one policy and one routing strategy over one trace.

    Arrivals are released when the clock reaches them, routed, screened by
    admission on the worker they were routed to, and queued there. Each worker
    keeps its own clock; the run loop always advances the worker whose next
    step is due earliest, so a long prefill on one GPU does not slow another.

    Args:
        trace: Arrivals, in arrival order.
        policy: Scheduling policy, one of POLICIES.
        seconds: Simulated horizon.
        workers: Fleet size. One reproduces Part 2 exactly.
        strategy: Routing strategy, one of router.STRATEGIES.
        prefix_cache: Whether a warm prefix removes prefill work and KV blocks.
            Off reproduces Part 2, where a resident prefix is only an
            admission discount and the GPU still prefills every token.
        stale: Telemetry lag to inject into one worker, or None.
        seed: Seed for the routing draws.

    Returns:
        Totals, per-worker KV and placement counts, and a per-second timeline.
    """
    fleet = [Worker(name=f"w{i}") for i in range(workers)]
    rng = random.Random(seed)

    metrics = RunMetrics(
        policy=policy,
        strategy=strategy,
        workers=workers,
        seed=seed,
        prefix_cache=prefix_cache,
        stale=stale,
    )
    for worker in fleet:
        metrics.routed_by_worker[worker.name] = 0
        metrics.kv_block_seconds[worker.name] = 0.0
        metrics.prefix_block_seconds[worker.name] = 0.0

    records = {req.id: req for req in trace}
    owner: dict[str, Worker] = {}
    first_token_at: dict[str, float] = {}
    accounts: dict[str, list[int]] = {}

    clock = 0.0
    index = 0

    while clock < seconds:
        # Advance to the next event before releasing arrivals, so a request is
        # admitted against the state the worker is actually in when it lands.
        active = [w for w in fleet if w.waiting or w.running]
        due: Worker | None = None
        if active:
            due = min(active, key=lambda w: (w.next_step_at, w.name))
            advanced = max(clock, due.next_step_at)
        else:
            if index >= len(trace):
                break
            # Idle: jump to the next arrival instead of burning empty steps.
            # Never behind the last step any worker finished -- that work has
            # already happened, and an arrival cannot be served before it.
            advanced = max(clock, trace[index].arrival_t, *(w.next_step_at for w in fleet))

        _integrate(metrics, fleet, clock, advanced - clock)
        clock = advanced
        if clock >= seconds:
            break

        while index < len(trace) and trace[index].arrival_t <= clock:
            req = trace[index]
            index += 1
            metrics.arrived += 1
            metrics.bucket(clock).arrived += 1

            pending = _to_pending(req, clock)
            worker = route(
                req,
                pending,
                fleet,
                strategy=strategy,
                rng=rng,
                clock=clock,
                stale=stale,
                prefix_cache=prefix_cache,
            )
            if worker is None:
                # H4: the fleet refused it. Admission's own code for the same
                # condition is 503, so the two paths land in the same counter.
                metrics.rejected_by_code[503] = metrics.rejected_by_code.get(503, 0) + 1
                metrics.shed_by_router += 1
                metrics.bucket(clock).rejected += 1
                continue

            cached = warm_tokens(worker, req, prefix_cache)
            snapshot = _snapshot(
                req,
                worker,
                cached,
                worker.ttft_samples,
                len(worker.waiting) > worker.previous_queue,
                accounts,
            )
            shed, code, _retry = should_shed(pending, snapshot)
            if shed:
                metrics.rejected_by_code[code] = metrics.rejected_by_code.get(code, 0) + 1
                metrics.bucket(clock).rejected += 1
                continue

            metrics.admitted += 1
            metrics.bucket(clock).admitted += 1
            worker.routed += 1
            metrics.routed_by_worker[worker.name] += 1
            owner[req.id] = worker
            account = accounts.setdefault(req.tenant, [0, 0])
            account[0] += req.prompt_tokens + req.max_new_tokens
            account[1] += 1

            prompt_tokens = req.prompt_tokens
            if prefix_cache and req.prefix_hash is not None:
                warm = min(SHARED_PREFIX_TOKENS, req.prompt_tokens)
                if cached:
                    # Resident here: this request prefills and holds its tail
                    # only, and joins the blocks already standing.
                    prompt_tokens -= warm
                    metrics.cached_prefill_tokens += warm
                _claim_prefix(worker, req.id, req.prefix_hash, warm)
            if req.prefix_hash is not None:
                worker.seen_prefixes.add(req.prefix_hash)

            worker.waiting.append(
                ScheduledRequest(
                    request_id=req.id,
                    arrived_at=req.arrival_t,
                    prompt_tokens=prompt_tokens,
                    priority=req.priority,
                    tenant_id=req.tenant,
                    max_output_tokens=req.max_new_tokens,
                )
            )

        if due is None:
            continue

        due.previous_queue = len(due.waiting)
        result = step(
            due.waiting,
            due.running,
            TOKEN_BUDGET,
            policy=policy,
            drr=due.drr if policy == "drr" else None,
            kv=due.kv,
            decode_slots=DECODE_SLOTS,
        )
        # A token exists once the step that produced it has run, so the clock
        # this step ends on is the one TTFT is measured against.
        finished = clock + step_seconds(result.prefill_tokens)
        due.next_step_at = finished
        metrics.steps += 1
        metrics.preempts += len(result.preempted_request_ids)
        metrics.wasted_decode_tokens += result.wasted_decode_tokens
        metrics.prefill_tokens += result.prefill_tokens

        for request_id in result.decoded_request_ids:
            # Time to first token, measured from arrival. A later preemption
            # discards the tokens but not the fact that the client saw one.
            if request_id not in first_token_at:
                first_token_at[request_id] = finished
                ttft = finished - records[request_id].arrival_t
                due.ttft_samples.append(ttft)
                metrics.ttft_by_tenant.setdefault(
                    records[request_id].tenant, []
                ).append(ttft)
                metrics.bucket(finished).ttfts.append(ttft)

        for request_id in result.completed_request_ids:
            metrics.completed += 1
            metrics.bucket(finished).completed += 1
            done = records[request_id]
            account = accounts[done.tenant]
            account[0] -= done.prompt_tokens + done.max_new_tokens
            account[1] -= 1
            _release_prefix(owner[request_id], request_id)

    metrics.seconds = min(clock, seconds)
    metrics.peak_blocks_by_worker = {w.name: w.peak_blocks for w in fleet}
    return metrics


def report(runs: list[RunMetrics]) -> str:
    """Render the Part 2 comparison table."""
    tenants = sorted({t for run in runs for t in run.ttft_by_tenant})
    rows: list[tuple[str, list[str]]] = [
        ("completed req/s", [f"{r.completed / r.seconds:.2f}" for r in runs]),
        ("completed", [str(r.completed) for r in runs]),
    ]
    for tenant in tenants:
        # The sample count travels with the percentile: a p99 over a handful of
        # requests says more about how few got served than about latency.
        rows.append(
            (
                f"{tenant} p99 TTFT (s)",
                [
                    f"{percentile(r.ttft_by_tenant.get(tenant, []), 99):.2f}"
                    f" (n={len(r.ttft_by_tenant.get(tenant, []))})"
                    for r in runs
                ],
            )
        )
    rows += [
        ("preempts/s", [f"{r.preempts / r.seconds:.2f}" for r in runs]),
        ("preempts", [str(r.preempts) for r in runs]),
        ("wasted decode tokens", [f"{r.wasted_decode_tokens:,}" for r in runs]),
        ("requests rejected", [str(r.rejected) for r in runs]),
        ("  of which 429 / 503", [
            f"{r.rejected_by_code.get(429, 0)} / {r.rejected_by_code.get(503, 0)}" for r in runs
        ]),
        ("admitted", [str(r.admitted) for r in runs]),
        ("unfinished at horizon", [str(r.in_flight) for r in runs]),
        ("steps", [str(r.steps) for r in runs]),
    ]

    width = max(len(label) for label, _ in rows) + 2
    header = "metric".ljust(width) + "".join(r.policy.rjust(14) for r in runs)
    lines = [header, "-" * len(header)]
    lines += [
        label.ljust(width) + "".join(value.rjust(14) for value in values)
        for label, values in rows
    ]
    return "\n".join(lines)


def router_report(runs: list[RunMetrics]) -> str:
    """Render the Part 3 comparison table: one column per routing strategy."""
    names = sorted({name for run in runs for name in run.routed_by_worker})
    rows: list[tuple[str, list[str]]] = [
        ("completed req/s", [f"{r.completed / r.seconds:.2f}" for r in runs]),
        ("completed", [str(r.completed) for r in runs]),
        ("p99 TTFT (s)", [f"{percentile(r.ttfts, 99):.2f}" for r in runs]),
        ("p50 TTFT (s)", [f"{percentile(r.ttfts, 50):.2f}" for r in runs]),
    ]
    for name in names:
        rows.append(
            (
                f"routed to {name}",
                [
                    f"{r.routed_by_worker.get(name, 0)}"
                    f" ({_share(r, name):.0%})"
                    for r in runs
                ],
            )
        )
    rows += [
        ("routing imbalance", [f"{imbalance(r):.3f}" for r in runs]),
        ("KV block-seconds", [f"{r.kv_total_block_seconds:,.0f}" for r in runs]),
        ("  of which prefix", [f"{r.prefix_total_block_seconds:,.0f}" for r in runs]),
        ("peak blocks (fleet)", [
            f"{sum(r.peak_blocks_by_worker.values()):,}" for r in runs
        ]),
        ("prefill tokens", [f"{r.prefill_tokens:,}" for r in runs]),
        ("prefill saved by cache", [f"{r.cached_prefill_tokens:,}" for r in runs]),
        ("shed %", [f"{100 * r.rejected / max(r.arrived, 1):.1f}%" for r in runs]),
        ("  by router / admit", [
            f"{r.shed_by_router} / {r.rejected - r.shed_by_router}" for r in runs
        ]),
        ("preempts", [str(r.preempts) for r in runs]),
        ("wasted decode tokens", [f"{r.wasted_decode_tokens:,}" for r in runs]),
        ("unfinished at horizon", [str(r.in_flight) for r in runs]),
    ]

    width = max(len(label) for label, _ in rows) + 2
    header = "metric".ljust(width) + "".join(r.strategy.rjust(18) for r in runs)
    lines = [header, "-" * len(header)]
    lines += [
        label.ljust(width) + "".join(value.rjust(18) for value in values)
        for label, values in rows
    ]
    return "\n".join(lines)


def _share(run: RunMetrics, name: str) -> float:
    placed = sum(run.routed_by_worker.values())
    return run.routed_by_worker.get(name, 0) / placed if placed else 0.0


def imbalance(run: RunMetrics) -> float:
    """Spread of placements across the fleet: 0 is even, 1 is one worker.

    (max - min) / total, so it does not depend on how many requests were
    admitted and can be compared across strategies and traces.
    """
    counts = list(run.routed_by_worker.values())
    placed = sum(counts)
    if not counts or placed == 0:
        return 0.0

    return (max(counts) - min(counts)) / placed


def as_dict(run: RunMetrics) -> dict:
    """Flatten one run into JSON, including the seed that produced it."""
    return {
        "policy": run.policy,
        "strategy": run.strategy,
        "workers": run.workers,
        "seed": run.seed,
        "prefix_cache": run.prefix_cache,
        "stale": None
        if run.stale is None
        else {
            "worker": run.stale.worker,
            "lag_s": run.stale.lag_s,
            "hidden": run.stale.hidden,
        },
        "seconds": run.seconds,
        "arrived": run.arrived,
        "admitted": run.admitted,
        "completed": run.completed,
        "completed_per_s": run.completed / run.seconds if run.seconds else 0.0,
        "in_flight_at_horizon": run.in_flight,
        "steps": run.steps,
        "preempts": run.preempts,
        "preempts_per_s": run.preempts / run.seconds if run.seconds else 0.0,
        "wasted_decode_tokens": run.wasted_decode_tokens,
        "rejected": run.rejected,
        "rejected_by_code": {str(k): v for k, v in sorted(run.rejected_by_code.items())},
        "shed_by_router": run.shed_by_router,
        "shed_pct": 100 * run.rejected / run.arrived if run.arrived else 0.0,
        "ttft_p50_s": percentile(run.ttfts, 50),
        "ttft_p99_s": percentile(run.ttfts, 99),
        "ttft_p99_by_tenant": {
            tenant: percentile(samples, 99)
            for tenant, samples in sorted(run.ttft_by_tenant.items())
        },
        "ttft_n_by_tenant": {
            tenant: len(samples) for tenant, samples in sorted(run.ttft_by_tenant.items())
        },
        "routed_by_worker": dict(sorted(run.routed_by_worker.items())),
        "routing_imbalance": imbalance(run),
        "kv_block_seconds_by_worker": dict(sorted(run.kv_block_seconds.items())),
        "kv_block_seconds": run.kv_total_block_seconds,
        "prefix_block_seconds": run.prefix_total_block_seconds,
        "peak_blocks_by_worker": dict(sorted(run.peak_blocks_by_worker.items())),
        "peak_blocks": sum(run.peak_blocks_by_worker.values()),
        "prefill_tokens": run.prefill_tokens,
        "cached_prefill_tokens": run.cached_prefill_tokens,
        "timeline": [
            {
                "t": bucket_index * BUCKET_S,
                "arrived": bucket.arrived,
                "admitted": bucket.admitted,
                "completed": bucket.completed,
                "rejected": bucket.rejected,
                "ttft_p99_s": percentile(bucket.ttfts, 99),
                "ttft_p50_s": percentile(bucket.ttfts, 50),
                "kv_blocks": bucket.kv_blocks,
            }
            for bucket_index, bucket in sorted(run.timeline.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="traces/mixed.jsonl")
    parser.add_argument("--seconds", type=float, default=HORIZON_S)
    parser.add_argument("--policy", default="all", choices=("all", *POLICIES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--strategy", default="least_loaded", choices=("all", *STRATEGIES))
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help="a warm prefix removes prefill work and KV blocks, not just admission cost",
    )
    parser.add_argument("--stale-worker", default=None, help="e.g. w1")
    parser.add_argument("--stale-lag", type=float, default=15.0)
    parser.add_argument(
        "--stale-hidden",
        action="store_true",
        help="report the stale view as fresh, so H6 cannot see the lag",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json", default=None, help="also write results here")
    args = parser.parse_args()

    if args.workers > 1 and args.policy == "all":
        parser.error("--workers > 1 compares routing, so pick one --policy")
    if args.workers == 1 and args.strategy == "all":
        parser.error("--strategy all needs more than one worker to mean anything")

    trace = load_trace(args.trace)
    policies = POLICIES if args.policy == "all" else (args.policy,)
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    stale = (
        None
        if args.stale_worker is None
        else StaleSpec(args.stale_worker, args.stale_lag, args.stale_hidden)
    )

    runs = [
        run(
            trace,
            policy,
            args.seconds,
            workers=args.workers,
            strategy=strategy,
            prefix_cache=args.prefix_cache,
            stale=stale,
            seed=args.seed,
        )
        for policy in policies
        for strategy in strategies
    ]

    print(f"{len(trace)} requests over {args.seconds:g}s simulated seconds")
    print(f"{args.workers} worker(s), seed {args.seed}\n")
    print(report(runs) if args.workers == 1 else router_report(runs))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps([as_dict(r) for r in runs], indent=2))


if __name__ == "__main__":
    main()
