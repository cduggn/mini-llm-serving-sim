"""Routing cases for router.pick(req, workers) — the simulated exercise router.

Each case builds a fleet of telemetry views and asks where one request goes.
The result is a WorkerView, or a Shed when no worker would accept it.
Telemetry is supplied directly; scraping, freshness tracking and the live
prefix index belong to the gateway this was extracted from and are not
exercised here.
"""

import random

import pytest

from state import PendingRequest
from router import (
    STALE_AFTER_S,
    STRATEGIES,
    RouteRequest,
    Shed,
    WorkerView,
    pick,
)


def _view(name: str, **overrides) -> WorkerView:
    """Supply healthy, fresh telemetry for everything the case is not testing."""
    values = dict(
        healthy=True,
        age_s=0.0,
        running=1,
        waiting=1,
        free_kv_blocks=4_000,
        total_kv_blocks=8_192,
        ttft_p50_s=0.15,
        ttft_p99_s=0.30,
        queue_growing=False,
    )
    values.update(overrides)
    return WorkerView(name=name, **values)


def _req(
    prefix_hash: str | None = None,
    n_in: int = 200,
    n_out: int = 100,
    priority: int | None = 0,
    deadline_s: float = 30.0,
) -> RouteRequest:
    return RouteRequest(
        pending=PendingRequest(
            n_in=n_in,
            n_out=n_out,
            deadline_s=deadline_s,
            tenant="chat",
            token_ids=[1] * n_in,
            messages=[],
            stream=False,
            body={},
            deadline_at=10.0 + deadline_s,
            priority=priority,
            request_class="interactive",
        ),
        prefix_hash=prefix_hash,
    )


# --- The four strategies ---------------------------------------------------


def test_least_loaded_picks_the_shortest_queue():
    """Load is what this arrival waits behind: running plus waiting."""
    busy = _view("busy", running=8, waiting=8)
    quiet = _view("quiet", running=1, waiting=1)

    assert pick(_req(), [busy, quiet], strategy="least_loaded") is quiet


def test_least_loaded_breaks_an_equal_queue_on_free_memory():
    """Equal queues are separated by which worker the request still fits in."""
    tight = _view("tight", running=2, waiting=2, free_kv_blocks=500)
    roomy = _view("roomy", running=2, waiting=2, free_kv_blocks=6_000)

    assert pick(_req(), [tight, roomy], strategy="least_loaded") is roomy


def test_random_reaches_every_worker():
    """Random spreads over the fleet rather than favouring a position."""
    workers = [_view("a"), _view("b"), _view("c")]
    rng = random.Random(3)

    chosen = {pick(_req(), workers, strategy="random", rng=rng).name for _ in range(60)}

    assert chosen == {"a", "b", "c"}


def test_random_is_reproducible_from_a_seed():
    """Two runs of one seed route identically, so experiments can be compared."""
    workers = [_view("a"), _view("b"), _view("c")]

    first = [pick(_req(), workers, strategy="random", rng=random.Random(11)).name]
    second = [pick(_req(), workers, strategy="random", rng=random.Random(11)).name]

    assert first == second


def test_p2c_never_picks_the_worst_of_three():
    """The most loaded worker only wins if both draws land on it, and they cannot."""
    workers = [
        _view("worst", running=20, waiting=20),
        _view("middle", running=5, waiting=5),
        _view("best", running=0, waiting=0),
    ]
    rng = random.Random(5)

    chosen = {pick(_req(), workers, strategy="p2c", rng=rng).name for _ in range(60)}

    assert chosen == {"middle", "best"}


def test_prefix_then_load_prefers_a_warm_worker():
    """A resident prefix is prefill the request does not have to pay for."""
    warm = _view("warm", running=6, waiting=6, cached_prefixes={"agent": 150})
    cold = _view("cold", running=0, waiting=0)

    req = _req(prefix_hash="agent")

    assert pick(req, [warm, cold], strategy="prefix_then_load") is warm


def test_prefix_then_load_breaks_a_tie_on_load():
    """Cache affinity chooses the set; load chooses inside it."""
    busy = _view("busy", running=9, waiting=9, cached_prefixes={"agent": 150})
    quiet = _view("quiet", running=1, waiting=1, cached_prefixes={"agent": 150})

    req = _req(prefix_hash="agent")

    assert pick(req, [busy, quiet], strategy="prefix_then_load") is quiet


def test_a_prefix_miss_falls_back_to_load():
    """With nothing cached anywhere every worker ties at zero tokens."""
    someone_elses = _view("other", running=0, waiting=0, cached_prefixes={"chat": 150})
    quiet = _view("quiet", running=4, waiting=4)
    workers = [quiet, someone_elses]

    req = _req(prefix_hash="agent")

    assert pick(req, workers, strategy="prefix_then_load") is pick(
        req, workers, strategy="least_loaded"
    )


# --- H6: unknown is not idle -----------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "unknown",
    [
        pytest.param(dict(healthy=False), id="unhealthy"),
        pytest.param(dict(age_s=STALE_AFTER_S + 1), id="stale"),
        pytest.param(dict(running=None), id="missing-running-count"),
        pytest.param(dict(free_kv_blocks=None), id="missing-free-memory"),
        pytest.param(dict(ttft_p50_s=None), id="missing-latency"),
    ],
)
def test_an_unknown_worker_is_never_chosen_over_a_known_one(unknown, strategy):
    """H6: a worker that stopped reporting reads as empty, and is not.

    Every case makes the unknown worker look like the most attractive target
    in the fleet — no queue, no running work — which is exactly the shape a
    silent worker has.
    """
    dark = _view("dark", **(dict(running=0, waiting=0) | unknown))
    known = _view("known", running=12, waiting=12)

    assert pick(_req(), [dark, known], strategy=strategy) is known


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_an_all_unknown_fleet_still_routes(strategy):
    """H6 excludes unknown workers; when that is all there is, it cannot.

    Refusing here would turn a scrape outage into a total outage, so the
    fallback routes anyway rather than deadlocking the fleet.
    """
    workers = [_view("a", age_s=60.0), _view("b", age_s=60.0)]

    chosen = pick(_req(), workers, strategy=strategy)

    assert isinstance(chosen, WorkerView)
    assert chosen.name in {"a", "b"}


# --- H4: do not bounce an overloaded request -------------------------------


def test_a_worker_that_would_shed_is_skipped():
    """H4: an idle worker with no memory left is still a dead end."""
    full = _view("full", running=0, waiting=0, free_kv_blocks=0)
    busy = _view("busy", running=12, waiting=12)

    assert pick(_req(), [full, busy], strategy="least_loaded") is busy


def test_a_fleet_that_would_all_shed_returns_a_refusal():
    """H4: refuse once, here, instead of bouncing the request A to B to A."""
    workers = [
        _view("a", free_kv_blocks=0),
        _view("b", free_kv_blocks=0),
    ]

    result = pick(_req(), workers, strategy="least_loaded")

    assert isinstance(result, Shed)
    assert result.code == 503
    assert result.retry_after == 2


def test_admission_gates_beyond_memory_also_exclude_a_worker():
    """H4 reuses should_shed, so every gate it applies costs a worker its turn.

    Here it is the tail-latency gate: a growing queue with a p99 four times
    its p50 refuses batch work, and the idle-looking worker is the one with
    the bad tail.
    """
    laggy = _view("laggy", running=0, waiting=0, queue_growing=True, ttft_p99_s=1.0)
    steady = _view("steady", running=12, waiting=12)

    assert pick(_req(priority=20), [laggy, steady], strategy="least_loaded") is steady


def test_an_empty_fleet_is_a_refusal():
    """No workers is the degenerate case of every worker refusing."""
    result = pick(_req(), [], strategy="least_loaded")

    assert isinstance(result, Shed)
    assert result.code == 503 and result.retry_after == 2


# --- Contract --------------------------------------------------------------


def test_an_unknown_strategy_is_rejected():
    """A misspelled strategy must fail loudly, not silently route by default."""
    with pytest.raises(ValueError):
        pick(_req(), [_view("a")], strategy="least-loaded")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_picking_does_not_modify_the_fleet(strategy):
    """Routing reads telemetry; it neither reorders the fleet nor reserves."""
    workers = [_view("a", running=5, waiting=5), _view("b")]
    before = list(workers)

    pick(_req(), workers, strategy=strategy)

    assert workers == before
