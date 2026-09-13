"""Simulation harness tests: the clock, the metrics, and the admission bridge."""

import json

import pytest

from serve import (
    STEP_BASE_S,
    TraceRequest,
    Worker,
    _snapshot,
    _to_pending,
    load_trace,
    percentile,
    run,
    step_seconds,
)


def _trace_request(request_id: str, **overrides) -> TraceRequest:
    values = dict(
        id=request_id,
        arrival_t=0.0,
        priority=0,
        prompt_tokens=100,
        max_new_tokens=8,
        prefix_hash=None,
        timeout_s=30.0,
        tenant="interactive",
    )
    values.update(overrides)
    return TraceRequest(**values)


def test_decode_only_step_costs_the_base_pass():
    """A step carrying no prefill is one forward pass, whatever else it did."""
    assert step_seconds(0) == STEP_BASE_S


def test_prefill_makes_a_step_more_expensive():
    """Prefill time scales with tokens, which is why long prompts hurt others."""
    full_chunk = step_seconds(2048)

    assert full_chunk > step_seconds(256) > STEP_BASE_S
    assert full_chunk == pytest.approx(STEP_BASE_S + 2048 / 20_000)


@pytest.mark.parametrize(
    "values, q, expected",
    [
        ([], 99, None),  # Nothing measured yet.
        ([1.0], 99, 1.0),
        ([1.0, 2.0, 3.0, 4.0], 50, 2.0),
        (list(range(1, 101)), 99, 99),
    ],
)
def test_percentile_uses_nearest_rank(values, q, expected):
    """Percentiles never interpolate, and an empty sample reports NaN."""
    result = percentile(values, q)

    if expected is None:
        assert result != result  # NaN.
    else:
        assert result == expected


def test_load_trace_orders_by_arrival(tmp_path):
    """A trace is replayed in arrival order regardless of file order."""
    path = tmp_path / "t.jsonl"
    lines = [
        json.dumps(_trace_request("late", arrival_t=2.0).__dict__),
        json.dumps(_trace_request("early", arrival_t=1.0).__dict__),
    ]
    path.write_text("\n".join(lines) + "\n")

    assert [req.id for req in load_trace(path)] == ["early", "late"]


def test_snapshot_discounts_a_resident_prefix():
    """A cached prefix is memory we do not have to find again."""
    req = _trace_request("agent", prompt_tokens=4000, max_new_tokens=512)
    worker = Worker()

    cold = _snapshot(req, worker, 0, [], False, {})
    warm = _snapshot(req, worker, 3500, [], False, {})

    assert cold.required_kv_blocks > warm.required_kv_blocks
    assert warm.cached_prefix_tokens == 3500


def test_pending_request_carries_the_timeout_admission_measures():
    """The queue-wait gate compares its estimate against this deadline."""
    req = _trace_request("r", timeout_s=5.0, priority=20)

    pending = _to_pending(req, clock=1.0)

    assert pending.deadline_s == 5.0 and pending.deadline_at == 6.0
    assert pending.priority == 20 and pending.n_in == 100


@pytest.mark.parametrize("policy", ["fcfs", "priority", "drr"])
def test_light_workload_completes_under_every_policy(policy):
    """With capacity to spare, no policy sheds, preempts, or wastes work."""
    trace = [
        _trace_request(f"r{i}", arrival_t=float(i), prompt_tokens=200, max_new_tokens=8)
        for i in range(5)
    ]

    metrics = run(trace, policy, seconds=60.0)

    assert metrics.arrived == 5 and metrics.admitted == 5
    assert metrics.completed == 5 and metrics.in_flight == 0
    assert metrics.rejected == 0 and metrics.preempts == 0
    assert metrics.wasted_decode_tokens == 0


def test_ttft_is_recorded_per_tenant():
    """Time to first token is grouped by tenant so classes can be compared."""
    trace = [
        _trace_request("i0", tenant="interactive", arrival_t=0.0),
        _trace_request("b0", tenant="batch", priority=20, arrival_t=0.0),
    ]

    metrics = run(trace, "priority", seconds=60.0)

    assert set(metrics.ttft_by_tenant) == {"interactive", "batch"}
    assert all(t > 0 for samples in metrics.ttft_by_tenant.values() for t in samples)


def test_idle_clock_jumps_to_the_next_arrival():
    """An empty server does not burn steps waiting for work to show up."""
    trace = [_trace_request("late", arrival_t=30.0, max_new_tokens=1)]

    metrics = run(trace, "fcfs", seconds=60.0)

    assert metrics.completed == 1
    # Two steps: one prefill, one decode. Without the jump this would be
    # thousands of empty steps before the arrival.
    assert metrics.steps == 2


def test_horizon_stops_the_run():
    """Work still running when the clock runs out is reported, not finished.

    The output fits in memory but not in the horizon: 2,000 decode steps take
    twelve simulated seconds. A larger output would be refused by admission
    rather than started, which is a different behaviour.
    """
    trace = [_trace_request("long", max_new_tokens=2_000)]

    metrics = run(trace, "fcfs", seconds=1.0)

    assert metrics.seconds <= 1.0
    assert metrics.completed == 0 and metrics.in_flight == 1
