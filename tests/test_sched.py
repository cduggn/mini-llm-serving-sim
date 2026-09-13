"""Scheduler ordering, prefill, and decode tests; GPU execution is not involved."""

import pytest

from collections import OrderedDict

from sched import (
    DrrState,
    KvPool,
    ScheduledRequest,
    StepResult,
    _select_fcfs,
    _select_priority,
    step,
)


def _req(request_id: str, **overrides) -> ScheduledRequest:
    values = dict(arrived_at=1.0, prompt_tokens=100, priority=20)
    values.update(overrides)
    return ScheduledRequest(request_id=request_id, **values)


def test_fcfs_uses_arrival_time_regardless_of_priority():
    """An earlier low-priority request wins even if the list is unsorted."""
    earlier = _req("earlier", arrived_at=1.0, priority=20)
    later = _req("later", arrived_at=2.0, priority=0)

    assert _select_fcfs([later, earlier]) is earlier


def test_priority_beats_arrival_time():
    """Priority zero runs before an earlier priority-twenty request."""
    earlier = _req("earlier", arrived_at=1.0, priority=20)
    urgent = _req("urgent", arrived_at=2.0, priority=0)

    assert _select_priority([earlier, urgent]) is urgent


def test_equal_priority_uses_arrival_time():
    """Arrival time breaks equal priorities even if the list is unsorted."""
    earlier = _req("earlier", arrived_at=1.0, priority=10)
    later = _req("later", arrived_at=2.0, priority=10)

    assert _select_priority([later, earlier]) is earlier


@pytest.mark.parametrize("select", [_select_fcfs, _select_priority])
def test_exact_ties_preserve_queue_order(select):
    """Equal ordering keys keep the first queued request, not the lowest ID."""
    first = _req("z", arrived_at=1.0, priority=10)
    second = _req("a", arrived_at=1.0, priority=10)

    assert select([first, second]) is first


@pytest.mark.parametrize("select", [_select_fcfs, _select_priority])
def test_empty_queue_returns_none(select):
    """No waiting work means there is no selection."""
    assert select([]) is None


@pytest.mark.parametrize("select", [_select_fcfs, _select_priority])
def test_selection_does_not_modify_queue(select):
    """Selection neither removes requests nor reorders the supplied list."""
    later = _req("later", arrived_at=2.0, priority=20)
    earlier = _req("earlier", arrived_at=1.0, priority=0)
    waiting = [later, earlier]

    assert select(waiting) is earlier
    assert len(waiting) == 2
    assert waiting[0] is later
    assert waiting[1] is earlier


def test_large_prompt_continues_across_steps():
    """A 32,000-token prompt advances by at most 2,048 tokens per step."""
    req = _req("large", prompt_tokens=32_000)
    waiting, running = [req], []

    assert step(waiting, running, 2048) == StepResult("large", 2048, 0)
    assert req.prefill_done == 2048
    assert waiting == []
    assert len(running) == 1 and running[0] is req

    assert step(waiting, running, 2048) == StepResult("large", 2048, 0)
    assert req.prefill_done == 4096
    assert len(running) == 1 and running[0] is req


def test_final_chunk_uses_only_remaining_prompt_tokens():
    """Finishing prefill leaves unused budget and keeps the request running."""
    req = _req("finishing", prompt_tokens=5000, prefill_done=4500)
    waiting, running = [], [req]

    assert step(waiting, running, 2048) == StepResult("finishing", 500, 1548)
    assert req.prefill_done == 5000
    assert running == [req]
    assert req.decode_done == 0
    assert step(waiting, running, 2048) == StepResult(
        None, 0, 2047, decoded_request_ids=("finishing",)
    )
    assert req.decode_done == 1


def test_only_one_prefill_chunk_per_step():
    """A short prompt does not spend its leftover budget on a second prompt."""
    first = _req("first", arrived_at=1.0)
    second = _req("second", arrived_at=2.0)
    waiting, running = [first, second], []

    assert step(waiting, running, 2048) == StepResult("first", 100, 1948)
    assert waiting == [second]
    assert running == [first]
    assert second.prefill_done == 0


def test_zero_budget_does_not_start_or_advance_requests():
    """No tokens available means no progress or queue transitions."""
    new = _req("new")
    active = _req("active", prefill_done=50)
    waiting, running = [new], [active]

    assert step(waiting, running, 0) == StepResult(None, 0, 0)
    assert waiting == [new] and running == [active]
    assert new.prefill_done == 0 and active.prefill_done == 50


def test_empty_step_preserves_budget():
    """An empty scheduler has no work to charge to the step budget."""
    assert step([], [], 2048) == StepResult(None, 0, 2048)


@pytest.mark.parametrize("policy, chosen_id", [("fcfs", "active"), ("priority", "urgent")])
def test_policy_considers_waiting_and_unfinished_running_work(policy, chosen_id):
    """A partially prefilled request competes with new arrivals under the policy."""
    active = _req("active", arrived_at=1.0, priority=20, prefill_done=50)
    urgent = _req("urgent", arrived_at=2.0, priority=0)
    waiting, running = [urgent], [active]

    assert step(waiting, running, 25, policy=policy) == StepResult(chosen_id, 25, 0)
    assert active.prefill_done == (75 if policy == "fcfs" else 50)
    assert urgent.prefill_done == (25 if policy == "priority" else 0)


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_running_prefill_wins_exact_cross_queue_ties(policy):
    """An exact ordering tie continues running work before an identical arrival."""
    active = _req("active", prefill_done=25)
    new = _req("new")

    assert step([new], [active], 25, policy=policy) == StepResult("active", 25, 0)
    assert new.prefill_done == 0


@pytest.mark.parametrize("budget", [-1, 1.5, True, None])
def test_invalid_budget_does_not_mutate_state(budget):
    """Reject invalid token budgets before starting work."""
    req = _req("waiting")
    waiting, running = [req], []

    with pytest.raises(ValueError, match="budget"):
        step(waiting, running, budget)

    assert waiting == [req] and running == [] and req.prefill_done == 0


def test_unsupported_policy_does_not_mutate_state():
    """Unsupported policies are explicit errors, not a silent fallback."""
    req = _req("waiting")
    waiting, running = [req], []

    with pytest.raises(ValueError, match="policy"):
        step(waiting, running, 2048, policy="sjf")

    assert waiting == [req] and running == [] and req.prefill_done == 0


@pytest.mark.parametrize(
    "overrides, error_field",
    [
        ({"prompt_tokens": 0}, "prompt_tokens"),
        ({"prompt_tokens": -1}, "prompt_tokens"),
        ({"prompt_tokens": 1.5}, "prompt_tokens"),
        ({"prompt_tokens": True}, "prompt_tokens"),
        ({"prefill_done": -1}, "prefill_done"),
        ({"prefill_done": 101}, "prefill_done"),
        ({"prefill_done": 1.5}, "prefill_done"),
        ({"prefill_done": True}, "prefill_done"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"max_output_tokens": -1}, "max_output_tokens"),
        ({"max_output_tokens": 1.5}, "max_output_tokens"),
        ({"max_output_tokens": True}, "max_output_tokens"),
        ({"max_output_tokens": None}, "max_output_tokens"),
        ({"decode_done": -1}, "decode_done"),
        ({"decode_done": 129}, "decode_done"),
        ({"decode_done": 1.5}, "decode_done"),
        ({"decode_done": True}, "decode_done"),
        ({"decode_done": None}, "decode_done"),
        ({"decode_done": 1}, "decode_done"),
    ],
)
def test_invalid_request_counts_reject_before_any_progress(overrides, error_field):
    """Validate all requests before advancing even the first eligible request."""
    valid = _req("valid", arrived_at=0.0)
    invalid = _req("invalid", **overrides)
    waiting, running = [valid], [invalid]

    with pytest.raises(ValueError, match=error_field):
        step(waiting, running, 2048)

    assert waiting == [valid] and running == [invalid]
    assert valid.prefill_done == 0


def test_large_prefill_reserves_capacity_for_existing_decoders():
    """Three decoders each progress while a huge prompt gets 2,045 token slots."""
    large = _req("large", prompt_tokens=32_000)
    decoders = [_req(f"d{i}", prefill_done=100) for i in range(3)]
    waiting, running = [large], list(decoders)

    for count in (1, 2):
        result = step(waiting, running, 2048)

        assert result == StepResult("large", 2045, 0, ("d0", "d1", "d2"))
        assert large.prefill_done == count * 2045
        assert [req.decode_done for req in decoders] == [count] * 3


@pytest.mark.parametrize("policy, expected_id", [("fcfs", "earlier"), ("priority", "urgent")])
def test_scarce_budget_decodes_by_policy_without_starting_prefill(policy, expected_id):
    """When decoders outnumber slots, only the selected decoder advances."""
    earlier = _req("earlier", arrived_at=1.0, priority=20, prefill_done=100)
    urgent = _req("urgent", arrived_at=2.0, priority=0, prefill_done=100)
    large = _req("large", prompt_tokens=32_000)
    waiting, running = [large], [urgent, earlier]

    assert step(waiting, running, 1, policy=policy) == StepResult(
        None, 0, 0, (expected_id,)
    )
    assert waiting == [large] and large.prefill_done == 0
    assert earlier.decode_done == (1 if policy == "fcfs" else 0)
    assert urgent.decode_done == (1 if policy == "priority" else 0)


def test_decode_priority_ties_use_arrival_then_queue_order():
    """Equal-priority decoders serve earlier arrivals, keeping exact ties stable."""
    later = _req("later", arrived_at=2.0, priority=10, prefill_done=100)
    first = _req("z", arrived_at=1.0, priority=10, prefill_done=100)
    second = _req("a", arrived_at=1.0, priority=10, prefill_done=100)

    result = step([], [later, first, second], 2, policy="priority")

    assert result.decoded_request_ids == ("z", "a")
    assert later.decode_done == 0


def test_decode_only_step_generates_one_token_each_despite_large_budget():
    """Extra capacity does not generate multiple tokens for a decoder in one step."""
    first = _req("first", prefill_done=100)
    second = _req("second", prefill_done=100)
    running = [first, second]

    assert step([], running, 2048) == StepResult(None, 0, 2046, ("first", "second"))
    assert first.decode_done == second.decode_done == 1


def test_prefill_completing_this_step_waits_to_decode():
    """Capture eligibility before prefill, even if its completion leaves spare slots."""
    new = _req("new", prompt_tokens=10)
    existing = _req("existing", prefill_done=100)
    waiting, running = [new], [existing]

    assert step(waiting, running, 20) == StepResult("new", 10, 9, ("existing",))
    assert new.prefill_done == 10 and new.decode_done == 0
    assert step(waiting, running, 20) == StepResult(None, 0, 18, ("existing", "new"))
    assert new.decode_done == 1


def test_completed_output_is_removed_once_without_extra_tokens():
    """Reaching the output limit removes that request in place and reports it once."""
    finishing = _req("finishing", prefill_done=100, max_output_tokens=2, decode_done=1)
    ongoing = _req("ongoing", prefill_done=100)
    running = [finishing, ongoing]
    original_list = running

    assert step([], running, 10) == StepResult(
        None, 0, 8, ("finishing", "ongoing"), ("finishing",)
    )
    assert running is original_list and running == [ongoing]
    assert finishing.decode_done == 2
    assert step([], running, 10) == StepResult(None, 0, 9, ("ongoing",))
    assert finishing.decode_done == 2


def test_zero_budget_does_not_advance_decoder():
    """An existing decoder consumes nothing and remains running at zero budget."""
    req = _req("decoder", prefill_done=100)
    running = [req]

    assert step([], running, 0) == StepResult(None, 0, 0)
    assert req.decode_done == 0 and running == [req]


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 100, 2048])
def test_mixed_step_conserves_token_budget(budget):
    """Every token slot is charged to prefill, charged to decode, or returned unused."""
    prompt = _req("prompt", prompt_tokens=50)
    running = [_req(f"d{i}", prefill_done=100) for i in range(3)]

    result = step([prompt], running, budget)

    assert result.prefill_tokens + len(result.decoded_request_ids) + result.tokens_remaining == budget
    assert result.tokens_remaining >= 0


def test_drr_alternates_tenants_instead_of_draining_first_arrivals():
    """FCFS would run A1, A2, A3, then B1; DRR gives B the second turn."""
    tenant_a = [_req(f"A{i}", arrived_at=float(i), tenant_id="A") for i in (1, 2, 3)]
    tenant_b = _req("B1", arrived_at=4.0, tenant_id="B")
    waiting, running, state = [*tenant_a, tenant_b], [], DrrState(quantum=100)

    order = [
        step(waiting, running, 1000, policy="drr", drr=state).prefill_request_id
        for _ in range(4)
    ]

    assert order == ["A1", "B1", "A2", "A3"]


def test_drr_priority_does_not_cross_tenants():
    """A priority-zero request waits for its own tenant's turn."""
    batch = _req("batch", arrived_at=1.0, priority=20, tenant_id="A")
    urgent = _req("urgent", arrived_at=2.0, priority=0, tenant_id="B")

    result = step([batch, urgent], [], 1000, policy="drr", drr=DrrState())

    assert result.prefill_request_id == "batch"


def test_drr_orders_one_tenant_by_priority_then_arrival():
    """Fairness applies between tenants; priority still orders a tenant's own work."""
    batch = _req("batch", arrived_at=1.0, priority=20, tenant_id="A")
    urgent = _req("urgent", arrived_at=2.0, priority=0, tenant_id="A")

    result = step([batch, urgent], [], 1000, policy="drr", drr=DrrState())

    assert result.prefill_request_id == "urgent"


def test_drr_chunks_may_overdraw_credit_and_repay_it_later():
    """Chunks stay full-size while turns alternate; debt is repaid in whole rounds."""
    first = _req("A", arrived_at=1.0, prompt_tokens=3000, tenant_id="A")
    second = _req("B", arrived_at=2.0, prompt_tokens=3000, tenant_id="B")
    waiting, running, state = [first, second], [], DrrState(quantum=500)

    results = [step(waiting, running, 1000, policy="drr", drr=state) for _ in range(4)]

    assert [r.prefill_request_id for r in results] == ["A", "B", "A", "B"]
    assert [r.prefill_tokens for r in results] == [1000] * 4
    assert state.deficits == OrderedDict([("A", -500), ("B", -500)])


def test_drr_leaving_the_ring_forfeits_credit():
    """An idle tenant neither banks its unused quantum nor carries it back."""
    state = DrrState(quantum=2048)
    served = _req("served", arrived_at=1.0, tenant_id="A")
    waiting, running = [served], []

    assert step(waiting, running, 1000, policy="drr", drr=state).prefill_request_id == "served"
    assert state.deficits == OrderedDict([("A", 1948)])

    assert step(waiting, running, 1000, policy="drr", drr=state).prefill_request_id is None
    assert state.deficits == OrderedDict()

    waiting.append(_req("later", arrived_at=9.0, tenant_id="A"))
    step(waiting, running, 1000, policy="drr", drr=state)

    # One fresh quantum less the new chunk and the token "served" decoded,
    # not the forfeited 1948 added to it.
    assert state.deficits == OrderedDict([("A", 1947)])


def test_drr_decode_tokens_end_a_tenants_turn():
    """Generated tokens are charged too, so decoding spends a tenant's credit."""
    decoder = _req("a_decoder", arrived_at=1.0, tenant_id="A", prefill_done=100)
    prompt = _req("a_prompt", arrived_at=2.0, tenant_id="A", prompt_tokens=50)
    other = _req("b_prompt", arrived_at=3.0, tenant_id="B")
    waiting, running, state = [prompt, other], [decoder], DrrState(quantum=51)

    first = step(waiting, running, 1000, policy="drr", drr=state)

    # The 50-token chunk left one credit; the decoded token spent it.
    assert first == StepResult("a_prompt", 50, 949, ("a_decoder",))
    assert state.deficits == OrderedDict([("B", 51), ("A", 0)])

    second = step(waiting, running, 1000, policy="drr", drr=state)

    assert second.prefill_request_id == "b_prompt"


def test_drr_repays_deep_debt_without_iterating_per_token():
    """Rounds are computed directly, so a large debt costs no extra work."""
    req = _req("owing", tenant_id="A")
    state = DrrState(quantum=1, deficits=OrderedDict([("A", -1_000_000)]))

    result = step([req], [], 1000, policy="drr", drr=state)

    assert result.prefill_request_id == "owing"
    assert state.deficits == OrderedDict([("A", -99)])


def test_drr_newcomer_joins_behind_members_holding_credit():
    """A later tenant cannot jump ahead by arriving with an older request."""
    holding = _req("holding", arrived_at=5.0, tenant_id="A")
    newcomer = _req("newcomer", arrived_at=0.0, tenant_id="C")
    state = DrrState(quantum=100, deficits=OrderedDict([("A", 100)]))

    waiting, running = [newcomer, holding], []

    first = step(waiting, running, 1000, policy="drr", drr=state)

    # C joined at zero behind A, whose chunk then ended its turn.
    assert first.prefill_request_id == "holding"
    assert state.deficits == OrderedDict([("C", 0), ("A", 0)])

    second = step(waiting, running, 1000, policy="drr", drr=state)

    assert second.prefill_request_id == "newcomer"


@pytest.mark.parametrize(
    "policy, state, error",
    [
        ("drr", None, "drr state"),
        ("fcfs", DrrState(), "drr state"),
        ("priority", DrrState(), "drr state"),
        ("drr", OrderedDict(), "DrrState"),
        ("drr", DrrState(quantum=0), "quantum"),
        ("drr", DrrState(quantum=-1), "quantum"),
        ("drr", DrrState(quantum=1.5), "quantum"),
        ("drr", DrrState(quantum=True), "quantum"),
        ("drr", DrrState(deficits={"A": 0}), "deficits"),
        ("drr", DrrState(deficits=OrderedDict([("A", 1.5)])), "deficits"),
        ("drr", DrrState(deficits=OrderedDict([("A", True)])), "deficits"),
    ],
)
def test_invalid_drr_state_does_not_mutate_state(policy, state, error):
    """Reject unusable credit before any request or ring changes."""
    req = _req("waiting", tenant_id="A")
    waiting, running = [req], []

    with pytest.raises(ValueError, match=error):
        step(waiting, running, 1000, policy=policy, drr=state)

    assert waiting == [req] and running == [] and req.prefill_done == 0


def test_zero_budget_leaves_drr_credit_untouched():
    """A step with no capacity neither serves a turn nor charges a tenant."""
    req = _req("waiting", tenant_id="A")
    state = DrrState(deficits=OrderedDict([("A", 5)]))

    assert step([req], [], 0, policy="drr", drr=state) == StepResult(None, 0, 0)
    assert state.deficits == OrderedDict([("A", 5)])


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 100, 2048])
def test_drr_step_conserves_token_budget(budget):
    """Charging tenants does not create or lose token slots."""
    prompt = _req("prompt", prompt_tokens=50, tenant_id="A")
    running = [_req(f"d{i}", prefill_done=100, tenant_id="B") for i in range(3)]

    result = step([prompt], running, budget, policy="drr", drr=DrrState())

    assert result.prefill_tokens + len(result.decoded_request_ids) + result.tokens_remaining == budget
    assert result.tokens_remaining >= 0


def test_kv_blocks_are_claimed_as_prefill_advances():
    """Holdings round the prefilled tokens up to whole blocks each step."""
    req = _req("large", prompt_tokens=200)
    waiting, running, pool = [req], [], KvPool(total_blocks=100)

    step(waiting, running, 100, kv=pool)

    assert pool.held == {"large": 7}  # 100 tokens over 16-token blocks.
    assert pool.free_blocks == 93

    step(waiting, running, 100, kv=pool)

    assert req.prefill_done == 200
    assert pool.held == {"large": 13}


def test_kv_claims_no_extra_block_on_an_exact_boundary():
    """A token past a full block claims the next one; filling it exactly does not."""
    req = _req("exact", prompt_tokens=32)
    waiting, running, pool = [req], [], KvPool(total_blocks=10)

    step(waiting, running, 32, kv=pool)

    assert pool.held == {"exact": 2}

    step(waiting, running, 32, kv=pool)

    assert req.decode_done == 1
    assert pool.held == {"exact": 3}


def test_completion_releases_blocks_for_the_next_step():
    """Finished requests hand their blocks back to the pool."""
    req = _req("short", prompt_tokens=32, max_output_tokens=1)
    waiting, running, pool = [req], [], KvPool(total_blocks=4)

    step(waiting, running, 32, kv=pool)
    result = step(waiting, running, 32, kv=pool)

    assert result.completed_request_ids == ("short",)
    assert pool.held == {} and pool.free_blocks == 4


def test_prefill_chunk_shrinks_to_available_blocks_then_stalls():
    """A chunk takes the room that exists; with none left the request waits."""
    req = _req("large", prompt_tokens=200)
    waiting, running, pool = [req], [], KvPool(total_blocks=4)

    assert step(waiting, running, 100, kv=pool) == StepResult("large", 64, 36)
    assert pool.held == {"large": 4} and pool.free_blocks == 0

    # No room and nothing to preempt: progress is kept, and the step is wasted.
    assert step(waiting, running, 100, kv=pool) == StepResult(None, 0, 100)
    assert req.prefill_done == 64 and running == [req]


def test_request_without_blocks_is_not_started():
    """An exhausted pool leaves new work queued instead of running empty-handed."""
    req = _req("queued")
    waiting, running = [req], []
    pool = KvPool(total_blocks=1, held={"other": 1})

    assert step(waiting, running, 100, kv=pool) == StepResult(None, 0, 100)
    assert waiting == [req] and running == []
    assert pool.held == {"other": 1}


def test_stalled_decoder_leaves_its_slot_to_the_next_request():
    """With nothing evictable, a blocked request waits without spending capacity.

    The only other request is more important, so preemption is refused and the
    stall is genuine.
    """
    full = _req("full", arrived_at=1.0, priority=20, prompt_tokens=32, prefill_done=32)
    spare = _req("spare", arrived_at=2.0, priority=0, prompt_tokens=30, prefill_done=30)
    running = [full, spare]
    pool = KvPool(total_blocks=4, held={"full": 2, "spare": 2})

    # One token slot: proof the stall did not spend it.
    result = step([], running, 1, kv=pool)

    assert result.decoded_request_ids == ("spare",)
    assert result.preempted_request_ids == ()
    assert full.decode_done == 0 and spare.decode_done == 1
    assert pool.held == {"full": 2, "spare": 2}  # Filled its last block.


@pytest.mark.parametrize(
    "policy, expected", [("fcfs", ("old", "mid")), ("priority", ("urgent", "mid"))]
)
def test_decode_slots_cap_concurrent_decoders_by_policy(policy, expected):
    """Only as many requests decode as there are slots, chosen by the policy."""
    old = _req("old", arrived_at=1.0, priority=20, prefill_done=100)
    mid = _req("mid", arrived_at=2.0, priority=10, prefill_done=100)
    urgent = _req("urgent", arrived_at=3.0, priority=0, prefill_done=100)
    running = [old, mid, urgent]

    result = step([], running, 10, policy=policy, decode_slots=2)

    assert result.decoded_request_ids == expected


def test_decode_slots_limit_the_capacity_reserved_for_decoding():
    """Reservation follows the slots actually used, not the decoder count."""
    decoders = [_req(f"d{i}", prefill_done=100) for i in range(3)]
    prompt = _req("prompt", arrived_at=0.0, prompt_tokens=10)

    result = step([prompt], decoders, 5, decode_slots=1)

    assert result == StepResult("prompt", 4, 0, ("d0",))


@pytest.mark.parametrize(
    "pool, error",
    [
        ({}, "KvPool"),
        (KvPool(total_blocks=0), "total_blocks"),
        (KvPool(total_blocks=-1), "total_blocks"),
        (KvPool(total_blocks=1.5), "total_blocks"),
        (KvPool(total_blocks=True), "total_blocks"),
        (KvPool(total_blocks=4, block_size=0), "block_size"),
        (KvPool(total_blocks=4, block_size=1.5), "block_size"),
        (KvPool(total_blocks=4, held={"a": -1}), "held"),
        (KvPool(total_blocks=4, held={"a": 1.5}), "held"),
        (KvPool(total_blocks=1, held={"a": 2}), "exceed"),
    ],
)
def test_invalid_kv_pool_does_not_mutate_state(pool, error):
    """Reject unusable block accounting before any request changes."""
    req = _req("waiting")
    waiting, running = [req], []

    with pytest.raises(ValueError, match=error):
        step(waiting, running, 100, kv=pool)

    assert waiting == [req] and running == [] and req.prefill_done == 0


@pytest.mark.parametrize("slots", [0, -1, 1.5, True, "2"])
def test_invalid_decode_slots_does_not_mutate_state(slots):
    """Reject unusable decode slot limits before any request changes."""
    req = _req("waiting")
    waiting, running = [req], []

    with pytest.raises(ValueError, match="decode_slots"):
        step(waiting, running, 100, decode_slots=slots)

    assert waiting == [req] and running == [] and req.prefill_done == 0


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 100, 2048])
def test_step_conserves_token_budget_under_memory_limits(budget):
    """Blocked work returns its capacity instead of losing or inventing slots."""
    prompt = _req("prompt", arrived_at=0.0, prompt_tokens=50)
    running = [_req(f"d{i}", prefill_done=100) for i in range(3)]
    pool = KvPool(total_blocks=32, held={"d0": 7, "d1": 7, "d2": 7})

    result = step([prompt], running, budget, kv=pool, decode_slots=2)

    assert result.prefill_tokens + len(result.decoded_request_ids) + result.tokens_remaining == budget
    assert result.tokens_remaining >= 0
    assert pool.free_blocks >= 0


def test_aborted_requests_are_dropped_even_at_zero_budget():
    """A disconnected client stops holding memory on the very next step."""
    live = _req("live", prompt_tokens=32, prefill_done=32, aborted=True)
    queued = _req("queued", aborted=True)
    waiting, running = [queued], [live]
    pool = KvPool(total_blocks=4, held={"live": 2})

    result = step(waiting, running, 0, kv=pool)

    assert result == StepResult(None, 0, 0, aborted_request_ids=("live", "queued"))
    assert waiting == [] and running == []
    assert pool.held == {} and pool.free_blocks == 4


def test_aborted_requests_neither_prefill_nor_decode():
    """Removal happens before scheduling, so no further tokens are generated."""
    gone = _req("gone", prefill_done=100, aborted=True)
    also_gone = _req("also_gone", aborted=True)
    waiting, running = [also_gone], [gone]

    result = step(waiting, running, 2048)

    assert result == StepResult(
        None, 0, 2048, aborted_request_ids=("gone", "also_gone")
    )
    assert waiting == [] and running == []
    assert gone.decode_done == 0 and also_gone.prefill_done == 0


def test_abort_releases_blocks_exactly_once():
    """A second step finds nothing left to free and reports no aborts."""
    keep = _req("keep", prompt_tokens=32, prefill_done=32)
    gone = _req("gone", prompt_tokens=32, prefill_done=32, aborted=True)
    running = [keep, gone]
    pool = KvPool(total_blocks=4, held={"keep": 2, "gone": 2})

    assert step([], running, 0, kv=pool).aborted_request_ids == ("gone",)
    assert pool.held == {"keep": 2} and pool.free_blocks == 2

    assert step([], running, 0, kv=pool).aborted_request_ids == ()
    assert pool.held == {"keep": 2}


def test_invalid_aborted_flag_does_not_mutate_state():
    """Only a boolean can decide whether a request is dropped."""
    req = _req("bad", aborted="yes")
    waiting, running = [req], []

    with pytest.raises(ValueError, match="aborted"):
        step(waiting, running, 2048)

    assert waiting == [req] and req.prefill_done == 0


def test_blocked_decoder_preempts_the_lowest_priority_request():
    """Exhausted memory evicts batch work so interactive work keeps decoding."""
    urgent = _req("urgent", arrived_at=1.0, priority=0, prompt_tokens=32, prefill_done=32)
    batch = _req(
        "batch", arrived_at=2.0, priority=20, prompt_tokens=32, prefill_done=32, decode_done=5
    )
    waiting, running = [], [urgent, batch]
    pool = KvPool(total_blocks=5, held={"urgent": 2, "batch": 3})

    result = step(waiting, running, 10, policy="priority", kv=pool)

    assert result == StepResult(
        None, 0, 9, ("urgent",), preempted_request_ids=("batch",), wasted_decode_tokens=5
    )
    # Requeued for recomputation: its prompt and its five tokens are gone.
    assert waiting == [batch] and running == [urgent]
    assert batch.prefill_done == 0 and batch.decode_done == 0
    assert pool.held == {"urgent": 3}


def test_victim_is_the_latest_arrival_among_equal_priorities():
    """Equal priority breaks toward evicting the newest work."""
    early = _req("early", arrived_at=0.5, prompt_tokens=32, prefill_done=32)
    mid = _req("mid", arrived_at=1.0, prompt_tokens=32, prefill_done=32)
    late = _req("late", arrived_at=2.0, prompt_tokens=32, prefill_done=32, decode_done=7)
    waiting, running = [], [early, mid, late]
    pool = KvPool(total_blocks=7, held={"early": 2, "mid": 2, "late": 3})

    result = step(waiting, running, 10, policy="priority", kv=pool)

    assert result.decoded_request_ids == ("early", "mid")
    assert result.preempted_request_ids == ("late",)
    assert result.wasted_decode_tokens == 7


def test_preemption_never_evicts_more_important_work():
    """Batch work stalls rather than reclaiming an interactive request's blocks."""
    interactive = _req(
        "interactive", arrived_at=1.0, priority=0, prompt_tokens=30, prefill_done=30
    )
    batch = _req("batch", arrived_at=2.0, priority=20, prompt_tokens=32, prefill_done=32)
    waiting, running = [], [interactive, batch]
    pool = KvPool(total_blocks=4, held={"interactive": 2, "batch": 2})

    result = step(waiting, running, 10, policy="priority", kv=pool)

    assert result.decoded_request_ids == ("interactive",)
    assert result.preempted_request_ids == () and result.wasted_decode_tokens == 0
    assert batch.decode_done == 0 and running == [interactive, batch]


def test_preemption_skips_victims_holding_no_blocks():
    """Evicting a request that frees nothing would leave the requester blocked."""
    blocked = _req("blocked", arrived_at=1.0, priority=0, prompt_tokens=32, prefill_done=32)
    holder = _req(
        "holder", arrived_at=2.0, priority=10, prompt_tokens=32, prefill_done=32, decode_done=3
    )
    ghost = _req("ghost", arrived_at=3.0, priority=20)  # Lowest priority, no blocks.
    waiting, running = [], [blocked, holder, ghost]
    pool = KvPool(total_blocks=5, held={"blocked": 2, "holder": 3})

    result = step(waiting, running, 10, policy="priority", kv=pool)

    assert result.preempted_request_ids == ("holder",)
    assert result.wasted_decode_tokens == 3
    assert ghost in running and ghost.prefill_done == 0


def test_full_pool_preempts_so_prefill_can_start():
    """A new prompt reclaims blocks, then takes whatever room they freed."""
    holder = _req(
        "holder", arrived_at=1.0, priority=20, prompt_tokens=32, prefill_done=32, decode_done=3
    )
    new = _req("new", arrived_at=2.0, priority=0, prompt_tokens=200)
    waiting, running = [new], [holder]
    pool = KvPool(total_blocks=3, held={"holder": 3})

    result = step(waiting, running, 100, kv=pool)

    assert result == StepResult(
        "new", 48, 52, preempted_request_ids=("holder",), wasted_decode_tokens=3
    )
    assert waiting == [holder] and running == [new]
    assert holder.prefill_done == 0 and holder.decode_done == 0


def test_partial_room_shrinks_the_chunk_without_preempting():
    """Preemption unblocks stalled work; it does not enlarge a chunk that fits."""
    holder = _req("holder", arrived_at=1.0, priority=0, prompt_tokens=30, prefill_done=30)
    new = _req("new", arrived_at=2.0, priority=20, prompt_tokens=200)
    waiting, running = [new], [holder]
    pool = KvPool(total_blocks=5, held={"holder": 2})

    result = step(waiting, running, 100, kv=pool)

    assert result == StepResult("new", 48, 51, ("holder",))
    assert result.preempted_request_ids == ()


def test_unmodeled_memory_never_preempts():
    """Without a pool nothing can run out, so nothing is evicted."""
    running = [_req(f"d{i}", prefill_done=100) for i in range(3)]

    result = step([], running, 1)

    assert result.decoded_request_ids == ("d0",)
    assert result.preempted_request_ids == () and result.wasted_decode_tokens == 0
