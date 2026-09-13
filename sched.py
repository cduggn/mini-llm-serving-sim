from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal


SchedulingPolicy = Literal["fcfs", "priority", "drr"]


@dataclass
class ScheduledRequest:
    """Request metadata and simulated prefill/decode progress.

    Token counts represent completed work, not remaining work. Prefill is
    complete when prefill_done equals prompt_tokens; generation finishes at
    max_output_tokens. Each request belongs to at most one scheduler queue.
    """

    request_id: str
    arrived_at: float  # Monotonic arrival time.
    prompt_tokens: int
    priority: int = 20
    tenant_id: str = "default"
    prefill_done: int = 0  # Tokens processed across previous steps.
    max_output_tokens: int = 128  # Simulated completion limit; no EOS sampling.
    decode_done: int = 0  # Output tokens generated across previous steps.
    aborted: bool = False  # Client disconnected; drop at the start of a step.


@dataclass(frozen=True)
class StepResult:
    """Work performed and unused token capacity for one scheduling step.

    Each decoded request ID accounts for one generated token, in execution
    order. Completed IDs identify requests removed from running, in queue
    order. Prefill and decode tokens together never exceed the step budget.

    Aborted and preempted IDs report requests removed for reasons other than
    completion: aborted requests are gone for good, while preempted ones
    return to waiting with their progress discarded. Counting preempts means
    taking the length of preempted_request_ids; wasted_decode_tokens totals
    the generated tokens those preemptions threw away.
    """

    prefill_request_id: str | None
    prefill_tokens: int
    tokens_remaining: int
    decoded_request_ids: tuple[str, ...] = ()
    completed_request_ids: tuple[str, ...] = ()
    aborted_request_ids: tuple[str, ...] = ()
    preempted_request_ids: tuple[str, ...] = ()
    wasted_decode_tokens: int = 0


@dataclass
class DrrState:
    """Per-tenant token credit carried across steps for the DRR policy.

    deficits is the round-robin ring; its first key holds the current turn.
    Only tenants with unfinished prefill are members. Work is charged after it
    runs, so credit may go negative and later rounds repay it. A head whose
    credit is spent moves to the tail. Leaving the ring forfeits credit and
    debt. Use one state per scheduler queue.
    """

    quantum: int = 2048  # Credit each ring member earns per round.
    deficits: OrderedDict[str, int] = field(default_factory=OrderedDict)


@dataclass
class KvPool:
    """Block accounting for one worker's KV cache.

    A request holds ceil(tokens / block_size) blocks, counting its prefilled
    prompt plus generated output. Blocks are claimed as work runs and released
    when the request leaves running. No memory is allocated; this is a counter
    over a fixed supply. Request IDs must be unique within a pool, and one pool
    belongs to one scheduler queue.
    """

    total_blocks: int
    block_size: int = 16
    held: dict[str, int] = field(default_factory=dict)  # request_id -> blocks.

    @property
    def free_blocks(self) -> int:
        """Blocks no request currently holds."""
        return self.total_blocks - sum(self.held.values())


def _blocks_for(tokens: int, block_size: int) -> int:
    """Round tokens up to whole blocks using integer arithmetic."""
    return (tokens + block_size - 1) // block_size


def _kv_room(pool: KvPool, req: ScheduledRequest, wanted: int) -> int:
    """Return how many of the wanted tokens this request can still store.

    Counts unused space in the blocks it already holds plus the pool's free
    blocks. Reads state only, so callers may ask before committing to work.
    """
    tokens = req.prefill_done + req.decode_done
    spare = pool.held.get(req.request_id, 0) * pool.block_size - tokens
    return min(wanted, spare + pool.free_blocks * pool.block_size)


def _kv_claim(pool: KvPool, req: ScheduledRequest) -> None:
    """Grow a request's holding to cover the tokens it has now.

    Called after progress is recorded. Never shrinks: generated tokens stay
    resident until the request releases everything.
    """
    tokens = req.prefill_done + req.decode_done
    pool.held[req.request_id] = _blocks_for(tokens, pool.block_size)


@dataclass
class _Preemptions:
    """Victims and discarded work accumulated while making room in one step."""

    request_ids: list[str] = field(default_factory=list)
    wasted_decode_tokens: int = 0


def _drop_aborted(
    waiting: list[ScheduledRequest],
    running: list[ScheduledRequest],
    kv: KvPool | None,
) -> tuple[str, ...]:
    """Remove disconnected requests from both queues and release their blocks.

    Runs before any scheduling, including at a zero budget, so a client that
    has gone away never receives another token or holds memory for another
    step. Blocks are released once: the request leaves both lists, and
    releasing a holding that does not exist is a no-op.

    Returns:
        Removed request IDs, running first, then waiting.
    """
    aborted = tuple(req.request_id for req in [*running, *waiting] if req.aborted)
    running[:] = [req for req in running if not req.aborted]
    waiting[:] = [req for req in waiting if not req.aborted]

    if kv is not None:
        for request_id in aborted:
            kv.held.pop(request_id, None)
    return aborted


def _select_victim(
    running: list[ScheduledRequest],
    requester: ScheduledRequest,
    pool: KvPool,
) -> ScheduledRequest | None:
    """Choose the lowest-priority request whose blocks can be reclaimed.

    Lowest priority is the highest numeric value, with the latest arrival
    breaking ties. Three requests are never chosen: the requester itself,
    anything more important than the requester, so batch work cannot evict
    interactive work, and anything holding no blocks, which would free
    nothing and leave the requester blocked.
    """
    candidates = [
        req for req in running
        if req is not requester
        and req.priority >= requester.priority
        and pool.held.get(req.request_id, 0) > 0
    ]
    return max(
        candidates,
        key=lambda req: (req.priority, req.arrived_at),
        default=None,
    )


def _preempt(
    victim: ScheduledRequest,
    waiting: list[ScheduledRequest],
    running: list[ScheduledRequest],
    pool: KvPool,
) -> int:
    """Discard a victim's progress and requeue it, returning tokens thrown away.

    Releases every block it holds and resets both phases, so running again
    means recomputing the prompt from scratch. This is recompute, not swap:
    nothing is copied elsewhere, and the generated tokens are lost.
    """
    wasted = victim.decode_done
    victim.prefill_done = 0
    victim.decode_done = 0
    pool.held.pop(victim.request_id, None)

    # Identity, not equality: two requests with identical fields compare equal.
    for index, candidate in enumerate(running):
        if candidate is victim:
            running.pop(index)
            break
    waiting.append(victim)
    return wasted


def _reclaim_blocks(
    requester: ScheduledRequest,
    wanted: int,
    waiting: list[ScheduledRequest],
    running: list[ScheduledRequest],
    pool: KvPool,
    record: _Preemptions,
) -> int:
    """Preempt until the requester can store wanted tokens, or nothing is left.

    Each round removes one request from running, so this terminates. It stops
    early when only more important work remains, leaving the requester
    stalled with its progress intact.

    Returns:
        Room finally available, which is below wanted when nothing was left
        to preempt.
    """
    room = _kv_room(pool, requester, wanted)
    while room < wanted:
        victim = _select_victim(running, requester, pool)
        if victim is None:
            return room

        record.wasted_decode_tokens += _preempt(victim, waiting, running, pool)
        record.request_ids.append(victim.request_id)
        room = _kv_room(pool, requester, wanted)
    return room


def _select_fcfs(
    waiting: list[ScheduledRequest],
) -> ScheduledRequest | None:
    """Choose the earliest arrival without changing the queue."""
    return min(waiting, key=lambda req: req.arrived_at, default=None)


def _select_priority(
    waiting: list[ScheduledRequest],
) -> ScheduledRequest | None:
    """Choose lowest numeric priority, then earliest arrival; exact ties stay FIFO."""
    return min(
        waiting,
        key=lambda req: (req.priority, req.arrived_at),
        default=None,
    )


def step(
    waiting: list[ScheduledRequest],
    running: list[ScheduledRequest],
    budget: int,
    *,
    policy: SchedulingPolicy = "fcfs",
    drr: DrrState | None = None,
    kv: KvPool | None = None,
    decode_slots: int | None = None,
) -> StepResult:
    """Simulate one token-budgeted scheduling step.

    Reserve one token slot per existing decoder, up to the available budget.
    Execute at most one prefill chunk, then advance eligible decoders once.
    Newly completed prefill becomes decode-eligible on the next step.

    Args:
        waiting: Queued requests. Newly started work is removed from this list.
        running: Started requests. Receives new work and loses completed work.
        budget: Nonnegative token slots shared by prefill and decode.
        policy: FCFS, numeric priority, or per-tenant DRR credit. Arrival time
            breaks ties, and priority orders requests within a DRR tenant.
        drr: Credit carried across steps. Required by, and only accepted by,
            the DRR policy; one state belongs to one scheduler queue.
        kv: Block supply for this queue, or None to leave KV memory unmodeled.
        decode_slots: Limit on requests decoding at once, or None for no limit.

    Returns:
        Prefill work, decoded/completed request IDs, requests removed by abort
        or preemption, and unused token capacity.

    Mutates:
        Request progress, both queue lists, any supplied DRR credit, and any
        supplied block holdings. Aborted requests are dropped even at a zero
        budget; otherwise a zero budget changes nothing. Requests reaching
        their output limit are removed from running and release their blocks.
        Preempted requests return to waiting with their progress discarded.

    Raises:
        ValueError: Invalid policy, credit, blocks, or token counts, before any
            state changes.

    This simulation assumes successful execution; it does not call a GPU.
    """
    _validate_step_inputs(waiting, running, budget, policy, drr, kv, decode_slots)

    # Disconnected clients are dropped before anything competes for capacity.
    aborted_ids = _drop_aborted(waiting, running, kv)
    preemptions = _Preemptions()

    if budget == 0:
        return StepResult(None, 0, 0, aborted_request_ids=aborted_ids)

    decoders = _select_decoders(running, policy, decode_slots)
    prefill_budget = _prefill_budget(budget, len(decoders))
    prefill_id = None
    prefill_tokens = 0
    if prefill_budget > 0:
        req = _select_prefill(waiting, running, policy, drr)
        if req is not None:
            if kv is not None:
                room = _kv_room(kv, req, prefill_budget)
                if room == 0:
                    # Preemption unblocks stalled work rather than enlarging a
                    # chunk that already fits, so it asks for a single token.
                    _reclaim_blocks(req, 1, waiting, running, kv, preemptions)
                    room = _kv_room(kv, req, prefill_budget)
                prefill_budget = room
            # Without room for even one token the request stays queued,
            # unstarted, holding nothing.
            if prefill_budget > 0:
                _start_request(req, waiting, running)
                prefill_tokens = _run_prefill(req, prefill_budget)
                prefill_id = req.request_id
                if kv is not None:
                    _kv_claim(kv, req)
                if drr is not None:
                    _charge(drr, req.tenant_id, prefill_tokens)

    remaining = budget - prefill_tokens
    decoded = _run_decode(decoders, remaining, kv, waiting, running, preemptions)
    if drr is not None:
        # Decode is charged after selection, so it affects the next step's turn.
        for req in decoded:
            _charge(drr, req.tenant_id, 1)
    completed_ids = _finish_completed(running, kv)

    return StepResult(
        prefill_request_id=prefill_id,
        prefill_tokens=prefill_tokens,
        tokens_remaining=remaining - len(decoded),
        decoded_request_ids=tuple(req.request_id for req in decoded),
        completed_request_ids=completed_ids,
        aborted_request_ids=aborted_ids,
        preempted_request_ids=tuple(preemptions.request_ids),
        wasted_decode_tokens=preemptions.wasted_decode_tokens,
    )


def _validate_step_inputs(
    waiting: list[ScheduledRequest],
    running: list[ScheduledRequest],
    budget: int,
    policy: SchedulingPolicy,
    drr: DrrState | None,
    kv: KvPool | None,
    decode_slots: int | None,
) -> None:
    """Reject unsupported policies or invalid token counts before any mutation."""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if policy not in ("fcfs", "priority", "drr"):
        raise ValueError("policy must be 'fcfs', 'priority', or 'drr'")

    # Credit supplied under another policy would be ignored, silently running
    # the default policy instead of the fair one the caller expected.
    if (policy == "drr") != (drr is not None):
        raise ValueError("drr state is required by, and only accepted by, policy 'drr'")
    if drr is not None:
        _validate_drr_state(drr)
    if kv is not None:
        _validate_kv_pool(kv)
    if decode_slots is not None and (
        not isinstance(decode_slots, int)
        or isinstance(decode_slots, bool)
        or decode_slots <= 0
    ):
        raise ValueError("decode_slots must be a positive integer or None")

    for req in [*running, *waiting]:
        _validate_request_progress(req)


def _validate_kv_pool(pool: KvPool) -> None:
    """Require a positive block supply and holdings that fit inside it."""
    if not isinstance(pool, KvPool):
        raise ValueError("kv must be a KvPool")

    for name in ("total_blocks", "block_size"):
        count = getattr(pool, name)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if any(
        not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 0
        for blocks in pool.held.values()
    ):
        raise ValueError("held must map request IDs to nonnegative block counts")

    # A pool lending more than it owns would let stalled work run.
    if sum(pool.held.values()) > pool.total_blocks:
        raise ValueError("held blocks must not exceed total_blocks")


def _validate_drr_state(state: DrrState) -> None:
    """Require a positive integer quantum and an ordered ring of integer credit."""
    if not isinstance(state, DrrState):
        raise ValueError("drr must be a DrrState")

    quantum = state.quantum
    if not isinstance(quantum, int) or isinstance(quantum, bool) or quantum <= 0:
        raise ValueError("quantum must be a positive integer")

    # Rotation needs insertion order and move_to_end, so a plain dict is unsafe.
    if not isinstance(state.deficits, OrderedDict) or any(
        not isinstance(credit, int) or isinstance(credit, bool)
        for credit in state.deficits.values()
    ):
        raise ValueError("deficits must be an OrderedDict of integer credit")


def _validate_request_progress(req: ScheduledRequest) -> None:
    """Require positive token limits and progress within each phase's limit.

    Decode progress requires completed prefill. Validation reads state only;
    callers validate every request before executing any work in the step.
    """
    if not isinstance(req.aborted, bool):
        raise ValueError("aborted must be a boolean")

    for name in ("prompt_tokens", "max_output_tokens"):
        limit = getattr(req, name)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError(f"{name} must be a positive integer")

    for name, limit in (
        ("prefill_done", req.prompt_tokens),
        ("decode_done", req.max_output_tokens),
    ):
        done = getattr(req, name)
        if not isinstance(done, int) or isinstance(done, bool) or not 0 <= done <= limit:
            raise ValueError(f"{name} must be an integer within its token limit")

    if req.decode_done > 0 and req.prefill_done < req.prompt_tokens:
        raise ValueError("decode_done requires complete prefill")


def _select_prefill(
    waiting: list[ScheduledRequest],
    running: list[ScheduledRequest],
    policy: SchedulingPolicy,
    drr: DrrState | None,
) -> ScheduledRequest | None:
    """Choose unfinished prefill; running work wins exact cross-queue ties."""
    candidates = [
        req for req in [*running, *waiting]
        if req.prefill_done < req.prompt_tokens
    ]
    if policy == "drr":
        return _select_drr(candidates, drr)

    select = _select_fcfs if policy == "fcfs" else _select_priority
    return select(candidates)


def _select_drr(
    candidates: list[ScheduledRequest],
    state: DrrState,
) -> ScheduledRequest | None:
    """Choose the next prompt for the first ring member holding credit.

    Sync ring membership, grant whole rounds only when nobody has credit, and
    rotate members without credit to the tail. Within the chosen tenant,
    priority then arrival decides, so fairness applies between tenants only.
    """
    _sync_ring(candidates, state)
    if not state.deficits:
        return None

    _grant_rounds(state)
    head = next(iter(state.deficits))
    while state.deficits[head] <= 0:  # Terminates: _grant_rounds leaves credit.
        state.deficits.move_to_end(head)
        head = next(iter(state.deficits))

    return _select_priority([req for req in candidates if req.tenant_id == head])


def _sync_ring(candidates: list[ScheduledRequest], state: DrrState) -> None:
    """Drop members without prefill work; append newcomers by earliest arrival.

    Removal forfeits leftover credit and debt, so idle time is neither banked
    nor punished. Newcomers start at zero behind the existing members.
    """
    backlogged = {req.tenant_id for req in candidates}
    # Collect first: a dict cannot shrink while it is being iterated.
    for tenant in [t for t in state.deficits if t not in backlogged]:
        del state.deficits[tenant]

    for req in sorted(candidates, key=lambda req: req.arrived_at):
        state.deficits.setdefault(req.tenant_id, 0)


def _grant_rounds(state: DrrState) -> None:
    """If no member has credit, grant the fewest whole rounds that create some.

    Every member receives the same rounds, preserving relative debt. The count
    is computed directly, so deep debt costs no extra iterations.
    """
    if any(credit > 0 for credit in state.deficits.values()):
        return

    rounds = min(-credit // state.quantum + 1 for credit in state.deficits.values())
    for tenant in state.deficits:
        state.deficits[tenant] += rounds * state.quantum


def _charge(state: DrrState, tenant: str, tokens: int) -> None:
    """Debit completed work, ending the head's turn once its credit is spent.

    Tenants outside the ring have no prefill waiting and are not charged.
    Charging after execution lets a chunk overdraw credit by up to one step
    of work; later rounds repay it.
    """
    if tenant not in state.deficits:
        return

    state.deficits[tenant] -= tokens
    if state.deficits[tenant] <= 0 and tenant == next(iter(state.deficits)):
        state.deficits.move_to_end(tenant)


def _start_request(
    req: ScheduledRequest,
    waiting: list[ScheduledRequest],
    running: list[ScheduledRequest],
) -> None:
    """Move newly selected work into running; continuing work stays in place."""
    for index, candidate in enumerate(waiting):
        if candidate is req:
            running.append(waiting.pop(index))
            return


def _run_prefill(req: ScheduledRequest, budget: int) -> int:
    """Advance one validated prompt by at most the available prefill budget.

    Mutates prefill_done and returns tokens consumed. Completing prefill does
    not generate output or change queue membership in this helper.
    """
    chunk = min(req.prompt_tokens - req.prefill_done, budget)
    req.prefill_done += chunk
    return chunk


def _select_decoders(
    running: list[ScheduledRequest],
    policy: SchedulingPolicy,
    slots: int | None = None,
) -> list[ScheduledRequest]:
    """Capture and order decode-ready requests without changing running.

    Called before prefill so newly finished prompts wait until the next step.
    Completed outputs are excluded. Stable sorting preserves queue order for
    exact ties; a scarce budget serves the first requests in this result.

    DRR orders decoders by priority; tenant-fair decode ordering is not
    modeled. Requests beyond the concurrent decode slots are excluded, so the
    policy decides who generates tokens this step and who waits.
    """
    candidates = [
        req for req in running
        if req.prefill_done == req.prompt_tokens
        and req.decode_done < req.max_output_tokens
    ]
    if policy == "fcfs":
        ordered = sorted(candidates, key=lambda req: req.arrived_at)
    else:
        ordered = sorted(candidates, key=lambda req: (req.priority, req.arrived_at))
    return ordered if slots is None else ordered[:slots]


def _prefill_budget(budget: int, decoder_count: int) -> int:
    """Leave one token slot per existing decoder, capped by the step budget.

    Inputs are nonnegative integers. When decoders fill all available slots,
    prefill receives zero; otherwise it may spend the unreserved capacity.
    """
    return max(0, budget - decoder_count)


def _run_decode(
    decoders: list[ScheduledRequest],
    budget: int,
    kv: KvPool | None,
    waiting: list[ScheduledRequest],
    running: list[ScheduledRequest],
    record: _Preemptions,
) -> list[ScheduledRequest]:
    """Advance up to budget eligible requests by one output token each.

    Args:
        decoders: Requests ordered by policy, with prefill complete and output
            remaining. Captured before this step's prefill work.
        budget: Nonnegative number of token slots available for decoding.
        kv: Block supply to claim from, or None to leave memory unmodeled.
        waiting: Queue that preempted victims rejoin.
        running: Queue that preempted victims leave.
        record: Collects victims and the decode tokens they discarded.

    Returns:
        Requests advanced, in execution order, so callers can report their IDs
        and charge their tenants. Each consumes one slot.

    Mutates:
        Each selected request's decode_done and its block holding, plus any
        queue changes made by preemption. Completion cleanup is handled
        separately. A request needing a block the pool cannot supply preempts
        lower-priority work to obtain one; when nothing may be preempted it
        stalls with its progress intact and consumes no slot, leaving that
        capacity to the requests behind it. A request preempted earlier in
        this step never decodes.
    """
    selected: list[ScheduledRequest] = []
    for req in decoders:
        if len(selected) == budget:
            break
        if req.request_id in record.request_ids:
            continue
        if kv is not None and _kv_room(kv, req, 1) < 1:
            if _reclaim_blocks(req, 1, waiting, running, kv, record) < 1:
                continue

        req.decode_done += 1
        if kv is not None:
            _kv_claim(kv, req)
        selected.append(req)
    return selected


def _finish_completed(
    running: list[ScheduledRequest],
    kv: KvPool | None,
) -> tuple[str, ...]:
    """Remove completed outputs from running and return their IDs in queue order.

    Updates the list in place so existing references see the removals. Request
    progress is retained for inspection, but the blocks are released for the
    next step. Removed requests cannot decode again on subsequent steps.
    """
    completed = tuple(
        req.request_id for req in running
        if req.decode_done == req.max_output_tokens
    )
    running[:] = [req for req in running if req.decode_done < req.max_output_tokens]
    if kv is not None:
        for request_id in completed:
            kv.held.pop(request_id, None)
    return completed
