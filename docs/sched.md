# `sched.py` — the scheduler

> Which request gets the GPU next, and what happens when memory runs out.

One call to `step()` is one GPU turn. The module is an ordered sequence of
decisions about a fixed token budget. Nothing here touches a GPU: every
"token" is an integer, and every "block" is a counter.

---

## Data structures

### `ScheduledRequest`
Simulated progress for one request. Token counts are **work completed**, not
work remaining.

| Field | Meaning |
|---|---|
| `request_id` | Unique within a queue and a block pool |
| `arrived_at` | Monotonic arrival time; breaks ordering ties |
| `prompt_tokens` / `prefill_done` | Prefill is complete when these are equal |
| `max_output_tokens` / `decode_done` | Generation stops at the limit; no EOS sampling |
| `priority` | **Lower number is more important.** 0 = interactive, 20 = batch |
| `tenant_id` | The DRR fairness unit |
| `aborted` | Client disconnected; dropped at the start of the next step |

### `StepResult` (frozen)
What one step did. The scheduler **accumulates nothing** — the caller owns the
totals, which keeps hidden state out of the scheduler.

| Field | Feeds the metric |
|---|---|
| `prefill_request_id`, `prefill_tokens` | prefill throughput |
| `decoded_request_ids` | one entry per generated token |
| `completed_request_ids` | completed requests / second |
| `aborted_request_ids` | `aborted_freed` = its length |
| `preempted_request_ids` | `preempts` = its length |
| `wasted_decode_tokens` | tokens thrown away by preemption |

### `DrrState`
Per-tenant token credit that survives between steps. `deficits` is an
`OrderedDict` doing double duty as the **ring**: its first key holds the
current turn. One structure means the order and the credit cannot drift apart.

### `KvPool`
Block accounting: `total_blocks`, `block_size` (16), and `held`, a map of
request ID to blocks. `free_blocks` is a computed property. A request holds
`ceil(tokens / block_size)` blocks where `tokens = prefill_done + decode_done`.

---

## One step, gate by gate

### 1. Validate everything, change nothing
Budget, policy, DRR credit, block pool, decode slots and every request's
counters are checked **before any mutation**.

**Why:** a step that failed halfway would leave requests half-advanced and
blocks half-claimed, with no way to tell how far it got.
**Consequence:** invalid input is always a clean `ValueError`, and state is
guaranteed untouched.

### 2. Drop aborted clients
`_drop_aborted` removes `req.aborted` requests from **both** queues and
releases their blocks.

**Why it runs first:** memory held for someone who hung up is memory stolen
from someone still waiting.
**Consequence:** it happens **even at a zero budget**. Releasing is idempotent:
the request leaves both lists, and popping a holding that isn't there is a
no-op, so blocks are freed exactly once.

### 3. Zero budget stops here
Returns immediately, reporting only the aborts.

### 4. Capture the decoders, then cap them
`_select_decoders` takes running requests with prefill complete and output
remaining, sorts by policy (arrival for FCFS; priority then arrival
otherwise), and truncates to `decode_slots`.

**Why capture before prefill:** a prompt that finishes prefilling this step
must not also decode this step. One step, one phase transition.
**Consequence:** `decode_slots` is the concurrency limit — real vLLM's
`max_num_seqs`. Requests beyond the cap keep their memory but generate
nothing, and **the policy decides who is inside the cap**.

### 5. Reserve one token per decoder
`prefill_budget = max(0, budget - len(decoders))`.

**Why:** this is the "don't let one huge prompt block decoding" rule,
implemented as a *reservation* rather than as leftovers — stronger than the
assignment requires.
**Consequence:** with 3 decoders and a 2,048 budget, prefill gets 2,045 and
each decoder gets its token. Prefill only reaches zero when the decoders
number as much as the entire budget.

### 6. Pick exactly one prompt to prefill
Candidates are unfinished prefills from `running` **and** `waiting`, with
running listed first so a continuing chunk wins an exact tie.

| Policy | Rule | Failure mode it has |
|---|---|---|
| `fcfs` | Earliest arrival | Head-of-line blocking: a 32k prompt holds the slot ~16 steps |
| `priority` | Lowest number, then arrival | Batch work can starve |
| `drr` | Ring head's tenant, then priority within it | Priority does not cross tenants |

**Consequence:** at most **one request starts per step**, so a 400-token
prompt leaves ~1,600 tokens unused. This caps admission throughput and shapes
every number in the report.

### 7. Memory check, then preemption as a last resort
`_kv_room` asks how many tokens fit, counting spare space inside blocks the
request already holds plus the pool's free blocks.

- **Some room** → the chunk **shrinks to fit**. Nothing is evicted.
- **Zero room** → `_reclaim_blocks` asks for **one token's worth**, not a full chunk.

**Why only one token:** preemption exists to unblock stalled work, not to
optimise chunk size. Asking for a full 2,048 would evict far more than needed.

**The victim rule** (`_select_victim`) excludes three candidates, then takes
the worst of the rest:

1. **Not the requester itself** — it cannot free its own way forward.
2. **Not anything more important** (`priority >= requester.priority`) — batch
   work can never evict interactive work. *This safeguard is not in the
   assignment and is the one that matters most in production.*
3. **Not anything holding zero blocks** — evicting it frees nothing and would
   leave the requester still blocked.
4. Of what remains: **highest priority number, then latest arrival**.

`_preempt` discards the victim's progress (`prefill_done = 0`,
`decode_done = 0`), releases its blocks and requeues it. **This is recompute,
not swap:** nothing is copied anywhere and the generated tokens are gone.
`wasted_decode_tokens` records them.

Removal from `running` is by **identity, not equality** — two dataclass
requests with identical fields compare equal, so `list.remove` would drop the
wrong one.

### 8. Run the chunk
Advance by the room available, claim blocks to match, charge the tenant's DRR
credit.

### 9. Decode, one token each
For each captured decoder:
- **Skip** it if it was preempted earlier in this same step.
- If it cannot claim a block, **try preemption**.
- If that fails, **stall without spending a budget slot**, leaving the
  capacity to the request behind it.

**Consequence:** stalls are now rare. They happen only when nothing may be
evicted — that is, when every other request is more important.

### 10. Release on completion
`_finish_completed` removes requests at their output limit and frees their
blocks, so the space is available on the next step.

---

## DRR in detail

Classic deficit round robin, adapted for chunked prefill.

1. **Sync the ring.** Tenants with no unfinished prefill are removed, losing
   both credit and debt. Newcomers join the **back** at zero credit, ordered by
   earliest arrival, so they cannot jump ahead.
2. **Grant rounds.** If nobody has credit, add the fewest whole rounds of
   `quantum` that give someone credit. The count is computed with
   `-credit // quantum + 1`, so deep debt costs no extra iterations.
3. **Rotate** past members without credit; the first with credit takes the
   prefill slot.
4. **Charge after the work runs** — both prefill chunks *and* decode tokens.
   Credit may go negative; later rounds repay it.
5. A head whose credit reaches zero **moves to the tail immediately**, not at
   the next step.

**Why charge after:** classic DRR only sends a job once credit covers all of
it. A 32k prompt would wait many rounds before doing anything. Chunks are at
most one budget, so we allow an overdraft of one step's work instead — chunks
stay full-size and long-run fairness is unchanged.

**Why charge decode:** a 10-page answer consumes GPU work for thousands of
steps. If decode were free, heavy generators would never pay for it.

---

## Known limitations (say these out loud in the report)

1. **Deadlock is reachable.** If the blocked request is the most important one
   and nothing may be evicted, it stalls forever. Admission control in
   `admit.py` is what prevents reaching that state — the two layers are
   load-bearing together.
2. **Thrash is reachable.** A preempted request returns to `waiting`, gets
   selected again, and may evict whatever displaced it. Real systems damp this;
   we do not.
3. **DRR double-charges the victim.** It paid for prefill that was thrown
   away, and pays again to recompute. The tenant is penalised for being evicted.
4. **Wasted prefill is not counted.** Only decode tokens are, per the spec, so
   the true waste is larger than the metric reports.
5. **A chunk can be undone in the step that ran it**, if the prefilling request
   is the lowest-priority thing in memory.
6. **Fairness is short-term.** A tenant forfeits its debt when it goes idle, so
   DRR is fair over a busy period, not across a session. The long-term version
   is VTC.
7. **Catch-up claims.** A request placed in `running` without a matching `held`
   entry is charged for everything it owes the moment it next does work, since
   `_kv_room` computes negative spare space. A pool cannot be fooled by
   hand-built state.

---

## Assignment questions this file answers

| Question | Location |
|---|---|
| Where do I prioritise interactive traffic? | `_select_priority`, plus the victim rule refusing to evict upward |
| Where do I stop one tenant monopolising the GPU? | `_select_drr` |
| Where do I preempt a request? | `_preempt` |
| Where do I protect KV memory (after entry)? | `_kv_room`, `_reclaim_blocks` |

---

## Tests

`tests/test_sched.py` — 57 tests covering ordering, chunked prefill, the decode
reservation, DRR fairness and validation, block accounting, decode slots,
preemption victim selection, and abort cleanup. Every validation test asserts
that state is unchanged after the error.
