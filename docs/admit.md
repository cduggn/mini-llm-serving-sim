# `admit.py` — admission control

> Is the server too busy to safely accept this request?

`should_shed(req, snap)` returns `(shed, code, retry_after_seconds)`. It runs
six gates in a fixed order and returns the **first refusal**; if all pass, it
returns `(False, None, 0.0)`.

It does exactly one thing: say yes or no. It **does not** reserve capacity,
retry, queue, or route. Choosing a worker is `router.py`'s job, and failing
over is explicitly not admission's business.

---

## Inputs

### `req` — a `PendingRequest` (from `state.py`)
Reused from the class-7 gateway rather than redefining the assignment's
`Request`, so the same checks work against the real service.

| Field | Used for |
|---|---|
| `n_in` | Upper bound for the cached-prefix sanity check |
| `deadline_s` | The timeout the queue-wait rule measures against |
| `priority` | Lower is more important; `< 10` is protected from the tail-latency gate |

### `snap` — an `AdmissionSnapshot` (frozen)
Everything the decision reads, captured once. Every field is `| None`, because
a real gateway often has no fresh signal.

| Group | Fields |
|---|---|
| Queue | `queue_length`, `queue_growing` |
| Latency | `ttft_p50_s`, `ttft_p99_s` |
| Memory | `kv_usage`, `cached_prefix_tokens`, `free_kv_blocks`, `required_kv_blocks` |
| Tenant tokens | `tenant_tokens_used`, `tenant_tokens_reserved`, `tenant_token_limit` |
| Tenant requests | `tenant_requests_used`, `tenant_requests_reserved`, `tenant_request_limit` |

`queue_length` counts work **waiting ahead** on the candidate replica. It
excludes running requests, this arrival, and unrelated replicas. `used` is
dispatched work; `reserved` is accepted work not yet settled. Both are counted
against the same allowance window.

---

## The gates, in order

### 1. Tenant token allowance → `429`
`committed = used + reserved`; refuse when `committed * 100 >= limit * 95`.

**Why first:** ten requests are not ten times the same amount of work. A tenant
can exhaust a token budget with a single enormous prompt while its request
count still looks harmless, so **tokens are checked before requests**.
**Detail:** the comparison is multiplied out rather than dividing, so there is
no floating-point rounding at the 95% boundary. `limit == 0` means no
allowance at all and refuses immediately.
**Consequence:** `429` means *you* are over budget — retrying elsewhere will
not help.

### 2. Tenant request allowance → `429`
The same rule applied to request counts. Catches a flood of tiny requests that
would never trip the token budget.

### 3. Doomed-request check → `503`
`expected_wait = queue_length * ttft_p50_s`. Refuse when that exceeds **half**
the request's deadline.

**Why half and not all:** the estimate only covers queue wait. The request
still has to prefill and decode after it reaches the front, so spending more
than half the budget waiting means it almost certainly misses.
**Consequence:** work that would time out anyway never enters. It occupies no
memory and produces no output that anyone is still waiting for.

### 4. KV pressure and prefix reuse → `503`
Below **8% free** (`kv_usage > 0.92`), refuse a request whose prefix is **not**
already resident. Admit it when `cached_prefix_tokens > 0`.

**Why the exception:** a cached prefix means much of its KV is already in
memory, so it costs a fraction of what a cold request would. Under pressure,
the cheap request is exactly the one still worth taking.
**Consequence:** this is the admission-side half of prefix awareness; the
router exploits the same property when choosing a worker.

### 5. Absolute block capacity → `503`
Refuse when `required_kv_blocks > free_kv_blocks`.

**Beyond the assignment.** Gate 4 is a *percentage* rule, and a large enough
request can exceed real capacity while usage still sits under 92%. Without this
gate, such a request would be admitted only to stall or be preempted inside the
scheduler.

### 6. Tail latency vs interactive traffic → `503`
When `ttft_p99_s > 4 * ttft_p50_s` **and** the queue is growing, refuse
`priority >= 10` and keep `priority < 10`.

**Why both conditions:** a bad p99 on its own may be a blip. A bad p99 *while
the queue grows* means the server is falling behind and the tail will get
worse. Shedding batch traffic is what buys interactive traffic its latency back.
**Detail:** a `None` priority is treated as `10` — unprotected. Protection is
opt-in, never the default, and it exempts a request from **this gate only**.
A priority-0 request is still refused by gates 1 to 5.

---

## Cross-cutting design decisions

**Fail closed.** Every gate refuses with `503` when the signal it needs is
missing, malformed, or out of range — an unknown `p50` refuses even with an
empty queue.
*Why:* an unknown signal is not a healthy one. Treating missing data as "fine"
is exactly the failure this course's H6 rule warns about on the routing side.
*Cost:* a monitoring outage sheds traffic. That is the intended trade.

**Order matters.** Cheap, request-local, definitive checks come before
fleet-wide estimates. Tenant quota is a fact; expected queue wait is a
prediction. Facts first.

**Status codes.**

| Code | Meaning | Should the client retry elsewhere? |
|---|---|---|
| `429` | Tenant allowance exhausted | No — the limit follows the tenant |
| `503` | Server capacity or missing signal | Yes, after `retry_after` |

The assignment allows `503/529`; this implementation uses `503` throughout.

**`retry_after` is always `1.0`** — a placeholder until the accounting service
can supply a real delay based on when allowance actually refills.

---

## Required situations, and where each is handled

| Situation | Result | Gate |
|---|---|---|
| Tenant used 96% of token allowance | `429` | 1 |
| Tenant used 96% of request allowance | `429` | 2 |
| Likely wait > half the timeout | `503` | 3 |
| 5% KV left, new prefix | `503` | 4 |
| 5% KV left, prefix already cached | **accept** | 4 |
| Terrible tail latency, interactive | **accept** | 6 |
| Terrible tail latency, batch | `503` | 6 |

---

## The short-answer question

Each production pattern the course names maps onto one of these rules:

- **DALL·E's 5-minute cancellation** → gate 3. Work that cannot finish inside
  the user's time budget is refused up front rather than computed and then
  thrown away. The cheapest request is the one never admitted.
- **Anthropic's late-capacity behaviour** → gates 4 and 5, plus the scheduler's
  preemption. Capacity is protected by degrading what is accepted as memory
  tightens, keeping already-admitted work alive instead of failing everything
  equally.
- **Cloudflare overload protection** → gate 6. Under overload, shed by class
  rather than at random, so the traffic that a human is actively waiting on
  keeps its latency while background work absorbs the loss.

---

## Known limitations

1. **Snapshot freshness is not checked here.** `should_shed` trusts whatever it
   is handed. Whoever builds the snapshot must reject stale data — the same
   problem as the router's T3 stale-telemetry trace.
2. **Priority is trusted.** A client can send `priority: 0` and buy itself an
   exemption from gate 6. Real deployments derive priority from the
   authenticated tenant.
3. **No reservation.** Two concurrent requests can both be admitted against the
   same free capacity. The accounting service's `reserved` counters exist to
   close this, but nothing takes the reservation yet.
4. **`retry_after` carries no information**, so clients cannot pace themselves
   usefully and may retry into the same wall.
5. **The block estimate must agree with the scheduler's.** `required_kv_blocks`
   is computed by the caller, while `sched.py` charges `ceil(tokens / 16)` as
   work runs. If the two disagree, admission is either too generous or too
   strict.

---

## Assignment questions this file answers

| Question | Location |
|---|---|
| Where do I prevent accepting work that will time out? | Gate 3 |
| Where do I protect KV memory (before entry)? | Gates 4 and 5 |
| Where do I prioritise interactive traffic? | Gate 6 |
| Where do I prevent one tenant monopolising? | Gates 1 and 2 (`sched.py` DRR handles it after entry) |
| Where do I exploit shared prefixes? | Gate 4 (the router does the rest) |

---

## Tests

`tests/test_admit.py` — 20 test functions, heavily parametrised. They cover all
seven required situations, the ordering between gates, and a malformed or
missing value for every field in the snapshot.
