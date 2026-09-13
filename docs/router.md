# `router.py` — request routing

> Which worker gets this request, and when no worker should.

`pick(req, workers)` reads one telemetry view per worker and returns the
`WorkerView` to send the request to, or a `Shed` when the answer is nobody.
Four strategies decide **which** acceptable worker wins; two rules decide
**whether there is one to win at all**, and those two outrank every strategy.

    H6  An unknown worker is never treated as an idle one.
    H4  A request no worker would accept is refused here, not bounced.

It does exactly one thing: name a destination. It **does not** retry, queue,
reserve capacity, record a placement, or re-route the request later. Once
`pick` returns, the decision is final — there is no second opinion anywhere in
the path, which is precisely why H4 has to be answered before the request
leaves rather than after a worker refuses it.

This is the simulated router for Part 3. The vLLM gateway this was extracted
from has its own live-service router: different file, different job, different
tests.

---

## Inputs

### `req` — a `RouteRequest` (frozen)

| Field | Used for |
|---|---|
| `pending` | A `PendingRequest`, handed to `should_shed` **unchanged** |
| `prefix_hash` | The workload's identity for a shared prompt; `None` means there is nothing to reuse anywhere |

`pending` is passed through rather than re-derived so the router and the
gateway ask admission the identical question about the identical request. A
router that predicted refusals using its own copy of the rule would drift from
the rule the worker actually applies, and the drift would only show up as
requests being sent to workers that then reject them.

### `workers` — a `WorkerView` per worker (frozen)

| Group | Fields |
|---|---|
| Identity | `name` |
| Link | `healthy`, `age_s` |
| Queue | `running`, `waiting`, `queue_growing` |
| Memory | `free_kv_blocks`, `total_kv_blocks`, `cached_prefixes` |
| Latency | `ttft_p50_s`, `ttft_p99_s` |

`waiting` counts work queued **ahead of this arrival** and excludes running
requests. `cached_prefixes` maps a prefix hash to the tokens of that prefix
resident on that worker; an absent hash is not cached there.

**Every load signal is `| None`.** This is not defensive boilerplate. Telemetry
goes missing as easily as it goes wrong, and `waiting = 0` and `waiting = None`
are opposite facts: the first says *this worker told us it has nothing queued*,
the second says *this worker is not telling us anything*. Collapsing the two —
`waiting or 0`, a dataclass default of `0`, a JSON parser that fills gaps — is
a real bug, and it is a bug with a specific shape: the missing worker becomes
the most attractive target in the fleet, so every request piles onto the one
machine there is the least reason to trust. `healthy` and `age_s` describe the
**link**, not the worker, so a view can be unhealthy or stale while still
carrying the last numbers that arrived. Those numbers are exactly the trap.

### Constants

| Constant | Value | Why |
|---|---|---|
| `STALE_AFTER_S` | `5.0` | Telemetry older than this is unknown, not current. The scrape interval sets the floor; this leaves a few intervals of grace |
| `BLOCK_SIZE` | `16` | Matches `sched.py`'s `KvPool` and `serve.py` |
| `ROUTER_SEED` | `7` | Seeded at import, so one trace routes identically on every run and two strategies compare on the same draws |
| `_QUOTA_UNUSED` | `1` | A tenant allowance that always passes. See below |

Nothing here reads the global `random` module. `pick(rng=...)` varies the draws
deliberately; omitting it gets the module's seeded generator.

---

## The decision, in order

### 0. No fleet → `503`

`Shed(503, retry_after=2.0, reason="no_workers")`. An empty list is treated as
the degenerate case of every worker refusing, not as an error or an empty
result.

**Why:** the caller's question is "where does this go", and "nowhere, come back
in two seconds" is a complete answer to it. Raising instead would make an
operational condition — a fleet that scaled to zero, a config that resolved no
backends — indistinguishable from a programming mistake, and would force every
caller to wrap `pick` in a `try`. The `reason` field separates the two cases for
anyone reading logs; the code and the delay are identical because the client's
correct behaviour is identical. Pinned by `test_an_empty_fleet_is_a_refusal`.

### 1. H6 — unknown is not idle (`_eligible`, `_is_unknown`)

Runs **first**, before admission and before any strategy. `_eligible` keeps only
the workers we can vouch for and reports whether the whole fleet is dark.

`_is_unknown` has three ways to fail, treated identically because they have
identical consequences:

1. `healthy` is false — the worker is failing its health check.
2. `age_s > STALE_AFTER_S` — the numbers are too old to describe the present.
3. Any of `running`, `waiting`, `free_kv_blocks`, `total_kv_blocks`,
   `ttft_p50_s`, `ttft_p99_s` is `None` — a signal the decision needs is
   missing.

**Why it runs before the strategies:** a strategy cannot forget a safety rule it
never had the chance to break. If each of the four had to remember to exclude
dark workers, the rule would hold in four places and would eventually hold in
three. Here it holds once, and `test_an_unknown_worker_is_never_chosen_over_a_known_one`
runs all five unknown shapes against all four strategies — 20 cases — with the
dark worker deliberately built as `running=0, waiting=0`, the exact shape a
silent worker presents.

**The defence in depth.** `_load_key` returns `(math.inf, 0.0)` for a view
missing `running` or `waiting`, so an unknown worker sorts as **maximally
loaded** and can never win a load comparison by looking empty. That guard is
unreachable through `pick`, because `_eligible` already removed those views —
which is the point. `_eligible` is the rule; `_load_key` is the rule still
holding when the guard is bypassed, by a future caller reaching a strategy
directly or by the all-unknown fallback below.

**The one exception.** When *every* worker is unknown, `_eligible` returns the
full fleet and flags `fleet_is_dark`. Excluding unknown workers when unknown
workers are all there is would turn a monitoring outage into a total outage.
`test_an_all_unknown_fleet_still_routes` pins that a 60-second-stale two-worker
fleet still gets a placement under every strategy.

### 2. H4 — do not bounce (`_admissible`, `_would_shed`, `_snapshot`)

`_admissible` keeps the workers that **would accept this request if it arrived**,
asked before choosing rather than discovered after being refused. If none
survive, `pick` returns `Shed(503, retry_after=2.0)`.

**Why ask `should_shed` rather than reimplement it:** the router's prediction and
the worker's decision have to be the same rule or the prediction is worthless.
Re-expressing "is there room" as a KV check would silently ignore gates 3 and 6
— a worker with a doomed queue or a blown tail would keep winning placements.
`test_admission_gates_beyond_memory_also_exclude_a_worker` pins this with the
tail-latency gate specifically: an idle worker with `queue_growing` and a p99
four times its p50 loses to a worker carrying 24 requests, because for batch
traffic the idle-looking one is a dead end. The refusal *code* is discarded —
the router only needs to know whether this worker is a dead end, not why.

**Why refuse once instead of bouncing:** the alternative is A refuses, the client
retries, B refuses, the client retries A, and the request burns fleet capacity
being rejected instead of being told to come back later. Every bounce costs a
connection, an admission evaluation and a round trip on a fleet that is by
definition already out of room. One `503` with `retry_after=2.0` gives the
client the one thing the bounce never does: the information that waiting, not
shopping around, is the productive response. `test_a_fleet_that_would_all_shed_returns_a_refusal`
pins the code and the delay; `test_a_worker_that_would_shed_is_skipped` pins the
ordinary case, where an idle worker with `free_kv_blocks=0` loses to a busy one
that can actually take the work.

**What the snapshot says** (`_snapshot`), per candidate:

| Snapshot field | Source |
|---|---|
| `queue_length` | `view.waiting` |
| `ttft_p50_s`, `ttft_p99_s`, `queue_growing` | passed through |
| `kv_usage` | `1.0 - free_kv_blocks / total_kv_blocks`, or `None` |
| `cached_prefix_tokens` | `_cached_tokens` |
| `required_kv_blocks` | `_required_blocks` |
| tenant quota (six fields) | neutralised — see below |

Missing signals are passed through as `None` rather than filled in, so
`should_shed` keeps failing closed on them. `kv_usage` is only computed when
`total_kv_blocks` is truthy, so a zero or missing total yields `None` and
admission refuses rather than dividing. `_cached_tokens` is capped at
`req.pending.n_in`: admission treats a resident prefix as a discount on this
request's own input and validates `0 <= cached <= n_in`, so an over-large
number from a worker would cost that worker its turn on a `503` that describes
the telemetry rather than the worker.

`_required_blocks` is `ceil(max(n_in - cached + n_out, 0) / 16)` — a resident
prefix costs nothing. This has to agree with what `serve.py` hands admission and
with the `ceil(tokens / 16)` that `sched.py` charges as work actually runs. If
the three disagree, the router predicts refusals that never happen, or misses
ones that do.

### 3. The strategy (`_choose`)

Everything reaching here is known and would accept the request. A strategy
answers only the cheap question — which of several acceptable workers is the
best of them — and it cannot shed and cannot pick a dark worker, because
neither is its decision to make.

---

## The four strategies

| Strategy | Rule | What it is for |
|---|---|---|
| `random` | Uniform draw | The baseline. No telemetry, no herd, no correlation between routers |
| `least_loaded` | `min` by `(running + waiting, -free_kv_blocks)` | The obvious answer, and the one that herds |
| `p2c` | Draw two, take the less loaded | Most of least-loaded's benefit without the herd |
| `prefix_then_load` | Largest resident prefix, ties by load | Cache affinity; the default |

**`least_loaded`** sorts on work in front first — running plus waiting is what
this arrival queues behind — then breaks ties on free blocks, because the
request has to *fit* as well as queue (`test_least_loaded_breaks_an_equal_queue_on_free_memory`:
equal queues of 4, and the worker with 6,000 free blocks beats the one with
500). An exact tie on both keeps the first worker listed, which is the only
place the order of `workers` matters.

**`p2c`** exists because least-loaded has a failure mode that gets worse as the
fleet gets busier: every router reads the same telemetry, agrees on the same
best worker, and sends it everything — and the herd lands a full scrape interval
before the numbers that would have warned them off. Two random draws remove
nearly all of that for the cost of one comparison, because the worst worker now
only wins if both draws land on it. `test_p2c_never_picks_the_worst_of_three`
pins that over 60 routes across three workers.

**The honest note on p2c:** with two workers, `rng.sample(candidates, 2)` draws
both of them, so p2c *is* least-loaded — the full scan, with a shuffle in front
of it. Part 3 runs two workers. So any p2c-versus-least-loaded result on this
fleet measures nothing about sampling; it measures the tie-break order and the
RNG. The `t1_unique_prefix` trace therefore contrasts **random against p2c**,
where the difference is real, and the p2c-versus-least-loaded comparison is
deferred to a fleet large enough for it to mean something. A strategy that
degenerates at the size you actually deploy is worth naming before someone
reads a null result as evidence.

**`prefix_then_load`** takes the maximum resident prefix and then applies
`_least_loaded` inside that set. A warm worker skips the prefill for those
tokens, which is the most expensive thing the request would otherwise ask for;
load still decides between equally warm workers, so affinity concentrates
traffic only as far as the queues allow (`test_prefix_then_load_prefers_a_warm_worker`
sends the request to a worker carrying 12 requests over an idle cold one;
`test_prefix_then_load_breaks_a_tie_on_load` splits two equally warm workers by
queue).

Two cases need no special handling and get none. A **cache miss** ties every
candidate at zero tokens, so the strategy is exactly `_least_loaded` —
`test_a_prefix_miss_falls_back_to_load` asserts the two strategies return the
same worker rather than describing the behaviour. And a **warm worker with no
memory left** cannot win, because H4 removed it two steps earlier. That is what
it buys to run the safety rules first: the strategies have no error cases.

---

## Decisions worth arguing about

### Failing open on a dark fleet

`_admissible` is **skipped entirely** when `fleet_is_dark`. This is the single
most challengeable line in the file and it deserves the argument in both
directions.

*The case against.* Admission exists to protect workers from work they cannot
do. Skipping it means that during a telemetry outage the router will happily
place requests on workers that may be at 100% KV, with growing queues and
deadlines that cannot be met. The failure is silent: no `503` is emitted, so
the metric that would show the problem is the one that stays flat.

*The case for.* `should_shed` fails closed on every missing signal by design —
an unknown `p50` refuses even with an empty queue. A dark fleet supplies
nothing but missing signals, so asking admission about it does not produce a
capacity answer; it produces `503` for **every request to every worker for as
long as the scrape is down**. That is not admission control, it is an outage
with an HTTP code, and it is triggered by the monitoring system rather than by
the workers. It also directly contradicts H6's own fallback one step earlier:
`_eligible` deliberately refuses to let a monitoring failure become a total
failure, and then H4 would impose exactly that total failure by another route.
The workers themselves have not stopped working; only our view of them has
stopped arriving.

*Why the trade lands this way.* The request has already passed the gateway's own
admission pass before it reached the router, so this is a second opinion being
skipped, not the only one. The worker is still free to refuse on arrival — it
has the numbers we lack. And the blast radius is bounded by how long telemetry
is down, whereas the alternative's blast radius is *all traffic* for the same
duration, which is strictly worse for the same cause.

*What would change the answer.* If the workers had no admission of their own —
if this were the last gate rather than the second — failing open would be
indefensible and the right move would be a degraded-mode cap: route, but only
at a fraction of normal concurrency. That is the version worth building if the
t3 results show the fail-open window doing real damage.

### Tenant quota is neutralised in the H4 snapshot

`_snapshot` sends `tenant_tokens_used=0`, `tenant_tokens_reserved=0`,
`tenant_token_limit=_QUOTA_UNUSED` and the same three for requests, so gates 1
and 2 always pass: zero committed against an allowance of one is 0% used.

**Why:** a `429` follows the client to every worker in the fleet. Tenant
allowance belongs to the tenant, not to a worker, so it is not a *routing*
input — there is no worker that makes an over-budget tenant under budget, and
letting gate 1 fire here would empty the candidate list for a reason that has
nothing to do with capacity. The result would be an H4 `503` (`no_worker_would_accept`)
where the honest answer is a `429` the gateway should have returned once,
before routing. The router asks admission the **capacity** question only, and
supplies quota inputs that pass so gates 1 and 2 cannot stand in for a capacity
answer.

**Why `1` and not `0`:** `_check_allowance` treats `limit == 0` as *no allowance
at all* and refuses immediately. Zero would neutralise the gate in the wrong
direction.

**The cost:** this only holds while the gateway really does settle quota before
calling `pick`. Nothing in `router.py` enforces that, and a caller who skipped
the gateway's admission pass would get a router that ignores tenant limits
entirely.

### The block size is duplicated, not imported

`BLOCK_SIZE = 16` is written out here, and again in `serve.py`, and again as
`KvPool.block_size`'s default in `sched.py`.

**Why:** `serve.py` is the router's caller in Part 3, so importing
`serve.BLOCK_SIZE` would close an import cycle. `sched.py` does not export the
value at module level at all — it is a dataclass field default, reachable only
through a `KvPool` instance the router has no reason to hold.

**The cost, stated plainly:** four copies of one number. Agreement used to be
by comment rather than by construction, which is the weaker kind — the first
divergence would have shown up as the router predicting admission decisions
that do not match the ones the worker makes: quiet, plausible, and hard to
trace back to a constant. `admit.md` lists the same hazard from the admission
side.

`tests/test_constants_agree.py::test_every_copy_of_the_block_size_agrees` now
pins all four — the three module constants and `KvPool.block_size`'s default —
so the copies are still copies, but drift fails a test instead of quietly
changing what a gate means. A shared constants module would be the fuller fix
and would cost an import cycle to get.

### Reads only

`pick` reserves nothing, records nothing, and mutates no view — including not
reordering the `workers` list it is handed, which `_prefix_then_load` and
`_least_loaded` would be free to do since they build their own lists.
`test_picking_does_not_modify_the_fleet` pins it for all four strategies.

**The consequence, which is a real limitation:** two requests routed in the same
scrape interval see identical telemetry and can both be sent to the same worker
against the same free capacity. This is the routing-side twin of admission's
missing reservation. Closing it needs in-flight placements tracked between
scrapes, which is deliberately out of scope for the exercise.

### An unknown strategy raises

`pick` validates `strategy` against `STRATEGIES` and raises `ValueError` before
touching the fleet. A misspelled strategy is a bug in the experiment harness,
and silently routing by the default would produce results labelled with a
strategy that never ran — the worst possible outcome for a comparison. Pinned by
`test_an_unknown_strategy_is_rejected`.

---

## Status codes

| Code | Meaning | Should the client retry elsewhere? |
|---|---|---|
| `503` | No worker would accept, or no workers exist | No — retry **later**, after `retry_after` |

`Shed` carries only `503`. A `429` would mean the tenant is over budget, and by
construction that decision was settled by the gateway before `pick` was called.
`retry_after=2.0` is a placeholder in the same sense as `admit.py`'s `1.0`: the
fleet cannot yet estimate when capacity actually returns.

---

## Known limitations

1. **No reservation between scrapes.** Concurrent routes see the same telemetry
   and can oversubscribe the same worker. Bounded by the scrape interval.
2. **The fail-open window is unbounded.** A dark fleet routes on hope for as
   long as telemetry is down, with no degraded-mode cap on concurrency.
3. **p2c degenerates at two workers.** Drawing two from two is the full scan.
4. **`STALE_AFTER_S` is a fixed 5 seconds**, not derived from the observed
   scrape interval. A slower scrape would mark healthy workers unknown; a
   faster one wastes the grace.
5. **`cached_prefixes` is exact-match on a hash.** A request sharing 90% of its
   prompt with a resident prefix under a different hash gets no credit, so
   affinity is all-or-nothing.
6. **One prefix per request.** `prefix_hash` is a single value, so a request
   that could reuse two separate cached segments is credited for neither.
7. **Three copies of the block size**, agreeing by comment. See above.
8. **Health and freshness are trusted as reported.** The router cannot tell a
   worker that is down from a scraper that is down, and treats both the same —
   correct for H6, but it means a broken scraper darkens a healthy fleet.

---

## Assignment questions this file answers

| Question | Location |
|---|---|
| Where do I avoid sending work to a worker that has gone quiet? | `_eligible` / `_is_unknown` (H6), with `_load_key` behind it |
| Where do I avoid bouncing a request around the fleet? | `_admissible` and the `Shed` in `pick` (H4) |
| Where do I exploit shared prefixes? | `_prefix_then_load`, with `_cached_tokens` feeding admission's gate 4 |
| Where do I avoid herding onto one worker? | `_power_of_two` |
| Where does routing stop and admission start? | `_would_shed` — the router predicts, it never re-implements |

---

## Tests

`tests/test_router_exercise.py` — 16 test functions, 41 cases after
parametrisation. Coverage: each of the four strategies and its tie-break; H6
across five unknown shapes × four strategies plus the all-unknown fallback; H4
for a single dead-end worker, a fleet that would all shed, and an admission gate
beyond memory; the empty fleet; strategy validation; and read-only behaviour for
every strategy.

Telemetry is supplied directly. Scraping, freshness tracking and the live prefix
index are not exercised here — those belong to the live gateway router this was
extracted from, which has its own tests there.

---

## Experiments

Three traces, generated by `traces/gen_router.py`, each sized against
`serve.py`'s simulated hardware so that exactly one routing question is
answerable from it:

| Trace | Contrast | The question |
|---|---|---|
| `t1_unique_prefix.jsonl` | `random` vs `p2c` | With every prefix distinct, does reading load beat not reading it? |
| `t2_shared_prefix.jsonl` | `least_loaded` vs `prefix_then_load` | Is cache affinity worth the load imbalance it creates? |
| `t3_stale.jsonl` | `least_loaded` on 15-second-old telemetry | What H6 is actually preventing |

**Results are pending.** The runs are being carried out separately and no
numbers are quoted here until they exist; `t3` in particular is the one that
turns H6 from an assertion into a measurement.
