# mini-llm-serving-sim

A simulated LLM serving system — admission control, scheduling and routing — with
no GPU, no network and no model. Every "token" is an integer and every "KV block"
is a counter, so the whole thing runs in about a second and the policy decisions
are the only variable.

The four policy modules were extracted from a live vLLM gateway, unchanged. The
point of the simulator is to put them under load you can reproduce exactly and
measure what they actually do, rather than what the comments claim.

```bash
make setup && make test     # 304 tests
make part2                  # scheduler comparison: FCFS vs priority vs DRR
make part3                  # routing experiments -> results/ and plots/
```

Python 3.11+. The simulation itself is pure standard library; `pytest` and
`matplotlib` are only needed for the tests and the two plots.

---

## The request path

Each request goes **route → admit → schedule**, in that order, and each stage
owns one decision. Nothing retries and nothing reserves capacity.

**1. `router.py` — which worker?** &nbsp;*(answers Q6, Q7, Q8)*

`pick(req, workers)` returns a `WorkerView` or a `Shed`. Two safety rules run
first, so by the time a strategy sees the candidate list every worker on it is
both known and willing, and the strategies have no error cases:

- `_eligible` / `_is_unknown` — **H6**, an unknown worker is never treated as an
  idle one. Unhealthy, stale beyond 5 s, or missing any load signal all count as
  unknown. If *every* worker is unknown the whole fleet is used anyway, so a
  monitoring outage cannot become a total outage.
- `_admissible` / `_would_shed` — **H4**, ask `should_shed` whether this worker
  *would* accept before dispatching. If none would, refuse once with
  `Shed(503, retry_after=2.0)` rather than letting the client bounce around a
  fleet that is already out of room.
- `_choose` — one of `random`, `least_loaded`, `p2c`, `prefix_then_load`.

**2. `admit.py` — should this be accepted at all?** &nbsp;*(answers Q1–Q4, Q6)*

`should_shed(req, snap)` runs six gates in a fixed order and returns the first
refusal as `(shed, code, retry_after)`. It says yes or no and nothing else — no
reserving, no retrying, no routing. Every gate fails closed: a missing signal
refuses, because an unknown signal is not a healthy one.

| Gate | Function | Refuses when |
|---|---|---|
| 1–2 | `_check_allowance` | tenant is at ≥95% of its token or request budget → `429` |
| 3 | `_check_queue_wait` | expected queue wait exceeds half the request's deadline → `503` |
| 4–5 | `_check_kv_pressure`, `_check_kv_capacity` | under 8% free KV and the prefix is cold, or the request simply will not fit → `503` |
| 6 | `_check_tail_latency` | p99 > 4× p50 *and* the queue is growing → shed batch, keep interactive |

**3. `sched.py` — who gets the GPU next?** &nbsp;*(answers Q2–Q5)*

One `step()` is one GPU turn: validate everything before mutating anything, drop
aborted clients, capture and cap the decoders, reserve each of them a token,
prefill exactly one prompt, and preempt only if memory demands it.

- `_select_fcfs` / `_select_priority` / `_select_drr` — who takes the single
  prefill slot this step.
- `_kv_room`, `_reclaim_blocks` — block accounting; a chunk shrinks to fit before
  anything is evicted.
- `_preempt`, victim chosen by `_select_victim` — preemption is **recompute, not
  swap**, so a victim's generated tokens are gone. The victim rule never evicts
  anything more important than the requester.

### The eight questions, and where each is answered

| # | Question | Where |
|---|---|---|
| Q1 | Where do I prevent accepting work that will time out? | `admit._check_queue_wait` (gate 3) — expected queue wait vs half of `req.deadline_s` |
| Q2 | Where do I protect KV memory? | `admit._check_kv_pressure` and `_check_kv_capacity` before entry; `sched._kv_room` and `_reclaim_blocks` after |
| Q3 | Where do I prioritise interactive traffic? | `admit._check_tail_latency` (gate 6) sheds batch under a blown tail; `sched._select_priority` orders the GPU; `_select_victim` refuses to evict upward |
| Q4 | Where do I stop one tenant monopolising? | `admit._check_allowance` (gates 1–2, 429) before entry; `sched._select_drr` shares the prefill slot after |
| Q5 | Where do I preempt a request? | `sched._preempt`, triggered by `_reclaim_blocks`, victim chosen by `_select_victim` |
| Q6 | Where do I exploit shared prefixes? | `router._prefix_then_load` picks the warm worker; `_cached_tokens` feeds admission gate 4 |
| Q7 | Where do I avoid sending work to a worker that has gone quiet? | `router._eligible` / `_is_unknown` (H6), with `_load_key` sorting an unknown view as maximally loaded behind it |
| Q8 | Where do I avoid bouncing a request around the fleet? | `router._admissible` / `_would_shed` (H4) — refuses once with `Shed(503, retry_after=2.0)` rather than letting the client shop around |

---

## Files

The three policy modules above are the point of the repo. Everything else exists
to drive them or to check them.

| Path | Responsible for |
|---|---|
| `router.py` | Picking a worker, or refusing. H6 and H4, then one of four strategies. |
| `admit.py` | Saying yes or no to one request. Six gates, first refusal wins, fails closed. |
| `sched.py` | One GPU turn: chunked prefill, decode, KV blocks, preemption. FCFS / priority / DRR. |
| `state.py` | The `PendingRequest` shape shared by admission and routing, lifted from the gateway so the same `should_shed` runs against both. |
| `serve.py` | The harness — simulated clock, per-worker queues, the admission bridge, metrics. Defines the simulated hardware. |
| `experiments.py` | Runs Part 2 and T1/T2/T3, writes `results/*.json` + `*.csv` and the two plots. Re-runs Part 2 first and fails loudly if the published table has moved. |
| `traces/gen_mixed.py` | Part 2 workload generator — seeded, exact class proportions (interactive / batch / agents). |
| `traces/gen_router.py` | The three routing workloads: unique prefixes (T1), shared prefix (T2), stale telemetry (T3). |
| `traces/*.jsonl` | The generated traces themselves, committed so results reproduce without regenerating. |
| `tests/` | 304 tests. Ordering, chunked prefill, DRR fairness, block accounting, victim selection, all six admission gates against malformed input, H6 across five unknown shapes × four strategies. |
| `results/`, `plots/` | Output of `make part3` — every number quoted in the report, plus the two figures. |
| `docs/` | The reasoning behind each module, gate by gate, including the limitations. |
| `REPORT.pdf` | The two-page write-up of the results. |

### Simulated hardware

Set in `serve.py`. Token budget 2,048/step, 32 decode slots, 8,192 KV blocks of
16 tokens. Step cost is `0.006 + prefill_tokens / 20,000` seconds — prefill work
steals decode throughput, which is the single most transferable result here.

**Capacity is ~6.15 req/s and the useful band is narrow.** Below ~6 req/s the
server absorbs everything and all three policies score identically; above ~8 the
admission gate sheds three quarters of the traffic and you are measuring
admission rather than scheduling. Part 2 runs at 7.

---

## What the experiments found

Full numbers and the reasoning are in [`REPORT.pdf`](REPORT.pdf) and
[`docs/part2-findings.md`](docs/part2-findings.md). The three results worth
knowing up front:

**DRR wastes the most decode work while preempting the least.** 524 wasted tokens
across 10 preemptions, against priority's 125 across 17 and FCFS's 158 across 114
— 52.4 tokens per preempt vs 7.4 and 1.4. Fairness is the cause: DRR lets a batch
request decode continuously until memory pressure evicts it, and preemption here
is recompute, so every generated token is thrown away. The policy that refuses to
starve anyone is the one that guarantees its victims have something to lose.

**On two workers, power-of-two-choices *is* least-loaded.** Drawing two from two
is the full scan. The p2c-vs-random result is real (8.4× better p99), but any
p2c-vs-least-loaded claim at N=2 measures the tie-break order and the RNG, so the
comparison is re-run at 4 and 8 workers.

**The stale-telemetry guard is wrong at N=2 and right at N=8.** H6 excludes a
worker whose telemetry has gone quiet. At two workers that takes half the fleet
offline and drives p99 to 23.35 s with 29.4% shed — much worse than the 8.17 s it
was preventing. At eight workers the same guard costs nothing and avoids a 65 s
p99. H6 costs one worker in N, so it costs more than the failure it prevents at
N=2 and almost nothing at N=8.

---

## Make targets

| Target | What it does |
|---|---|
| `make setup` | Create `.venv` and install `requirements.txt` |
| `make test` | Full suite |
| `make trace` | Regenerate `traces/mixed.jsonl` only |
| `make part2` | Regenerate the trace, run all three schedulers, print the comparison |
| `make part2-overload` | The 8 req/s congestion-collapse sensitivity case |
| `make router-traces` | Regenerate the three routing traces and the fleet-sweep copies |
| `make part3` | Part 2 + T1/T2/T3 into `results/` and `plots/` |
| `make all` | Tests, then everything |

Running one experiment by hand:

```bash
export PYTHONPATH=.
python serve.py --trace traces/t1_unique_prefix.jsonl --seconds 90 \
    --policy drr --workers 2 --strategy all
python serve.py --trace traces/t2_shared_prefix.jsonl --seconds 100 \
    --policy drr --workers 2 --strategy all --prefix-cache
python serve.py --trace traces/t3_stale.jsonl --seconds 120 \
    --policy drr --workers 2 --strategy all --stale-worker w1 --stale-lag 15
```

Useful `serve.py` flags: `--workers`, `--strategy` (or `all`), `--prefix-cache`,
`--stale-worker` / `--stale-lag` / `--stale-hidden`, `--seed`, `--json`.

Seeds are fixed at 7 throughout. The traces are inputs to a comparison, not a
sample, so the same seed every time is the point.

## Known gaps

Nothing reserves capacity, so two requests inside one scrape interval can both be
admitted against the same free blocks. `retry_after` is a placeholder and carries
no real estimate of when capacity returns. Priority is trusted as sent. Wasted
*prefill* is not counted, so true waste is larger than the metric reports. Each
module's `docs/` page lists its own limitations in full.
