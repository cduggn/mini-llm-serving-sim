# mini-llm-serving-sim

A simulated LLM serving system: admission control, scheduling and routing, with no
GPU, no network and no model. Tokens are integers and KV blocks are counters, so a
run takes about a second and the policy is the only variable.

```bash
make setup && make test
make part2      # scheduler comparison: FCFS vs priority vs DRR
make part3      # routing experiments -> results/ and plots/
```

Python 3.11+. The simulation is standard library only; `pytest` and `matplotlib`
are for the tests and plots.

## Request path

```
trace arrival -> router.pick -> admit.should_shed -> worker queue -> sched.step
                 (which worker?)  (accept or refuse?)                (who runs next?)
```

`serve.py` drives the loop. Nothing retries and nothing reserves capacity.

---

## Core files

### `serve.py` — the harness

Replays a trace against one or more workers and records the metrics.

- **Input:** a JSONL trace, one request per line:
  `id, arrival_t, priority, prompt_tokens, max_new_tokens, prefix_hash, timeout_s, tenant`
- **Per arrival:** route, then admit, then append to the chosen worker's `waiting` list.
- **Per step:** the worker whose next step is due earliest runs `sched.step()`.
  Its clock advances by `0.006 + prefill_tokens / 20,000` seconds.
- **Outputs:** TTFT per tenant, rejections by code, preemptions, wasted decode tokens, KV block-seconds, requests per worker.

Simulated hardware: 2,048-token budget per step, 32 decode slots, 8,192 KV blocks of 16 tokens.

### `router.py` — which worker?

`pick(req, workers, strategy=...)` returns a `WorkerView` or a `Shed(503)`.

- **Input:** `RouteRequest(pending, prefix_hash)` and one `WorkerView` per worker:
  `healthy, age_s, running, waiting, free_kv_blocks, total_kv_blocks, ttft_p50_s, ttft_p99_s, queue_growing, cached_prefixes`

| Check | Refuses / excludes when |
|---|---|
| **H6** `_eligible` | worker is unhealthy, telemetry is older than 5 s, or any load signal is missing. If *every* worker is unknown, all are kept (fail open). |
| **H4** `_admissible` | `should_shed` says this worker would refuse (capacity only, not quota). If none are left: `Shed(503, retry_after=2.0)`. |

Then one strategy picks from the remaining workers:

| Strategy | Picks |
|---|---|
| `random` | any worker |
| `least_loaded` | fewest `running + waiting`, ties broken by more free KV |
| `p2c` | the less loaded of two workers drawn at random |
| `prefix_then_load` | the most cached prefix tokens, then least loaded |

### `admit.py` — accept or refuse?

`should_shed(req, snap)` runs the gates in order and returns the first refusal as
`(shed, code, retry_after)`. Every gate **fails closed**: a missing or invalid signal refuses with `503`.

- **Input:** `PendingRequest` (`n_in`, `n_out`, `deadline_s`, `priority`) and `AdmissionSnapshot`.

| # | Gate | Snapshot inputs | Refuses when |
|---|---|---|---|
| 1 | token allowance | `tenant_tokens_used`, `_reserved`, `_limit` | used + reserved ≥ 95% of limit → `429` |
| 2 | request allowance | `tenant_requests_used`, `_reserved`, `_limit` | used + reserved ≥ 95% of limit → `429` |
| 3 | queue wait | `queue_length`, `ttft_p50_s` | `queue_length × p50 > 0.5 × deadline_s` → `503` |
| 4 | KV pressure | `kv_usage`, `cached_prefix_tokens` | KV > 92% used and no cached prefix → `503` |
| 5 | KV capacity | `free_kv_blocks`, `required_kv_blocks` | required > free → `503` |
| 6 | tail latency | `queue_growing`, `ttft_p99_s`, `ttft_p50_s` | priority ≥ 10, queue growing and p99 > 4 × p50 → `503` |

Priority < 10 (interactive) is exempt from gate 6 only.

### `sched.py` — who runs next?

`step(waiting, running, budget, policy=..., drr=..., kv=..., decode_slots=...)` simulates one forward pass
and returns a `StepResult`.

- **Input:** the two queues of `ScheduledRequest`, the token budget, a policy (`fcfs` / `priority` / `drr`),
  a `DrrState` (DRR only), a `KvPool`, and the decode slot limit.

One step, in order:

1. Drop aborted requests and free their blocks.
2. Pick decoders (prefill done, output remaining), sorted by policy and capped at `decode_slots`.
3. Reserve 1 token per decoder; the rest of the budget goes to **one** prefill chunk.
4. Choose the prefill request: FCFS by arrival, priority by value then arrival, or DRR by tenant credit.
5. If KV is short, shrink the chunk to fit, preempting only if even one token won't fit.
6. Advance each decoder by one token, preempting or stalling if no block is free.
7. Remove finished requests and free their blocks.

| Check | Rule |
|---|---|
| KV room `_kv_room` | a request can only hold as many tokens as there are free blocks |
| Preemption `_select_victim` | lowest priority, latest arrival; never a more important request, never one holding no blocks |
| Recompute `_preempt` | the victim loses all progress and returns to `waiting`; its tokens count as wasted |
| DRR credit `_charge` | each tenant earns 2,048 credit per round, is charged after the work runs, and may go into debt |

---

## Where each question is answered

| # | Question | Where |
|---|---|---|
| Q1 | Avoid accepting work that will time out | `admit` gate 3 |
| Q2 | Protect KV memory | `admit` gates 4–5; `sched._kv_room`, `_reclaim_blocks` |
| Q3 | Prioritise interactive traffic | `admit` gate 6; `sched._select_priority`, `_select_victim` |
| Q4 | Stop one tenant monopolising | `admit` gates 1–2; `sched._select_drr` |
| Q5 | Preempt a request | `sched._preempt` via `_reclaim_blocks` |
| Q6 | Exploit shared prefixes | `router._prefix_then_load`; `cached_prefix_tokens` in gate 4 |
| Q7 | Avoid a worker that has gone quiet | `router` H6 |
| Q8 | Avoid bouncing a request around the fleet | `router` H4 |

## Key findings

Full numbers are in [`REPORT.pdf`](REPORT.pdf) and [`docs/part2-findings.md`](docs/part2-findings.md).

- **DRR preempts least but wastes most:** about 52 decode tokens lost per preemption, against 7 for priority and 1 for FCFS. Preemption is recompute, and DRR lets batch work decode for a long time before it's evicted.
- **p2c equals least_loaded at 2 workers:** drawing two from two is a full scan, so p2c is compared again at 4 and 8 workers.
- **H6 hurts at 2 workers and helps at 8:** excluding one stale worker drops half of a 2-worker fleet (p99 23 s, 29% shed) but costs almost nothing at 8 workers.

---

## Other files

| Path | Purpose |
|---|---|
| `state.py` | `PendingRequest`, the request shape shared by admission and routing |
| `experiments.py` | Runs Part 2 and T1/T2/T3, writes `results/` and `plots/` |
| `traces/` | Seeded workload generators and the committed `.jsonl` traces |
| `tests/` | Unit tests for every module |
| `docs/` | Per-module reasoning and limitations |
| `REPORT.pdf` | Write-up of the results |

## Running by hand

```bash
export PYTHONPATH=.
python serve.py --trace traces/mixed.jsonl --policy all
python serve.py --trace traces/t2_shared_prefix.jsonl --seconds 100 \
    --policy drr --workers 2 --strategy all --prefix-cache
python serve.py --trace traces/t3_stale.jsonl --seconds 120 \
    --policy drr --workers 2 --strategy all --stale-worker w1 --stale-lag 15
```

Flags: `--policy`, `--workers`, `--strategy` (or `all`), `--prefix-cache`,
`--stale-worker` / `--stale-lag` / `--stale-hidden`, `--seed`, `--json`. Seeds are fixed at 7.

## Known gaps

- Nothing reserves capacity, so two arrivals can be admitted against the same free blocks.
- `retry_after` is a placeholder.
- Output length is always `max_new_tokens` (no end-of-sequence token).
- Wasted prefill isn't counted.
