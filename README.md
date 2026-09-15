# mini-llm-serving-sim

A simulated LLM serving system: admission, scheduling and routing.

No GPU, no network, no model. Tokens are integers and KV blocks are counters,
so a run takes about a second.

```bash
make setup && make test
make part2      # schedulers: FCFS vs priority vs DRR
make part3      # routing experiments -> results/, plots/
```

Python 3.11+.

---

## Flow

`serve.py` runs this loop for every request and owns the clock.

```
             trace (JSONL)
                  │
                  ▼
        ┌───────────────────┐
        │     router.py     │   which worker?
        └─────────┬─────────┘
                  │ ──── no worker would accept ────► 503
                  ▼
        ┌───────────────────┐
        │     admit.py      │   accept or refuse?
        └─────────┬─────────┘
                  │ ──── gate fails ────► 429 / 503
                  ▼
        ┌───────────────────┐
        │   worker queue    │   waiting list
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │     sched.py      │   who runs this step?
        └─────────┬─────────┘
                  ▼
              tokens out
```

---

## Core files

### serve.py

The harness. Replays a trace and records metrics.

**Input:** one request per trace line

```
id, arrival_t, priority, prompt_tokens, max_new_tokens, prefix_hash, timeout_s, tenant
```

**What it does**

- Routes, admits and queues each arrival.
- Steps whichever worker is due next.
- Step time = `0.006 s + prefill_tokens / 20,000`.

**Simulated hardware**

- 2,048 tokens per step
- 32 decode slots
- 8,192 KV blocks × 16 tokens

---

### router.py

Picks a worker, or refuses.

**Input:** the request, plus one `WorkerView` per worker

```
healthy, age_s, running, waiting, free_kv_blocks,
total_kv_blocks, ttft_p50_s, ttft_p99_s, queue_growing, cached_prefixes
```

**Checks**

| Rule | Removes a worker when |
|---|---|
| H6 | unhealthy, stale > 5 s, or a signal is missing |
| H4 | admission says it would refuse |

- If every worker is unknown, H6 keeps them all.
- If H4 leaves nobody, return `503`, retry after 2 s.

**Strategies**

| Name | Picks |
|---|---|
| `random` | any worker |
| `least_loaded` | fewest running + waiting |
| `p2c` | better of two random workers |
| `prefix_then_load` | most cached prefix, then least loaded |

---

### admit.py

Accepts or refuses one request.

**Input:** `PendingRequest` and `AdmissionSnapshot`

- Gates run in order. The first refusal wins.
- A missing or invalid signal refuses (`503`).

| # | Gate | Reads | Refuses when |
|---|---|---|---|
| 1 | Tokens | tenant tokens used, reserved, limit | ≥ 95% of limit → `429` |
| 2 | Requests | tenant requests used, reserved, limit | ≥ 95% of limit → `429` |
| 3 | Queue wait | queue length, p50 TTFT | queue × p50 > half the deadline |
| 4 | KV pressure | KV usage, cached prefix | > 92% used, no cached prefix |
| 5 | KV capacity | free blocks, required blocks | required > free |
| 6 | Tail latency | queue growing, p99, p50 | batch, queue growing, p99 > 4 × p50 |

Interactive requests (priority < 10) skip gate 6.

---

### sched.py

Runs one step on one worker.

**Input**

- `waiting` and `running` queues
- token budget
- policy: `fcfs`, `priority` or `drr`
- KV pool and decode slot limit

**One step**

1. Drop aborted requests.
2. Pick decoders, up to 32.
3. Reserve 1 token per decoder.
4. Give the rest to one prefill chunk.
5. Advance each decoder by one token.
6. Remove finished requests.

**Checks**

| Check | Rule |
|---|---|
| KV room | a chunk shrinks to fit free blocks |
| Preemption | evict lowest priority, never a more important request |
| Recompute | the victim restarts from zero |
| DRR | 2,048 credit per tenant per round, debt allowed |

---

## Other files

| Path | Purpose |
|---|---|
| `state.py` | shared `PendingRequest` shape |
| `experiments.py` | runs Part 2 and T1–T3 |
| `traces/` | workload generators and traces |
| `tests/` | unit tests |
| `docs/` | per-module notes |
| `docs/questions.md` | which code answers each assignment question |
| `REPORT.pdf` | results write-up |

---

## Run by hand

```bash
export PYTHONPATH=.
python serve.py --trace traces/mixed.jsonl --policy all
python serve.py --trace traces/t2_shared_prefix.jsonl --seconds 100 \
    --policy drr --workers 2 --strategy all --prefix-cache
```

Flags: `--policy`, `--workers`, `--strategy`, `--prefix-cache`,
`--stale-worker`, `--stale-lag`, `--seed`, `--json`.

---

## Known gaps

- Nothing reserves capacity.
- `retry_after` is a placeholder.
- Output is always `max_new_tokens` long.
- Wasted prefill isn't counted.
