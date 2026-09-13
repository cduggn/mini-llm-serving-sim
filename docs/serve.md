# `serve.py` — the simulated serving system

> One clock, one admission gate, one scheduler. Part 2's workload runs here.

`serve.py` is the runner the assignment implies but never names. Per arrival
the path is **route → admit → a worker queue**, and every worker advances
through `step()`. Part 2 uses a single worker so that **scheduling policy is
the only variable**; Part 3 keeps the scheduler fixed, adds a second worker,
and makes `router.pick` the only variable instead. `route()` is the whole
seam, and Part 2's numbers are unchanged by it — `make part2` still prints the
table in `docs/part2-findings.md` line for line.

---

## The clock

`step()` has no notion of time, but TTFT is measured in seconds. The cost model:

```
step_seconds = 0.006 + prefill_tokens / 20,000
```

A decode-only pass costs 6ms whatever the batch size, because every sequence
advances together. Prefill adds time in proportion to the tokens it processes,
so a full 2,048-token chunk costs ~108ms — eighteen decode passes.

**Why this matters more than it looks:** a flat per-step duration would have
erased the central effect of the experiment. Prefill work directly steals
decode throughput. At light load steps average ~8ms and decode delivers
~4,000 tokens/s; under prefill pressure steps average ~43ms and decode
collapses to ~744 tokens/s. Same hardware, 5× less generation.

When the server is idle the clock **jumps to the next arrival** instead of
burning empty 6ms steps.

## Simulated hardware

| Setting | Value | Why |
|---|---|---|
| `TOKEN_BUDGET` | 2,048 / step | The assignment's prefill budget |
| `DECODE_SLOTS` | 32 | Concurrent sequences, vLLM's `max_num_seqs` |
| `KV_TOTAL_BLOCKS` | 8,192 (131k tokens) | **Must hold `DECODE_SLOTS` × average request.** See below. |
| `BLOCK_SIZE` | 16 | Matches `sched.py` |

## Admission bridge

`should_shed` needs an `AdmissionSnapshot` that nothing else produces, so
`serve.py` assembles one per arrival:

- **Percentiles** come from the last 64 TTFT samples. Until samples exist, a
  cold-start estimate stands in — without it, the fail-closed queue-wait gate
  would reject *every* request at t=0.
- **`cached_prefix_tokens`** is 3,500 once an agent prefix has been seen. This
  is how the shared prefix matters in Part 2: it lowers `required_kv_blocks`,
  so agents stay admissible under memory pressure. The scheduler still
  prefills all 4,000 tokens — prefix-aware caching is Part 3's job.
- **Tenant limits** are deliberately generous. Quota refusals would mask the
  capacity rules, and Part 2 is about scheduling, not billing. Every rejection
  in the results below is a 503; not one is a 429.

---

## Calibration

Two mistakes, both instructive, before the numbers meant anything.

**1. The KV pool was too small (2,048 blocks).** 32 decode slots × ~2,100
tokens per request needs ~4,200 blocks. A pool that cannot hold one full batch
preempts continuously: 348–423 preempts for 37–118 admitted requests, each
request evicted 4–10 times. Almost nothing completed, and the policy
comparison drowned in thrash. **The sizing rule is that KV must cover the
decode concurrency you allow** — otherwise `decode_slots` is a lie.

**2. The offered load was 5× capacity.** At 8 req/s, 77% of requests were
rejected, so "completed/s" measured admission rather than scheduling.

**Capacity, derived:** the average request is 1,750 prefill tokens and 400
decode tokens.

```
prefill        1,750 / 20,000            = 0.0875 s
decode         (400 / 32 slots) × 0.006  = 0.0750 s
                                   total ≈ 0.1625 s per request  →  ~6.15 req/s
```

A sweep at 4, 5, 6 and 7 req/s confirmed it. Below 6 the server absorbs
everything: zero preemptions, near-zero rejections, and all three policies
score identically. **7 req/s (~1.14× capacity) is the headline rate** — mild
overload, where all six required metrics are non-zero and policy actually
decides outcomes.

---

## Results — `traces/mixed.jsonl`, 7 req/s, 60 simulated seconds

444 requests (311 interactive / 89 batch / 44 agents), seed 7. All three
policies replay the identical trace.

> **`docs/part2-findings.md` is the canonical write-up of these results.** The table
> and answer below are repeated there with the calibration evidence; update
> that file first if the numbers ever change.

| Metric | FCFS | Priority | DRR |
|---|---|---|---|
| **completed req/s** | 3.53 | 5.70 | **5.77** |
| completed | 212 | 342 | 346 |
| **interactive p99 TTFT** | 10.99 s | **0.24 s** | 0.38 s |
| **batch p99 TTFT** | 11.99 s | 6.33 s | **2.70 s** |
| agents p99 TTFT | 11.54 s | 0.74 s | 0.97 s |
| **preempts/s** | 1.90 | 0.28 | **0.17** |
| preempts | 114 | 17 | 10 |
| **wasted decode tokens** | 158 | 125 | **524** |
| **requests rejected** | 195 | 70 | 69 |
| *tokens wasted per preempt* | *1.4* | *7.4* | *52.4* |

**FCFS collapses under overload.** One 8,000-token prompt holds the single
prefill slot for four steps while everything queues behind it, so *every*
class lands near an 11s p99 — interactive traffic included. It also rejects
195 requests, nearly triple the others, because the queue-wait gate correctly
predicts that arrivals behind that backlog are doomed.

**Priority does exactly what it promises, and charges batch for it.**
Interactive p99 of 0.24s is 46× better than FCFS. Batch pays with a 6.33s p99.

**DRR trades a little interactive latency for balance.** Interactive p99 rises
to 0.38s, but batch p99 *improves* to 2.70s — less than half priority's — and
throughput is the highest of the three.

---

## Which scheduler wastes the most decode work?

**DRR**, by a factor of three or more, and it does so while preempting the
*fewest* times. It discarded 524 decode tokens across just 10 preemptions,
against priority's 125 across 17 and FCFS's 158 across 114. The separation is
entirely in the cost of each eviction: DRR loses 52 tokens per preemption,
priority 7.4, FCFS 1.4. The cause is DRR's own guarantee. Because every tenant
is given a steady share of the GPU, a batch request finishes its prefill and
then decodes continuously, accumulating hundreds of generated tokens before
memory pressure finally evicts it — and preemption is recompute, so all of
them are thrown away. FCFS, by contrast, is so congested that its victims have
barely started generating: they hold KV blocks from a finished prefill but have
almost no output to lose. Priority starves batch work at the *prefill* stage,
so it never reaches a state worth losing either. The uncomfortable conclusion
is that fairness and wasted work pull against each other — the policy that
refuses to starve anyone is precisely the policy that guarantees its victims
have something valuable to lose.

*Caveat: one trace, one seed. The ordering held at 8 req/s too (FCFS 2.7,
priority 8.8, DRR 20.8 tokens per preempt), so the effect is not an artefact
of a single load point, but a seed sweep would make it solid.*

---

---

# Part 3 — routing

Four things changed, and only the fourth can move a Part 2 number.

**1. `route()` replaced `pick_worker()`.** It builds one `WorkerView` per
worker and calls `router.pick`, then maps the chosen name back to the worker
that owns the queues. A `Shed` is counted as a 503, the same code admission
returns for the same condition, so the router's H4 refusal and admission's
own refusal land in one counter. Admission still runs afterwards on the
chosen worker: the router asks it only the *capacity* question with quota
inputs that pass, so tenant limits are still settled here.

**2. `_view()` is the adapter**, and therefore where staleness is injected. A
view carries the worker's live scrape with `age_s = 0`, unless a `StaleSpec`
names it — then it carries the scrape from `lag_s` ago, with the age either
reported honestly or reported as zero. Reporting it as zero is the only way
to see what `pick` would do *without* H6: the rule is unconditional inside the
router, so the experiment lies to it rather than disabling it.

**3. Every worker has its own clock.** The loop is a next-event loop: the
worker whose next step is due earliest takes it, and the global clock follows
that worker. A long prefill on one GPU therefore does not slow another one
down. With a single worker this is exactly the Part 2 loop.

**4. `--prefix-cache` makes a warm prefix real.** In Part 2 a resident prefix
is only an admission discount; the GPU still prefills all 4,000 tokens. With
the flag on:

- a worker advertises a prefix only while it is **actually holding the
  blocks** — the last request using it drains, the prefix goes with it, and
  the worker goes cold. Without this, one stray request makes a worker warm
  forever and prefix affinity has nothing left to choose between workers.
- a request landing on a warm worker prefills and holds **only its tail**;
- the prefix itself is charged to the worker **once**, as its own holding in
  the KV pool, for as long as any request carrying it is in flight there.

This flag is off by default and Part 2 never sets it. **It is only meaningful
on a trace with genuinely shared prefixes** — with per-request unique hashes
every request would claim a residency holding covering its own prompt, which
double-charges its KV.

### New flags

| Flag | Default | What it does |
|---|---|---|
| `--workers` | 1 | Fleet size. 1 is Part 2. |
| `--strategy` | `least_loaded` | One of `router.STRATEGIES`, or `all` |
| `--prefix-cache` | off | A warm prefix removes prefill work and KV blocks |
| `--stale-worker` | none | e.g. `w1` — its telemetry arrives late |
| `--stale-lag` | 15.0 | How late, in seconds |
| `--stale-hidden` | off | Report the stale view as fresh, so H6 cannot see it |
| `--seed` | 7 | Routing draws |
| `--json` | none | Also write the run as JSON |

`--workers > 1` needs a single `--policy`, because the table then compares
strategies rather than schedulers.

### New metrics

`KV block-seconds` is the headline memory number: blocks held, integrated over
time. Peak alone cannot tell apart a strategy that holds the same blocks twice
as long. `routing imbalance` is (max − min) / total placements, so it does not
depend on how much was admitted. Every run also records a per-second timeline
— arrivals, completions, rejections, p99 TTFT, fleet KV — which is what
`plots/soak.png` is drawn from.

---

## Limitations

1. **Prefix caching understates cold prefill.** The first request of a
   residency episode prefills its whole prompt *and* the worker is charged for
   the prefix holding, so the prefix is counted twice while that request runs.
   The overcharge falls on prefix-aware routing, which is the safe direction
   for the experiment it is read by.
2. **A stale worker's prefix index is not stale**, only its load scrape. The
   two are different data paths in a real gateway, and only the load path is
   what T3 is about.
3. **No disconnects.** The abort path in `sched.py` is tested but unexercised
   here, so the three policies differ only by scheduling.
4. **TTFT is kept at the first token even if that request is later preempted.**
   The client did see a token; a real recompute regenerates the same text.
5. **`used` versus `reserved` is not modeled separately** — everything admitted
   and unfinished counts as reserved until it completes.
6. **p99 over small samples.** Counts travel with every percentile in the
   output for this reason.

---

## Running it

```bash
make part2            # regenerate the trace, run all three policies
make part2-overload   # the 8 req/s congestion-collapse sensitivity case
make trace            # regenerate traces/mixed.jsonl only
make router-traces    # regenerate the three Part 3 traces and the fleet sweeps
make part3            # Part 2 + T1/T2/T3, into results/ and plots/
make test             # full suite, 320 tests
```

`make part3` runs `experiments.py`, which re-runs Part 2 first and fails
loudly in its output if the published table has moved. Results land in
`results/` (JSON and CSV per experiment, plus `summary.json`) and the soak
plot in `plots/soak.png`.

One experiment by hand, for varying it:

```bash
.venv/bin/python serve.py --trace traces/t1_unique_prefix.jsonl --seconds 90 \
    --policy drr --workers 2 --strategy all
.venv/bin/python serve.py --trace traces/t2_shared_prefix.jsonl --seconds 100 \
    --policy drr --workers 2 --strategy all --prefix-cache
.venv/bin/python serve.py --trace traces/t3_stale.jsonl --seconds 120 \
    --policy drr --workers 2 --strategy all --stale-worker w1 --stale-lag 15
```

Raw equivalents, for varying the experiment (run from `class7/` so the
`gateway` package imports):

```bash
.venv/bin/python traces/gen_mixed.py --rate 7 --out traces/mixed.jsonl --seed 7
.venv/bin/python serve.py --trace traces/mixed.jsonl --policy all
```

`serve.py` also accepts `--policy fcfs|priority|drr` and `--seconds`;
`gen_mixed.py` accepts `--rate`, `--seconds` and `--seed`. Flag tables and the
warning about the narrow useful load band are in **`docs/part2-findings.md`**.
Tests are in `tests/test_serve.py`.
