# Part 2 findings — scheduler comparison

**Workload:** `traces/mixed.jsonl` — 444 requests over 60 simulated seconds at
7 req/s, seed 7 (311 interactive / 89 batch / 44 agents). All three policies
replay the identical trace, so **scheduling policy is the only variable**.

**Reproduce:** `make part2` — full details under
[Running the experiment](#running-the-experiment) below.

---

## Results

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
| *derived: tokens wasted per preempt* | *1.4* | *7.4* | *52.4* |

Every rejection was a 503 (capacity). Not one was a 429, because tenant limits
were set generously on purpose — quota refusals would have masked the capacity
rules this part is about.

### Reading the table

**FCFS collapses under overload.** A single 8,000-token prompt holds the one
prefill slot for four steps while everything queues behind it, so *every* class
lands near an 11-second p99 — interactive traffic included. It also rejects 195
requests, nearly triple the others, because the queue-wait gate correctly
predicts that arrivals behind that backlog are doomed. Throughput is 39% below
the other two.

**Priority does exactly what it promises, and sends batch the bill.**
Interactive p99 of 0.24s is **46× better than FCFS**. Batch pays for it with a
6.33s p99.

**DRR trades a little interactive latency for balance.** Interactive p99 rises
to 0.38s, but batch p99 *improves* to 2.70s — less than half of priority's —
and it posts the highest throughput of the three.

---

## Which scheduler wastes the most decode work?

**DRR**, by a factor of three or more, and it does so while preempting the
**fewest** times. It discarded 524 decode tokens across just 10 preemptions,
against priority's 125 across 17 and FCFS's 158 across 114. The separation is
entirely in the cost of each eviction: DRR loses 52 tokens per preemption,
priority 7.4, FCFS 1.4. The cause is DRR's own guarantee. Because every tenant
receives a steady share of the GPU, a batch request finishes its prefill and
then decodes continuously, accumulating hundreds of generated tokens before
memory pressure finally evicts it — and preemption is recompute, so every one
of those tokens is thrown away. FCFS, by contrast, is so congested that its
victims have barely started generating: they hold KV blocks from a finished
prefill but have almost no output to lose. Priority starves batch work at the
*prefill* stage, so it never reaches a state worth losing either. The
uncomfortable conclusion is that fairness and wasted work pull against each
other — the policy that refuses to starve anyone is precisely the policy that
guarantees its victims have something valuable to lose.

---

## Findings from calibration

Three results that came out of getting the experiment to mean anything.

**1. Prefill work destroys decode throughput.** At light load a step averages
~8ms and decode delivers ~4,000 tokens/s. Under prefill pressure a step
averages ~43ms and decode collapses to ~744 tokens/s — **5× less generation on
identical hardware**. This is the single most transferable result here, and it
is why the clock charges `0.006 + prefill_tokens / 20,000` per step rather than
a flat duration.

**2. KV must cover the decode concurrency you allow, or `decode_slots` is a
lie.** The first run sized the pool at 2,048 blocks while 32 decode slots ×
~2,100 tokens per request needs ~4,200. A pool that cannot hold one full batch
preempts continuously: **348–423 preempts for 37–118 admitted requests**, each
request evicted 4 to 10 times. Almost nothing completed and the policy
comparison drowned in thrash.

**3. Capacity is ~6.15 req/s, and the useful regime is a narrow band above it.**

```
average request:  1,750 prefill tokens,  400 decode tokens
prefill          1,750 / 20,000             = 0.0875 s
decode           (400 / 32 slots) × 0.006   = 0.0750 s
                                      total ≈ 0.1625 s  →  ~6.15 req/s
```

A sweep at 4, 5, 6 and 7 req/s confirmed it. **Below 6 req/s the server absorbs
everything** — zero preemptions, near-zero rejections, and all three policies
score identically. **At 8 req/s (5× capacity) 77% of requests are rejected**,
so "completed/s" measures admission rather than scheduling. 7 req/s (~1.14×
capacity) is the only rate where all six required metrics are non-zero and
policy actually decides outcomes.

---

## Caveats

1. **One trace, one seed.** The per-preempt ordering held at 8 req/s as well
   (FCFS 2.7, priority 8.8, DRR 20.8 tokens per preempt), so the effect is not
   an artefact of a single load point — but a seed sweep would make it solid.
2. **p99 over small samples.** Sample counts travel with every percentile in
   the tool's output for this reason; FCFS's batch p99 rests on 41 requests.
3. **One worker.** Routing effects belong to Part 3 and are absent here.
4. **Shared prefixes only affect admission**, lowering `required_kv_blocks` for
   agent requests. The scheduler still prefills all 4,000 tokens; prefix-aware
   caching is Part 3's job.
5. **No client disconnects.** The abort path is tested but unexercised, so the
   policies differ only by scheduling.

---

## Running the experiment

Everything here is simulated: no GPU, no vLLM replicas, nothing to start
first. The whole run takes about a second.

```bash
cd class7
make part2
```

That regenerates the trace and runs all three policies over it, printing the
comparison table above.

| Command | What it does |
|---|---|
| `make part2` | The headline experiment: regenerate the trace, run all three policies |
| `make part2-overload` | The 8 req/s congestion-collapse sensitivity case |
| `make trace` | Regenerate `traces/mixed.jsonl` only |
| `make test` | Full suite, 278 tests, including the harness tests |

### Raw equivalents

Use these to vary the experiment. `make` exports `PYTHONPATH` to the project
root, so run these from `class7/` for the `gateway` package to import.

```bash
.venv/bin/python traces/gen_mixed.py --rate 7 --out traces/mixed.jsonl --seed 7
.venv/bin/python serve.py --trace traces/mixed.jsonl --policy all
```

| Flag | Applies to | Default | Notes |
|---|---|---|---|
| `--rate` | `gen_mixed.py` | 8.0 | Arrivals per second — see the warning below |
| `--seed` | `gen_mixed.py` | 7 | Same seed, same trace |
| `--out` | `gen_mixed.py` | `traces/mixed.jsonl` | Where to write |
| `--trace` | `serve.py` | `traces/mixed.jsonl` | Which trace to replay |
| `--policy` | `serve.py` | `all` | Or one of `fcfs`, `priority`, `drr` |
| `--seconds` | both | 60.0 | Simulated horizon |

**The useful band is narrow.** Below ~6 req/s the server absorbs everything:
no preemptions, almost no rejections, and all three policies score identically.
Above ~8 req/s admission sheds three quarters of the traffic and the table
measures admission rather than scheduling. Editing `DECODE_SLOTS` or
`KV_TOTAL_BLOCKS` in `serve.py` moves the band — **if you raise the decode
slots, raise the KV pool with them**, or the run thrashes (finding 2 above).

---

## Files

| Path | Role |
|---|---|
| `traces/gen_mixed.py` | Trace generator (seeded, exact class proportions) |
| `traces/mixed.jsonl` | The 7 req/s headline workload |
| `traces/mixed_overload.jsonl` | 8 req/s congestion-collapse sensitivity case |
| `serve.py` | Clock, worker, admission bridge, metrics |
| `serve.md` | How the harness works and why it is calibrated this way |
| `docs/sched.md`, `docs/admit.md` | Gate-by-gate reference for the two modules under test |
