#!/usr/bin/env python3
"""Generate the three Part 3 router traces.

Part 2 held the router fixed and varied the scheduler. Part 3 inverts that:
one scheduler, one pair of workers, and the routing rule as the only variable.
Each trace here exists to make exactly one routing question answerable, and is
sized against `serve.py`'s simulated hardware rather than picked by feel:

    STEP_BASE_S            0.006 s   a decode-only pass, any batch size
    PREFILL_TOKENS_PER_S   20,000    prefill time is linear in tokens
    DECODE_SLOTS           32        concurrent sequences per worker
    KV_TOTAL_BLOCKS        8,192     131,072 KV tokens per worker
    SHARED_PREFIX_TOKENS   3,500     credited once a prefix is resident

**t1_unique_prefix** — random vs power-of-two-choices. Every request carries a
distinct `prefix_hash`, so prefix affinity has nothing to bite on and the two
routers differ only in how they read load. Two classes at wildly different
cost (a 500-token prompt against a 6,000-token one) give the heavy tail that
makes random routing hurt: a random router can hand one worker four long
prompts in a row while the other idles, and p99 TTFT records it. Both classes
sit at priority 0 on purpose — under a priority scheduler a starved class
would inflate queue depth for reasons that have nothing to do with routing,
and queue depth is the signal the router reads.

**t2_shared_prefix** — least-loaded vs prefix-then-load. 40% of requests carry
one prefix of 3,500 tokens inside a 4,000-token prompt, which is the same
shape Part 2's agent class used and the same number `serve.py` credits back.
The other 60% are small and unique, so there is nothing else to reuse. See
`kv_arithmetic()` for why the assignment's target is reachable here.

**t3_stale** — least-loaded fed 15-second-old telemetry. The workload is
deliberately ordinary; the staleness lives in the harness, not the file. What
the harness needs from the trace, and what this generator guarantees:

  * A horizon of 120 s, eight full staleness windows, so the effect is not one
    unlucky burst. Poisson arrivals still supply the burstiness, but eight
    windows average the arrival noise out.
  * Residency that straddles the staleness window. Outputs of 768-2,304
    tokens are 768-2,304 decode steps; at ~148 steps/s that is 5-16 s on the
    worker, mean ~10.6 s. If requests drained in well under 15 s the lag would
    be self-correcting and there would be nothing to show.
  * Headroom of exactly the right size. At 4.5 req/s split evenly each worker
    carries ~25 concurrent sequences against DECODE_SLOTS=32 (~79%). A router
    that believes B is empty for 15 s sends B roughly double that, ~59, which
    B cannot hold: the queue grows, KV pressure rises, admission starts
    shedding, while A drains toward ~12. A fresh-telemetry run of the same
    trace stays inside 32 on both workers, so the overload is attributable to
    the lag rather than to the offered load.
  * Prompts kept short (500-1,500) so prefill is only ~11% of wall time. The
    bottleneck is decode-slot occupancy, which is what a load-reporting
    telemetry field actually measures, so the stale field and the failure mode
    describe the same quantity.
  * One tenant and one priority, so neither DRR nor priority can be blamed for
    the imbalance.

Field names match `gen_mixed.py` exactly; `serve.TraceRequest.from_json`
parses every line.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SHARED_PREFIX_TOKENS = 3_500  # Mirrors serve.SHARED_PREFIX_TOKENS.
SHARED_PREFIX_HASH = "shared-prefix-a"  # The one prefix t2 reuses.

# Prompt and output ranges are inclusive, in tokens. `prefix` is "unique" for a
# per-request hash nothing can reuse, or "shared" for the one t2 prefix.
TRACES = {
    "t1": {
        "out": "traces/t1_unique_prefix.jsonl",
        "rate": 14.0,
        "seconds": 90.0,
        "seed_offset": 0,
        "classes": {
            "interactive": {
                "share": 0.85,
                "priority": 0,
                "prompt": (200, 800),
                "output": (64, 256),
                "timeout_s": 10.0,
                "prefix": "unique",
            },
            "heavy": {
                "share": 0.15,
                "priority": 0,
                "prompt": (4_000, 8_000),
                "output": (512, 1_024),
                "timeout_s": 60.0,
                "prefix": "unique",
            },
        },
    },
    "t2": {
        "out": "traces/t2_shared_prefix.jsonl",
        "rate": 10.0,
        "seconds": 100.0,
        "seed_offset": 100,
        "classes": {
            "interactive": {
                "share": 0.60,
                "priority": 0,
                "prompt": (200, 800),
                "output": (64, 256),
                "timeout_s": 10.0,
                "prefix": "unique",
            },
            # 3,500 shared + 500 unique, the Part 2 agent shape.
            "agents": {
                "share": 0.40,
                "priority": 10,
                "prompt": (4_000, 4_000),
                "output": (128, 256),
                "timeout_s": 30.0,
                "prefix": "shared",
            },
        },
    },
    "t3": {
        "out": "traces/t3_stale.jsonl",
        "rate": 4.5,
        "seconds": 120.0,
        "seed_offset": 200,
        "classes": {
            "steady": {
                "share": 1.00,
                "priority": 0,
                "prompt": (500, 1_500),
                "output": (768, 2_304),
                "timeout_s": 60.0,
                "prefix": "unique",
            },
        },
    },
}


def arrival_times(rate: float, seconds: float, rng: random.Random) -> list[float]:
    """Return Poisson arrival times within the horizon, in order.

    Gaps are exponential, which is what makes the queue bursty: requests do not
    arrive evenly, so a backlog can form even when the mean rate is servable.
    """
    times: list[float] = []
    clock = 0.0
    while True:
        clock += rng.expovariate(rate)
        if clock >= seconds:
            return times
        times.append(clock)


def class_deck(classes: dict, count: int, rng: random.Random) -> list[str]:
    """Return exactly proportioned class labels, shuffled.

    Sampling each arrival independently would let t2's 40% shared-prefix share
    drift by a dozen requests, and the KV result is computed against that share.
    """
    deck: list[str] = []
    for name, spec in list(classes.items())[:-1]:
        deck += [name] * round(spec["share"] * count)

    last = list(classes)[-1]
    deck += [last] * (count - len(deck))
    rng.shuffle(deck)
    return deck


def build(trace: str, rate: float, seconds: float, seed: int) -> list[dict]:
    """Build one trace: arrivals, classes, then per-request token counts."""
    classes = TRACES[trace]["classes"]
    rng = random.Random(seed + TRACES[trace]["seed_offset"])
    times = arrival_times(rate, seconds, rng)
    deck = class_deck(classes, len(times), rng)

    records = []
    for index, (arrival, name) in enumerate(zip(times, deck)):
        spec = classes[name]
        if spec["prefix"] == "shared":
            prefix_hash = SHARED_PREFIX_HASH
        else:
            prefix_hash = f"{trace}-uniq-{index:05d}"
        records.append(
            {
                "id": f"{name[:3]}-{index:05d}",
                "arrival_t": round(arrival, 6),
                "priority": spec["priority"],
                "prompt_tokens": rng.randint(*spec["prompt"]),
                "max_new_tokens": rng.randint(*spec["output"]),
                "prefix_hash": prefix_hash,
                "timeout_s": spec["timeout_s"],
                "tenant": name,
            }
        )
    return records


def kv_arithmetic(records: list[dict]) -> str:
    """Report t2's KV budget under both routers, in KV tokens.

    A prefix-unaware least-loaded router cannot count on a hit, so every
    sharer materialises its whole 4,000-token prompt. Routing sharers together
    materialises the 3,500-token prefix once and charges the rest only for
    their unique tail. The assignment's target is

        KV(prefix_then_load) < 0.4 x KV(least_loaded)

    which this trace clears with room to spare.
    """
    sharers = [r for r in records if r["prefix_hash"] == SHARED_PREFIX_HASH]
    others = [r for r in records if r["prefix_hash"] != SHARED_PREFIX_HASH]

    tail = sum(r["prompt_tokens"] - SHARED_PREFIX_TOKENS + r["max_new_tokens"]
               for r in sharers)
    full = sum(r["prompt_tokens"] + r["max_new_tokens"] for r in sharers)
    rest = sum(r["prompt_tokens"] + r["max_new_tokens"] for r in others)

    least_loaded = full + rest
    prefix_then_load = SHARED_PREFIX_TOKENS + tail + rest
    ratio = prefix_then_load / least_loaded
    return (
        f"  KV(least_loaded)     {least_loaded:>9,} tokens "
        f"({len(sharers)} sharers x full prompt + {len(others)} others)\n"
        f"  KV(prefix_then_load) {prefix_then_load:>9,} tokens "
        f"({SHARED_PREFIX_TOKENS:,} prefix once + tails + others)\n"
        f"  ratio {ratio:.3f}  target < 0.400  "
        f"{'PASS' if ratio < 0.4 else 'FAIL'}"
    )


def write(trace: str, out: str, rate: float, seconds: float, seed: int) -> None:
    """Write one trace and print the shape a reader needs to trust it."""
    records = build(trace, rate, seconds, seed)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    counts: dict[str, int] = {}
    for record in records:
        counts[record["tenant"]] = counts.get(record["tenant"], 0) + 1
    shares = ", ".join(f"{k} {v} ({v / len(records):.0%})" for k, v in counts.items())
    distinct = len({r["prefix_hash"] for r in records})
    prompt_tokens = sum(r["prompt_tokens"] for r in records)

    print(f"{len(records)} requests over {seconds:g}s -> {path}  (seed {seed + TRACES[trace]['seed_offset']})")
    print(f"  {shares}")
    print(f"  {distinct} distinct prefixes; {prompt_tokens / seconds:,.0f} prompt tokens/s offered")
    if trace == "t2":
        print(kv_arithmetic(records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "trace",
        choices=[*TRACES, "all"],
        help="which router trace to write",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--rate", type=float, default=None, help="arrivals per second")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    names = list(TRACES) if args.trace == "all" else [args.trace]
    if args.trace == "all" and args.out is not None:
        parser.error("--out names a single file; pick one trace")

    for name in names:
        spec = TRACES[name]
        write(
            name,
            args.out or spec["out"],
            args.rate if args.rate is not None else spec["rate"],
            args.seconds if args.seconds is not None else spec["seconds"],
            args.seed,
        )


if __name__ == "__main__":
    main()
