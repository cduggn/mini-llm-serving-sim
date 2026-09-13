#!/usr/bin/env python3
"""Generate traces/mixed.jsonl: the Part 2 workload.

Three classes share one Poisson arrival process, with class proportions exact
rather than sampled, so every run sees the same mix. Arrival times are the
only randomness that matters to the comparison: the same trace drives FCFS,
priority and DRR, so the scheduling policy is the only variable between runs.

Fields follow the assignment's request shape. Two of them are choices the
assignment does not make: `tenant`, without which DRR has no fairness unit,
and `timeout_s`, which admission measures its queue-wait estimate against.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# Prompt and output ranges are inclusive, in tokens.
CLASSES = {
    "interactive": {
        "share": 0.70,
        "priority": 0,
        "prompt": (200, 800),
        "output": (64, 256),
        "timeout_s": 5.0,
        "shared_prefix": False,
    },
    "batch": {
        "share": 0.20,
        "priority": 20,
        "prompt": (2_000, 8_000),
        "output": (512, 2_048),
        "timeout_s": 60.0,
        "shared_prefix": False,
    },
    # The assignment fixes the prompt at 4,000 with 3,500 shared, but names
    # neither an output range nor a priority. Agents sit between the two other
    # classes: background work that a person is still ultimately waiting on.
    "agents": {
        "share": 0.10,
        "priority": 10,
        "prompt": (4_000, 4_000),
        "output": (128, 512),
        "timeout_s": 30.0,
        "shared_prefix": True,
    },
}

AGENT_PREFIX_HASH = "agent-prefix-a"  # One prefix shared by every agent request.


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


def class_deck(count: int, rng: random.Random) -> list[str]:
    """Return exactly proportioned class labels, shuffled.

    Sampling each arrival independently would let the 10% agent share drift by
    several requests in a short trace, changing how much prefix reuse exists.
    """
    deck: list[str] = []
    for name, spec in list(CLASSES.items())[:-1]:
        deck += [name] * round(spec["share"] * count)

    last = list(CLASSES)[-1]
    deck += [last] * (count - len(deck))
    rng.shuffle(deck)
    return deck


def build(rate: float, seconds: float, seed: int) -> list[dict]:
    """Build one trace: arrivals, classes, then per-request token counts."""
    rng = random.Random(seed)
    times = arrival_times(rate, seconds, rng)
    deck = class_deck(len(times), rng)

    trace = []
    for index, (arrival, name) in enumerate(zip(times, deck)):
        spec = CLASSES[name]
        trace.append(
            {
                "id": f"{name[:3]}-{index:05d}",
                "arrival_t": round(arrival, 6),
                "priority": spec["priority"],
                "prompt_tokens": rng.randint(*spec["prompt"]),
                "max_new_tokens": rng.randint(*spec["output"]),
                "prefix_hash": AGENT_PREFIX_HASH if spec["shared_prefix"] else None,
                "timeout_s": spec["timeout_s"],
                "tenant": name,
            }
        )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="traces/mixed.jsonl")
    parser.add_argument("--rate", type=float, default=8.0, help="arrivals per second")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    trace = build(args.rate, args.seconds, args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in trace:
            handle.write(json.dumps(record) + "\n")

    counts: dict[str, int] = {}
    for record in trace:
        counts[record["tenant"]] = counts.get(record["tenant"], 0) + 1
    shares = ", ".join(f"{k} {v} ({v / len(trace):.0%})" for k, v in counts.items())
    print(f"{len(trace)} requests over {args.seconds:g}s -> {path}")
    print(f"  {shares}")


if __name__ == "__main__":
    main()
