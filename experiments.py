#!/usr/bin/env python3
"""Run Part 2 and the three Part 3 routing experiments, and write results/.

Everything here is a driver: the simulation lives in `serve.py`, the routing
rules in `router.py`, and the workloads in `traces/`. This file only decides
which runs to make, writes them to `results/` in a form the report can be
written from without re-running anything, and draws `plots/soak.png`.

Every run is seeded. The routing seed is `serve.DEFAULT_SEED`; the trace seeds
are the ones `traces/gen_router.py` and `traces/gen_mixed.py` were invoked
with, and both are recorded in `results/summary.json` beside the numbers.

    python experiments.py            # everything
    python experiments.py part2 t2   # just those
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from router import STRATEGIES
from serve import (
    DEFAULT_SEED,
    RunMetrics,
    StaleSpec,
    as_dict,
    load_trace,
    run,
)


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

# Part 3 holds the scheduler fixed so routing is the only variable. DRR is the
# choice because Part 2 measured it as the best all-rounder, and because a
# short queue is what makes a routing difference visible rather than drowned
# in scheduling backlog. Every experiment is also run under the other two
# policies, and those runs are kept in the JSON so the conclusions can be
# checked against a scheduler that is not this one.
HEADLINE_POLICY = "drr"
POLICIES = ("fcfs", "priority", "drr")

# Seeds the traces were generated with, for the provenance block.
TRACE_SEEDS = {
    "traces/mixed.jsonl": {"generator": "traces/gen_mixed.py", "seed": 7, "rate": 7.0},
    "traces/t1_unique_prefix.jsonl": {"generator": "traces/gen_router.py t1", "seed": 7, "rate": 14.0},
    "traces/t2_shared_prefix.jsonl": {"generator": "traces/gen_router.py t2", "seed": 107, "rate": 10.0},
    "traces/t3_stale.jsonl": {"generator": "traces/gen_router.py t3", "seed": 207, "rate": 4.5},
}

SHARED_PREFIX_HASH = "shared-prefix-a"
SHARED_PREFIX_TOKENS = 3_500

# The table PART2_FINDINGS.md publishes. Part 2 is re-run every time so that a
# change to the routing seam that quietly moved these numbers is caught here
# rather than in the write-up.
PART2_EXPECTED = {
    "fcfs": {"completed": 212, "preempts": 114, "wasted": 158, "rejected": 195},
    "priority": {"completed": 342, "preempts": 17, "wasted": 125, "rejected": 70},
    "drr": {"completed": 346, "preempts": 10, "wasted": 524, "rejected": 69},
}

CSV_COLUMNS = [
    "experiment",
    "arm",
    "policy",
    "strategy",
    "workers",
    "prefix_cache",
    "stale",
    "seed",
    "seconds",
    "arrived",
    "admitted",
    "completed",
    "completed_per_s",
    "shed_pct",
    "shed_by_router",
    "ttft_p50_s",
    "ttft_p99_s",
    "routing_imbalance",
    "kv_block_seconds",
    "prefix_block_seconds",
    "peak_blocks",
    "preempts",
    "wasted_decode_tokens",
    "prefill_tokens",
    "cached_prefill_tokens",
    "in_flight_at_horizon",
]


def record(run_metrics: RunMetrics, experiment: str, arm: str) -> dict:
    """Flatten one run and label it with the experiment arm it belongs to."""
    row = as_dict(run_metrics)
    row["experiment"] = experiment
    row["arm"] = arm
    row["stale"] = (
        None
        if run_metrics.stale is None
        else f"{run_metrics.stale.worker}@{run_metrics.stale.lag_s:g}s"
        + ("(hidden)" if run_metrics.stale.hidden else "")
    )
    return row


def write(name: str, rows: list[dict]) -> None:
    """Write one experiment's rows as JSON and as a flat CSV.

    The timelines are stripped here and kept in `results/timelines.json`: they
    are the only part of a result that is long, and nothing but the plot reads
    them.
    """
    RESULTS.mkdir(exist_ok=True)
    slim = [{k: v for k, v in row.items() if k != "timeline"} for row in rows]
    (RESULTS / f"{name}.json").write_text(json.dumps(slim, indent=2))

    with (RESULTS / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(slim)


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


# --- Part 2 ----------------------------------------------------------------


def part2() -> tuple[list[dict], dict]:
    """Re-run the scheduler comparison and check it against the write-up."""
    trace = load_trace(ROOT / "traces/mixed.jsonl")
    rows = []
    reproduced = {}
    for policy in POLICIES:
        metrics = run(trace, policy, 60.0, workers=1, seed=DEFAULT_SEED)
        rows.append(record(metrics, "part2", policy))
        expected = PART2_EXPECTED[policy]
        reproduced[policy] = {
            "completed": [metrics.completed, expected["completed"]],
            "preempts": [metrics.preempts, expected["preempts"]],
            "wasted_decode_tokens": [metrics.wasted_decode_tokens, expected["wasted"]],
            "rejected": [metrics.rejected, expected["rejected"]],
            "matches_published": (
                metrics.completed == expected["completed"]
                and metrics.preempts == expected["preempts"]
                and metrics.wasted_decode_tokens == expected["wasted"]
                and metrics.rejected == expected["rejected"]
            ),
        }

    write("part2", rows)
    return rows, {
        "all_match": all(v["matches_published"] for v in reproduced.values()),
        "per_policy": reproduced,
    }


# --- T1: does p2c balance better than random? ------------------------------


def t1() -> tuple[list[dict], dict]:
    """Four strategies on a trace with nothing to reuse but load to read.

    Two workers is the fleet the assignment asks for, and it is also the fleet
    where power-of-two-choices cannot differ from least-loaded: sampling two
    of two workers is scanning all of them. The sweep re-runs the same
    workload shape on four and eight workers, with the offered rate scaled to
    keep load per worker constant, because that is the only place where the
    p2c claim -- nearly all of the benefit of scanning, at the cost of two
    probes -- can be told apart from least-loaded at all.
    """
    trace = load_trace(ROOT / "traces/t1_unique_prefix.jsonl")
    rows = []
    for policy in POLICIES:
        for strategy in STRATEGIES:
            metrics = run(
                trace,
                policy,
                90.0,
                workers=2,
                strategy=strategy,
                seed=DEFAULT_SEED,
            )
            rows.append(record(metrics, "t1", f"{policy}/{strategy}/2w"))

    sweep = []
    for workers, path in ((4, "traces/t1_unique_prefix_4w.jsonl"), (8, "traces/t1_unique_prefix_8w.jsonl")):
        fleet_trace = load_trace(ROOT / path)
        for strategy in STRATEGIES:
            metrics = run(
                fleet_trace,
                HEADLINE_POLICY,
                90.0,
                workers=workers,
                strategy=strategy,
                seed=DEFAULT_SEED,
            )
            row = record(metrics, "t1_fleet_sweep", f"{HEADLINE_POLICY}/{strategy}/{workers}w")
            rows.append(row)
            sweep.append(row)

    write("t1_balance", rows)

    headline = {
        row["strategy"]: row
        for row in rows
        if row["arm"].startswith(f"{HEADLINE_POLICY}/") and row["workers"] == 2
    }
    return rows, {
        "policy": HEADLINE_POLICY,
        "workers": 2,
        "p99_ttft_s": {k: v["ttft_p99_s"] for k, v in headline.items()},
        "routing_imbalance": {k: v["routing_imbalance"] for k, v in headline.items()},
        "completed_per_s": {k: v["completed_per_s"] for k, v in headline.items()},
        "shed_pct": {k: v["shed_pct"] for k, v in headline.items()},
        "p2c_vs_random_p99_ratio": ratio(
            headline["p2c"]["ttft_p99_s"], headline["random"]["ttft_p99_s"]
        ),
        "p2c_equals_least_loaded_on_two_workers": (
            headline["p2c"]["completed"] == headline["least_loaded"]["completed"]
        ),
        "fleet_sweep": {
            f"{row['workers']}w/{row['strategy']}": {
                "ttft_p99_s": row["ttft_p99_s"],
                "routing_imbalance": row["routing_imbalance"],
                "completed_per_s": row["completed_per_s"],
            }
            for row in sweep
        },
    }


# --- T2: does prefix-aware routing save KV? --------------------------------


def _analytic_kv(path: Path) -> dict:
    """The prefix-unaware arithmetic the assignment's target assumes.

    Counted straight off the trace: a router that cannot see prefixes pays for
    every sharer's whole prompt, while one that routes sharers together pays
    for the prefix once. Nothing here is measured; it is the ceiling the
    harness runs are compared against.
    """
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    sharers = [r for r in records if r["prefix_hash"] == SHARED_PREFIX_HASH]
    others = [r for r in records if r["prefix_hash"] != SHARED_PREFIX_HASH]

    full = sum(r["prompt_tokens"] + r["max_new_tokens"] for r in sharers)
    tail = sum(
        r["prompt_tokens"] - SHARED_PREFIX_TOKENS + r["max_new_tokens"] for r in sharers
    )
    rest = sum(r["prompt_tokens"] + r["max_new_tokens"] for r in others)
    unaware = full + rest
    aware = SHARED_PREFIX_TOKENS + tail + rest
    return {
        "sharers": len(sharers),
        "others": len(others),
        "kv_tokens_prefix_unaware": unaware,
        "kv_tokens_prefix_aware": aware,
        "ratio": ratio(aware, unaware),
    }


def t2() -> tuple[list[dict], dict]:
    """Two framings of the same question, because they do not agree.

    Framing A is the assignment's: the baseline router is prefix-unaware, so
    every sharer materialises its whole prompt and the prefix is paid for once
    per request. Framing B gives every strategy the same prefix cache and
    changes only the routing rule, which is the comparison that isolates
    routing -- and which cannot reach the assignment's target, because a
    two-worker fleet holding the prefix twice against once has a floor of 0.5
    on the prefix itself and far higher than that on total KV.
    """
    trace = load_trace(ROOT / "traces/t2_shared_prefix.jsonl")
    rows = []
    for policy in POLICIES:
        for strategy in STRATEGIES:
            for cache in (True, False):
                metrics = run(
                    trace,
                    policy,
                    100.0,
                    workers=2,
                    strategy=strategy,
                    prefix_cache=cache,
                    seed=DEFAULT_SEED,
                )
                arm = f"{policy}/{strategy}/{'cache' if cache else 'nocache'}"
                rows.append(record(metrics, "t2", arm))

    write("t2_prefix", rows)

    def find(strategy: str, cache: bool) -> dict:
        want = f"{HEADLINE_POLICY}/{strategy}/{'cache' if cache else 'nocache'}"
        return next(row for row in rows if row["arm"] == want)

    aware = find("prefix_then_load", True)
    unaware = find("least_loaded", False)
    same_mechanism = find("least_loaded", True)

    return rows, {
        "policy": HEADLINE_POLICY,
        "workers": 2,
        "framing_a_assignment": {
            "baseline": "least_loaded, prefix cache off (each sharer materialises its full prompt)",
            "treatment": "prefix_then_load, prefix cache on",
            "kv_block_seconds": {
                "baseline": unaware["kv_block_seconds"],
                "treatment": aware["kv_block_seconds"],
                "ratio": ratio(aware["kv_block_seconds"], unaware["kv_block_seconds"]),
            },
            "peak_blocks": {
                "baseline": unaware["peak_blocks"],
                "treatment": aware["peak_blocks"],
                "ratio": ratio(aware["peak_blocks"], unaware["peak_blocks"]),
            },
            "kv_block_seconds_per_completed": {
                "baseline": ratio(unaware["kv_block_seconds"], unaware["completed"]),
                "treatment": ratio(aware["kv_block_seconds"], aware["completed"]),
                "ratio": ratio(
                    ratio(aware["kv_block_seconds"], aware["completed"]),
                    ratio(unaware["kv_block_seconds"], unaware["completed"]),
                ),
            },
            "target": 0.4,
        },
        "framing_b_same_mechanism": {
            "baseline": "least_loaded, prefix cache on",
            "treatment": "prefix_then_load, prefix cache on",
            "kv_block_seconds": {
                "baseline": same_mechanism["kv_block_seconds"],
                "treatment": aware["kv_block_seconds"],
                "ratio": ratio(
                    aware["kv_block_seconds"], same_mechanism["kv_block_seconds"]
                ),
            },
            "prefix_block_seconds": {
                "baseline": same_mechanism["prefix_block_seconds"],
                "treatment": aware["prefix_block_seconds"],
                "ratio": ratio(
                    aware["prefix_block_seconds"], same_mechanism["prefix_block_seconds"]
                ),
            },
            "prefill_tokens": {
                "baseline": same_mechanism["prefill_tokens"],
                "treatment": aware["prefill_tokens"],
                "ratio": ratio(aware["prefill_tokens"], same_mechanism["prefill_tokens"]),
            },
            "floor_for_two_workers": 0.5,
        },
        "analytic_prefix_unaware": _analytic_kv(ROOT / "traces/t2_shared_prefix.jsonl"),
        "by_strategy_cache_on": {
            row["strategy"]: {
                "kv_block_seconds": row["kv_block_seconds"],
                "prefix_block_seconds": row["prefix_block_seconds"],
                "peak_blocks": row["peak_blocks"],
                "prefill_tokens": row["prefill_tokens"],
                "cached_prefill_tokens": row["cached_prefill_tokens"],
                "ttft_p99_s": row["ttft_p99_s"],
                "completed": row["completed"],
            }
            for row in rows
            if row["arm"].startswith(f"{HEADLINE_POLICY}/") and row["prefix_cache"]
        },
    }


# --- T3: stale telemetry ---------------------------------------------------


def t3() -> tuple[list[dict], dict]:
    """Worker w1's load telemetry arrives 15 s late, claiming it is empty.

    Three arms per strategy. `fresh` is the control. `stale_hidden` reports
    the 15-second-old numbers as current, which is the only way to see what a
    router does without H6 -- the rule is unconditional inside pick(), so the
    experiment lies about the age rather than disabling the rule.
    `stale_honest` reports the age truthfully, and H6 then classes w1 as
    unknown rather than idle.
    """
    trace = load_trace(ROOT / "traces/t3_stale.jsonl")
    arms = {
        "fresh": None,
        "stale_hidden": StaleSpec("w1", 15.0, hidden=True),
        "stale_honest": StaleSpec("w1", 15.0, hidden=False),
    }
    rows = []
    for policy in POLICIES:
        for strategy in STRATEGIES:
            for arm, stale in arms.items():
                metrics = run(
                    trace,
                    policy,
                    120.0,
                    workers=2,
                    strategy=strategy,
                    stale=stale,
                    seed=DEFAULT_SEED,
                )
                rows.append(record(metrics, "t3", f"{policy}/{strategy}/{arm}"))

    # H6 excludes the dark worker, and on a two-worker fleet that is half the
    # capacity. The same workload shape on eight workers, with the rate scaled
    # to hold load per worker constant, says whether the cost of the rule is
    # the rule or the fleet size.
    sweep_trace = load_trace(ROOT / "traces/t3_stale_8w.jsonl")
    sweep = []
    for strategy in STRATEGIES:
        for arm, stale in arms.items():
            metrics = run(
                sweep_trace,
                HEADLINE_POLICY,
                120.0,
                workers=8,
                strategy=strategy,
                stale=stale,
                seed=DEFAULT_SEED,
            )
            row = record(metrics, "t3_fleet_sweep", f"{HEADLINE_POLICY}/{strategy}/{arm}/8w")
            rows.append(row)
            sweep.append(row)

    write("t3_stale", rows)

    headline = {
        (row["strategy"], row["arm"].rsplit("/", 1)[1]): row
        for row in rows
        if row["arm"].startswith(f"{HEADLINE_POLICY}/") and row["experiment"] == "t3"
    }
    return rows, {
        "policy": HEADLINE_POLICY,
        "workers": 2,
        "stale_worker": "w1",
        "lag_s": 15.0,
        "by_strategy": {
            strategy: {
                arm: {
                    "ttft_p99_s": headline[(strategy, arm)]["ttft_p99_s"],
                    "kv_block_seconds": headline[(strategy, arm)]["kv_block_seconds"],
                    "peak_blocks": headline[(strategy, arm)]["peak_blocks"],
                    "shed_pct": headline[(strategy, arm)]["shed_pct"],
                    "completed": headline[(strategy, arm)]["completed"],
                    "share_to_w1": ratio(
                        headline[(strategy, arm)]["routed_by_worker"].get("w1", 0),
                        sum(headline[(strategy, arm)]["routed_by_worker"].values()),
                    ),
                }
                for arm in arms
            }
            for strategy in STRATEGIES
        },
        "fleet_sweep_8w": {
            f"{row['strategy']}/{row['arm'].split('/')[2]}": {
                "ttft_p99_s": row["ttft_p99_s"],
                "shed_pct": row["shed_pct"],
                "completed": row["completed"],
                "kv_block_seconds": row["kv_block_seconds"],
                "share_to_w1": ratio(
                    row["routed_by_worker"].get("w1", 0),
                    sum(row["routed_by_worker"].values()),
                ),
            }
            for row in sweep
        },
    }


# --- The soak plot ---------------------------------------------------------


def _smooth(values: list[float], window: int) -> list[float]:
    """Centred rolling mean, NaNs skipped, so an empty second is not a zero."""
    out = []
    for index in range(len(values)):
        lo = max(0, index - window // 2)
        chunk = [v for v in values[lo : index + window // 2 + 1] if v == v]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def soak(t1_rows: list[dict], t3_rows: list[dict]) -> list[str]:
    """Draw plots/soak.png from the timelines the runs already recorded."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    soak_rows = [
        row
        for row in t1_rows
        if row["experiment"] == "t1" and row["arm"].startswith(f"{HEADLINE_POLICY}/")
    ]
    stale_rows = [
        row
        for row in t3_rows
        if row["arm"].startswith(f"{HEADLINE_POLICY}/")
        and row["arm"].endswith(("stale_hidden", "stale_honest", "fresh"))
        and row["strategy"] == "least_loaded"
    ]

    PLOTS.mkdir(exist_ok=True)
    written = []

    # Per-second counts at 14 req/s are mostly arrival noise. A five-second
    # rolling mean leaves the drift, which is what a soak is read for.
    window = 5
    styles = {
        "random": ("-", "#c1440e", 1.6),
        "least_loaded": ("--", "#1f6f8b", 1.6),
        "p2c": (":", "#2a9d8f", 2.2),
        "prefix_then_load": ("-.", "#6a4c93", 1.6),
    }

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    base = soak_rows[0]["timeline"][:-1]
    times = [point["t"] for point in base]
    axes[0].fill_between(
        times,
        _smooth([p["arrived"] for p in base], window),
        alpha=0.18,
        color="0.45",
        label="offered (arrivals/s)",
    )
    for row in soak_rows:
        line = row["timeline"][:-1]  # The final second is a partial bucket.
        style, colour, width = styles[row["strategy"]]
        shared = dict(linestyle=style, color=colour, linewidth=width, label=row["strategy"])
        axes[0].plot([p["t"] for p in line], _smooth([p["completed"] for p in line], window), **shared)
        axes[1].plot([p["t"] for p in line], _smooth([p["ttft_p99_s"] for p in line], window), **shared)
        axes[2].plot([p["t"] for p in line], _smooth([p["kv_blocks"] for p in line], window), **shared)

    axes[0].set_ylabel("requests / s")
    axes[0].set_title(
        "T1 soak -- 1,296 requests over 90 s at 14 req/s, 2 workers, DRR, seed 7\n"
        "offered load against completions (5 s rolling mean)",
        fontsize=10,
    )
    axes[1].set_ylabel("p99 TTFT (s)")
    axes[1].set_yscale("log")
    axes[1].set_title(
        "latency over the soak -- random drifts away after ~55 s and does not come back",
        fontsize=10,
    )
    axes[2].set_ylabel("KV blocks held (fleet)")
    axes[2].set_xlabel("simulated seconds")
    axes[2].set_title("KV occupancy over the soak", fontsize=10)
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.legend(fontsize=8, ncol=2, loc="upper left")
    fig.text(
        0.01,
        0.005,
        "On a two-worker fleet p2c samples both workers, so p2c, least_loaded and "
        "prefix_then_load are the same router and their curves coincide.",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(PLOTS / "soak.png", dpi=140)
    plt.close(fig)
    written.append(str(PLOTS / "soak.png"))

    if stale_rows:
        arm_styles = {
            "fresh": ("-", "#1f6f8b"),
            "stale_hidden": ("-", "#c1440e"),
            "stale_honest": ("--", "#6a4c93"),
        }
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for row in stale_rows:
            line = row["timeline"][:-1]
            arm = row["arm"].rsplit("/", 1)[1]
            style, colour = arm_styles[arm]
            axes[0].plot(
                [p["t"] for p in line],
                _smooth([p["ttft_p99_s"] for p in line], 5),
                linewidth=1.5,
                linestyle=style,
                color=colour,
                label=arm,
            )
            axes[1].plot(
                [p["t"] for p in line],
                _smooth([p["rejected"] for p in line], 5),
                linewidth=1.5,
                linestyle=style,
                color=colour,
                label=arm,
            )
        axes[0].set_ylabel("p99 TTFT (s), per second")
        axes[0].set_yscale("log")
        axes[0].set_title(
            "T3: least_loaded with w1's telemetry 15 s late (120 s, 4.5 req/s, seed 7)",
            fontsize=10,
        )
        axes[1].set_ylabel("rejections / s")
        axes[1].set_xlabel("simulated seconds")
        for axis in axes:
            axis.grid(alpha=0.25, linewidth=0.5)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS / "t3_stale.png", dpi=140)
        plt.close(fig)
        written.append(str(PLOTS / "t3_stale.png"))

    return written


def timelines(rows: list[dict]) -> None:
    """Keep the per-second series the plot is drawn from, so it can be redrawn."""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "timelines.json").write_text(
        json.dumps(
            {
                f"{row['experiment']}:{row['arm']}": row["timeline"]
                for row in rows
                if row["arm"].startswith(f"{HEADLINE_POLICY}/")
            },
            indent=2,
        )
    )


def main() -> None:
    wanted = set(sys.argv[1:]) or {"part2", "t1", "t2", "t3"}
    summary: dict = {
        "seeds": {
            "router_seed": DEFAULT_SEED,
            "note": "router.ROUTER_SEED is a module default; every run here passes seed explicitly",
            "traces": TRACE_SEEDS,
        },
        "headline_policy": HEADLINE_POLICY,
        "strategies": list(STRATEGIES),
    }
    kept: list[dict] = []

    if "part2" in wanted:
        rows, summary["part2"] = part2()
        kept += rows
        print("part2:", "reproduced" if summary["part2"]["all_match"] else "DIVERGED")

    t1_rows: list[dict] = []
    if "t1" in wanted:
        t1_rows, summary["t1"] = t1()
        kept += t1_rows
        print("t1 p99 TTFT:", summary["t1"]["p99_ttft_s"])

    if "t2" in wanted:
        _rows, summary["t2"] = t2()
        kept += _rows
        print(
            "t2 ratio (assignment framing):",
            round(summary["t2"]["framing_a_assignment"]["kv_block_seconds"]["ratio"], 3),
        )

    t3_rows: list[dict] = []
    if "t3" in wanted:
        t3_rows, summary["t3"] = t3()
        kept += t3_rows

    if t1_rows and t3_rows:
        summary["plots"] = soak(t1_rows, t3_rows)
        timelines(t1_rows + t3_rows)
        print("plots:", summary.get("plots"))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RESULTS}/summary.json")


if __name__ == "__main__":
    main()
