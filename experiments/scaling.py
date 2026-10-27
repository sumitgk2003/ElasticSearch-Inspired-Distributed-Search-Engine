"""Deliverable 3: latency and throughput for 1, 2, 4 and 8 shards.

Runs a closed-loop workload against clusters of increasing size and plots
mean / p50 / p95 / p99 latency and queries per second.  An injected network
delay with exponential jitter on every shard call approximates separate
machines; because the coordinator waits for the slowest shard, the tail grows
with the shard count (expected max of S exponentials ~ jitter * H_S).

    python -m experiments.scaling --shards 1 2 4 8 --duration 20 --jitter-ms 2
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dse.cluster import Cluster
from dse.loadgen import run_load, summarize, to_rows

from .common import add_corpus_args, load_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_corpus_args(ap)
    ap.add_argument("--shards", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--delay-ms", type=float, default=1.0)
    ap.add_argument("--jitter-ms", type=float, default=2.0)
    ap.add_argument("--stats-modes", nargs="+", default=["dfs", "local"])
    ap.add_argument("--coordinators", type=int, default=1,
                    help="coordinator processes; >1 removes the single-coordinator CPU ceiling")
    a = ap.parse_args()
    docs, queries = load_corpus(a)
    out = Path(a.out)

    summaries, per_request = [], []
    for S in a.shards:
        with Cluster(Path(a.work) / f"scaling_s{S}", num_shards=S, replicas=a.replicas,
                     delay_ms=a.delay_ms, jitter_ms=a.jitter_ms, timeout_s=10,
                     coordinators=a.coordinators) as c:
            c.build(docs)
            c.start()
            for mode in a.stats_modes:
                recs = asyncio.run(run_load(c.urls, queries, a.concurrency, a.duration, a.warmup,
                                            body_extra={"stats_mode": mode}))
                s = summarize(recs, a.duration)
                s.update(shards=S, stats_mode=mode, coordinators=a.coordinators, docs=len(docs))
                summaries.append(s)
                per_request += [dict(r, shards=S, stats_mode=mode) for r in to_rows(recs)]
                print(f"S={S} {mode:5s} qps={s['qps']:.1f} mean={s['mean_ms']:.1f} "
                      f"p95={s['p95_ms']:.1f} p99={s['p99_ms']:.1f} ms")

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "scaling_summary.csv", index=False)
    pd.DataFrame(per_request).to_csv(out / "scaling_requests.csv", index=False)
    plot(summary, out / "scaling.png")


def plot(summary: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for mode, g in summary.groupby("stats_mode"):
        g = g.sort_values("shards")
        for col, style in (("mean_ms", "-"), ("p95_ms", "--"), ("p99_ms", ":")):
            ax1.plot(g["shards"], g[col], style, marker="o", label=f"{mode} {col[:-3]}")
        ax2.plot(g["shards"], g["qps"], marker="o", label=mode)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(summary["shards"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("shards")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax1.set_ylabel("latency (ms)")
    ax1.set_title("Query latency vs. shard count")
    ax2.set_ylabel("queries / second")
    ax2.set_title("Throughput vs. shard count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
