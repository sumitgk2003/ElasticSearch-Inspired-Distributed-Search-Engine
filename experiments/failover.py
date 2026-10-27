"""Deliverable 4: failure demonstration under a running workload.

Timeline (defaults):
  t = 5 s   SIGKILL one copy of shard 0     -> queries reroute to the replica
  t = 12 s  SIGKILL the last copy of shard 0 -> responses carry partial: true
            and missing_shards = [0]
Writes a per-request log, the coordinator's event log, and a timeline plot.

    python -m experiments.failover --shards 4 --duration 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dse.cluster import Cluster
from dse.loadgen import run_load, summarize, to_rows

from .common import add_corpus_args, load_corpus


async def scenario(c: Cluster, queries, a, log):
    shard = 0
    primary, replica = c.allocation[shard][0], c.allocation[shard][1]
    t0 = time.time()

    async def at(t, label, fn):
        await asyncio.sleep(t)
        await asyncio.to_thread(fn)
        log.append({"t": time.time() - t0 - a.warmup, "action": label})
        print(f"[{log[-1]['t']:5.1f}s] {label}")

    load = asyncio.create_task(run_load(c.url, queries, a.concurrency, a.duration, a.warmup))
    await asyncio.gather(
        at(a.warmup + a.kill_replica_at, f"kill copy {shard}@{primary} (primary)",
           lambda: c.kill_copy(shard, primary)),
        at(a.warmup + a.kill_all_at, f"kill copy {shard}@{replica} (last copy of shard {shard})",
           lambda: c.kill_copy(shard, replica)))
    return await load, t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_corpus_args(ap)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--kill-replica-at", type=float, default=5.0)
    ap.add_argument("--kill-all-at", type=float, default=12.0)
    ap.add_argument("--timeout", type=float, default=0.5, help="per shard-call timeout (s)")
    ap.add_argument("--heartbeat", type=float, default=0.25)
    a = ap.parse_args()
    docs, queries = load_corpus(a)
    out = Path(a.out)

    actions = []
    with Cluster(Path(a.work) / "failover", num_shards=a.shards, replicas=1,
                 timeout_s=a.timeout, heartbeat_s=a.heartbeat) as c:
        c.build(docs)
        c.start()
        print("allocation:", c.allocation)
        records, t0 = asyncio.run(scenario(c, queries, a, actions))
        sample = c.search(queries[0])
        events = c.events()
    for e in events:
        e["t"] = e["t"] - t0 - a.warmup

    df = pd.DataFrame(to_rows(records))
    df.to_csv(out / "failover_requests.csv", index=False)
    (out / "failover_events.json").write_text(json.dumps({"actions": actions, "coordinator_events": events,
                                                          "sample_response_after_failure": sample}, indent=2))
    phases = [("both copies up", 0, a.kill_replica_at), ("one copy down", a.kill_replica_at, a.kill_all_at),
              ("all copies down", a.kill_all_at, a.duration)]
    rows = []
    for name, lo, hi in phases:
        part = [r for r in records if lo <= r.t_start < hi]
        s = summarize(part, hi - lo)
        rows.append({"phase": name, **{k: s.get(k) for k in
                     ("requests", "qps", "error_rate", "partial_rate", "mean_ms", "p99_ms")},
                     "requests_with_retry": sum(r.retries > 0 for r in part)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "failover_summary.csv", index=False)
    print(table.to_string(index=False))
    print("sample response after failure:",
          json.dumps({k: sample[k] for k in ("status", "partial", "shards")}))
    plot(df, actions, out / "failover.png")


def plot(df: pd.DataFrame, actions, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {"ok": "tab:blue", "partial": "tab:orange"}
    for status, g in df.groupby("status"):
        ax.scatter(g["t_start"], g["latency_ms"], s=4, color=colors.get(status, "tab:red"), label=status)
    retried = df[df["retries"] > 0]
    ax.scatter(retried["t_start"], retried["latency_ms"], s=30, facecolors="none",
               edgecolors="black", label="retried on replica")
    for act in actions:
        ax.axvline(act["t"], color="gray", ls="--")
        ax.text(act["t"], ax.get_ylim()[1] * 0.95, " " + act["action"], fontsize=7, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Queries during replica failure")
    ax.legend(markerscale=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
