"""Deliverable 2: ranking agreement with the single-node index.

For each statistics strategy (global DFS, periodic refresh, shard-local) and
each scorer, compare the distributed top-k against a single-node index over
the same corpus using top-k overlap, Kendall's tau and NDCG@k.

The periodic strategy is measured twice: right after a refresh (fresh) and
after new documents were indexed but before the next refresh (stale), which
is where it differs from DFS.

    python -m experiments.ranking --shards 4 8
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from dse.analysis import analyze
from dse.build import build_single
from dse.cluster import Cluster
from dse.metrics import kendall_tau_at_k, ndcg_at_k, overlap_at_k

from .common import add_corpus_args, load_corpus


def evaluate(cluster: Cluster, reference, queries, k, scorer, stats_mode, label, shards):
    rows = []
    for q in queries:
        ref = [d for d, _ in reference.search(analyze(q), k, scorer=scorer)]
        res = cluster.search(q, k, scorer=scorer, stats_mode=stats_mode)
        got = [h["doc_id"] for h in res["hits"]]
        rows.append({"shards": shards, "scorer": scorer, "strategy": label, "query": q,
                     "overlap": overlap_at_k(ref, got, k), "kendall_tau": kendall_tau_at_k(ref, got, k),
                     "ndcg": ndcg_at_k(ref, got, k), "exact": ref == got})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_corpus_args(ap)
    ap.add_argument("--shards", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--update-frac", type=float, default=0.2,
                    help="fraction of the corpus held back and indexed live, to make periodic stats stale")
    a = ap.parse_args()
    docs, queries = load_corpus(a)

    rng = random.Random(0)
    docs = docs[:]
    rng.shuffle(docs)
    n_late = int(len(docs) * a.update_frac)
    initial, late = docs[n_late:], docs[:n_late]

    rows = []
    for S in a.shards:
        with Cluster(Path(a.work) / f"ranking_s{S}", num_shards=S, replicas=1,
                     refresh_s=3600, timeout_s=30) as c:
            c.build(initial)
            c.start()
            c.refresh_stats()
            ref = build_single(initial)
            for scorer in ("bm25", "tfidf"):
                rows += evaluate(c, ref, queries, a.k, scorer, "dfs", "global (DFS)", S)
                rows += evaluate(c, ref, queries, a.k, scorer, "periodic", "periodic (fresh)", S)
                rows += evaluate(c, ref, queries, a.k, scorer, "local", "shard-local", S)
            if late:
                # New documents arrive; periodic stats are now stale until the next refresh.
                for i in range(0, len(late), 2000):
                    c.index_documents(late[i:i + 2000])
                ref_all = build_single(initial + late)
                for scorer in ("bm25", "tfidf"):
                    rows += evaluate(c, ref_all, queries, a.k, scorer, "dfs", "global (DFS), after writes", S)
                    rows += evaluate(c, ref_all, queries, a.k, scorer, "periodic", "periodic (stale)", S)
                    rows += evaluate(c, ref_all, queries, a.k, scorer, "local", "shard-local, after writes", S)
        print(f"shards={S} done")

    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "ranking_per_query.csv", index=False)
    table = (df.groupby(["shards", "scorer", "strategy"], sort=False)
               [["overlap", "kendall_tau", "ndcg", "exact"]].mean().round(4).reset_index()
               .rename(columns={"overlap": f"top{a.k}_overlap", "ndcg": f"ndcg@{a.k}",
                                "exact": "identical_rankings"}))
    table.to_csv(Path(a.out) / "ranking_table.csv", index=False)
    (Path(a.out) / "ranking_table.md").write_text(table.to_markdown(index=False) if _has_tabulate()
                                                  else table.to_string(index=False))
    print(table.to_string(index=False))


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
