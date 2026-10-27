"""Ranking-agreement metrics: distributed ranking vs. single-node reference."""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
from scipy.stats import kendalltau


def overlap_at_k(ref: Sequence[str], got: Sequence[str], k: int) -> float:
    ref, got = list(ref[:k]), list(got[:k])
    if not ref:
        return 1.0 if not got else 0.0
    return len(set(ref) & set(got)) / len(ref)


def kendall_tau_at_k(ref: Sequence[str], got: Sequence[str], k: int) -> float:
    """Kendall's tau-b over the union of both top-k lists.

    Items missing from one list are placed tied at rank k (just below its
    last item), the usual way to compare truncated rankings.
    """
    ref, got = list(ref[:k]), list(got[:k])
    if ref == got:
        return 1.0
    items = list(dict.fromkeys(ref + got))
    rr = {d: i for i, d in enumerate(ref)}
    gr = {d: i for i, d in enumerate(got)}
    x = [rr.get(d, k) for d in items]
    y = [gr.get(d, k) for d in items]
    tau = kendalltau(x, y).statistic
    return 0.0 if tau is None or math.isnan(tau) else float(tau)


def ndcg_at_k(ref: Sequence[str], got: Sequence[str], k: int) -> float:
    """NDCG@k using the reference ranking as graded relevance:
    the reference's i-th result has relevance k - i, others 0."""
    ref = list(ref[:k])
    if not ref:
        return 1.0
    rel: Dict[str, float] = {d: k - i for i, d in enumerate(ref)}
    dcg = sum(rel.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(got[:k]))
    idcg = sum(rel[d] / math.log2(i + 2) for i, d in enumerate(ref))
    return dcg / idcg


def percentiles(latencies_ms: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(latencies_ms, dtype=float)
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99", "max")}
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}
