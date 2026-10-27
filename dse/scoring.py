"""BM25 and TF-IDF scoring plus the corpus statistics they depend on.

Both scorers take their collection statistics (N, avgdl, df) as an argument
instead of reading them from the index.  That is what lets a shard score with
either its own local statistics or the global statistics handed to it by the
coordinator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np

BM25_K1 = 1.2
BM25_B = 0.75


@dataclass
class CorpusStats:
    num_docs: int = 0
    total_len: int = 0
    df: Dict[str, int] = field(default_factory=dict)

    @property
    def avgdl(self) -> float:
        return self.total_len / self.num_docs if self.num_docs else 0.0

    @staticmethod
    def merge(parts: Iterable["CorpusStats"]) -> "CorpusStats":
        out = CorpusStats()
        for p in parts:
            out.num_docs += p.num_docs
            out.total_len += p.total_len
            for t, n in p.df.items():
                out.df[t] = out.df.get(t, 0) + n
        return out

    def to_json(self) -> dict:
        return {"num_docs": self.num_docs, "total_len": self.total_len, "df": self.df}

    @staticmethod
    def from_json(d: dict) -> "CorpusStats":
        return CorpusStats(int(d["num_docs"]), int(d["total_len"]), {k: int(v) for k, v in d["df"].items()})


def bm25_idf(n_docs: int, df: int) -> float:
    # Lucene's non-negative BM25 IDF.
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def tfidf_idf(n_docs: int, df: int) -> float:
    # Smoothed IDF (as in scikit-learn): always > 0.
    return math.log((1.0 + n_docs) / (1.0 + df)) + 1.0


def bm25_term(tf: np.ndarray, dl: np.ndarray, idf: float, avgdl: float,
              k1: float = BM25_K1, b: float = BM25_B) -> np.ndarray:
    norm = k1 * (1.0 - b + b * dl / avgdl)
    return idf * tf * (k1 + 1.0) / (tf + norm)


def tfidf_term(tf: np.ndarray, idf: float) -> np.ndarray:
    # Log-scaled term frequency, no length normalisation (simple baseline).
    return (1.0 + np.log(tf)) * idf


SCORERS = ("bm25", "tfidf")
