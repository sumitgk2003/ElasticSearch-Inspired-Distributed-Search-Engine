"""In-memory inverted index for one shard (or for the single-node baseline).

Postings are kept as growable Python lists while documents are being added
and are frozen into NumPy arrays on first use, so scoring a term is a few
vectorised operations over its postings list.
"""
from __future__ import annotations

import heapq
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .analysis import analyze
from .scoring import CorpusStats, bm25_idf, bm25_term, tfidf_idf, tfidf_term

Hit = Tuple[str, float]  # (doc_id, score)


class InvertedIndex:
    def __init__(self) -> None:
        self.doc_ids: List[str] = []          # internal id -> external doc id
        self._lens: List[int] = []            # internal id -> document length
        self._ext: Dict[str, int] = {}        # external doc id -> internal id
        self._post_ids: Dict[str, List[int]] = {}
        self._post_tfs: Dict[str, List[int]] = {}
        self.total_len = 0
        # Frozen array views, rebuilt lazily after writes.
        self._arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._len_arr: Optional[np.ndarray] = None

    # ------------------------------------------------------------- writing
    def add(self, doc_id: str, text: str) -> bool:
        """Index one document.  Returns False if the id is already present."""
        if doc_id in self._ext:
            return False
        terms = analyze(text)
        iid = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self._ext[doc_id] = iid
        self._lens.append(len(terms))
        self.total_len += len(terms)
        for term, tf in Counter(terms).items():
            self._post_ids.setdefault(term, []).append(iid)
            self._post_tfs.setdefault(term, []).append(tf)
            self._arrays.pop(term, None)
        self._len_arr = None
        return True

    def add_many(self, docs: Iterable[Tuple[str, str]]) -> int:
        return sum(self.add(d, t) for d, t in docs)

    # ------------------------------------------------------------- reading
    def __len__(self) -> int:
        return len(self.doc_ids)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._ext

    @property
    def num_terms(self) -> int:
        return len(self._post_ids)

    def df(self, term: str) -> int:
        return len(self._post_ids.get(term, ()))

    def local_stats(self, terms: Optional[Iterable[str]] = None) -> CorpusStats:
        """N, total length and df -- for the given terms, or for all terms."""
        if terms is None:
            df = {t: len(ids) for t, ids in self._post_ids.items()}
        else:
            df = {t: self.df(t) for t in set(terms)}
        return CorpusStats(len(self.doc_ids), self.total_len, df)

    def _postings(self, term: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        arr = self._arrays.get(term)
        if arr is None:
            ids = self._post_ids.get(term)
            if not ids:
                return None
            arr = (np.asarray(ids, dtype=np.int64), np.asarray(self._post_tfs[term], dtype=np.float64))
            self._arrays[term] = arr
        return arr

    def _lengths(self) -> np.ndarray:
        if self._len_arr is None:
            self._len_arr = np.asarray(self._lens, dtype=np.float64)
        return self._len_arr

    def search(self, query_terms: Sequence[str], k: int = 10,
               stats: Optional[CorpusStats] = None, scorer: str = "bm25") -> List[Hit]:
        """Score already-analyzed query terms and return the local top-k.

        ``stats`` supplies N / avgdl / df.  When it is None the shard's own
        local statistics are used, which is what makes rankings drift.
        Ties are broken by doc id so that results are fully deterministic and
        comparable between the distributed and single-node engines.
        """
        if not self.doc_ids or k <= 0:
            return []
        if stats is None:
            stats = self.local_stats(query_terms)
        n_docs, avgdl = stats.num_docs, stats.avgdl or 1.0
        lengths = self._lengths()
        acc: Optional[np.ndarray] = None
        # Query term frequency is honoured; iteration order is fixed (first
        # occurrence) so float sums are identical on every node.
        for term, qtf in Counter(query_terms).items():
            post = self._postings(term)
            if post is None:
                continue
            ids, tfs = post
            df = stats.df.get(term, len(ids))
            if scorer == "bm25":
                contrib = bm25_term(tfs, lengths[ids], bm25_idf(n_docs, df), avgdl)
            elif scorer == "tfidf":
                contrib = tfidf_term(tfs, tfidf_idf(n_docs, df))
            else:
                raise ValueError(f"unknown scorer {scorer!r}")
            if acc is None:
                acc = np.zeros(len(self.doc_ids), dtype=np.float64)
            acc[ids] += qtf * contrib
        if acc is None:
            return []
        # Every matching term contributes a strictly positive score, so the
        # non-zero accumulators are exactly the matching documents.
        cand = np.flatnonzero(acc)
        cand_scores = acc[cand]
        if len(cand) > k:
            # Keep everything tied with the k-th score so tie-breaking by
            # doc id below stays exact.
            kth = np.partition(cand_scores, len(cand) - k)[len(cand) - k]
            keep = cand_scores >= kth
            cand, cand_scores = cand[keep], cand_scores[keep]
        hits = [(self.doc_ids[i], float(s)) for i, s in zip(cand.tolist(), cand_scores.tolist())]
        return heapq.nsmallest(k, hits, key=lambda h: (-h[1], h[0]))

    # --------------------------------------------------------- persistence
    def to_state(self) -> dict:
        """Plain-data snapshot, used for disk storage and replica recovery."""
        return {"doc_ids": self.doc_ids, "lens": self._lens,
                "post_ids": self._post_ids, "post_tfs": self._post_tfs}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.to_state(), f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        with open(path, "rb") as f:
            return cls.from_state(pickle.load(f))

    @classmethod
    def from_state(cls, state: dict) -> "InvertedIndex":
        idx = cls()
        idx.doc_ids = state["doc_ids"]
        idx._lens = state["lens"]
        idx._ext = {d: i for i, d in enumerate(idx.doc_ids)}
        idx._post_ids = state["post_ids"]
        idx._post_tfs = state["post_tfs"]
        idx.total_len = sum(idx._lens)
        return idx
