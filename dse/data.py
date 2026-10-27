"""Corpus and query loading.

Supported inputs:
* MS MARCO style TSV: ``id<TAB>text`` (collection.tsv / queries.tsv)
* JSONL with ``{"id": ..., "text": ...}`` (or ``_id`` / ``title`` + ``text``,
  as in BEIR datasets)
* CSV with a header row, e.g. Kaggle's parthplc/ms-marco-dataset
  (columns ``answers, query, finalpassage``); pick the column with ``column``
* a synthetic Zipf-distributed corpus for tests and offline runs
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

Doc = Tuple[str, str]

csv.field_size_limit(sys.maxsize)


def read_tsv(path: Path, limit: Optional[int] = None) -> Iterator[Doc]:
    with open(path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
            if limit is not None and i >= limit:
                return
            if len(row) >= 2:
                yield row[0], " ".join(row[1:])


def read_jsonl(path: Path, limit: Optional[int] = None) -> Iterator[Doc]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            d = json.loads(line)
            doc_id = str(d.get("id", d.get("_id")))
            text = " ".join(x for x in (d.get("title", ""), d.get("text", "")) if x)
            yield doc_id, text


# Column used when a CSV is loaded without naming one, in order of preference.
CSV_TEXT_COLUMNS = ("text", "passage", "finalpassage", "query")


def read_csv(path: Path, limit: Optional[int] = None, column: Optional[str] = None) -> Iterator[Doc]:
    """One document per row.  Uses an ``id`` column if present, otherwise the
    row number; rows whose text was already seen are skipped."""
    seen = set()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if column is None:
            column = next((c for c in CSV_TEXT_COLUMNS if c in reader.fieldnames), None)
            if column is None:
                raise ValueError(f"{path}: pass column=, one of {reader.fieldnames}")
        n = 0
        for i, row in enumerate(reader):
            if limit is not None and n >= limit:
                return
            text = (row.get(column) or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            n += 1
            yield row.get("id") or f"p{i}", text


def load(path: Path, limit: Optional[int] = None, column: Optional[str] = None) -> List[Doc]:
    path = Path(path)
    if path.suffix == ".csv":
        return list(read_csv(path, limit, column))
    reader = read_jsonl if path.suffix in (".jsonl", ".json") else read_tsv
    return list(reader(path, limit))


def synthetic_corpus(num_docs: int = 5000, vocab_size: int = 20000, mean_len: int = 60,
                     seed: int = 7) -> List[Doc]:
    """Documents whose words follow a Zipf law, like natural text."""
    rng = random.Random(seed)
    vocab = [_word(i) for i in range(vocab_size)]
    weights = [1.0 / (r + 1) for r in range(vocab_size)]
    docs = []
    for i in range(num_docs):
        n = max(5, int(rng.expovariate(1.0 / mean_len)))
        docs.append((f"d{i}", " ".join(rng.choices(vocab, weights, k=n))))
    return docs


def synthetic_queries(docs: List[Doc], num_queries: int = 200, seed: int = 11,
                      min_terms: int = 2, max_terms: int = 4) -> List[Doc]:
    """Queries sampled from document text, so every query has matches."""
    rng = random.Random(seed)
    out = []
    for q in range(num_queries):
        words = rng.choice(docs)[1].split()
        n = min(len(words), rng.randint(min_terms, max_terms))
        out.append((f"q{q}", " ".join(rng.sample(words, n))))
    return out


def _word(i: int) -> str:
    # Deterministic pronounceable pseudo-words: "bako", "tiruse", ...
    cons, vows = "bcdfghjklmnprstvwz", "aeiou"
    s = ""
    i += 1
    while i:
        i, r = divmod(i, len(cons) * len(vows))
        s += cons[r % len(cons)] + vows[r // len(cons)]
    return s + "x"   # suffix keeps the stemmer and stop-word list out of the way
