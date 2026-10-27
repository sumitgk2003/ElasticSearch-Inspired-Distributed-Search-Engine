"""Shared CLI options: which corpus and query set an experiment runs on."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from dse import data


def add_corpus_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--corpus", help="collection TSV/JSONL/CSV (e.g. MS MARCO collection.tsv)")
    ap.add_argument("--queries", help="queries TSV/JSONL/CSV (e.g. MS MARCO queries.dev.tsv)")
    ap.add_argument("--corpus-column", default=None,
                    help="CSV column holding document text (default: auto, e.g. finalpassage)")
    ap.add_argument("--query-column", default="query", help="CSV column holding query text")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N documents")
    ap.add_argument("--num-docs", type=int, default=20000, help="synthetic corpus size")
    ap.add_argument("--num-queries", type=int, default=200)
    ap.add_argument("--out", default="results")
    ap.add_argument("--work", default="work")


def load_corpus(a: argparse.Namespace) -> Tuple[List[data.Doc], List[str]]:
    if a.corpus:
        docs = data.load(a.corpus, a.limit, a.corpus_column)
    else:
        docs = data.synthetic_corpus(a.num_docs)
    if a.queries:
        queries = [q for _, q in data.load(a.queries, a.num_queries, a.query_column)]
    else:
        queries = [q for _, q in data.synthetic_queries(docs, a.num_queries)]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    return docs, queries
