"""Partition a corpus into N shard indexes on disk, plus the baseline index."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from . import data
from .index import InvertedIndex
from .sharding import shard_for


def shard_path(index_dir: Path, shard: int) -> Path:
    return Path(index_dir) / f"shard_{shard}.pkl"


def build_shards(docs: List[data.Doc], num_shards: int, index_dir: Path) -> List[InvertedIndex]:
    shards = [InvertedIndex() for _ in range(num_shards)]
    for doc_id, text in docs:
        shards[shard_for(doc_id, num_shards)].add(doc_id, text)
    index_dir = Path(index_dir)
    for s, idx in enumerate(shards):
        idx.save(shard_path(index_dir, s))
    meta = {"num_shards": num_shards, "num_docs": len(docs),
            "docs_per_shard": [len(s) for s in shards]}
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return shards


def build_single(docs: List[data.Doc]) -> InvertedIndex:
    idx = InvertedIndex()
    idx.add_many(docs)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", help="TSV or JSONL file; omit for a synthetic corpus")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    docs = data.load(a.corpus, a.limit) if a.corpus else data.synthetic_corpus()
    t = time.perf_counter()
    shards = build_shards(docs, a.shards, Path(a.out))
    print(f"indexed {len(docs)} docs into {a.shards} shards "
          f"{[len(s) for s in shards]} in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
