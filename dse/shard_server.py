"""Shard server: one OS process serving one copy (primary or replica) of a shard.

Endpoints
  GET  /health            liveness + shard identity, used by the heartbeat
  POST /stats             local N / total length / df (for DFS and refresh)
  POST /search            local top-k for already-analyzed query terms
  POST /index             add documents (writes are replicated by the coordinator)
  GET  /snapshot          full index state, used to recover a new replica
  POST /admin/delay       change injected network delay / jitter at runtime
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .index import InvertedIndex
from .scoring import CorpusStats


class StatsRequest(BaseModel):
    terms: Optional[List[str]] = None   # None -> statistics for every term


class SearchRequest(BaseModel):
    terms: List[str]
    k: int = 10
    scorer: str = "bm25"
    stats: Optional[dict] = None        # global CorpusStats; None -> local stats


class IndexRequest(BaseModel):
    docs: List[List[str]]               # [[doc_id, text], ...]


class DelayRequest(BaseModel):
    delay_ms: float = 0.0
    jitter_ms: float = 0.0


def create_app(index: InvertedIndex, shard: int, node: str, copy_id: str,
               delay_ms: float = 0.0, jitter_ms: float = 0.0, seed: Optional[int] = None) -> FastAPI:
    app = FastAPI(title=f"shard {shard} on {node}")
    rng = random.Random(seed)
    net = {"delay_ms": delay_ms, "jitter_ms": jitter_ms}

    async def network_delay() -> None:
        # Simulated network hop: fixed delay plus exponential jitter, which
        # produces the long right tail seen on real networks.
        d = net["delay_ms"] + (rng.expovariate(1.0 / net["jitter_ms"]) if net["jitter_ms"] > 0 else 0.0)
        if d > 0:
            await asyncio.sleep(d / 1000.0)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "shard": shard, "node": node, "copy": copy_id,
                "docs": len(index), "pid": os.getpid()}

    @app.post("/stats")
    async def stats(req: StatsRequest) -> dict:
        await network_delay()
        return index.local_stats(req.terms).to_json()

    @app.post("/search")
    async def search(req: SearchRequest) -> dict:
        await network_delay()
        t = time.perf_counter()
        st = CorpusStats.from_json(req.stats) if req.stats is not None else None
        hits = index.search(req.terms, req.k, st, req.scorer)
        return {"shard": shard, "copy": copy_id, "hits": hits,
                "took_ms": (time.perf_counter() - t) * 1000.0}

    @app.post("/index")
    async def add(req: IndexRequest) -> dict:
        n = sum(index.add(d, t) for d, t in req.docs)
        return {"indexed": n, "docs": len(index)}

    @app.get("/snapshot")
    async def snapshot() -> dict:
        return index.to_state()

    @app.post("/admin/delay")
    async def set_delay(req: DelayRequest) -> dict:
        net.update(delay_ms=req.delay_ms, jitter_ms=req.jitter_ms)
        return net

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one shard copy")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--copy-id", default=None)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--index", help="path to the shard's index file")
    src.add_argument("--recover-from", help="URL of a live copy to copy the index from")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    if a.index:
        index = InvertedIndex.load(a.index)
    else:
        # Peer recovery: copy the current index (including writes made after
        # the shard files were built) from a healthy copy.
        r = httpx.get(a.recover_from.rstrip("/") + "/snapshot", timeout=300)
        r.raise_for_status()
        index = InvertedIndex.from_state(r.json())
    copy_id = a.copy_id or f"{a.shard}@{a.node}"
    app = create_app(index, a.shard, a.node, copy_id, a.delay_ms, a.jitter_ms, a.seed)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
