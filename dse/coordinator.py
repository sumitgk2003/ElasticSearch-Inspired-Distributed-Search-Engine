"""Coordinator: receives queries, scatters them to one live copy of every shard,
gathers the per-shard top-k lists and merges them into a global top-k.

Statistics modes (how N / avgdl / df are obtained for scoring):
  dfs       pre-query phase asks every shard for the query terms' df, like
            Elasticsearch's ``dfs_query_then_fetch``; always exact, one extra
            round trip per query
  periodic  global statistics for all terms are pulled from the shards every
            ``refresh_interval`` seconds and reused; no extra round trip, but
            can be stale after writes
  local     every shard scores with its own statistics (Elasticsearch's
            default ``query_then_fetch``); rankings drift between shards

Failure handling: a heartbeat task marks copies up/down; each shard request
goes to a healthy copy (round-robin over replicas) and is retried on the next
copy when it times out or errors.  If no copy of a shard answers, the response
is flagged ``partial: true`` with the missing shards listed, or the query
fails with HTTP 503 when partial results are disallowed.
"""
from __future__ import annotations

import argparse
import asyncio
import heapq
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .analysis import analyze
from .scoring import SCORERS, CorpusStats
from .sharding import shard_for

STATS_MODES = ("dfs", "periodic", "local")


@dataclass
class Copy:
    shard: int
    node: str
    url: str
    primary: bool = False
    healthy: bool = True
    stale: bool = False      # missed a write; must be recovered, not revived

    @property
    def id(self) -> str:
        return f"{self.shard}@{self.node}"


Routing = Dict[int, List[Copy]]


class ShardUnavailable(Exception):
    pass


@dataclass
class CoordinatorConfig:
    timeout_s: float = 1.0               # per shard call
    heartbeat_interval_s: float = 0.5
    heartbeat_timeout_s: float = 0.5
    refresh_interval_s: float = 5.0      # periodic statistics refresh
    default_stats_mode: str = "dfs"
    allow_partial: bool = True


@dataclass
class Event:
    t: float
    kind: str
    detail: Dict[str, Any] = field(default_factory=dict)


class Coordinator:
    def __init__(self, routing: Routing, config: Optional[CoordinatorConfig] = None) -> None:
        self.routing = routing
        self.cfg = config or CoordinatorConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._rr = itertools.count()
        self._tasks: List[asyncio.Task] = []
        self.global_stats: Optional[CorpusStats] = None
        self.global_stats_at: float = 0.0
        self.events: List[Event] = []

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        # aiohttp rather than httpx for coordinator->shard RPC: in profiling,
        # httpx's connection pool used ~90% of the coordinator's CPU and capped
        # a 4-shard fan-out at ~230 QPS, versus ~2100 QPS with aiohttp.
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._refresh_loop()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self.session:
            await self.session.close()

    def set_routing(self, routing: Routing) -> None:
        self.routing = routing
        self._event("routing_updated", copies=sum(len(c) for c in routing.values()))

    def _event(self, kind: str, **detail: Any) -> None:
        self.events.append(Event(time.time(), kind, detail))

    # ------------------------------------------------------------- health
    async def _heartbeat_loop(self) -> None:
        while True:
            await self.check_health()
            await asyncio.sleep(self.cfg.heartbeat_interval_s)

    async def check_health(self) -> None:
        copies = [c for cs in self.routing.values() for c in cs]

        async def ping(c: Copy) -> None:
            try:
                async with self.session.get(c.url + "/health", timeout=_timeout(self.cfg.heartbeat_timeout_s)) as r:
                    ok = r.status == 200
            except RPC_ERRORS:
                ok = False
            if ok and not c.healthy and not c.stale:
                c.healthy = True
                self._event("copy_up", copy=c.id, source="heartbeat")
            elif not ok and c.healthy:
                self._mark_down(c, "heartbeat")

        await asyncio.gather(*(ping(c) for c in copies))

    def _mark_down(self, c: Copy, source: str) -> None:
        if c.healthy:
            c.healthy = False
            self._event("copy_down", copy=c.id, source=source)

    def cluster_health(self) -> dict:
        shards = {}
        for s, cs in sorted(self.routing.items()):
            shards[s] = {"copies": len(cs), "healthy": sum(c.healthy for c in cs),
                         "primary": next((c.node for c in cs if c.primary), None)}
        missing = [s for s, v in shards.items() if v["healthy"] == 0]
        degraded = [s for s, v in shards.items() if 0 < v["healthy"] < v["copies"]]
        status = "red" if missing else "yellow" if degraded else "green"
        return {"status": status, "num_shards": len(shards), "shards": shards,
                "unavailable_shards": missing, "degraded_shards": degraded}

    # --------------------------------------------------- shard RPC + retry
    def _candidates(self, shard: int) -> List[Copy]:
        healthy = [c for c in self.routing.get(shard, []) if c.healthy]
        if not healthy:
            return []
        start = next(self._rr) % len(healthy)   # spread reads over replicas
        return healthy[start:] + healthy[:start]

    async def _post(self, c: Copy, path: str, payload: dict, timeout_s: float) -> dict:
        async with self.session.post(c.url + path, json=payload, timeout=_timeout(timeout_s)) as r:
            r.raise_for_status()
            return await r.json()

    async def _call_shard(self, shard: int, path: str, payload: dict,
                          trace: List[dict]) -> Tuple[dict, Copy]:
        for c in self._candidates(shard):
            t = time.perf_counter()
            try:
                return await self._post(c, path, payload, self.cfg.timeout_s), c
            except RPC_ERRORS as e:
                trace.append({"shard": shard, "copy": c.id, "error": type(e).__name__,
                              "after_ms": round((time.perf_counter() - t) * 1000, 2)})
                self._mark_down(c, "request")
        raise ShardUnavailable(shard)

    async def _scatter(self, path: str, payload_for: Any, trace: List[dict]
                       ) -> Tuple[Dict[int, Tuple[dict, Copy]], List[int]]:
        shards = sorted(self.routing)

        async def one(s: int):
            try:
                return s, await self._call_shard(s, path, payload_for(s), trace)
            except ShardUnavailable:
                return s, None

        done = await asyncio.gather(*(one(s) for s in shards))
        ok = {s: r for s, r in done if r is not None}
        return ok, [s for s, r in done if r is None]

    # ------------------------------------------------------------ statistics
    async def collect_stats(self, terms: Optional[Sequence[str]], trace: List[dict]
                            ) -> Tuple[CorpusStats, List[int]]:
        payload = {"terms": list(terms) if terms is not None else None}
        ok, missing = await self._scatter("/stats", lambda s: payload, trace)
        return CorpusStats.merge(CorpusStats.from_json(r) for r, _ in ok.values()), missing

    async def refresh_stats(self) -> CorpusStats:
        stats, missing = await self.collect_stats(None, [])
        self.global_stats, self.global_stats_at = stats, time.time()
        self._event("stats_refreshed", num_docs=stats.num_docs, terms=len(stats.df), missing=missing)
        return stats

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.refresh_interval_s)
            # Pulling every term's df from every shard is expensive, so only
            # refresh once periodic statistics are actually in use (the first
            # periodic query, or an explicit refresh, populates them).
            if self.global_stats is None:
                continue
            try:
                await self.refresh_stats()
            except Exception as e:  # keep the loop alive; the old stats stay in use
                self._event("stats_refresh_failed", error=repr(e))

    # ---------------------------------------------------------------- search
    async def search(self, query: str, k: int = 10, scorer: str = "bm25",
                     stats_mode: Optional[str] = None, allow_partial: Optional[bool] = None) -> dict:
        t0 = time.perf_counter()
        stats_mode = stats_mode or self.cfg.default_stats_mode
        allow_partial = self.cfg.allow_partial if allow_partial is None else allow_partial
        if stats_mode not in STATS_MODES:
            raise ValueError(f"stats_mode must be one of {STATS_MODES}")
        if scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}")
        terms = analyze(query)
        trace: List[dict] = []
        stats_ms = 0.0

        stats_json: Optional[dict] = None
        if stats_mode == "dfs":
            ts = time.perf_counter()
            stats, _ = await self.collect_stats(terms, trace)
            stats_ms = (time.perf_counter() - ts) * 1000
            stats_json = stats.to_json()
        elif stats_mode == "periodic":
            if self.global_stats is None:
                await self.refresh_stats()
            g = self.global_stats
            stats_json = {"num_docs": g.num_docs, "total_len": g.total_len,
                          "df": {t: g.df.get(t, 0) for t in set(terms)}}

        tq = time.perf_counter()
        payload = {"terms": terms, "k": k, "scorer": scorer, "stats": stats_json}
        ok, missing = await self._scatter("/search", lambda s: payload, trace)
        query_ms = (time.perf_counter() - tq) * 1000

        # Gather: per-shard lists are already sorted by (-score, doc_id), so a
        # k-way heap merge yields the global order directly.
        streams = [[(-score, doc_id, s) for doc_id, score in r["hits"]] for s, (r, _) in ok.items()]
        top = list(itertools.islice(heapq.merge(*streams), k))
        hits = [{"doc_id": d, "score": -neg, "shard": s} for neg, d, s in top]

        total = len(self.routing)
        result = {
            "query": query, "terms": terms, "k": k, "scorer": scorer, "stats_mode": stats_mode,
            "hits": hits,
            "partial": bool(missing),
            "shards": {"total": total, "successful": len(ok), "failed": len(missing),
                       "missing": missing},
            "served_by": {s: c.id for s, (_, c) in ok.items()},
            "retries": trace,
            "took_ms": round((time.perf_counter() - t0) * 1000, 3),
            "phases_ms": {"stats": round(stats_ms, 3), "query": round(query_ms, 3)},
        }
        if missing:
            self._event("partial_result" if allow_partial else "search_failed",
                        query=query, missing=missing)
            if not allow_partial:
                result["status"] = "error"
                raise SearchFailed(result)
        result["status"] = "partial" if missing else "ok"
        return result

    # ----------------------------------------------------------------- writes
    async def index_documents(self, docs: Sequence[Tuple[str, str]]) -> dict:
        """Route each document to its shard and write it to every live copy,
        primary first.  A copy that misses a write is marked stale."""
        num_shards = len(self.routing)
        by_shard: Dict[int, List[List[str]]] = {}
        for d, t in docs:
            by_shard.setdefault(shard_for(d, num_shards), []).append([d, t])
        report: Dict[int, dict] = {}

        async def write(c: Copy, batch: List[List[str]]) -> bool:
            try:
                await self._post(c, "/index", {"docs": batch}, max(self.cfg.timeout_s, 10.0))
                return True
            except RPC_ERRORS:
                c.stale = True
                self._mark_down(c, "write")
                return False

        for s, batch in by_shard.items():
            live = sorted((c for c in self.routing[s] if c.healthy), key=lambda c: not c.primary)
            if not live:
                report[s] = {"docs": len(batch), "acked_copies": 0}
                continue
            if await write(live[0], batch):              # primary
                acks = 1 + sum(await asyncio.gather(*(write(c, batch) for c in live[1:])))
            else:
                acks = 0
            report[s] = {"docs": len(batch), "acked_copies": acks}
        return {"shards": report, "failed_shards": [s for s, r in report.items() if r["acked_copies"] == 0]}


RPC_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, ValueError)


def _timeout(seconds: float) -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=seconds)


class SearchFailed(Exception):
    def __init__(self, result: dict) -> None:
        super().__init__(f"shards unavailable: {result['shards']['missing']}")
        self.result = result


# ------------------------------------------------------------- routing I/O
def routing_from_json(d: dict) -> Routing:
    return {int(s): [Copy(int(s), c["node"], c["url"].rstrip("/"), bool(c.get("primary", False)))
                     for c in copies]
            for s, copies in d["shards"].items()}


def routing_to_json(r: Routing) -> dict:
    return {"shards": {str(s): [asdict(c) for c in cs] for s, cs in sorted(r.items())}}


# --------------------------------------------------------------- HTTP API
class SearchBody(BaseModel):
    query: str
    k: int = 10
    scorer: str = "bm25"
    stats_mode: Optional[str] = None
    allow_partial: Optional[bool] = None


class DocsBody(BaseModel):
    docs: List[List[str]]


def create_app(coord: Coordinator) -> FastAPI:
    app = FastAPI(title="coordinator")

    @app.on_event("startup")
    async def _start() -> None:
        await coord.start()

    @app.on_event("shutdown")
    async def _stop() -> None:
        await coord.stop()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/search")
    async def search(body: SearchBody):
        try:
            return await coord.search(body.query, body.k, body.scorer, body.stats_mode, body.allow_partial)
        except SearchFailed as e:
            return JSONResponse(e.result, status_code=503)
        except ValueError as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=400)

    @app.post("/documents")
    async def documents(body: DocsBody) -> dict:
        return await coord.index_documents([(d, t) for d, t in body.docs])

    @app.get("/cluster/health")
    async def cluster_health() -> dict:
        return coord.cluster_health()

    @app.get("/cluster/routing")
    async def get_routing() -> dict:
        return routing_to_json(coord.routing)

    @app.put("/cluster/routing")
    async def put_routing(body: dict) -> dict:
        coord.set_routing(routing_from_json(body))
        await coord.check_health()
        return coord.cluster_health()

    @app.post("/admin/refresh")
    async def refresh() -> dict:
        s = await coord.refresh_stats()
        return {"num_docs": s.num_docs, "terms": len(s.df)}

    @app.get("/events")
    async def events(since: float = 0.0) -> list:
        return [asdict(e) for e in coord.events if e.t > since]

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the coordinator")
    ap.add_argument("--routing", required=True, help="routing table JSON file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--stats-mode", choices=STATS_MODES, default="dfs")
    ap.add_argument("--timeout", type=float, default=1.0)
    ap.add_argument("--heartbeat", type=float, default=0.5)
    ap.add_argument("--refresh", type=float, default=5.0)
    ap.add_argument("--no-partial", action="store_true")
    a = ap.parse_args()
    with open(a.routing) as f:
        routing = routing_from_json(json.load(f))
    cfg = CoordinatorConfig(timeout_s=a.timeout, heartbeat_interval_s=a.heartbeat,
                            heartbeat_timeout_s=min(a.heartbeat, a.timeout),
                            refresh_interval_s=a.refresh, default_stats_mode=a.stats_mode,
                            allow_partial=not a.no_partial)
    uvicorn.run(create_app(Coordinator(routing, cfg)), host=a.host, port=a.port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
