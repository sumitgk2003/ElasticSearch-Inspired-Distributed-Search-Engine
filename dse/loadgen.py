"""Closed-loop asyncio load generator.

``concurrency`` virtual clients each send a query, wait for the answer, and
immediately send the next one, for ``duration_s`` seconds.  Every request is
recorded so latency percentiles, throughput and failure behaviour over time
can be analysed afterwards.
"""
from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Union

import aiohttp

from .metrics import percentiles


@dataclass
class Record:
    t_start: float          # seconds since the run started
    latency_ms: float
    status: str             # ok | partial | error | http_<code> | exception name
    retries: int = 0
    missing: int = 0


async def run_load(url: Union[str, Sequence[str]], queries: Sequence[str], concurrency: int = 8,
                   duration_s: float = 10.0, warmup_s: float = 1.0, k: int = 10,
                   body_extra: Optional[dict] = None, timeout_s: float = 30.0) -> List[Record]:
    """``url`` may be a list of coordinators; clients are spread over them."""
    body_extra = body_extra or {}
    urls = [url] if isinstance(url, str) else list(url)
    cycle = itertools.cycle(queries)
    records: List[Record] = []
    # aiohttp: httpx's client-side overhead would cap the measured throughput.
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector,
                                     timeout=aiohttp.ClientTimeout(total=timeout_s)) as client:
        t0 = time.perf_counter()
        measure_from = t0 + warmup_s
        stop_at = measure_from + duration_s

        async def worker(url: str) -> None:
            while True:
                start = time.perf_counter()
                if start >= stop_at:
                    return
                q = next(cycle)
                status, retries, missing = "ok", 0, 0
                try:
                    async with client.post(url + "/search", json={"query": q, "k": k, **body_extra}) as r:
                        d = await r.json()
                    if r.status == 200:
                        status = d.get("status", "ok")
                        retries, missing = len(d.get("retries", [])), len(d["shards"]["missing"])
                    else:
                        status = f"http_{r.status}"
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                    status = type(e).__name__
                end = time.perf_counter()
                if start >= measure_from:
                    records.append(Record(start - measure_from, (end - start) * 1000, status, retries, missing))

        await asyncio.gather(*(worker(urls[i % len(urls)]) for i in range(concurrency)))
    return records


def summarize(records: Sequence[Record], duration_s: Optional[float] = None) -> Dict[str, float]:
    if not records:
        return {"requests": 0}
    if duration_s is None:
        duration_s = max(r.t_start + r.latency_ms / 1000 for r in records)
    ok = [r for r in records if r.status in ("ok", "partial")]
    out = {"requests": len(records), "qps": len(ok) / duration_s,
           "error_rate": 1 - len(ok) / len(records),
           "partial_rate": sum(r.status == "partial" for r in records) / len(records)}
    out.update({f"{k}_ms": v for k, v in percentiles([r.latency_ms for r in ok]).items()})
    return out


def to_rows(records: Sequence[Record]) -> List[dict]:
    return [asdict(r) for r in records]
