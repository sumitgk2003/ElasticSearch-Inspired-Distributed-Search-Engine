"""Launch and control a local cluster: one OS process per shard copy plus a
coordinator process.  Used by the experiments and the Kaggle notebook.

Nodes are logical: a node is a name, and every shard copy allocated to it runs
as its own process tagged with that name.  Killing a node kills all of its
copies, which is how a machine failure is simulated on a single host.

    with Cluster("work/c4", num_shards=4, replicas=1) as c:
        c.build(docs)
        c.start()
        c.search("distributed systems")
        c.kill_node("node-1")
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import httpx

from . import allocation
from .build import build_shards, shard_path
from .data import Doc


@dataclass
class Proc:
    shard: int
    node: str
    port: int
    popen: subprocess.Popen

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        return self.popen.poll() is None


_next_port = 20000 + (os.getpid() % 1000) * 20


def _free_port() -> int:
    # Hand out ports below the OS ephemeral range (49152+ on macOS, 32768+ on
    # Linux).  A port from bind(0) is ephemeral, so before the shard process
    # binds it an outgoing connection -- or a TCP self-connect from the health
    # poller -- can take it, and startup then hangs.
    global _next_port
    while True:
        port, _next_port = _next_port, _next_port + 1
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port


def _wait_http(url: str, proc: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process for {url} exited with code {proc.returncode}")
        try:
            if httpx.get(url + "/health", timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"{url} did not become healthy in {timeout}s")


class Cluster:
    def __init__(self, work_dir: str | Path, num_shards: int, replicas: int = 1,
                 nodes: Optional[Sequence[str]] = None, delay_ms: float = 0.0,
                 jitter_ms: float = 0.0, stats_mode: str = "dfs", timeout_s: float = 1.0,
                 heartbeat_s: float = 0.5, refresh_s: float = 5.0, allow_partial: bool = True,
                 coordinators: int = 1) -> None:
        self.work = Path(work_dir)
        self.index_dir = self.work / "index"
        self.log_dir = self.work / "logs"
        self.num_shards = num_shards
        self.replicas = replicas
        # Default: enough nodes that every copy of a shard can sit on its own node.
        n_nodes = max(replicas + 1, min(num_shards * (replicas + 1), 4))
        self.nodes: List[str] = list(nodes) if nodes else [f"node-{i}" for i in range(n_nodes)]
        self.delay_ms, self.jitter_ms = delay_ms, jitter_ms
        self.coord_args = ["--stats-mode", stats_mode, "--timeout", str(timeout_s),
                           "--heartbeat", str(heartbeat_s), "--refresh", str(refresh_s)]
        if not allow_partial:
            self.coord_args.append("--no-partial")
        # Several coordinator processes can share the routing table, like
        # Elasticsearch's coordinating-only nodes; each runs its own heartbeat.
        self.coord_ports = [_free_port() for _ in range(coordinators)]
        self.allocation: allocation.Allocation = {}
        self.procs: Dict[Tuple[int, str], Proc] = {}
        self.coordinators: List[subprocess.Popen] = []
        self._seed = 0

    # --------------------------------------------------------------- build
    def build(self, docs: List[Doc]) -> List[int]:
        shards = build_shards(docs, self.num_shards, self.index_dir)
        return [len(s) for s in shards]

    # --------------------------------------------------------- processes
    def _spawn_copy(self, shard: int, node: str, recover_from: Optional[str] = None) -> Proc:
        port = _free_port()
        self._seed += 1
        src = ["--recover-from", recover_from] if recover_from else ["--index", str(shard_path(self.index_dir, shard))]
        cmd = [sys.executable, "-m", "dse.shard_server", "--shard", str(shard), "--node", node,
               "--port", str(port), "--delay-ms", str(self.delay_ms),
               "--jitter-ms", str(self.jitter_ms), "--seed", str(self._seed), *src]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"shard{shard}_{node}.log", "ab")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=_env())
        proc = Proc(shard, node, port, p)
        self.procs[(shard, node)] = proc
        return proc

    def start(self) -> None:
        self.allocation = allocation.allocate(self.num_shards, self.replicas, self.nodes)
        allocation.validate(self.allocation)
        started = [self._spawn_copy(s, n) for s, ns in self.allocation.items() for n in ns]
        for p in started:
            _wait_http(p.url, p.popen)
        routing_file = self.work / "routing.json"
        routing_file.write_text(json.dumps(self.routing_json(), indent=2))
        for i, port in enumerate(self.coord_ports):
            log = open(self.log_dir / f"coordinator{i}.log", "ab")
            self.coordinators.append(subprocess.Popen(
                [sys.executable, "-m", "dse.coordinator", "--routing", str(routing_file),
                 "--port", str(port), *self.coord_args],
                stdout=log, stderr=subprocess.STDOUT, env=_env()))
        for url, p in zip(self.urls, self.coordinators):
            _wait_http(url, p)
        self.wait_green()

    def stop(self) -> None:
        procs = [p.popen for p in self.procs.values()] + self.coordinators
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs.clear()
        self.coordinators = []

    def __enter__(self) -> "Cluster":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------ routing
    @property
    def urls(self) -> List[str]:
        return [f"http://127.0.0.1:{p}" for p in self.coord_ports]

    @property
    def url(self) -> str:
        return self.urls[0]

    def routing_json(self) -> dict:
        shards = {}
        for s, nodes in sorted(self.allocation.items()):
            shards[str(s)] = [{"node": n, "url": self.procs[(s, n)].url, "primary": i == 0}
                              for i, n in enumerate(nodes) if (s, n) in self.procs]
        return {"shards": shards}

    def push_routing(self) -> dict:
        for url in self.urls:
            r = httpx.put(url + "/cluster/routing", json=self.routing_json(), timeout=10)
            r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        return httpx.get(self.url + "/cluster/health", timeout=5).json()

    def wait_green(self, timeout: float = 30.0) -> dict:
        deadline = time.time() + timeout
        while True:
            h = self.health()
            if h["status"] == "green" or time.time() > deadline:
                return h
            time.sleep(0.1)

    # ----------------------------------------------------- fault injection
    def kill_copy(self, shard: int, node: str) -> None:
        """SIGKILL one shard copy (a crashed replica)."""
        p = self.procs[(shard, node)]
        if p.alive:
            os.kill(p.popen.pid, signal.SIGKILL)
            p.popen.wait()

    def kill_node(self, node: str) -> List[int]:
        """SIGKILL every copy hosted on ``node``.  Returns the affected shards."""
        hit = [s for (s, n) in self.procs if n == node]
        for s in hit:
            self.kill_copy(s, node)
        return hit

    def kill_shard(self, shard: int) -> None:
        """Kill all copies of one shard, forcing partial results."""
        for n in self.allocation[shard]:
            self.kill_copy(shard, n)

    def set_delay(self, delay_ms: float, jitter_ms: float = 0.0) -> None:
        for p in self.procs.values():
            if p.alive:
                httpx.post(p.url + "/admin/delay", json={"delay_ms": delay_ms, "jitter_ms": jitter_ms})

    # -------------------------------------------------- membership changes
    def _apply(self, plan: allocation.RebalancePlan) -> allocation.RebalancePlan:
        allocation.validate(plan.allocation)
        old_procs = dict(self.procs)
        for shard, node in plan.added:
            source = next((p for (s, _), p in old_procs.items() if s == shard and p.alive), None)
            if source is None:
                raise RuntimeError(f"no live copy of shard {shard} to recover from")
            p = self._spawn_copy(shard, node, recover_from=source.url)
            _wait_http(p.url, p.popen)
        self.allocation = plan.allocation
        self.push_routing()                       # route to the new copies first ...
        for shard, node in plan.removed:          # ... then retire the old ones
            p = self.procs.pop((shard, node), None)
            if p and p.alive:
                p.popen.terminate()
                p.popen.wait()
        return plan

    def add_node(self, node: str) -> allocation.RebalancePlan:
        self.nodes.append(node)
        return self._apply(allocation.rebalance(self.allocation, self.nodes, self.replicas))

    def remove_node(self, node: str) -> allocation.RebalancePlan:
        """Node left (or died): re-replicate its copies onto the remaining nodes."""
        self.nodes.remove(node)
        self.kill_node(node)
        return self._apply(allocation.rebalance(self.allocation, self.nodes, self.replicas))

    # ------------------------------------------------------------- client
    def search(self, query: str, k: int = 10, **kw) -> dict:
        r = httpx.post(self.url + "/search", json={"query": query, "k": k, **kw}, timeout=30)
        return r.json()

    def index_documents(self, docs: Sequence[Doc]) -> dict:
        r = httpx.post(self.url + "/documents", json={"docs": [list(d) for d in docs]}, timeout=120)
        r.raise_for_status()
        return r.json()

    def refresh_stats(self) -> dict:
        return httpx.post(self.url + "/admin/refresh", timeout=60).json()

    def events(self, since: float = 0.0) -> list:
        return httpx.get(self.url + "/events", params={"since": since}, timeout=5).json()


def _env() -> dict:
    # Make ``python -m dse...`` work from any working directory.
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env
