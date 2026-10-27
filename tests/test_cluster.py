"""End-to-end tests against real shard and coordinator processes."""
import time

import pytest

from dse import data
from dse.analysis import analyze
from dse.build import build_single
from dse.cluster import Cluster


@pytest.fixture(scope="module")
def docs():
    return data.synthetic_corpus(1500, seed=5)


@pytest.fixture()
def cluster(tmp_path, docs):
    c = Cluster(tmp_path, num_shards=4, replicas=1, timeout_s=1.0, heartbeat_s=0.2, refresh_s=3600)
    c.build(docs)
    c.start()
    yield c
    c.stop()


def test_distributed_matches_single_node(cluster, docs):
    ref = build_single(docs)
    for _, q in data.synthetic_queries(docs, 20, seed=9):
        for mode in ("dfs", "periodic"):
            got = [(h["doc_id"], h["score"]) for h in cluster.search(q, 10, stats_mode=mode)["hits"]]
            assert got == ref.search(analyze(q), 10)


def test_failover_to_replica(cluster, docs):
    q = docs[0][1]
    before = cluster.search(q)
    cluster.kill_node(cluster.allocation[0][0])
    after = cluster.search(q)
    assert after["status"] == "ok" and not after["partial"]
    assert after["hits"] == before["hits"]
    time.sleep(0.6)
    assert cluster.health()["status"] == "yellow"


def test_partial_results_when_all_copies_down(cluster, docs):
    cluster.kill_shard(2)
    r = cluster.search(docs[0][1])
    assert r["partial"] is True and r["status"] == "partial"
    assert r["shards"]["missing"] == [2] and r["shards"]["successful"] == 3
    assert all(h["shard"] != 2 for h in r["hits"])
    strict = cluster.search(docs[0][1], allow_partial=False)
    assert strict["status"] == "error" and strict["shards"]["missing"] == [2]


def test_writes_replicate_and_recovery_copies_them(cluster):
    cluster.index_documents([("fresh-1", "zzyzx quokka"), ("fresh-2", "quokka marsupial")])
    for _ in range(4):   # reads rotate over both copies of each shard
        hits = {h["doc_id"] for h in cluster.search("quokka")["hits"]}
        assert hits == {"fresh-1", "fresh-2"}
    # A node leaves: its copies are rebuilt from live peers, including the new docs.
    gone = cluster.allocation[0][0]
    plan = cluster.remove_node(gone)
    assert plan.added and all(gone not in c for c in cluster.allocation.values())
    assert cluster.wait_green()["status"] == "green"
    for _ in range(4):
        assert {h["doc_id"] for h in cluster.search("quokka")["hits"]} == {"fresh-1", "fresh-2"}


def test_node_join_rebalances(cluster):
    plan = cluster.add_node("node-new")
    assert any(n == "node-new" for _, n in plan.added)
    assert cluster.wait_green()["status"] == "green"
    assert len(cluster.search("bax")["shards"]["missing"]) == 0
