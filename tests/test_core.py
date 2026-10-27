import heapq

import pytest

from dse import allocation, data
from dse.analysis import analyze
from dse.build import build_single
from dse.index import InvertedIndex
from dse.metrics import kendall_tau_at_k, ndcg_at_k, overlap_at_k
from dse.scoring import CorpusStats
from dse.sharding import shard_for


@pytest.fixture(scope="module")
def corpus():
    docs = data.synthetic_corpus(2000, seed=3)
    return docs, [q for _, q in data.synthetic_queries(docs, 60, seed=4)]


def sharded(docs, n):
    shards = [InvertedIndex() for _ in range(n)]
    for d, t in docs:
        shards[shard_for(d, n)].add(d, t)
    return shards


def distributed_search(shards, terms, k, scorer="bm25", global_stats=True):
    stats = CorpusStats.merge(s.local_stats(terms) for s in shards) if global_stats else None
    hits = [h for s in shards for h in s.search(terms, k, stats, scorer)]
    return heapq.nsmallest(k, hits, key=lambda h: (-h[1], h[0]))


def test_analyzer():
    assert analyze("The Running dogs, and THE cats!") == ["run", "dog", "cat"]


def test_shard_routing_is_stable_and_balanced():
    ids = [f"doc{i}" for i in range(4000)]
    assert [shard_for(i, 4) for i in ids[:5]] == [shard_for(i, 4) for i in ids[:5]]
    counts = [sum(shard_for(i, 4) == s for i in ids) for s in range(4)]
    assert min(counts) > 850


def test_duplicate_doc_is_rejected():
    idx = InvertedIndex()
    assert idx.add("a", "hello world")
    assert not idx.add("a", "again")
    assert len(idx) == 1


def test_save_load_roundtrip(tmp_path, corpus):
    docs, queries = corpus
    idx = build_single(docs[:300])
    idx.save(tmp_path / "i.pkl")
    again = InvertedIndex.load(tmp_path / "i.pkl")
    for q in queries[:10]:
        assert idx.search(analyze(q), 10) == again.search(analyze(q), 10)
    assert InvertedIndex.from_state(idx.to_state()).search(analyze(queries[0]), 5) == idx.search(analyze(queries[0]), 5)


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("n_shards", [1, 2, 4, 8])
def test_global_stats_match_single_node_exactly(corpus, scorer, n_shards):
    docs, queries = corpus
    single = build_single(docs)
    shards = sharded(docs, n_shards)
    for q in queries:
        terms = analyze(q)
        assert distributed_search(shards, terms, 10, scorer) == single.search(terms, 10, scorer=scorer)


def test_local_stats_drift(corpus):
    docs, queries = corpus
    single = build_single(docs)
    shards = sharded(docs, 8)
    diffs = sum(distributed_search(shards, analyze(q), 10, global_stats=False) != single.search(analyze(q), 10)
                for q in queries)
    assert diffs > 0


def test_metrics():
    ref = list("abcdefghij")
    assert overlap_at_k(ref, ref, 10) == kendall_tau_at_k(ref, ref, 10) == ndcg_at_k(ref, ref, 10) == 1.0
    swapped = list("bacdefghij")
    assert overlap_at_k(ref, swapped, 10) == 1.0
    assert 0.8 < kendall_tau_at_k(ref, swapped, 10) < 1.0
    assert 0.9 < ndcg_at_k(ref, swapped, 10) < 1.0
    assert overlap_at_k(ref, list("abcdexyzuv"), 10) == 0.5
    assert ndcg_at_k(ref, list("xyzuvwqrst"), 10) == 0.0


def _no_shared_nodes(alloc):
    allocation.validate(alloc)


@pytest.mark.parametrize("shards,replicas,nodes", [(1, 1, 2), (4, 1, 4), (8, 1, 4), (8, 2, 5), (3, 1, 2)])
def test_allocation_constraints(shards, replicas, nodes):
    names = [f"n{i}" for i in range(nodes)]
    alloc = allocation.allocate(shards, replicas, names)
    _no_shared_nodes(alloc)
    assert all(len(c) == replicas + 1 for c in alloc.values())
    loads = [sum(n in c for c in alloc.values()) for n in names]
    assert max(loads) - min(loads) <= 1


def test_rebalance_on_node_leave_promotes_and_rereplicates():
    names = ["n0", "n1", "n2", "n3"]
    alloc = allocation.allocate(4, 1, names)
    plan = allocation.rebalance(alloc, ["n1", "n2", "n3"], replicas=1)
    _no_shared_nodes(plan.allocation)
    assert all(len(c) == 2 and "n0" not in c for c in plan.allocation.values())
    lost_primary = [s for s, c in alloc.items() if c[0] == "n0"]
    assert {s for s, _ in plan.promoted} == set(lost_primary)
    # survivors keep their copies: only copies that lived on n0 are re-created
    assert len(plan.added) == sum("n0" in c for c in alloc.values())


def test_rebalance_on_node_join_moves_copies():
    alloc = allocation.allocate(4, 1, ["n0", "n1"])
    plan = allocation.rebalance(alloc, ["n0", "n1", "n2"], replicas=1)
    _no_shared_nodes(plan.allocation)
    loads = [sum(n in c for c in plan.allocation.values()) for n in ["n0", "n1", "n2"]]
    assert max(loads) - min(loads) <= 1 and loads[2] > 0
    assert len(plan.added) == len(plan.removed)


def test_rebalance_reports_unassigned_when_too_few_nodes():
    plan = allocation.rebalance({0: ["a", "b"]}, ["a"], replicas=1)
    assert plan.unassigned == [0] and plan.allocation == {0: ["a"]}
