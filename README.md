# Elasticsearch-Inspired Distributed Search Engine

Distributed Systems course project — Sumit Kothalkar · Sowreddy Pranavi.
A small distributed full-text search engine: documents are hash-partitioned
across N shards, every shard is replicated on a different node, and a
coordinator answers queries by scatter-gather with BM25 / TF-IDF scored on
corpus-wide statistics. See `Elastic_search_corrected.pdf` for the proposal.

## Layout

| Path | What it is |
| --- | --- |
| `dse/analysis.py` | tokenizer, stop words, NLTK Porter stemmer (same at index and query time) |
| `dse/index.py` | per-shard in-memory inverted index (postings with tf, doc lengths), save/load, snapshot |
| `dse/scoring.py` | BM25 (k1 = 1.2, b = 0.75), TF-IDF, `CorpusStats` (N, avgdl, df) and merging |
| `dse/sharding.py` | `shard = crc32(doc_id) mod N` |
| `dse/build.py` | split a corpus into shard index files |
| `dse/allocation.py` | place primaries/replicas (no two copies of a shard on one node), rebalance on join/leave |
| `dse/shard_server.py` | FastAPI process serving one shard copy; injectable delay + jitter |
| `dse/coordinator.py` | scatter-gather, DFS / periodic / local statistics, heartbeat, failover, partial results, replicated writes |
| `dse/cluster.py` | launches and controls a local cluster; kill copies/nodes, add/remove nodes |
| `dse/loadgen.py` | closed-loop asyncio load generator |
| `dse/metrics.py` | top-k overlap, Kendall's τ, NDCG@k, latency percentiles |
| `experiments/` | `ranking.py`, `scaling.py`, `failover.py` — one script per deliverable |
| `notebooks/kaggle_cluster.ipynb` | Kaggle runner: demo + all three experiments |
| `tests/` | unit tests and end-to-end tests against real processes |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
pytest -q                                         # 27 tests, ~30 s
```

## Running

```python
from dse import data
from dse.cluster import Cluster

docs = data.load("collection.tsv", limit=100_000)   # or data.synthetic_corpus(20_000)
with Cluster("work/demo", num_shards=4, replicas=1) as c:
    c.build(docs)
    c.start()                                        # 8 shard processes + coordinator
    c.search("what is a distributed system", k=10, stats_mode="dfs")
    c.kill_node("node-0")                            # fails over to replicas
    c.kill_shard(1)                                  # -> "partial": true, "missing": [1]
```

HTTP API of the coordinator (`c.url`):

* `POST /search` `{"query", "k", "scorer": "bm25"|"tfidf", "stats_mode": "dfs"|"periodic"|"local", "allow_partial"}`
* `POST /documents` `{"docs": [[id, text], ...]}` — routed by hash, written to every live copy
* `GET /cluster/health` (green / yellow / red), `GET|PUT /cluster/routing`, `POST /admin/refresh`, `GET /events`

A search response carries `hits`, `status` (`ok` | `partial`), `partial`,
`shards: {total, successful, failed, missing}`, `served_by`, `retries`,
`took_ms` and `phases_ms`. With `allow_partial: false`, a missing shard returns
HTTP 503 instead.

Experiments (any corpus: `--corpus collection.tsv --queries queries.tsv --limit N`; default is synthetic):

```bash
python -m experiments.ranking  --shards 4 8                 # results/ranking_table.csv
python -m experiments.scaling  --shards 1 2 4 8 --coordinators 2   # results/scaling.png
python -m experiments.failover --shards 4 --duration 20     # results/failover.png
```

## Design notes

* **Statistics strategies.** `dfs` asks every shard for the query terms' df
  before scoring (one extra round trip; always exact). `periodic` pulls all
  statistics every `refresh_s` seconds (no extra round trip; stale after
  writes). `local` lets each shard use its own statistics (rankings drift).
* **Determinism.** Ties are broken by doc id and per-term contributions are
  summed in query order, so with global statistics the distributed scores are
  bit-identical to the single-node index — the tests assert exact equality.
* **Failover.** Reads are round-robined over healthy copies. A failed or
  timed-out call marks the copy down and is retried on the next copy; the
  heartbeat brings copies back. A copy that misses a write is marked stale and
  is not revived; a new copy is built by peer recovery (`/snapshot`).
* **Nodes are logical.** Each shard copy is its own process tagged with a node
  name; killing a node kills all its copies.

## Deviations from the write-up

* Coordinator → shard RPC uses **aiohttp** instead of httpx. Profiling showed
  httpx's connection pool using ~90 % of the coordinator's CPU (≈230 QPS for a
  4-shard fan-out vs ≈2100 QPS with aiohttp). FastAPI is unchanged; httpx is
  still used for control-plane calls.
* `Cluster(coordinators=n)` can run several coordinator processes (like
  Elasticsearch's coordinating-only nodes). On one machine a single Python
  coordinator otherwise becomes the bottleneck, hiding the shard-count effect.

## Limitations

All processes share one machine: traffic is loopback, and 8 shards × 2 copies
compete for a few cores, so throughput scaling depends on the corpus being
large enough for per-shard work to dominate. Injected delay/jitter stands in
for real network hops. Compare trends, not absolute latencies. Documents
cannot be updated or deleted once indexed.
