# Project Proposal & Write-Up: Elasticsearch-Inspired Distributed Search Engine

**Project ID:** 37  
**Title:** Elasticsearch-Inspired Distributed Search Engine  
**Domain:** Distributed Systems, Information Retrieval, Performance & Reliability Engineering  

---

## 1. Project Description

Modern search engines operate over web-scale document collections that far exceed the storage, memory, and throughput capabilities of any single physical machine. Distributed search systems—such as Elasticsearch and Apache Solr—address these limitations by partitioning inverted indexes across multiple independent nodes (shards) and coordinating concurrent query execution across the cluster.

This project focuses on the architecture, implementation, and empirical evaluation of an **Elasticsearch-Inspired Distributed Search Engine**. The system adopts a decoupled, multi-tier coordinator-worker architecture:

* **Coordinator (Gateway Node):** Serves as the client entry point, exposing search and indexing APIs. It maintains cluster routing tables and active shard health state. When a query is received, it executes the **scatter-gather pattern**: asynchronously scattering the query to relevant shards, awaiting responses with timeout thresholds, and gathering/merging intermediate top-$k$ ranked candidates using a bounded min-heap before streaming the global top-$k$ results to the client.
* **Shard Worker Nodes (Primary & Replica):** Each shard hosts an isolated slice of the global inverted index (lexicon dictionary, posting lists with term frequencies, document lengths, and positions). Each primary shard is paired with active secondary replicas that mirror the index state for high availability and load distribution.
* **Consistent Distributed Ranking:** Information retrieval ranking algorithms such as **BM25** and **TF-IDF** depend on global corpus statistics, specifically Inverse Document Frequency ($\text{IDF} = \ln(1 + \frac{N - n + 0.5}{n + 0.5})$) and average document length ($\text{avgdl}$). In a partitioned distributed environment, calculating BM25 strictly on local shard statistics causes non-deterministic ranking divergence. This project implements a **two-phase query execution (Distributed Frequency Search / DFS) or periodic statistics synchronization** mechanism to ensure mathematical equivalence with a centralized monolithic search index.
* **Fault Tolerance, Failover, and Partial Results:** The coordinator incorporates proactive heartbeat failure detection. If a primary shard becomes unresponsive, query traffic is routed to surviving replicas. If all replicas of a shard are unavailable, the system supports configurable degradation semantics: either failing fast or returning an explicit partial-results envelope (`status: "PARTIAL_RESULT"` with metadata identifying missing shards and confidence bounds).

---

## 2. Goals

The key objectives of this project are divided into four core areas:

### A. Distributed Indexing & Query Coordination
1. **Document Sharding & Partitioning:** Implement deterministic hash-based document routing ($\text{hash}(\text{doc\_id}) \pmod S$) to partition a benchmark corpus across $S$ independent shards.
2. **Inverted Index Implementation:** Develop a performant in-memory and disk-backed inverted index on each shard supporting efficient term lookup and postings list intersection.
3. **Scatter-Gather Coordination:** Implement asynchronous, non-blocking scatter-gather query orchestration at the coordinator with thread-safe top-$k$ result aggregation.

### B. Consistent Distributed Ranking & Mathematical Parity
4. **Uniform Global Scoring:** Implement BM25 and TF-IDF scoring functions backed by synchronized global collection statistics ($N$, $\text{avgdl}$, and global term frequencies $\text{df}_t$).
5. **Single-Node Rank Parity Validation:** Quantitatively verify that query results from the multi-shard cluster match a single-node reference index identically (achieving NDCG@$k = 1.0$ and Kendall’s rank correlation $\tau = 1.0$).

### C. Replication, Fault Tolerance & Graceful Degradation
6. **Primary-Replica Synchronization:** Implement shard index replication with hot-standby replicas.
7. **Dynamic Failover Routing:** Detect shard failures via timeouts/heartbeats and transparently redirect client queries to surviving replicas.
8. **Graceful Partial-Result Degradation:** Demonstrate resilience when an entire shard is lost by delivering partial search results accompanied by diagnostic metadata (`total_shards`, `successful_shards`, `failed_shards`).

### D. Quantitative Benchmarking & Systems Analysis
9. **Performance Profiling Across Shard Configurations:** Measure median latency ($p50$), tail latencies ($p95, p99$), and query throughput (QPS) under varying shard counts ($S \in \{1, 2, 4, 8\}$).
10. **Tail-Latency Straggler Analysis:** Characterize the **scatter-gather tax**—the impact of slowest-shard latency amplification and network serialization overheads as cluster scale increases.

---

## 3. Tools and Technologies

| Layer | Tools & Technologies | Role & Purpose |
| :--- | :--- | :--- |
| **Programming Language** | **Python 3.11+** (or **Go**) | Core implementation of coordinator and shard daemons using asynchronous primitives (`asyncio` / goroutines). |
| **Networking & IPC** | **gRPC & Protocol Buffers** / **FastAPI (HTTP/2)** | Strongly-typed, low-overhead binary RPC communication between coordinator and shard nodes for high-speed scatter-gather operations. |
| **Text Processing & Tokenization** | Custom Lexer + `regex`, `PyStemmer` / `NLTK` | Document normalization, Porter Stemming, stopword removal, and inverted index token pipeline. |
| **Ranking Algorithms** | **BM25** ($k_1 \in [1.2, 2.0], b=0.75$) & **TF-IDF** | Term-weighting and probabilistic document relevance scoring models. |
| **Containerization & Orchestration** | **Docker & Docker Compose** | Multi-node cluster simulation with isolated network bridges, allowing reproducible startup of coordinator, primary shards, and replicas. |
| **Benchmarking & Load Testing** | **Locust** / **wrk** | Multi-threaded load generation to profile latency distributions ($p50, p90, p95, p99$) and throughput under sustained request concurrency. |
| **Profiling & System Metrics** | `cProfile`, `py-spy`, `psutil` | CPU profiling, memory utilization tracking, and identifying serialization / merge bottlenecks. |
| **Data Analysis & Visualization** | **Pandas, Matplotlib, Seaborn** | Generating empirical graphs (throughput vs. shard count, latency percentiles, and failover timelines). |
| **Benchmark Datasets** | **MS MARCO** (Passage Ranking) / **Cranfield** / **Antique** | Standard IR collections with realistic document lengths, vocabularies, and pre-formulated query sets. |

---

## 4. Special System Requirements

To simulate a true distributed environment and execute rigorous performance profiling, the project requires the following computing environment:

1. **Multi-Node / Multi-Container Environment:**
   * **Docker Network Isolation:** Deployment using Docker Compose with bridged networking to emulate isolated network stacks for each shard and coordinator.
   * **Network Emulation (Traffic Control):** Use of Linux `tc` / `netem` to inject synthetic latency, packet jitter, and bandwidth limits to model realistic data center network hops between the coordinator and shards.
   * **RCE / Cloud / Multi-Core Server:** Access to a multi-core Linux host or university Remote Computing Environment (RCE) / cloud VM (e.g., AWS EC2 `c5.2xlarge` or local workstation with $\ge 8$ physical cores, $\ge 16$ GB RAM) to ensure that running multiple shard daemons concurrently does not suffer from synthetic CPU thread throttling or scheduler starvation during tail-latency ($p99$) measurements.
2. **Process Orchestration & Fault-Injection Tooling:**
   * Programmatic control over node lifecycles (via Docker socket API or POSIX signals) to inject hard crashes (`SIGKILL`), delayed responses, and network drops during active query streams.

---

## 5. Expected Outcomes

The project will deliver a complete software implementation, automated test pipelines, and an empirical research report:

### A. Core Software Deliverables
1. **Fully Functional Distributed Search Cluster:** A production-grade coordinator and shard worker system supporting dynamic cluster formation, hash-based document routing, and parallel query execution.
2. **Consistent Distributed Ranking Module:** A two-phase statistics collection engine providing verified score parity with a single-node ground truth index.
3. **Resilient Failover & Degradation Engine:** An operational failure handling system with automatic replica query re-routing and structured partial-result metadata reporting.
4. **Single-Command Orchestration:** A turnkey `docker-compose.yml` environment for rapid deployment and automated teardown across arbitrary cluster topologies ($1, 2, 4, 8$ shards with $R \ge 1$ replicas).

### B. Experimental Benchmarks & Technical Evaluation
1. **Rank Fidelity Proof:** Tabulated evaluations confirming identical ranking outputs (NDCG@10 $= 1.0$) between the distributed cluster and monolithic single-node index.
2. **Scalability & Latency Profiles:**
   * **Throughput (QPS):** Quantifying parallel query processing capacity gains as shard count increases.
   * **Tail Latency Analysis:** Highlighting the impact of the **straggler problem** on $p95$ and $p99$ tail latencies in scatter-gather queries.
   * **Scatter-Gather Trade-Off:** Identifying the empirical workload threshold where sharding overhead surpasses parallel search gains.
3. **Fault Injection Case Study:** Recorded traces demonstrating cluster behavior during shard failure: successful zero-downtime failover to replicas and explicit partial-result status codes when all shard copies are terminated.
