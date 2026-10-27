# Results

| Folder | What it is | Use it for |
| --- | --- | --- |
| `kaggle_final/` | Kaggle run on MS MARCO (`parthplc/ms-marco-dataset`: 119,930 passages from `train.csv`, queries from `valid.csv`), after the periodic-refresh fix | **The final numbers in the report** |
| `kaggle_repeat_run/` | A second full "Save & Run All" of the same notebook after the fix; its log also contains the rebalancing demo output | Reproducibility check: same ranking table, throughput within 2%, p99 within 8% of `kaggle_final/` |
| `kaggle_before_refresh_fix/` | Same Kaggle setup, before the fix: the coordinator pulled every term's statistics from every shard every 5 s even when unused | Before/after comparison of tail latency |
| `local_synthetic/` | Laptop runs (10 cores) on the synthetic Zipf corpus, used during development | Sanity checks only |

Files in each folder:

* `ranking_table.csv` / `.md` — top-10 overlap, Kendall's τ, NDCG@10 and the share of identical rankings vs. the single-node index, per shard count, scorer and statistics strategy; `ranking_per_query.csv` has every query.
* `scaling_summary.csv`, `scaling.png` — QPS and mean / p50 / p95 / p99 latency for 1, 2, 4 and 8 shards; `scaling_requests.csv` has every request.
* `failover_summary.csv`, `failover.png`, `failover_events.json` — the replica-kill run: per-phase summary, latency timeline, coordinator event log.
* `kaggle_final/demo_output.txt` — output of the interactive demo, kill-node and rebalancing cells.
* `kaggle_final/kaggle_run.log` — Kaggle's log of the notebook run.

`kaggle_before_refresh_fix/scaling_summary.csv` was rebuilt from `scaling_requests.csv`, because the summary file was not downloaded from that run. The ranking tables of the two Kaggle runs are identical, as expected: the fix only affects latency.
