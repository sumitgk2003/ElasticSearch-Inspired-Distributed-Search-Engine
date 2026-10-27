 shards scorer                   strategy  top10_overlap  kendall_tau  ndcg@10  identical_rankings
      4   bm25               global (DFS)         1.0000       1.0000   1.0000               1.000
      4   bm25           periodic (fresh)         1.0000       1.0000   1.0000               1.000
      4   bm25                shard-local         0.9095       0.7624   0.9650               0.035
      4  tfidf               global (DFS)         1.0000       1.0000   1.0000               1.000
      4  tfidf           periodic (fresh)         1.0000       1.0000   1.0000               1.000
      4  tfidf                shard-local         0.8805       0.7434   0.9504               0.040
      4   bm25 global (DFS), after writes         1.0000       1.0000   1.0000               1.000
      4   bm25           periodic (stale)         0.9930       0.9760   0.9982               0.695
      4   bm25  shard-local, after writes         0.9150       0.7759   0.9680               0.035
      4  tfidf global (DFS), after writes         1.0000       1.0000   1.0000               1.000
      4  tfidf           periodic (stale)         0.9920       0.9835   0.9984               0.830
      4  tfidf  shard-local, after writes         0.8850       0.7407   0.9536               0.040
      8   bm25               global (DFS)         1.0000       1.0000   1.0000               1.000
      8   bm25           periodic (fresh)         1.0000       1.0000   1.0000               1.000
      8   bm25                shard-local         0.8575       0.6510   0.9330               0.000
      8  tfidf               global (DFS)         1.0000       1.0000   1.0000               1.000
      8  tfidf           periodic (fresh)         1.0000       1.0000   1.0000               1.000
      8  tfidf                shard-local         0.8590       0.6718   0.9349               0.020
      8   bm25 global (DFS), after writes         1.0000       1.0000   1.0000               1.000
      8   bm25           periodic (stale)         0.9930       0.9760   0.9982               0.695
      8   bm25  shard-local, after writes         0.8715       0.6797   0.9436               0.015
      8  tfidf global (DFS), after writes         1.0000       1.0000   1.0000               1.000
      8  tfidf           periodic (stale)         0.9920       0.9835   0.9984               0.830
      8  tfidf  shard-local, after writes         0.8570       0.6753   0.9406               0.005