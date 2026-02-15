# Performance Profile

- Parse/tokenize: `O(n*m)`
- Similarity: KNN-pruned pairwise
- Cluster/metrics: `O(E log V)` typical
- Optimize: `O(I*A)` (`I` iterations, `A` actions)

Heuristics:
- `η≈8` KNN target; higher threshold reduces false joins
- noise↑: lower threshold, raise `max_clusters`, raise `max_iter`
- corpora↑: raise `chunk-size`, lower `max-clusters`
- speed↑: cut `max-iter` before threshold reached

Failure signatures:
- bridges=0 → inspect references and cross-domain thresholds
- orphans↑ → broaden terms/stopwords
- unstable output → pin snapshot + fixed flags

Guardrails:
- hash `analysis.json` between runs
- monitor final score, `topo.eta`, orphans, bridge count
