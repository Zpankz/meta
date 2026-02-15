# Performance Profile (Condensed)

Complexity envelope
- parsing/tokenization: O(n * m)
- graph pairwise sim: reduced by KNN prune after pruning
- clustering/metrics: O(E log V) typical
- optimize loop: O(I * A) where I=iterations, A=actions

Density control
- KNN target η≈8 (or user-tuned via cluster balance)
- stronger edge threshold reduces false joins

Tuning
- high noise: lower threshold, raise `max_clusters`, increase `max_iter`
- large corpus: raise `chunk-size`, lower `max-clusters`
- slow runs: reduce `--max-iter` before quality threshold is reached

Failure signatures
- zero bridges: inspect references + threshold/cross-domain signals
- orphan-heavy graph: lower concept filters or expand stopword exclusions
- unstable output: pin corpus snapshot and rerun with same flags

Regression guard
- compare `analysis.json` hashes between runs
- compare `final_score`, `topo.eta`, `orphans`, bridge count
