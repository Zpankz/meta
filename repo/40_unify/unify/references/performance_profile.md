# Performance and Tuning Profile

## Complexity Model

The baseline complexity is driven by:

- document count
- unique token volume
- TF-IDF vector dimensions
- Louvain partitioning effort on KNN-pruned edges

Approximate behavior:

- `n` files and `m` extracted concepts produce O(n*m) parse overhead for feature extraction.
- KNN pruning limits graph adjacency to O(k*n) and keeps graph search tractable.
- MCTS depth grows with `block-size` and `max-iter` as a control lever, not as a fixed accuracy guarantee.

## Tuning Recipes

- Larger corpora: raise `chunk-size` and lower `max-clusters` to reduce churn.
- Deeply hierarchical codebases: raise `max-clusters` and enable explicit `--verbose`.
- Strict reproducibility: fix seeds, keep defaults unchanged across runs, compare `index.md` diffs only.

## Failure Signatures

- `No modules emitted`: often indicates `--analyze-only` or empty signal extraction.
- `Shallow bridges`: tune threshold upward and verify reference graph quality.
- `Excessive bridge fanout`: consider lowering `max-clusters` and re-optimizing after filtering.
