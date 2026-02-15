# Modular Patterns

- Core-first orientation: `specific → cluster_base → core`; bridges are cross-cluster.
- Decompose by `cluster_base = intra_cluster_hub`, `specific = split(file, block_size)`.
- Optional bridge module only when cross-cluster evidence is stable.
- Cycles: detect + prune weakest edge, then recompute depth.
- Load order: `core → cluster_base → specific → bridge`.
- Hyperedge: replace repeated pairwise links with one bridge when concept-set stable.
