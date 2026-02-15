# Modular Patterns (Condensed)

Bridge-first pattern
- emit cluster-bounded modules first, then bridge modules
- enforce only required dependencies in core→cluster→specific→bridge flow

Core extraction pattern
- concepts across multiple files become shared abstractions
- keep only high-overlap/high-signal concepts

Decomposition pattern
- cluster_base = intra-cluster hub
- specific = file/blocks (split on size threshold)
- optional bridge-only module for cross-cluster evidence

Dependency orientation pattern
- specific depends on base
- cluster_base depends on core overlap
- bridge depends on both cluster bases

Cycle discipline
- detect-cycle loop
- remove weakest directed edge
- recompute depths

Load ordering pattern
- tier 0 core -> tier 1 cluster_base -> tier 2 specific -> tier 3 bridge
- prefer minimal always-on core to cap cold-start cost

Hyperedge compression pattern
- one bridge can represent multiple pairwise cross-links if shared concept set is stable
