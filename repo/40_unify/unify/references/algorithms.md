# Algorithm Reference

## Contents

- [TF-IDF Concept Extraction](#tf-idf-concept-extraction)
- [Graph Construction](#graph-construction)
- [Community Detection](#community-detection)
- [PageRank](#pagerank)
- [Topology Metrics](#topology-metrics)
- [DAG Construction](#dag-construction)
- [MCTS-UCB1 Optimization](#mcts-ucb1-optimization)
- [Convergence Criteria](#convergence-criteria)

---

## TF-IDF Concept Extraction

Each file is tokenized into terms after camelCase/snake_case splitting and
stop word removal. Source-weighted tokens:

| Source | Weight Multiplier |
|--------|-------------------|
| File name tokens | ×3 |
| Section/function names | ×2 |
| Body content | ×1 |

TF-IDF formula per document d and term t:

```
TF(t,d) = count(t,d) / |d|
IDF(t) = log(1 + N / DF(t))
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

Top 20 terms retained per file as the concept vector.

---

## Graph Construction

Nodes = files. Edges = weighted relatedness.

Edge detection priority (higher weight overrides lower):

1. **Explicit reference** (w=1.0): import/require, markdown links, wikilinks,
   name mentions in content
2. **Structural similarity** (w=0.8): ≥2 shared function/class names
3. **Concept overlap** (w=0.7): Jaccard(concepts_a, concepts_b) ≥ 0.15,
   capped at w = min(0.7, jaccard × 2)
4. **Domain co-membership** (w=0.5): same domain classification, only if
   no higher-weight edge exists
5. **Lexical co-occurrence** (w=0.3): shared rare bigrams (future extension)

Jaccard similarity:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

---

## Community Detection

Primary: Louvain method (modularity maximization) via networkx.

Fallback: Label propagation (if Louvain unavailable).

Cluster count capped at `--max-clusters` (default 12). Overflow clusters
merged into the largest compatible cluster by domain.

Domain assigned by majority vote of member file domains.

Hub = member with highest PageRank within cluster.

---

## PageRank

Standard PageRank with weight-aware damping:

```
PR(v) = (1 - d) / N + d × Σ(PR(u) × w(u,v) / Σ w(u,*))
```

Parameters:
- d = 0.85 (damping factor)
- 50 iterations
- Weight-aware: edge weights influence rank flow

Node role assignment:
- Hub: degree > mean + 2σ
- Bridge: betweenness centrality > 90th percentile
- Leaf: degree ≤ 2
- Orphan: degree = 0

---

## Topology Metrics

Three health indicators:

**η (eta) — Edge density**
```
η = |E| / |V|
Target: η ≥ 4
```
Measures interconnectedness. Below 4 indicates sparse, disconnected context.

**φ (phi) — Isolation ratio**
```
φ = |{v : deg(v) = 0}| / |V|
Target: φ < 0.20
```
Fraction of completely unconnected files. Above 0.20 indicates significant
orphaned content.

**κ (kappa) — Clustering coefficient**
```
κ = avg(C_v) for all v where deg(v) ≥ 2
C_v = 2 × |{e(u,w) : u,w ∈ N(v)}| / (deg(v) × (deg(v) - 1))
Target: κ > 0.30
```
Measures local clustering. Below 0.30 suggests weak community structure.

---

## DAG Construction

1. Directed edges derived from module dependencies:
   - Core → cluster_base (shared concepts)
   - Cluster_base → specific (cluster membership)
   - Cluster_base → bridge (cross-cluster connection)

2. Cycle detection via DFS. Cycles broken by removing the lowest-weight
   back-edge (max 100 iterations as safety).

3. Topological sort via Kahn's algorithm (iteratively remove nodes with
   in-degree 0).

4. Depth assignment via BFS from roots (in-degree = 0 nodes).

5. Critical path = longest dependency chain = max depth.

6. Progressive loading tiers:
   - Tier 0: Core modules (always loaded, ~100 tokens each)
   - Tier 1: Cluster bases (loaded on domain match)
   - Tier 2: Specific modules (loaded on demand)
   - Tier 3: Bridges (loaded on cross-domain query)

---

## MCTS-UCB1 Optimization

Monte Carlo Tree Search with Upper Confidence Bound for Trees.

**Selection** — UCB1 formula:

```
UCB1(a) = Q̄(a) + C × √(ln(N_parent) / N_a)
```

Where:
- Q̄(a) = average reward of action a
- C = √2 (exploration constant)
- N_parent = total visits
- N_a = visits to action a

**Actions** (restructuring moves):

| Action | Effect | When Beneficial |
|--------|--------|-----------------|
| merge_modules | Combine two similar modules | parsimony < 0.8 |
| prune_redundant | Remove fully redundant modules | redundancy < 0.9 |
| rebalance_cluster | Redistribute members | modularity < 0.7 |
| promote_to_core | Move high-reference module to core | load_efficiency < 0.8 |
| demote_from_core | Move low-reference core to specific | load_efficiency > 0.9 |
| add_bridge | Create bridge for uncovered cross-edge | bridge_coverage < 0.8 |
| remove_bridge | Remove low-value bridge | bridge_coverage > 0.9 |

**Scoring** — 8-dimensional objective:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Parsimony | 0.15 | Fewer modules relative to input |
| Redundancy | 0.20 | Lower concept overlap between modules |
| Connectivity | 0.15 | Graph density and isolation health |
| DAG validity | 0.10 | Acyclicity and root reachability |
| Bridge coverage | 0.10 | Cross-cluster edges have bridges |
| Load efficiency | 0.10 | Tier 0 < 5% of total content |
| Modularity | 0.10 | Community detection quality score |
| Depth ratio | 0.10 | Critical path ≤ ⌈log₂(n)⌉ |

Composite score = weighted sum, range [0, 1].

---

## Convergence Criteria

Pipeline terminates when ANY condition is met:

1. **Score threshold**: composite > `--threshold` (default 0.95)
2. **Plateau detection**: score delta < 0.001 for 5 consecutive iterations
3. **Budget exhaustion**: iteration count reaches `--max-iter` (default 50)

On plateau or budget exhaustion, the best-achieved structure is emitted
with a recommendation for manual review of lowest-scoring dimensions.
