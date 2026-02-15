# Algorithm Primitives

Deterministic pipeline:

1) Ingest
- scan supported text/code extensions
- encode each file into weighted term multiset
- duplicate IDs become path-stable (`a/b__file.md`)

2) Concept extraction
- tokenize(file name, sections, body)
- tf-idf over document-term counts
- keep top-K concepts/file

3) Similarity graph
- explicit references: max signal
- section overlap, concept Jaccard, domain overlap
- edge weight = capped additive sum
- keep only strongest K per node (KNN/prune)

4) Topology + clusters
- PageRank for salience
- Louvain (fallback label propagation)
- cluster caps + overflow merge

5) Decompose
- core: concepts across ≥3 files
- cluster bases: hub nodes
- specific: file blocks (line cap)
- bridges: cross-cluster weighted connectors

6) DAG + optimize
- orient dependencies for load order
- iterative cycle-breaking and MCTS-UCB1 moves
- objective: weighted 8-dim score

Core scoring dimensions
- parsimony, redundancy, connectivity, dag_validity, bridge_coverage,
  load_efficiency, modularity, depth_ratio

Execution semantics
- phase order: ingest → analyze → decompose → unify(DAG) → optimize.
- stable defaults: threshold 0.95, max_iter 50.
