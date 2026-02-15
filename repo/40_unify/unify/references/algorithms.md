# Algorithm Primitives

1. Ingest  
   - scan supported text/code files  
   - build stable file IDs (`<path>__<name>`)

2. Concept extraction  
   - tokenize filenames, headers, and body  
   - compute TF-IDF  
   - keep top-K terms/file

3. Similarity graph  
   - explicit references: highest signal  
   - section overlap + concept Jaccard + domain overlap  
   - cap/prune to top-K neighbors

4. Topology + clustering  
   - PageRank salience  
   - Louvain primary, label propagation fallback  
   - cluster cap + overflow merge

5. Decompose  
   - core: concepts in ≥3 files  
   - cluster bases: hubs  
   - specific: size-bounded blocks  
   - bridges: cross-cluster connectors

6. DAG + optimize  
   - orient deps for load order  
   - iterative cycle removal + MCTS-UCB1 moves  
   - maximize weighted 8-dim objective

Execution:
`ingest → analyze → decompose → unify(DAG) → optimize`
