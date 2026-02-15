---
name: unify
description: >
  Recursive context unification engine that ingests file directories, extracts
  concepts via TF-IDF, builds KNN-pruned knowledge graphs, clusters by domain
  using Louvain detection, decomposes into tiered modules, and converges via
  MCTS-UCB1 optimization. Use when analyzing, mapping, or restructuring a
  codebase, skill ecosystem, or file collection into a navigable hierarchy.
  Triggers on "unify", "map codebase", "analyze structure", "decompose into
  modules", or requests to understand cross-file relationships.
---

# Unify — Recursive Context Unification Engine

Transforms any directory of files into a scored, tiered module hierarchy 
with dependency DAG and cross-domain bridge detection.

## Quick Start

```bash
# Basic unification
python scripts/unify.py /path/to/target --output ./unified --verbose

# Analysis only (no decomposition)
python scripts/unify.py /path/to/target --analyze-only

# Custom thresholds
python scripts/unify.py /path/to/target --threshold 0.90 --max-iter 100
```

**Requirements**: `pip install networkx --break-system-packages`

## Pipeline Phases

| Phase | Name | Function |
|-------|------|----------|
| Φ0 | Ingest | Scan files, extract concepts via TF-IDF, classify domains |
| Φ1 | Analyze | Build weighted graph, KNN prune to η≈8, cluster via Louvain |
| Φ2 | Decompose | Split into tiered modules: core → cluster base → specific → bridge |
| Φ3 | Unify | Construct DAG with topological ordering and cycle breaking |
| Φ4 | Optimize | MCTS-UCB1 convergence with 8 structural actions |

## Output Structure

```
unified/
├── index.md            # Master index with tiered loading instructions
├── analysis.json       # Full structural analysis data
├── report.md           # Human-readable report with scores
├── core/               # Tier 0: Always-load shared abstractions
├── modules/            # Tier 1-2: Cluster bases and specific modules
│   ├── {domain}/       # Grouped by detected domain
│   └── ...
└── bridges/            # Tier 3: Cross-domain connection modules
```

## Scoring Dimensions (8)

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| parsimony | 0.15 | Module/file ratio via sigmoid curve |
| redundancy | 0.20 | Intra-cluster concept overlap (lower = better) |
| connectivity | 0.15 | Graph density η ≥ 4 and isolation φ < 0.20 |
| dag_validity | 0.10 | Acyclicity and root-reachability |
| bridge_coverage | 0.10 | Cross-cluster edges covered by bridge modules |
| load_efficiency | 0.10 | Tier 0 content < 5% of total |
| modularity | 0.10 | Louvain modularity on pruned graph |
| depth_ratio | 0.10 | Critical path ≤ ceil(log₂(n)) |

Composite = Σ(score × weight). Target: **≥ 0.95**.

## CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `target` | required | Directory to unify |
| `--output` | `./unified` | Output directory |
| `--threshold` | `0.95` | Convergence target |
| `--max-iter` | `50` | Max MCTS iterations |
| `--max-clusters` | `12` | Maximum cluster count |
| `--block-size` | `500` | Max lines per specific module |
| `--chunk-size` | `4096` | Bytes per chunk for large files |
| `--analyze-only` | false | Stop after Φ1, write analysis.json |
| `--verbose` | false | Print progress to stderr |

## Domain Classification

14 base domains with adaptive corpus-specific keyword discovery:

logic, data, interface, network, system, test, model, text,
reasoning, skill, medical, graph, security, learning

Files matching < 1 domain keyword are classified as `general`.
Adaptive detection adds corpus-specific bigrams post-ingest.

## MCTS Actions (8)

| Action | Target Dimension |
|--------|-----------------|
| merge_modules | parsimony |
| split_module | modularity |
| prune_redundant | redundancy |
| rebalance_cluster | modularity |
| promote_to_core | load_efficiency |
| demote_from_core | parsimony |
| add_bridge | bridge_coverage |
| remove_bridge | bridge_coverage |

UCB1 exploration-exploitation with gap-scaled stochastic deltas.

## Key Design Decisions

- **KNN edge pruning** (k=8): Controls graph density to η≈8, enabling 
  meaningful community structure. Without this, dense concept-overlap 
  graphs (η>40) yield near-zero modularity.
- **Sigmoid parsimony**: `1/(1+exp(2*(ratio-2)))` — graceful degradation. 
  Ratio 1.0→1.0, 1.5→0.73, 2.0→0.50, 3.0→0.12.
- **Intra-cluster redundancy only**: O(k²) per cluster instead of O(n²) 
  all-pairs, avoiding artificial inflation from cross-domain comparisons.
- **Word-boundary reference matching**: Names ≥4 chars use `\b` regex 
  boundaries, preventing 'core' from matching 'score', etc.

## Integration with Other Skills

- **graph**: Unify output DAG feeds directly into graph skill's 
  k-bisimulation compression for further hierarchy reduction.
- **abduct**: Run abductive analysis on unify's analysis.json to 
  identify latent architectural patterns.
- **critique**: Apply multi-lens evaluation to the convergence score 
  breakdown for structural quality assessment.
- **rpp**: Feed unify clusters into RPP for Pareto-optimized 
  hierarchical knowledge compression.

## Reference Files

- `references/algorithms.md` — Detailed algorithm specifications
- `references/patterns.md` — Architecture patterns and usage examples
