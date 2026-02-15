---
name: unify
description: >
  Recursive context unification engine that decomposes a target directory into an
  actionable DAG-oriented modular architecture with explicit cross-skill bridges.
triggers:
  - unify
  - map codebase
  - analyze structure
  - decompose into modules
  - build dependency DAG
  - optimize ontology graph
---

# Unify

`unify` converts a directory into a minimal progressive architecture:
`extract → cluster → decompose → bridge → optimize`.

## Contract

- Input: filesystem directory.
- Output: deterministic artifacts via `--output`.
- Order: `core → cluster_base → specific → bridge`.
- Failures are explicit via exit code + stderr.

## Invocation

```bash
python scripts/unify.py <target_dir> --output ./unified
python scripts/unify.py <target_dir> --analyze-only
python scripts/unify.py <target_dir> --threshold 0.95 --max-iter 80
```

| Arg | Default | Purpose |
| --- | --- | --- |
| target | required | Directory to ingest |
| `--output` | `./unified` | Destination for artifacts |
| `--threshold` | `0.95` | Target score convergence |
| `--max-iter` | `50` | Optimization budget |
| `--max-clusters` | `12` | Maximum clusters |
| `--block-size` | `500` | Max lines per specific module |
| `--chunk-size` | `4096` | Byte chunk size for large files |
| `--analyze-only` | false | Emit analysis only |
| `--verbose` | false | Print phase telemetry |

## Outputs

- `index.md`: progressive manifest
- `analysis.json`: graph/cluster/bridge metadata
- `report.md`: score and diagnostics
- `core/`, `modules/`, `bridges/`: emitted module tiers

## Optimization model

Objective (9): parsimony, redundancy, connectivity, DAG validity,
bridge coverage, load efficiency, modularity, depth ratio, convergence.

## Deterministic failure/recovery
- Empty input: exit `1` + parse error.
- Weak signal: rerun `--analyze-only`, `+max-clusters`, or `-threshold`.
- Cycle churn: `+max-iter`, `-block-size`.

## Runtime boundaries

- No source execution.
- Binary assets skipped.
- Excludes `.git`, `node_modules`, `venv`, `dist`, `build`.
- Fixed defaults + fixed snapshot => deterministic output.
- Runtime dependency: `networkx>=3.0`.

## Orchestration usage

Downstream: `graph`, `hierarchical-reasoning`, `ontolog`, `critique`.
`unify` emits portable Markdown/JSON contracts.
