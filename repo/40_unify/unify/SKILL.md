---
name: unify
description: >
  Recursive context unification engine that decomposes a target directory into an
  actionable DAG-oriented module architecture, preserving cross-domain bridges and
  progressive loading order.
triggers:
  - unify
  - map codebase
  - analyze structure
  - decompose into modules
  - build dependency DAG
  - optimize ontology graph
---

# Unify

`unify` converts a directory into a minimal, progressive knowledge architecture:

after extraction → concept clustering → decomposition → DAG construction →
iterative optimization.

## Contract (What this skill guarantees)
- Input is a filesystem directory.
- Output is a deterministic set of artifacts in the target folder.
- Dependencies are explicit and versioned via this package’s manifest.
- Core modules are emitted before cluster modules, cluster modules before specific
  modules, then bridge modules.
- Runtime failure is explicit via exit status and short stderr diagnostics.

## Invocation

```bash
python scripts/unify.py <target_dir> --output ./unified
python scripts/unify.py <target_dir> --analyze-only
python scripts/unify.py <target_dir> --threshold 0.95 --max-iter 80
```

Required dependency at runtime:
`networkx>=3.0`.

| Arg | Default | Purpose |
| --- | --- | --- |
| target | required | Directory to ingest |
| `--output` | `./unified` | Destination for artifacts |
| `--threshold` | `0.95` | Optimization convergence target |
| `--max-iter` | `50` | Optimization cap |
| `--max-clusters` | `12` | Maximum clusters |
| `--block-size` | `500` | Max lines per specific module |
| `--chunk-size` | `4096` | Byte chunk size for very large files |
| `--analyze-only` | false | Emit analysis only |
| `--verbose` | false | Print phase telemetry |

## Outputs

- `index.md`: progressive loading manifest (tiers and traversal order)
- `analysis.json`: graph, clusters, bridge, and topology metadata
- `report.md`: score breakdown + topology health diagnostics
- `core/`: always-load modules
- `modules/`: domain cluster and specific modules
- `bridges/`: cross-cluster interfaces

## Optimization model

The optimizer balances 8 weighted dimensions:
- parsimony
- redundancy
- connectivity
- dag validity
- bridge coverage
- load efficiency
- modularity
- depth ratio
- score quality convergence

`--threshold` is interpreted against the composite structure score.

## Failure modes and deterministic recovery

- Empty input: exit code 1 with a parse/readability message.
- Weak graph signal: re-run with `--analyze-only`, inspect `report.md`, then
  increase `--max-clusters` or reduce `--threshold`.
- Repeated cycle churn: increase `--max-iter` and reduce `--block-size`.

## Runtime boundaries

- This skill does not execute project code.
- Binary or non-text assets outside supported extensions are skipped.
- Hidden/system folders are excluded (`.git`, `node_modules`, `venv`, etc.).
- Results are deterministic for fixed defaults and stable corpus snapshots.

## Interface and orchestration

`unify` is used downstream by:
- `graph` for additional structural compression
- `hierarchical-reasoning` and `ontolog` for ontology alignment
- `critique` for objective quality review

`unify` is platform-agnostic and emits plain Markdown/JSON contracts.

## Reference files

- `references/getting_started.md` — minimal operator runbook
- `references/examples.md` — invocation patterns
- `references/reference_api.md` — extracted API surface
- `references/contract_guide.md` — reproducibility and artifact contract
