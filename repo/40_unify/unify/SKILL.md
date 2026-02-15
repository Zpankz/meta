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

## Interface Contract

### Inputs

- `target` (required positional argument): directory containing source files.
- `--output` (default: `./unified`): destination directory for generated artifacts.
- `--analyze-only`: stop after `analysis.json` and skip module synthesis.
- `--threshold` (default: `0.95`): quality convergence target.
- `--max-iter` (default: `50`): MCTS iteration cap.
- `--max-clusters` (default: `12`): cap for detected domain clusters.
- `--block-size` (default: `500`): hard cap for lines per final specific module.
- `--chunk-size` (default: `4096`): bytes per chunk for large files.
- `--verbose`: enable structured progress logs to stderr.

### Deterministic Outputs

When successful, unify emits:

- `index.md` — tiered loading manifest (core, domain modules, bridges).
- `analysis.json` — weighted graph, clustering assignments, confidence scores.
- `report.md` — run summary with scoring, module counts, and warnings.
- `core/` — shared abstractions always loaded first.
- `modules/` — domain-anchored and specific modules.
- `bridges/` — cross-domain connector modules.

### Return Semantics

- `exit code 0` indicates successful run and a valid persisted hierarchy.
- `exit code 1` indicates command-line validation failure (missing path, unreadable target).
- `exit code 2` indicates graph build failure after deterministic retry.
- `exit code 3` indicates optimization convergence timeout (`--max-iter` exhausted).
- Non-zero exits always include a concise stderr cause and a path to diagnostics in
  the output directory.

### Invocation Matrix

| Use case | Command pattern | Expected result |
|----------|----------------|-----------------|
| quick map | `python scripts/unify.py <corpus> --analyze-only` | `analysis.json` + `report.md`, no module emit |
| full decomposition | `python scripts/unify.py <corpus> --output ./unified` | full hierarchy + index |
| stricter convergence | `python scripts/unify.py <corpus> --threshold 0.98 --max-iter 120` | conservative topology, more iterations |
| diagnostic mode | `python scripts/unify.py <corpus> --verbose` | detailed phase-level telemetry |

## Failure Modes and Recovery

- **Empty/insufficient corpus**: if fewer than 10 meaningful tokens remain after parsing, run with `--analyze-only` and review `report.md` for skip reasons.
- **Degenerate graph**: if all pairwise scores are flat, increase `--chunk-size` to improve term signal and rerun with `--max-clusters`.
- **Bridge explosion**: if bridge count rises rapidly, increase `--threshold` or decrease `--max-iter` to reduce overfitting.
- **Cycle persistence**: when DAG depth remains unstable, compare `analysis.json` SCC view against expected canonical dependencies before lowering threshold.
- **Runtime stalls**: if runtime is high for large corpora, reduce `--block-size` and increase `--chunk-size` to reduce parse cost.

## Integration with Other Skills

- **graph**: Unify output DAG feeds directly into graph skill's 
  k-bisimulation compression for further hierarchy reduction.
- **abduct**: Run abductive analysis on unify's analysis.json to 
  identify latent architectural patterns.
- **critique**: Apply multi-lens evaluation to the convergence score 
  breakdown for structural quality assessment.
- **rpp**: Feed unify clusters into RPP for Pareto-optimized 
  hierarchical knowledge compression.

### Cross-agent Interoperability

`unify` is platform-agnostic and remains compatible with:

- Claude Code (`.claude`-compatible package shape)
- Codex (`.codex` tool entry and prompt-ready layout)
- Gemini and other CLI ingest paths through plain Markdown+JSON outputs
- Agent-skills runtime by referencing `.index` outputs as structured control surfaces

Keep `SKILL.md` + `references/` as the stable contract; scripts are treated as
capability extensions consumed only by explicit invocation.

## Canonical Artifact Checklist

- `SKILL.md` updated with interface contract and failure matrix.
- `reference` outputs include architecture assumptions and domain patterns.
- `scripts/unify.py` remains authoritative for execution.
- `manifest.json` and `bridge.json` declare deterministic runtime dependencies.
- `.index` artifacts generated by `meta` remain authoritative for orchestration.

## Reference Files

- `references/algorithms.md` — Detailed algorithm specifications
- `references/patterns.md` — Architecture patterns and usage examples
- `references/contract_guide.md` — Interface compatibility and contract scope
- `references/integration_playbook.md` — Orchestration workflows and dependency handoffs
- `references/usage_gallery.md` — Concrete invocation examples by scenario
- `references/performance_profile.md` — Complexity model and tuning recipes
- `references/getting_started.md` — Deterministic bootstrap and operating procedures
- `references/examples.md` — Scenario-level examples and expected outputs
- `references/reference_api.md` — Public API surface extracted from v3 analysis

## Skill-Seekers v3-Integrated Control Layer

`unify` has been optimized using the v3 pipeline in `skill-seekers`:

1. `skill-seekers analyze` with:
   - `--preset comprehensive`
   - `--enhance --enhance-level 2`
   - API reference + dependency + documentation + config extraction passes
2. `skill-seekers quality` evaluation loops to converge on score saturation.
3. `skill-seekers package` validation for multi-platform payload generation (`gemini`, `openai`, `markdown`) while preserving canonical `.skill` archive form.
4. `skill-seekers extract-test-examples` verification to confirm test-usage coverage assumptions.
5. `skill-seekers multilang --detect` validation for language discovery.

This loop is preserved as a maintenance pattern:

- Re-run `analyze` whenever `scripts/` or `SKILL.md` contracts change.
- Re-run `quality --report` after any interface additions.
- Update v3 reference payloads in `references/` only through deterministic generation.

## Canonical v3 Reference Surface

- `references/getting_started.md` — operational bootstrap and invocation order
- `references/examples.md` — scenario examples and expected artifacts
- `references/reference_api.md` — function and class API surface extraction from `scripts/unify.py`
- `quality_report.json` — current measured quality summary (refreshed per run)
