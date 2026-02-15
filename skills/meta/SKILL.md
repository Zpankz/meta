---
name: meta
description: "Run and validate recursive control-plane synthesis for the skill corpus: dependency graph extraction, DAG/ontology emission, issue tracking, and release packaging across all SKILL assets. This skill orchestrates decomposition, cross-skill synthesis, and deterministic governance with hard/soft dependency normalization."
context: fork
agent: governance
---
# Meta Skill

`meta` is the governance control-plane orchestrator for this skill corpus. It is responsible for
recursive decomposition, deterministic artifact generation, quality gating, and release packaging.

## Completion Promise (Initial User Prompt)

This skill explicitly fulfills the original request to:
1. **Classify and interconnect all provided skills** through textual concept extraction and graph analytics.
2. **Decompose and reorganize them into a modular, cluster-first structure** with symlink compatibility.
3. **Emit a complete control-plane artifact graph** (`.index/*`) with explicit hard/soft dependency semantics, bridge metadata, and ontology.
4. **Optimize recursively to a DAG-friendly topology** using cycle demotion and transitive-reduction while preserving compatibility.
5. **Package and validate the result** as a reusable skill bundle and plugin-ready skill package under strict checks (`MCTSR >= 95`, hard DAG acyclicity, no unresolved references, clean release manifest).

## Fast Path

```bash
python .index/interfaces/emit_index.py --base .
```

## Full Deterministic Loop (recommended)

```bash
python skills/meta/scripts/meta_cycle.py --base . --iterations 5 --strict --package
```

The loop emits `.index` artifacts, regenerates a BD-style issue ledger, and (when requested) creates `meta_release.zip`.

## Progressive Loading Model

1. Resolve compatibility aliases and manifest metadata.
2. Compute raw and reduced hard dependency graphs.
3. Build reverse dependencies and SCC condensation.
4. Promote uncovered couplings into `soft_refs` when necessary.
5. Emit ontology/control payloads and quality reports.
6. Record unresolveds and action items as an issue ledger.
7. Package release artifact set with checksums.

## RALPH Orchestration

`meta` runs a **R**ecurrentive, **A**nalytic, **L**earning-guided, **P**runing/hardening loop:

1. **R**equest replay: read the latest graph, manifests, and constraints.
2. **A**nalysis pass: emit raw graph artifacts (`skill_graph`, `clustered_graph`, ontology schema payloads).
3. **L**eveling: apply SCC decomposition, reverse-call synthesis, and bridge/soft-edge augmentation.
4. **P**runing: demote cycle edges to soft refs and apply transitive reduction on DAG edges.
5. **H**ardening: generate BD ledger, run strict validation, and package release artifacts.

Use this explicit form with:

```bash
python skills/meta/scripts/meta_cycle.py --base . --iterations 5 --strict --package
```

## Runtime dependencies

- Hard dependency: `system-skill`, `skill-orchestrator`
- Soft compatibility references: `skill-protocol`, `skill-updater`

## Issue Tracking (`BD` loop)

- Generate issue ledger:

```bash
python skills/meta/scripts/issue_ledger.py --base . --output .index/bd_ledger.json --strict
```

- Interpret severity tiers from `severity` (`critical`, `high`, `medium`, `low`, `info`) and address all
  `critical` findings before release.

## Packaging (no legacy bundle artifacts)

```bash
python skills/meta/scripts/package_release.py --base . --strict
```

Outputs `./.index/meta_release.zip` and `./.index/meta_release_manifest.json`.

## Inputs and outputs

### Inputs

- `skills/*/manifest.json`
- `skills/*/bridge.json`
- `repo/**/.skill`
- `README.md`

### Outputs

- `.index/main_index.yaml`
- `.index/main_index_payload.json`
- `.index/quality_report.md`
- `.index/processing_report.md`
- `.index/bridge_candidates.csv`
- `.index/bridge_index.json`
- `.index/clustered_graph.json`
- `.index/skill_graph.json`
- `.index/control_ontology.json`
- `.index/control_ontology.schema.json`
- `.index/hyperedges.json`
- `.index/issue_log.json`
- `.index/bd_ledger.json`
- `.index/meta_release.zip`
- `.index/meta_release_manifest.json`

## Multi-platform integration

- `agents/claude.json`: direct command for Claude Code runtime entry.
- `agents/codex.json`: direct command for Codex runtime entry.
- `agents/gemini.json`: direct command for Gemini CLI integration.
- `agents/agent-skills.json`: Agent Skills platform compatibility.
- `agents/openai.yaml`: OpenAI Agent interface compatibility.
- Repository-level `meta` package metadata is mirrored by `.claude-plugin/marketplace.json`.

## Notes

- This skill is deterministic: repeated runs over the same graph should converge and show stable metrics.
- If unresolved references appear, rebuild manifests first, then re-run the loop.
