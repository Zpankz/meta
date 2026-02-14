---
name: meta
description: Package skill-system graph artifacts into a deterministic, checksummed bundle for audit and distribution.
---

# Meta

Use this skill when you need a reproducible artifact build of governance outputs.

## Canonical flow

1. Validate required `.index` artifacts are present and consistent.
2. Build or refresh `.index/skill_bundle_manifest.json` with checksums.
3. Emit `.index/skill_bundle.zip` with a deterministic payload.
4. Preserve output references and fail fast on missing artifacts.

## Runtime behavior

- Hard dependency: `system-skill`, `skill-orchestrator`.
- Soft dependency: `skill-protocol`, `skill-updater`.
- Output: `.index/skill_bundle.zip` and `.index/skill_bundle_manifest.json`.
- Entry point: `scripts/package_bundle.py`.

## Inputs and outputs

### Inputs

- `.index/bridge_candidates.csv`
- `.index/bridge_index.json`
- `.index/control_ontology.json`
- `.index/control_ontology.schema.json`
- `.index/hyperedges.json`
- `.index/main_index_payload.json`
- `.index/main_index.yaml`
- `.index/skill_graph.json`

### Outputs

- `.index/skill_bundle_manifest.json`
- `.index/skill_bundle.zip`

## Notes

No new domain semantics are introduced. This skill only normalizes and packages already computed control/graph metadata to a deterministic archive.
