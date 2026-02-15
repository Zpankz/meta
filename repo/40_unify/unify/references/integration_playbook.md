# Integration Playbook

1. Baseline: `--analyze-only` on fixed snapshot; inspect `analysis.json` (clusters, isolates).
2. Decompose: run full pass with explicit `--output`; validate `index.md` + `bridges`.
3. Gate: `skill-seekers quality` and reproducibility check on same snapshot.
4. Handoff: pass `analysis.json` to `graph`, `hierarchical-reasoning`, `ontolog`, then `skill-orchestrator`.

Promote if:
- DAG exists
- bridge coverage > 0
- stable core/cluster modules
- no critical warnings
