# Integration Playbook (Condensed)

Use pattern:
1) Ingest-only baseline
- run `--analyze-only` first on a known snapshot
- inspect `analysis.json` for cluster count, isolated nodes, edge density

2) Controlled decomposition
- run full pass with explicit output path
- review `index.md` tiering + `bridges/` count

3) Quality gate
- run `skill-seekers quality` and enforce `A+`/no warning regressions
- require deterministic run reproducibility on same snapshot

4) Ontology + orchestration handoff
- feed `analysis.json` into `graph`/`hierarchical-reasoning`/`ontolog`
- route through `skill-orchestrator` for meta workflows if needed

5) Safe promote policy
- promote to production when: DAG exists, bridge coverage non-zero,
  `index.md` has stable core+cluster modules, no unresolved critical issues

Minimal checks
- core emitted > 0
- top cluster count reasonable (not >12 unless warranted)
- bridge count aligns with expected cross-domain references
