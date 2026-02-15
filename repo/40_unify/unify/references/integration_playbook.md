# Unify Integration Playbook

## Context Engineering Integration

`unify` is most effective when invoked between:

1. Raw corpus capture or folder snapshot.
2. Graph-based reasoning or compression layers.
3. Reasoning orchestration / orchestration-policy layers.

## Recommended Upstream → Downstream

- Upstream:
  - Raw file tree or generated corpus exports.
- Downstream:
  - `graph` for structural graph compression.
  - `ontolog` for schema and relation alignment.
  - `abduct` for latent relationship discovery.
  - `rpp` for reduction planning.
  - `skill-composer` for orchestrated reuse of generated modules.

## Suggested Workflow Template

1. Run `unify` in analyze-only mode to inspect initial domain signal.
2. Approve cluster/bridge expectations from `analysis.json`.
3. Run full unification with explicit output target.
4. Feed generated modules into downstream orchestration skills.
5. Re-run `meta` graph extraction when upstream skill graph changed.

## Quality Safety Gates

- Reject a run if no core tier is emitted.
- Reject if bridge count is zero while explicit cross-domain references remain unresolved.
- Require deterministic reproducibility checks on repeated runs of stable corpus snapshots.
