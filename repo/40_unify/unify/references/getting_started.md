# Getting Started with `unify`

This document defines the deterministic startup path for both exploratory and production
unification runs. Use it to reproduce stable outputs across environments.

## 1) Bootstrap

```bash
cd /Users/mikhail/Projects/Context-Engineering/Skills/repo/40_unify/unify
python scripts/unify.py --help
```

If output is not produced, verify `networkx` and project dependencies.

## 2) Analyze-only pass (safe baseline)

```bash
python scripts/unify.py /path/to/target --analyze-only --verbose
```

Expected artifacts:

- `analysis.json` (graph + domains + module candidates)
- `report.md` (diagnostic scoring)
- `index.md` (progressive load plan)

Use this before destructive decomposition when the input corpus is noisy.

## 3) Full decomposition pass (production)

```bash
python scripts/unify.py /path/to/target --output ./unified --max-iter 60 --threshold 0.95
```

Expected artifact layout:

- `unified/core/` — base modules loaded first
- `unified/modules/` — tiered domain modules
- `unified/bridges/` — cross-domain connectors

## 4) Quality and packaging gate

```bash
/Users/mikhail/.local/bin/skill-seekers quality repo/40_unify/unify --report
/Users/mikhail/.local/bin/skill-seekers package repo/40_unify/unify --target gemini --skip-quality-check --no-open
```

The second command creates cross-platform package outputs (`gemini`, `openai`,
`markdown`) used as external integration smoke tests.

## 5) Deterministic re-run checklist

- If output layout changes unexpectedly, rerun with:
  - fixed flags
  - fixed Python version
  - fixed working tree (prefer clean git state before invocation)
- Validate `analysis.json` for cluster and bridge drift.
- Confirm `quality_report.json` thresholds and check no warning regressions.

