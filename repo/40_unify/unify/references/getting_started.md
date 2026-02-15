# Getting Started

Bootstrap:

```bash
cd /path/to/repo/40_unify/unify
python scripts/unify.py --help
python scripts/unify.py /path/to/target --analyze-only --verbose
python scripts/unify.py /path/to/target --output ./unified --max-iter 60 --threshold 0.95
```

Replay check:
- fixed flags + fixed snapshot + clean git tree
- compare `analysis.json`, `report.md`, `quality_report.json`

Validation (on-demand):
- `skill-seekers quality repo/40_unify/unify --report`
- `skill-seekers package repo/40_unify/unify --target gemini --skip-quality-check --no-open`
