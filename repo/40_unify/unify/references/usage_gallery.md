# Usage Gallery (Condensed)

Quick start
```bash
python scripts/unify.py <target> --output ./unified
```

Analysis-only
```bash
python scripts/unify.py <target> --analyze-only --verbose
```

Strict optimization
```bash
python scripts/unify.py <target> --threshold 0.98 --max-iter 100 --max-clusters 10
```

Medical corpus snapshot
```bash
python scripts/unify.py /Users/mikhail/Projects/Context-Engineering/Skills --max-clusters 10 --output ./unified-med
```
Verify `core/`, `modules/{domain}/`, `bridges/`.

Quality loop
```bash
skill-seekers quality repo/40_unify/unify --report
skill-seekers package repo/40_unify/unify --target openai --skip-quality-check --no-open
```

Recovery examples
- unstable DAG: rerun with `--max-clusters +2`
- noisy clusters: raise threshold to 0.98 then lower by 0.03
- bridge starvation: inspect `analysis.json` cross-cluster edge map
