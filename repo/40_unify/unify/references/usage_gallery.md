# Usage Gallery

```bash
python scripts/unify.py <target> --output ./unified
python scripts/unify.py <target> --analyze-only --verbose
python scripts/unify.py <target> --threshold 0.98 --max-iter 100 --max-clusters 10
```

Medical snapshot:

```bash
python scripts/unify.py /path/to/repo --max-clusters 10 --output ./unified-med
```

Quality loop:

```bash
skill-seekers quality repo/40_unify/unify --report
skill-seekers package repo/40_unify/unify --target openai --skip-quality-check --no-open
```

Recovery:
- unstable DAG: `--max-clusters +2`
- noisy clusters: `--threshold 0.98` then `--threshold 0.95`
- bridge starvation: inspect `analysis.json` cross-cluster map
