# Usage Gallery

## Scenario 1: Medical Skills Corpus

```bash
python scripts/unify.py /Users/mikhail/Projects/Context-Engineering/Skills --output ./unified/clinical --max-clusters 10
```

Expected outputs:

- `unified/index.md` with `saq`, `cicm-saq-rubric`, and `pex` as sibling module groups.
- `unified/core/` containing ontology-like coordination primitives.
- `unified/bridges/` including cross links where SAQ and medical taxonomies intersect.

## Scenario 2: Repo Refactoring Pass

```bash
python scripts/unify.py . --analyze-only --output ./unified/analyze
```

Use this when you want a deterministic decomposition plan without emitting final modules.

## Scenario 3: High-noise corpus

```bash
python scripts/unify.py /path/to/mixed/repo \
  --threshold 0.90 --max-clusters 16 --block-size 350
```

Use the lower threshold and higher cluster cap to preserve ambiguous boundary concepts in
the first pass. Then tighten with a second run using stricter defaults.
