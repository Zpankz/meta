# Usage Examples for `unify`

## Example 1: Minimal corpus reorganization

```bash
python scripts/unify.py /Users/mikhail/Projects/Context-Engineering/Skills/repo/40_unify/unify --analyze-only --verbose
```

Use when you want a deterministic analysis pass before applying merge/split operations.

## Example 2: Full clinical-style decomposition

```bash
python scripts/unify.py /Users/mikhail/Projects/Context-Engineering/Skills --output ./unified-clinical --max-clusters 10 --threshold 0.95
```

Expected outputs include:

- `unified-clinical/core/` with shared abstractions
- `unified-clinical/modules/` grouped by learned domains
- `unified-clinical/bridges/` for cross-domain links
- `unified-clinical/report.md` with convergence details

## Example 3: Low-threshold exploratory pass

```bash
python scripts/unify.py /path/to/repo --threshold 0.90 --max-clusters 16 --max-iter 100 --block-size 350
```

Use for high-noise codebases where boundary preservation is preferable to strict sparsity.

## Example 4: Quality-driven hardening

```bash
/Users/mikhail/.local/bin/skill-seekers quality repo/40_unify/unify --report
/Users/mikhail/.local/bin/skill-seekers package repo/40_unify/unify --target openai --skip-quality-check --no-open
```

Track score deltas after each content or CLI contract change.

## Example 5: API + configuration verification

```bash
/Users/mikhail/.local/bin/skill-seekers analyze --directory repo/40_unify/unify --preset comprehensive --enhance-level 2 --verbose
```

This validates code surface extraction and documentation completeness used by
`references/reference_api.md` and `references/reference_patterns.md`.

