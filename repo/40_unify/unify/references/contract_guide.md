# Unify Contract Guide

## Purpose

`unify` normalizes a directory into a modular graph and progressive loading
plan.

## Observable Artifacts

- `analysis.json`
- `report.md`
- `index.md`
- `unified/core/**`
- `unified/modules/**`
- `unified/bridges/**`

## Non-Goals

- Security scanning
- Source execution
- Binary deduplication

## Reproducibility

Fixed snapshot + fixed flags produce stable module ordering; parser ties may
shift in mixed corpora.
