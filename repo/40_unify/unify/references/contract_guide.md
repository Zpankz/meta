# Unify Contract Guide

## Purpose

`unify` standardizes a directory into a modular knowledge graph and a
progressive loading plan. The contract below clarifies what downstream skills
may rely upon.

## Observable Artifacts

- `analysis.json`: complete graph and clustering metadata.
- `report.md`: execution summary and score breakdown.
- `index.md`: human-oriented loading plan.
- `unified/core/**`: load-at-start modules.
- `unified/modules/**`: cluster modules and specific modules.
- `unified/bridges/**`: cross-module connectors.

## Non-Goals

- It is not a security classifier.
- It does not execute source code.
- It does not perform semantic deduplication across binary assets.

## Minimal Reproducibility Contract

Given fixed inputs and deterministic environment flags, two runs with equal
algorithm defaults should produce compatible module layout and comparable score
ordering. Minor variations may occur with non-deterministic parsing of large,
mixed-format corpora.
