# Modular Patterns Reference

## Contents

- [Bridge Files](#bridge-files)
- [Modular Decomposition](#modular-decomposition)
- [Progressive Loading Architecture](#progressive-loading-architecture)
- [Domain Classification](#domain-classification)
- [File Type Handling](#file-type-handling)

---

## Bridge Files

Bridge files enable hyperedge connectivity between modules that were
separated during clustering. A bridge documents the interface contract
between two or more clusters.

### Structure

```markdown
# Bridge: {domain_a} ↔ {domain_b}

Connects cluster {id_a} ({domain_a}) with cluster {id_b} ({domain_b}).

## Shared Concepts

{comma-separated list of concepts appearing in both clusters}

## Connections

- `{module_a}` → `{module_b}` (weight={w})
- ...

## Composition Patterns

- Sequential (∘): {module_a} then {module_b}
- Parallel (⊗): {module_a} and {module_b} simultaneously
- Conditional (|): {module_a} if {condition} else {module_b}
```

### When to Create

- Two clusters share ≥1 edge with weight ≥ 0.3
- Shared concepts exist between the clusters
- Removing the edge would increase isolation (φ)

### Hyperedge Bridges

A single bridge can connect 3+ clusters when they share a common
abstraction. The bridge file lists all participating clusters and the
shared concept set that binds them.

---

## Modular Decomposition

### Decomposition Rules

1. **Core extraction**: concepts appearing in ≥3 files across ≥2 clusters
   become core modules (Tier 0)
2. **Cluster bases**: hub node of each cluster becomes the cluster base
   module (Tier 1)
3. **Specific modules**: individual files become specific modules (Tier 2),
   split into blocks of ≤200 lines if large
4. **Bridge modules**: cross-cluster connections documented as bridges (Tier 3)

### Block Splitting

Files exceeding `--block-size` (default 200 lines) are split:

```
original_file.py (600 lines)
  → original_file_block0.md (lines 1-200)
  → original_file_block1.md (lines 201-400)
  → original_file_block2.md (lines 401-600)
```

Split points respect structural boundaries when possible (function/class
definitions, markdown headers).

### Concept Grouping

Shared concepts are grouped using Jaccard similarity > 0.5 on their
file-membership sets. Overlapping concept groups merge into a single
core module rather than creating multiple near-identical cores.

Maximum 5 core modules to prevent Tier 0 bloat.

---

## Progressive Loading Architecture

### Tier Model

```
     Tier 0: Core (~5% of content)
     ┌────────────────────────────┐
     │ Shared types, primitives,  │ ← Always loaded
     │ foundational abstractions  │    ~100 tokens each
     └────────────┬───────────────┘
                  │
     Tier 1: Cluster Bases (~15% of content)
     ┌────────────┴───────────────┐
     │ Domain entry points,       │ ← Loaded on domain
     │ hub module summaries       │    match detection
     └────────────┬───────────────┘
                  │
     Tier 2: Specific Modules (~70% of content)
     ┌────────────┴───────────────┐
     │ Individual file modules,   │ ← Loaded on explicit
     │ detailed implementations   │    demand/reference
     └────────────┬───────────────┘
                  │
     Tier 3: Bridges (~10% of content)
     ┌────────────┴───────────────┐
     │ Cross-cluster connectors,  │ ← Loaded on cross-
     │ hyperedge documentation    │    domain query only
     └────────────────────────────┘
```

### index.md Structure

The index serves as the DAG traversal entry point:

1. Tier 0 content inline or as immediate references
2. Tier 1 listed with domain labels and load conditions
3. Tier 2 grouped by cluster/domain, referenced by path
4. Tier 3 listed with cluster pair labels
5. Full topological order appended for programmatic traversal

### Load Decision Logic

```
IF query matches Tier 0 concepts → load core modules
IF query matches domain keywords → load matching Tier 1
IF query references specific file → load that Tier 2 module
IF query spans multiple domains → load relevant Tier 3 bridges
```

---

## Domain Classification

Files are classified into 8 base domains by keyword overlap between
their TF-IDF concept vector and domain keyword sets:

| Domain | Representative Keywords |
|--------|------------------------|
| logic | algorithm, reason, infer, constraint, solver, optimize |
| data | schema, table, query, database, sql, transform, pipeline |
| interface | ui, component, render, layout, style, widget, route |
| network | http, api, request, endpoint, server, auth, token |
| system | config, deploy, docker, build, package, script, service |
| test | test, assert, expect, mock, fixture, coverage, verify |
| model | entity, relation, graph, node, edge, ontology, taxonomy |
| text | document, content, write, markdown, template, parse, nlp |

Files matching no domain default to `general`.

Classification uses the domain with the highest keyword overlap count.
Ties broken by alphabetical order.

---

## File Type Handling

### Structure Extraction by Extension

| Extension Group | Extracted Structures |
|-----------------|---------------------|
| `.py` | `def`, `class` definitions |
| `.js`, `.ts`, `.jsx`, `.tsx` | `function`, `class`, `const`, `export` |
| `.java`, `.kt`, `.swift` | `class`, `interface`, `enum`, `func` |
| `.go`, `.rs`, `.c`, `.cpp` | `func`, `fn`, `struct`, `type` |
| `.md`, `.txt` | Markdown headers (h1-h6) |
| `.yaml`, `.yml`, `.toml` | Top-level keys |

### Large File Chunking

Files exceeding 50KB are chunked:
- Chunk size: 4096 bytes
- Overlap: 512 bytes (ensures no concept split at boundary)
- Maximum 12 chunks analyzed (~48KB effective ceiling)

### Encoding

Files attempted in order: UTF-8 → Latin-1 → ASCII.
Unreadable files silently skipped.
