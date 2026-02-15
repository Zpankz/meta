# API Reference Index for `unify`

This reference was normalized from `skill-seekers` v3 code analysis output.

## Public Module

`scripts/unify.py` contains the executable pipeline for recursive context unification.

## Core API by pipeline phase

### Ingestion Layer

- `scan_directory(target_dir: str) -> list[Path]`
- `read_file_safe(path: Path) -> str`
- `extract_sections(content: str, extension: str) -> list[str]`
- `tokenize(text: str) -> list[str]`
- `compute_tfidf(documents: dict[str, list[str]], top_k: int = 20) -> dict[str, list[Concept]]`
- `extract_references(content: str, all_names: set[str], file_path: str) -> list[str]`

### Domain + Graph Layer

- `classify_domain(concepts: list[Concept], sections: list[str], adaptive_keywords: dict = None) -> str`
- `detect_adaptive_domains(nodes: list[FileNode], top_n: int = 20) -> dict[str, set[str]]`
- `ingest(target_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE, verbose: bool = False) -> list[FileNode]`
- `jaccard(set_a: set, set_b: set) -> float`
- `build_graph(nodes: list[FileNode], verbose: bool = False) -> tuple[nx.Graph, list[Edge]]`
- `cluster_graph(G: nx.Graph, nodes: list[FileNode], max_clusters: int = 12, verbose: bool = False) -> list[Cluster]`
- `compute_topology(G: nx.Graph, nodes: list[FileNode]) -> dict`

### Decomposition + Optimization Layer

- `decompose(nodes: list[FileNode], clusters: list[Cluster], G: nx.Graph, block_size: int = 200, verbose: bool = False) -> tuple[list[Module], list[dict]]`
- `build_dag(modules: list[Module], nodes: list[FileNode], G: nx.Graph, verbose: bool = False) -> tuple[nx.DiGraph, list[str]]`
- `score_structure(modules: list[Module], bridges: list[dict], dag: nx.DiGraph, G: nx.Graph, nodes: list[FileNode], clusters: list[Cluster]) -> tuple[float, dict]`
- `mcts_optimize(modules: list[Module], bridges: list[dict], dag: nx.DiGraph, G: nx.Graph, nodes: list[FileNode], clusters: list[Cluster], threshold: float = 0.95, max_iter: int = 50, verbose: bool = False) -> tuple[float, dict, list[dict]]`
- `simulate_action_v2(action, modules, bridges, dag, G, nodes, clusters, detail, iteration) -> float`
- `apply_action_v2(action, modules, bridges, dag, G, nodes, clusters)`
- `generate_outputs(...)`
- `gap_delta(dim, base_pos, base_neg)`
- `main()`

## Data models

- `Concept`
- `FileNode`
- `Edge`
- `Cluster`
- `Module`

Use this index when syncing external orchestrators, API adapters, or agent-facing
contract checklists.

