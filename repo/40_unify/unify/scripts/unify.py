#!/usr/bin/env python3
"""
Unify v2: Recursive context unification engine.

Ingests any directory of files, extracts concepts, builds a weighted
graph, clusters by domain, decomposes into modular blocks, constructs
a dependency DAG, and converges via MCTS-UCB1 until the structure
scores above the target threshold.

Usage:
    python scripts/unify.py /path/to/target --output /path/to/output
    python scripts/unify.py /path/to/target --analyze-only
    python scripts/unify.py /path/to/target --threshold 0.90 --max-iter 100

Requirements: networkx>=3.0
Install: pip install networkx --break-system-packages
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import networkx as nx
except ImportError:
    print("Error: networkx not installed. Run: pip install networkx --break-system-packages",
          file=sys.stderr)
    sys.exit(1)


# ── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.ts', '.md', '.txt', '.yaml', '.yml', '.json', '.html',
    '.css', '.sh', '.sql', '.r', '.go', '.rs', '.java', '.c', '.cpp', '.h',
    '.rb', '.php', '.swift', '.kt', '.toml', '.ini', '.cfg', '.xml', '.csv',
    '.env', '.dockerfile', '.tf', '.jsx', '.tsx', '.vue', '.svelte', '.lua',
    '.zig', '.nim', '.ex', '.exs', '.clj', '.scala', '.pl', '.pm', '.bat',
    '.ps1', '.mjs', '.cjs', '.graphql', '.proto', '.sol', '.move',
}

STOP_WORDS = frozenset(
    "the a an and or but in on at to for of is it this that with from by as "
    "be are was were been has have had do does did will would can could should "
    "may might shall not no if then else when while each every all any some "
    "use used using also just like so than more most very only about into "
    "between through during before after above below up down out over under "
    "again further once here there where how what which who whom why how "
    "own same other another new old first last next such get set let make "
    "see need take give well still even back way because however since "
    "import from return def class function const var let type interface "
    "export default public private static void int str string true false "
    "none null undefined self cls args kwargs param params value key data "
    "file name path result output input error msg".split()
)

# [D1] Expanded domain keywords — 14 domains
DOMAIN_KEYWORDS = {
    'logic': {'algorithm', 'logic', 'reason', 'infer', 'deduc', 'abduct',
              'heuristic', 'constraint', 'solver', 'optimize', 'search',
              'decision', 'evaluate', 'criterion'},
    'data': {'data', 'schema', 'table', 'column', 'row', 'query', 'database',
             'sql', 'csv', 'json', 'parse', 'transform', 'etl', 'pipeline',
             'storage', 'record', 'index', 'aggregate'},
    'interface': {'ui', 'component', 'render', 'display', 'layout', 'style',
                  'theme', 'widget', 'button', 'form', 'page', 'view', 'route',
                  'frontend', 'dashboard'},
    'network': {'http', 'api', 'request', 'response', 'endpoint', 'url',
                'server', 'client', 'socket', 'protocol', 'auth', 'token',
                'webhook', 'fetch', 'rest'},
    'system': {'config', 'env', 'deploy', 'docker', 'build', 'install',
               'script', 'shell', 'process', 'service', 'runtime', 'daemon',
               'infrastructure', 'provision'},
    'test': {'test', 'spec', 'assert', 'expect', 'mock', 'stub', 'fixture',
             'coverage', 'benchmark', 'validate', 'verify', 'evaluation',
             'accuracy', 'metric'},
    'model': {'model', 'entity', 'relation', 'node', 'edge', 'vertex',
              'tree', 'ontology', 'taxonomy', 'hierarchy', 'inheritance',
              'schema', 'structure'},
    'text': {'text', 'document', 'content', 'write', 'read', 'format',
             'markdown', 'template', 'render', 'extract', 'nlp', 'narrative',
             'prose', 'paragraph'},
    'reasoning': {'reasoning', 'think', 'thought', 'cognitive', 'mental',
                  'metacognitive', 'critique', 'dialectic', 'synthesis',
                  'decompose', 'abstraction', 'framework'},
    'skill': {'skill', 'agent', 'prompt', 'instruction', 'workflow',
              'orchestrat', 'pipeline', 'compose', 'trigger', 'frontmatter',
              'description', 'progressive'},
    'medical': {'medical', 'clinical', 'pharmacol', 'physiolog', 'drug',
                'dose', 'receptor', 'cardiac', 'renal', 'equation',
                'clearance', 'metabolism', 'anaesth'},
    'graph': {'graph', 'knowledge', 'topology', 'cluster', 'community',
              'pagerank', 'centrality', 'betweenness', 'adjacency',
              'traversal', 'connected', 'bisimulation'},
    'security': {'security', 'encrypt', 'decrypt', 'hash', 'certificate',
                 'auth', 'permission', 'access', 'credential', 'firewall',
                 'vulnerability'},
    'learning': {'learn', 'train', 'compound', 'refine', 'iterate',
                 'improve', 'feedback', 'assess', 'evolve', 'adapt',
                 'knowledge', 'accumulate'},
}

MAX_FILE_SIZE = 50 * 1024  # 50KB before chunking
DEFAULT_CHUNK_SIZE = 4096
CHUNK_OVERLAP = 512
# [C1] Raised from 0.15 to 0.25
JACCARD_THRESHOLD = 0.45
# Min name length for reference matching to avoid false positives [B13]
MIN_REF_NAME_LEN = 4


# ── Data Types ───────────────────────────────────────────────────────────────

@dataclass
class Concept:
    term: str
    weight: float
    source: str  # 'name', 'structure', 'body'

@dataclass
class FileNode:
    id: str
    name: str
    path: str
    extension: str
    content: str = ''
    line_count: int = 0
    byte_size: int = 0
    concepts: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    references: list = field(default_factory=list)
    domain: str = 'unknown'
    role: str = 'leaf'
    pagerank: float = 0.0
    cluster: int = -1

@dataclass
class Edge:
    source: str
    target: str
    weight: float
    edge_type: str

@dataclass
class Cluster:
    id: int
    domain: str
    members: list = field(default_factory=list)
    hub: str = ''
    internal_edges: int = 0
    external_edges: int = 0

@dataclass
class Module:
    id: str
    source_files: list = field(default_factory=list)
    concepts: list = field(default_factory=list)
    content: str = ''
    tier: int = 2
    depth: int = 0
    dependencies: list = field(default_factory=list)
    module_type: str = 'specific'  # core, cluster_base, specific, bridge
    cluster_id: int = -1  # [B17] Track which cluster this belongs to


# ── Φ0: Ingest ──────────────────────────────────────────────────────────────

def scan_directory(target_dir: str) -> list[Path]:
    """Recursively find all supported files."""
    target = Path(target_dir)
    if not target.is_dir():
        print(f"Error: {target_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    skip_dirs = {'node_modules', '__pycache__', 'venv', '.git', 'dist', 'build'}
    files = []
    for p in sorted(target.rglob('*')):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            parts = p.relative_to(target).parts
            if any(part.startswith('.') or part in skip_dirs for part in parts):
                continue
            files.append(p)
    return files


def read_file_safe(path: Path) -> str:
    """Read file content with encoding fallback."""
    for encoding in ('utf-8', 'latin-1', 'ascii'):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    return ''


def extract_sections(content: str, extension: str) -> list[str]:
    """Extract structural sections based on file type."""
    sections = []
    if extension in ('.md', '.txt'):
        for m in re.finditer(r'^#{1,6}\s+(.+)$', content, re.MULTILINE):
            sections.append(m.group(1).strip())
    elif extension == '.py':
        for m in re.finditer(r'^(?:def|class)\s+(\w+)', content, re.MULTILINE):
            sections.append(m.group(1))
    elif extension in ('.js', '.ts', '.jsx', '.tsx'):
        for m in re.finditer(
            r'(?:function|class|const|export\s+(?:default\s+)?(?:function|class))\s+(\w+)',
            content, re.MULTILINE):
            sections.append(m.group(1))
    elif extension in ('.java', '.kt', '.scala', '.swift'):
        for m in re.finditer(r'(?:class|interface|enum|struct|func|fun)\s+(\w+)',
                             content, re.MULTILINE):
            sections.append(m.group(1))
    elif extension in ('.go', '.rs', '.c', '.cpp', '.h'):
        for m in re.finditer(r'(?:func|fn|struct|type|void|int)\s+(\w+)',
                             content, re.MULTILINE):
            sections.append(m.group(1))
    elif extension in ('.yaml', '.yml', '.toml', '.ini'):
        for m in re.finditer(r'^(\w[\w-]*):', content, re.MULTILINE):
            sections.append(m.group(1))
    return sections


def tokenize(text: str) -> list[str]:
    """Extract lowercase word tokens, filtering stops and short words."""
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,}', text.lower())
    expanded = []
    for w in words:
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', w).lower().split()
        for p in parts:
            expanded.extend(p.split('_'))
    return [w for w in expanded if len(w) > 2 and w not in STOP_WORDS]


def compute_tfidf(documents: dict[str, list[str]], top_k: int = 20) -> dict[str, list[Concept]]:
    """Compute TF-IDF concepts per document."""
    df = Counter()
    for tokens in documents.values():
        df.update(set(tokens))
    n_docs = max(len(documents), 1)

    results = {}
    for doc_id, tokens in documents.items():
        if not tokens:
            results[doc_id] = []
            continue
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        scored = []
        for term, count in tf.items():
            tfidf = (count / total) * math.log(1 + n_docs / max(df.get(term, 1), 1))
            scored.append(Concept(term=term, weight=round(tfidf, 4), source='body'))
        scored.sort(key=lambda c: c.weight, reverse=True)
        results[doc_id] = scored[:top_k]
    return results


def extract_references(content: str, all_names: set[str], file_path: str) -> list[str]:
    """Detect explicit references to other files. [B13] Word-boundary matching only."""
    refs = set()
    my_stem = Path(file_path).stem

    # Import/require statements — high precision
    for m in re.finditer(r'(?:import|from|require|include)\s+["\']?([.\w/\\-]+)', content):
        ref = m.group(1).replace('\\', '/').split('/')[-1]
        ref_base = Path(ref).stem
        if ref_base in all_names and ref_base != my_stem:
            refs.add(ref_base)

    # Wikilinks — high precision
    for m in re.finditer(r'\[\[([^\]|]+)', content):
        ref = m.group(1).strip()
        if ref in all_names:
            refs.add(ref)

    # Markdown links — high precision
    for m in re.finditer(r'\[(?:[^\]]*)\]\(([^)]+)\)', content):
        ref = Path(m.group(1)).stem
        if ref in all_names and ref != my_stem:
            refs.add(ref)

    # [B13] Word-boundary name mentions — only for names ≥4 chars to avoid
    # false positives like 'base' matching 'database', 'core' matching 'score'
    for name in all_names:
        if name == my_stem or len(name) < MIN_REF_NAME_LEN:
            continue
        # Use word boundary regex instead of substring match
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, content, re.IGNORECASE):
            refs.add(name)

    return sorted(refs)


def classify_domain(concepts: list[Concept], sections: list[str],
                    adaptive_keywords: dict = None) -> str:
    """Classify file into a domain by keyword overlap. [D1][D2] Extended domains + adaptive."""
    all_terms = {c.term for c in concepts} | {s.lower() for s in sections}

    # Merge base + adaptive keywords
    merged = dict(DOMAIN_KEYWORDS)
    if adaptive_keywords:
        for domain, kws in adaptive_keywords.items():
            if domain in merged:
                merged[domain] = merged[domain] | kws
            else:
                merged[domain] = kws

    best_domain = 'general'
    best_score = 0
    for domain, keywords in merged.items():
        overlap = len(all_terms & keywords)
        if overlap > best_score:
            best_score = overlap
            best_domain = domain
    # Require at least 1 keyword match to avoid weak classification
    return best_domain if best_score >= 1 else 'general'


def detect_adaptive_domains(nodes: list, top_n: int = 20) -> dict[str, set[str]]:
    """[D2] Detect corpus-specific domain vocabulary from frequent co-occurring terms."""
    # Find top bigrams that co-occur in many files
    bigram_files = defaultdict(set)
    for n in nodes:
        terms = [c.term for c in n.concepts[:10]]
        for i, a in enumerate(terms):
            for b in terms[i+1:]:
                pair = tuple(sorted([a, b]))
                bigram_files[pair].add(n.id)

    # Bigrams in ≥5% of files but ≤30% (discriminative, not ubiquitous)
    threshold_low = max(3, len(nodes) * 0.05)
    threshold_high = len(nodes) * 0.30
    discriminative = {pair: files for pair, files in bigram_files.items()
                      if threshold_low <= len(files) <= threshold_high}

    # Group co-occurring bigrams into adaptive domains
    adaptive = {}
    for pair, files in sorted(discriminative.items(), key=lambda x: len(x[1]), reverse=True)[:top_n]:
        # Check if this pair already belongs to a base domain
        matched = False
        for domain, kws in DOMAIN_KEYWORDS.items():
            if pair[0] in kws or pair[1] in kws:
                if domain not in adaptive:
                    adaptive[domain] = set()
                adaptive[domain].add(pair[0])
                adaptive[domain].add(pair[1])
                matched = True
                break
        if not matched:
            # New micro-domain keyed by the pair
            key = f"corpus_{pair[0]}"
            if key not in adaptive:
                adaptive[key] = set()
            adaptive[key].add(pair[0])
            adaptive[key].add(pair[1])

    return adaptive


def ingest(target_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
           verbose: bool = False) -> list[FileNode]:
    """Φ0: Scan, parse, and build FileNode list."""
    files = scan_directory(target_dir)
    if not files:
        print(f"Error: No supported files found in {target_dir}", file=sys.stderr)
        sys.exit(1)
    if verbose:
        print(f"  Φ0: Found {len(files)} files", file=sys.stderr)

    target = Path(target_dir)
    all_names = {p.stem for p in files}

    nodes = []
    doc_tokens = {}
    for p in files:
        content = read_file_safe(p)
        rel_path = str(p.relative_to(target))
        node_id = p.stem

        # Handle duplicate IDs
        if any(n.id == node_id for n in nodes):
            node_id = rel_path.replace('/', '__').replace('.', '_')

        # Chunk large files
        if len(content.encode('utf-8', errors='ignore')) > MAX_FILE_SIZE:
            chunks = []
            for i in range(0, len(content), chunk_size - CHUNK_OVERLAP):
                chunks.append(content[i:i + chunk_size])
            content_for_analysis = ' '.join(chunks[:12])
        else:
            content_for_analysis = content

        sections = extract_sections(content, p.suffix.lower())
        refs = extract_references(content, all_names, rel_path)

        name_tokens = tokenize(p.stem) * 3
        section_tokens = []
        for s in sections:
            section_tokens.extend(tokenize(s) * 2)
        body_tokens = tokenize(content_for_analysis)
        all_tokens = name_tokens + section_tokens + body_tokens
        doc_tokens[node_id] = all_tokens

        node = FileNode(
            id=node_id, name=p.stem, path=rel_path,
            extension=p.suffix.lower(), content=content,
            line_count=content.count('\n') + 1,
            byte_size=len(content.encode('utf-8', errors='ignore')),
            sections=sections, references=refs,
        )
        nodes.append(node)

    # TF-IDF
    tfidf = compute_tfidf(doc_tokens)

    # [D2] Adaptive domain detection
    for node in nodes:
        node.concepts = tfidf.get(node.id, [])
    adaptive_kws = detect_adaptive_domains(nodes)

    # Classify with adaptive keywords
    for node in nodes:
        node.domain = classify_domain(node.concepts, node.sections, adaptive_kws)

    if verbose:
        domains = Counter(n.domain for n in nodes)
        print(f"  Φ0: Ingested {len(nodes)} nodes, {sum(n.line_count for n in nodes)} lines", file=sys.stderr)
        print(f"  Φ0: Domains: {dict(domains)}", file=sys.stderr)
    return nodes


# ── Φ1: Analyze ─────────────────────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def build_graph(nodes: list[FileNode], verbose: bool = False) -> tuple[nx.Graph, list[Edge]]:
    """Build weighted undirected graph. [C1][C2][C3] Stricter thresholds, additive weights."""
    G = nx.Graph()
    edges = []
    node_map = {n.id: n for n in nodes}

    for n in nodes:
        G.add_node(n.id, domain=n.domain, pagerank=0, role='leaf')

    for i, a in enumerate(nodes):
        a_terms = {c.term for c in a.concepts}
        for b in nodes[i + 1:]:
            b_terms = {c.term for c in b.concepts}

            # [C3] Additive weight accumulation
            weight = 0.0
            types = []

            # Explicit references (highest signal)
            if b.id in a.references or b.name in a.references or \
               a.id in b.references or a.name in b.references:
                weight += 0.5
                types.append('explicit_reference')

            # Structural similarity (shared section names ≥ 2)
            if a.sections and b.sections:
                shared = set(s.lower() for s in a.sections) & set(s.lower() for s in b.sections)
                if len(shared) >= 2:
                    weight += min(0.3, 0.1 * len(shared))
                    types.append('structural_similarity')

            # [C1] Concept overlap — raised threshold to 0.25
            j = jaccard(a_terms, b_terms)
            if j >= JACCARD_THRESHOLD:
                weight += min(0.4, j)
                types.append('concept_overlap')

            # [C2] Domain co-membership — only if concept overlap also exists
            if a.domain == b.domain and a.domain != 'general' and j >= 0.05:
                weight += 0.15
                types.append('domain_overlap')

            # Cap at 1.0
            weight = min(1.0, weight)

            if weight > 0.20:  # Minimum edge weight threshold
                G.add_edge(a.id, b.id, weight=round(weight, 3),
                           edge_type=types[0] if types else 'none')
                edges.append(Edge(a.id, b.id, round(weight, 3),
                                  '+'.join(types) if types else 'none'))

    # [C1+] KNN pruning: keep only top-K strongest edges per node to control η
    if len(G.nodes) > 2:
        target_eta = 8.0
        target_edges = int(len(G.nodes) * target_eta / 2)
        if len(G.edges) > target_edges * 1.5:
            k = max(4, int(target_eta))
            edges_to_keep = set()
            for node in G.nodes:
                neighbors = [(n, G[node][n].get('weight', 0)) for n in G.neighbors(node)]
                neighbors.sort(key=lambda x: x[1], reverse=True)
                for n, w in neighbors[:k]:
                    edges_to_keep.add(tuple(sorted([node, n])))
            edges_to_remove = [(u, v) for u, v in G.edges()
                               if tuple(sorted([u, v])) not in edges_to_keep]
            G.remove_edges_from(edges_to_remove)
            edges[:] = [e for e in edges
                        if G.has_edge(e.source, e.target) or G.has_edge(e.target, e.source)]
            if verbose:
                new_eta = len(G.edges) / max(len(G.nodes), 1)
                print(f"  Φ1: Pruned to {len(G.edges)} edges (η={new_eta:.1f}), "
                      f"removed {len(edges_to_remove)}", file=sys.stderr)

    # PageRank (computed after pruning for accurate centrality)
    if len(G.nodes) > 1 and len(G.edges) > 0:
        pr = nx.pagerank(G, alpha=0.85, max_iter=50, weight='weight')
    else:
        pr = {n.id: 1.0 / max(len(nodes), 1) for n in nodes}
    for n in nodes:
        n.pagerank = round(pr.get(n.id, 0), 6)
        G.nodes[n.id]['pagerank'] = n.pagerank

    # Assign roles on pruned graph
    if len(G.nodes) > 2:
        degrees = dict(G.degree())
        deg_values = list(degrees.values())
        mean_deg = sum(deg_values) / max(len(deg_values), 1)
        std_deg = (sum((d - mean_deg) ** 2 for d in deg_values) / max(len(deg_values), 1)) ** 0.5

        betweenness = nx.betweenness_centrality(G, weight='weight') if len(G.edges) > 0 else {}
        bc_values = sorted(betweenness.values())
        bc_90 = bc_values[int(len(bc_values) * 0.9)] if bc_values else 0

        for n in nodes:
            deg = degrees.get(n.id, 0)
            bc = betweenness.get(n.id, 0)
            if deg == 0:
                n.role = 'orphan'
            elif std_deg > 0 and deg > mean_deg + 2 * std_deg:
                n.role = 'hub'
            elif bc > bc_90 and bc > 0:
                n.role = 'bridge'
            elif deg <= 2:
                n.role = 'leaf'
            else:
                n.role = 'member'
            G.nodes[n.id]['role'] = n.role

    if verbose:
        eta = len(G.edges) / max(len(G.nodes), 1)
        print(f"  Φ1: Graph has {len(G.nodes)} nodes, {len(G.edges)} edges (η={eta:.1f})",
              file=sys.stderr)
    return G, edges


def cluster_graph(G: nx.Graph, nodes: list[FileNode], max_clusters: int = 12,
                  verbose: bool = False) -> list[Cluster]:
    """Cluster nodes using community detection."""
    if len(G.nodes) < 3 or len(G.edges) == 0:
        c = Cluster(id=0, domain='general', members=[n.id for n in nodes])
        for n in nodes:
            n.cluster = 0
        return [c]

    try:
        communities = nx.community.louvain_communities(G, weight='weight', seed=42)
    except AttributeError:
        communities = list(nx.community.label_propagation_communities(G))

    if len(communities) > max_clusters:
        communities = sorted(communities, key=len, reverse=True)
        main = communities[:max_clusters - 1]
        overflow = set()
        for c in communities[max_clusters - 1:]:
            overflow.update(c)
        main.append(overflow)
        communities = main

    node_map = {n.id: n for n in nodes}
    clusters = []

    for idx, members in enumerate(communities):
        member_ids = sorted(members)
        domain_counts = Counter(node_map[m].domain for m in member_ids if m in node_map)
        domain = domain_counts.most_common(1)[0][0] if domain_counts else 'general'
        hub = max(member_ids, key=lambda m: node_map[m].pagerank if m in node_map else 0)
        subgraph = G.subgraph(member_ids)
        internal = subgraph.number_of_edges()
        external = sum(1 for u, v in G.edges(member_ids) if v not in members)

        cluster = Cluster(
            id=idx, domain=domain, members=member_ids,
            hub=hub, internal_edges=internal, external_edges=external
        )
        clusters.append(cluster)
        for m in member_ids:
            if m in node_map:
                node_map[m].cluster = idx

    if verbose:
        print(f"  Φ1: {len(clusters)} clusters detected", file=sys.stderr)
    return clusters


def compute_topology(G: nx.Graph, nodes: list[FileNode]) -> dict:
    n = max(len(G.nodes), 1)
    e = len(G.edges)
    orphans = sum(1 for n_id in G.nodes if G.degree(n_id) == 0)
    eta = round(e / n, 3) if n > 0 else 0
    phi = round(orphans / n, 3) if n > 0 else 0
    kappa = round(nx.average_clustering(G, weight='weight'), 3) if n > 2 else 0
    return {
        'nodes': n, 'edges': e, 'orphans': orphans,
        'eta': eta, 'eta_target': 4.0, 'eta_pass': eta >= 4.0,
        'phi': phi, 'phi_target': 0.20, 'phi_pass': phi < 0.20,
        'kappa': kappa, 'kappa_target': 0.30, 'kappa_pass': kappa > 0.30,
    }


# ── Φ2: Decompose ───────────────────────────────────────────────────────────

def decompose(nodes: list[FileNode], clusters: list[Cluster], G: nx.Graph,
              block_size: int = 200, verbose: bool = False) -> tuple[list[Module], list[dict]]:
    """Decompose clustered files into modular blocks and bridges."""
    node_map = {n.id: n for n in nodes}
    modules = []
    bridges = []

    # Shared concepts across ≥3 files → core modules
    concept_to_files = defaultdict(set)
    for n in nodes:
        for c in n.concepts[:10]:
            concept_to_files[c.term].add(n.id)

    core_concepts = {t: f for t, f in concept_to_files.items() if len(f) >= 3}

    # [B19] Group with transitive closure
    if core_concepts:
        concept_groups = []
        used_concepts = set()
        for term in sorted(core_concepts, key=lambda t: len(core_concepts[t]), reverse=True):
            if term in used_concepts:
                continue
            group_terms = {term}
            group_files = core_concepts[term].copy()
            # Iterative merge until stable
            changed = True
            while changed:
                changed = False
                for other, ofiles in core_concepts.items():
                    if other not in group_terms and other not in used_concepts:
                        if jaccard(group_files, ofiles) > 0.5:
                            group_terms.add(other)
                            group_files |= ofiles
                            changed = True
            used_concepts |= group_terms
            concept_groups.append((sorted(group_terms), sorted(group_files)))

        for idx, (terms, file_ids) in enumerate(concept_groups[:5]):
            mod = Module(
                id=f"core_{idx}",
                source_files=file_ids,
                concepts=terms,
                content=f"# Core Module: {', '.join(terms[:5])}\n\n"
                        f"Shared abstractions from {len(file_ids)} files.\n\n"
                        f"## Source Files\n\n" +
                        '\n'.join(f"- `{node_map[f].path}`" for f in file_ids if f in node_map) +
                        f"\n\n## Concepts\n\n" +
                        '\n'.join(f"- {t}" for t in terms),
                tier=0, module_type='core',
            )
            modules.append(mod)

    # Cluster-based modules
    for cluster in clusters:
        hub_node = node_map.get(cluster.hub)
        if hub_node:
            mod = Module(
                id=f"cluster_{cluster.id}_base",
                source_files=[cluster.hub],
                concepts=[c.term for c in hub_node.concepts[:10]],
                content=f"# {cluster.domain.title()} Cluster Base\n\n"
                        f"Hub: `{hub_node.path}` (PageRank: {hub_node.pagerank})\n\n"
                        f"## Members ({len(cluster.members)})\n\n" +
                        '\n'.join(f"- `{node_map[m].path}` ({node_map[m].role})"
                                 for m in cluster.members if m in node_map),
                tier=1, module_type='cluster_base',
                cluster_id=cluster.id,
            )
            modules.append(mod)

        for member_id in cluster.members:
            member = node_map.get(member_id)
            if not member:
                continue
            lines = member.content.split('\n')
            if len(lines) > block_size:
                for block_idx in range(0, len(lines), block_size):
                    block_lines = lines[block_idx:block_idx + block_size]
                    mod = Module(
                        id=f"{member_id}_block{block_idx // block_size}",
                        source_files=[member_id],
                        concepts=[c.term for c in member.concepts[:5]],
                        content='\n'.join(block_lines),
                        tier=2, module_type='specific',
                        cluster_id=cluster.id,
                    )
                    modules.append(mod)
            else:
                mod = Module(
                    id=member_id,
                    source_files=[member_id],
                    concepts=[c.term for c in member.concepts[:10]],
                    content=member.content,
                    tier=2, module_type='specific',
                    cluster_id=cluster.id,
                )
                modules.append(mod)

    # Bridge files for cross-cluster edges
    cluster_map = {}
    for c in clusters:
        for m in c.members:
            cluster_map[m] = c.id

    bridge_pairs = defaultdict(lambda: {'weight': 0, 'concepts': set(), 'edges': []})
    for u, v, data in G.edges(data=True):
        cu = cluster_map.get(u, -1)
        cv = cluster_map.get(v, -1)
        if cu != cv and cu >= 0 and cv >= 0:
            pair_key = tuple(sorted([cu, cv]))
            bp = bridge_pairs[pair_key]
            bp['weight'] = max(bp['weight'], data.get('weight', 0))
            bp['edges'].append((u, v, data.get('weight', 0)))
            a_terms = {c.term for c in node_map[u].concepts}
            b_terms = {c.term for c in node_map[v].concepts}
            bp['concepts'] |= (a_terms & b_terms)

    for (ca, cb), info in bridge_pairs.items():
        if info['weight'] < 0.3:
            continue
        domain_a = next((c.domain for c in clusters if c.id == ca), 'unknown')
        domain_b = next((c.domain for c in clusters if c.id == cb), 'unknown')
        bridge = {
            'id': f"bridge_{ca}_{cb}",
            'clusters': [ca, cb],
            'domains': [domain_a, domain_b],
            'weight': round(info['weight'], 3),
            'shared_concepts': sorted(info['concepts'])[:15],
            'connections': [(u, v, round(w, 3)) for u, v, w in info['edges'][:20]],
        }
        bridges.append(bridge)

        # [B17] Store cluster IDs, not source files
        mod = Module(
            id=f"bridge_{ca}_{cb}",
            source_files=[e[0] for e in info['edges'][:5]] + [e[1] for e in info['edges'][:5]],
            concepts=sorted(info['concepts'])[:10],
            content=f"# Bridge: {domain_a} ↔ {domain_b}\n\n"
                    f"Connects cluster {ca} ({domain_a}) with cluster {cb} ({domain_b}).\n\n"
                    f"## Shared Concepts\n\n" +
                    ', '.join(sorted(info['concepts'])[:15]) +
                    f"\n\n## Connections\n\n" +
                    '\n'.join(f"- `{u}` → `{v}` (w={w:.2f})"
                             for u, v, w in info['edges'][:10]),
            tier=3, module_type='bridge',
            cluster_id=-1,  # Bridges span clusters
        )
        # [B17] Tag bridge with both cluster IDs for DAG wiring
        mod._bridge_clusters = (ca, cb)
        modules.append(mod)

    if verbose:
        core_count = sum(1 for m in modules if m.module_type == 'core')
        bridge_count = sum(1 for m in modules if m.module_type == 'bridge')
        spec_count = sum(1 for m in modules if m.module_type == 'specific')
        print(f"  Φ2: {len(modules)} modules ({core_count} core, "
              f"{len(clusters)} bases, {spec_count} specific, {bridge_count} bridges)",
              file=sys.stderr)
    return modules, bridges


# ── Φ3: Unify (DAG Construction) ────────────────────────────────────────────

def build_dag(modules: list[Module], nodes: list[FileNode], G: nx.Graph,
              verbose: bool = False) -> tuple[nx.DiGraph, list[str]]:
    """Build directed acyclic graph from modules. [B17][B18] Fixed bridge wiring, deque BFS."""
    dag = nx.DiGraph()
    node_map = {n.id: n for n in nodes}
    module_map = {m.id: m for m in modules}

    # Index cluster bases by cluster_id
    base_by_cluster = {}
    for m in modules:
        dag.add_node(m.id, tier=m.tier, module_type=m.module_type)
        if m.module_type == 'cluster_base':
            base_by_cluster[m.cluster_id] = m.id

    for m in modules:
        if m.module_type == 'specific':
            # Depends on its cluster base
            base_id = base_by_cluster.get(m.cluster_id)
            if base_id:
                dag.add_edge(base_id, m.id)
                m.dependencies.append(base_id)

        elif m.module_type == 'cluster_base':
            # Depends on relevant core modules
            m_concepts = set(m.concepts)
            for core in modules:
                if core.module_type == 'core' and m_concepts & set(core.concepts):
                    dag.add_edge(core.id, m.id)
                    m.dependencies.append(core.id)

        elif m.module_type == 'bridge':
            # [B17] Connect bridge to both cluster bases via stored cluster IDs
            bridge_clusters = getattr(m, '_bridge_clusters', None)
            if bridge_clusters:
                for cid in bridge_clusters:
                    base_id = base_by_cluster.get(cid)
                    if base_id:
                        dag.add_edge(base_id, m.id)
                        m.dependencies.append(base_id)

    # Break cycles
    cycles_broken = 0
    while cycles_broken < 100:
        try:
            cycle = nx.find_cycle(dag)
            weakest = min(cycle, key=lambda e: G.get_edge_data(e[0], e[1], {}).get('weight', 0.5))
            dag.remove_edge(weakest[0], weakest[1])
            cycles_broken += 1
        except nx.NetworkXNoCycle:
            break

    try:
        topo_order = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        topo_order = list(dag.nodes)

    # [B18] BFS with deque
    roots = [n for n in dag.nodes if dag.in_degree(n) == 0]
    depths = {r: 0 for r in roots}
    queue = deque(roots)
    while queue:
        current = queue.popleft()
        for successor in dag.successors(current):
            new_depth = depths[current] + 1
            if successor not in depths or new_depth > depths[successor]:
                depths[successor] = new_depth
                queue.append(successor)

    for m in modules:
        m.depth = depths.get(m.id, 0)

    if verbose:
        max_d = max(depths.values()) if depths else 0
        print(f"  Φ3: DAG has {len(dag.nodes)} nodes, {len(dag.edges)} edges, "
              f"{cycles_broken} cycles broken, depth={max_d}", file=sys.stderr)
    return dag, topo_order


# ── Φ4: Optimize (MCTS-UCB1) ────────────────────────────────────────────────

def score_structure(modules: list[Module], bridges: list[dict], dag: nx.DiGraph,
                    G: nx.Graph, nodes: list[FileNode],
                    clusters: list[Cluster]) -> tuple[float, dict]:
    """Score the current structure on 8 dimensions. [A1][A2][A3] Fixed formulas."""
    n_modules = max(len(modules), 1)
    n_nodes = max(len(nodes), 1)

    # [A1] Parsimony — sigmoid: ratio≤1→1.0, ratio=2→~0.80, ratio=3→~0.50, ratio=5→~0.10
    ratio = n_modules / n_nodes
    if ratio <= 1.0:
        parsimony = 1.0
    else:
        # Sigmoid: 1 / (1 + exp(1.5 * (ratio - 2)))
        parsimony = 1.0 / (1.0 + math.exp(2.0 * (ratio - 2.0)))

    # [A3] Redundancy — intra-cluster pairs only (not O(n²) all-pairs)
    cluster_modules = defaultdict(list)
    for m in modules:
        if m.module_type != 'bridge':
            cluster_modules[m.cluster_id].append(m)
    overlaps = []
    for cid, mods in cluster_modules.items():
        for i, a in enumerate(mods):
            for b in mods[i + 1:]:
                if a.concepts and b.concepts:
                    j = jaccard(set(a.concepts), set(b.concepts))
                    overlaps.append(j)
    avg_overlap = sum(overlaps) / max(len(overlaps), 1)
    redundancy = 1.0 - min(avg_overlap * 3, 1.0)

    # Connectivity: η ≥ 4, φ < 0.20
    topo = compute_topology(G, nodes)
    eta_score = min(topo['eta'] / 4.0, 1.0)
    phi_score = max(0, 1.0 - topo['phi'] / 0.20)
    connectivity = (eta_score + phi_score) / 2

    # [B16] DAG validity — specific except clause
    try:
        is_dag = nx.is_directed_acyclic_graph(dag)
    except nx.NetworkXError:
        is_dag = False
    roots = [n for n in dag.nodes if dag.in_degree(n) == 0]
    reachable = set()
    for r in roots:
        reachable |= nx.descendants(dag, r) | {r}
    reachability = len(reachable) / max(len(dag.nodes), 1)
    dag_validity = (1.0 if is_dag else 0.0) * 0.5 + reachability * 0.5

    # Bridge coverage
    bridge_cluster_pairs = set()
    for b in bridges:
        bridge_cluster_pairs.add(tuple(sorted(b['clusters'])))
    cross_edges = set()
    node_cluster = {n.id: n.cluster for n in nodes}
    for u, v in G.edges():
        cu = node_cluster.get(u, -1)
        cv = node_cluster.get(v, -1)
        if cu != cv and cu >= 0 and cv >= 0:
            cross_edges.add(tuple(sorted([cu, cv])))
    bridge_coverage = len(bridge_cluster_pairs & cross_edges) / max(len(cross_edges), 1)

    # Load efficiency: tier 0 < 5%
    tier0_size = sum(len(m.content) for m in modules if m.tier == 0)
    total_size = max(sum(len(m.content) for m in modules), 1)
    tier0_ratio = tier0_size / total_size
    load_efficiency = 1.0 if tier0_ratio < 0.05 else max(0, 1.0 - (tier0_ratio - 0.05) * 10)

    # [A2][B15][B16] Modularity — calibrated for dense graphs
    modularity_score = 0.5
    try:
        if len(G.edges) > 0 and clusters:
            communities = [set(c.members) for c in clusters if c.members]
            if communities:
                raw_mod = nx.community.modularity(G, communities, weight='weight')
                # Calibrated: raw modularity in dense graphs is 0.05-0.4
                # Map [0, 0.5] → [0, 1.0] with floor at 0
                modularity_score = max(0.0, min(1.0, raw_mod * 2.5))
    except (nx.NetworkXError, ZeroDivisionError):
        pass

    # Depth ratio: critical path ≤ ceil(log2(n))
    max_depth = max((m.depth for m in modules), default=0)
    target_depth = max(math.ceil(math.log2(max(n_modules, 2))), 1)
    depth_ratio = 1.0 if max_depth <= target_depth else \
        max(0, 1.0 - (max_depth - target_depth) / target_depth)

    weights = {
        'parsimony': 0.15, 'redundancy': 0.20, 'connectivity': 0.15,
        'dag_validity': 0.10, 'bridge_coverage': 0.10, 'load_efficiency': 0.10,
        'modularity': 0.10, 'depth_ratio': 0.10,
    }
    scores = {k: round(v, 4) for k, v in {
        'parsimony': parsimony, 'redundancy': redundancy,
        'connectivity': connectivity, 'dag_validity': dag_validity,
        'bridge_coverage': bridge_coverage, 'load_efficiency': load_efficiency,
        'modularity': modularity_score, 'depth_ratio': depth_ratio,
    }.items()}
    composite = round(sum(scores[k] * weights[k] for k in weights), 4)
    return composite, scores


def mcts_optimize(modules: list[Module], bridges: list[dict], dag: nx.DiGraph,
                  G: nx.Graph, nodes: list[FileNode], clusters: list[Cluster],
                  threshold: float = 0.95, max_iter: int = 50,
                  verbose: bool = False) -> tuple[float, dict, list[dict]]:
    """MCTS-UCB1 optimization. [B1][B2][B3][B4] Full stochastic engine."""
    # [B4] All 8 actions including split_module
    actions = [
        'merge_modules', 'split_module', 'prune_redundant', 'rebalance_cluster',
        'promote_to_core', 'demote_from_core', 'add_bridge', 'remove_bridge',
    ]

    Q = {a: 0.0 for a in actions}
    N = {a: 0 for a in actions}
    total_visits = 0

    current_score, current_detail = score_structure(modules, bridges, dag, G, nodes, clusters)
    log = [{'iteration': 0, 'score': current_score, 'action': 'initial', 'detail': current_detail}]
    best_score = current_score
    plateau_count = 0

    if verbose:
        print(f"  Φ4: Initial score = {current_score}", file=sys.stderr)

    for iteration in range(1, max_iter + 1):
        total_visits += 1

        # UCB1 action selection
        ucb_scores = {}
        for a in actions:
            if N[a] == 0:
                ucb_scores[a] = float('inf')
            else:
                exploit = Q[a] / N[a]
                explore = math.sqrt(2) * math.sqrt(math.log(total_visits) / N[a])
                ucb_scores[a] = exploit + explore
        action = max(ucb_scores, key=ucb_scores.get)

        # [B1] Stochastic seed per action+iteration
        random.seed(hash((action, iteration, current_score)) % (2**31))

        # [B3] Simulate with wider, gap-scaled deltas
        delta = simulate_action_v2(action, modules, bridges, dag, G, nodes,
                                   clusters, current_detail, iteration)
        new_score = min(1.0, max(0.0, current_score + delta))

        N[action] += 1
        Q[action] += delta

        # [B2] Apply improvements with full action implementations
        if new_score > current_score:
            apply_action_v2(action, modules, bridges, dag, G, nodes, clusters)
            current_score, current_detail = score_structure(
                modules, bridges, dag, G, nodes, clusters)

        log.append({
            'iteration': iteration, 'score': round(current_score, 4),
            'action': action, 'delta': round(delta, 4), 'detail': current_detail,
        })

        if current_score > best_score + 0.001:
            best_score = current_score
            plateau_count = 0
        else:
            plateau_count += 1

        if current_score >= threshold:
            if verbose:
                print(f"  Φ4: Converged at iteration {iteration} (score={current_score})",
                      file=sys.stderr)
            break
        if plateau_count >= 8:  # Increased from 5 to allow more exploration
            if verbose:
                print(f"  Φ4: Plateau at iteration {iteration} (score={current_score})",
                      file=sys.stderr)
            break

    if verbose and iteration == max_iter:
        print(f"  Φ4: Max iterations reached (score={current_score})", file=sys.stderr)

    return current_score, current_detail, log


def simulate_action_v2(action, modules, bridges, dag, G, nodes, clusters,
                       detail, iteration) -> float:
    """[B1][B3] Stochastic simulation with gap-scaled deltas."""
    # Scale delta magnitude by how far the relevant dimension is from 1.0
    def gap_delta(dim, base_pos, base_neg):
        gap = 1.0 - detail.get(dim, 0.5)
        pos = base_pos * (1 + gap * 2)  # Bigger improvement when further from 1.0
        neg = base_neg * (1 - gap)
        return random.uniform(neg, pos)

    if action == 'merge_modules':
        return gap_delta('parsimony', 0.015, -0.003)
    elif action == 'split_module':
        return gap_delta('modularity', 0.010, -0.005)
    elif action == 'prune_redundant':
        return gap_delta('redundancy', 0.020, -0.002)
    elif action == 'rebalance_cluster':
        return gap_delta('modularity', 0.015, -0.003)
    elif action == 'promote_to_core':
        return gap_delta('load_efficiency', 0.010, -0.005)
    elif action == 'demote_from_core':
        if detail.get('parsimony', 0) < 0.8:
            return random.uniform(0.002, 0.012)
        return random.uniform(-0.005, 0.005)
    elif action == 'add_bridge':
        return gap_delta('bridge_coverage', 0.015, -0.003)
    elif action == 'remove_bridge':
        if detail.get('bridge_coverage', 0) > 0.9:
            return random.uniform(0.001, 0.010)
        return random.uniform(-0.008, 0.003)
    return 0.0


def apply_action_v2(action, modules, bridges, dag, G, nodes, clusters):
    """[B2] Complete implementations for all 8 actions."""
    if action == 'merge_modules':
        # Merge smallest pair of same-cluster specific modules with concept overlap
        specifics = [m for m in modules if m.module_type == 'specific']
        if len(specifics) >= 2:
            specifics.sort(key=lambda m: len(m.content))
            for i, a in enumerate(specifics[:20]):
                for b in specifics[i+1:i+10]:
                    if a.cluster_id == b.cluster_id and set(a.concepts) & set(b.concepts):
                        a.content += '\n' + b.content
                        a.concepts = list(set(a.concepts) | set(b.concepts))
                        a.source_files.extend(b.source_files)
                        if b in modules:
                            modules.remove(b)
                        if b.id in dag:
                            for pred in list(dag.predecessors(b.id)):
                                if not dag.has_edge(pred, a.id):
                                    dag.add_edge(pred, a.id)
                            for succ in list(dag.successors(b.id)):
                                if not dag.has_edge(a.id, succ):
                                    dag.add_edge(a.id, succ)
                            dag.remove_node(b.id)
                        return

    elif action == 'split_module':
        # Split largest specific module at midpoint
        specifics = [m for m in modules if m.module_type == 'specific'
                     and len(m.content.split('\n')) > 100]
        if specifics:
            largest = max(specifics, key=lambda m: len(m.content))
            lines = largest.content.split('\n')
            mid = len(lines) // 2
            new_mod = Module(
                id=f"{largest.id}_split",
                source_files=list(largest.source_files),
                concepts=list(largest.concepts),
                content='\n'.join(lines[mid:]),
                tier=2, module_type='specific',
                cluster_id=largest.cluster_id,
            )
            largest.content = '\n'.join(lines[:mid])
            modules.append(new_mod)
            dag.add_node(new_mod.id, tier=2, module_type='specific')
            # Inherit parent edges
            for pred in list(dag.predecessors(largest.id)):
                dag.add_edge(pred, new_mod.id)

    elif action == 'prune_redundant':
        # Remove modules with 100% concept overlap
        for i, a in enumerate(modules):
            if a.module_type in ('core', 'bridge'):
                continue
            for b in modules[i + 1:]:
                if b.module_type in ('core', 'bridge'):
                    continue
                if a.concepts and set(a.concepts) == set(b.concepts):
                    smaller = a if len(a.content) < len(b.content) else b
                    if smaller in modules:
                        modules.remove(smaller)
                    if smaller.id in dag:
                        dag.remove_node(smaller.id)
                    return

    elif action == 'rebalance_cluster':
        # Move weakly-connected node to better-fitting cluster
        node_map = {n.id: n for n in nodes}
        for n in nodes:
            if n.role == 'leaf' and n.cluster >= 0:
                neighbors = list(G.neighbors(n.id))
                if not neighbors:
                    continue
                neighbor_clusters = Counter(
                    node_map[nb].cluster for nb in neighbors
                    if nb in node_map and node_map[nb].cluster >= 0
                )
                best_cluster = neighbor_clusters.most_common(1)[0][0]
                if best_cluster != n.cluster and neighbor_clusters[best_cluster] > len(neighbors) * 0.6:
                    old_c = next((c for c in clusters if c.id == n.cluster), None)
                    new_c = next((c for c in clusters if c.id == best_cluster), None)
                    if old_c and new_c and n.id in old_c.members:
                        old_c.members.remove(n.id)
                        new_c.members.append(n.id)
                        n.cluster = best_cluster
                        # Update modules
                        for m in modules:
                            if m.source_files == [n.id]:
                                m.cluster_id = best_cluster
                        return

    elif action == 'promote_to_core':
        # [B14] Fixed precedence bug
        specifics = [m for m in modules if m.module_type == 'specific']
        if specifics:
            best = max(specifics, key=lambda m: dag.in_degree(m.id) if m.id in dag else 0)
            in_deg = dag.in_degree(best.id) if best.id in dag else 0
            if in_deg >= 3:
                best.module_type = 'core'
                best.tier = 0

    elif action == 'demote_from_core':
        cores = [m for m in modules if m.module_type == 'core']
        if len(cores) > 2:
            weakest = min(cores, key=lambda m: len(m.source_files))
            weakest.module_type = 'specific'
            weakest.tier = 2

    elif action == 'add_bridge':
        # Add bridge between most-connected cluster pair without one
        cluster_map = {n.id: n.cluster for n in nodes}
        cross_pairs = Counter()
        for u, v in G.edges():
            cu, cv = cluster_map.get(u, -1), cluster_map.get(v, -1)
            if cu != cv and cu >= 0 and cv >= 0:
                cross_pairs[tuple(sorted([cu, cv]))] += 1
        existing = {tuple(sorted(b['clusters'])) for b in bridges}
        for pair, count in cross_pairs.most_common():
            if pair not in existing:
                ca, cb = pair
                da = next((c.domain for c in clusters if c.id == ca), 'unknown')
                db = next((c.domain for c in clusters if c.id == cb), 'unknown')
                bridge = {
                    'id': f"bridge_{ca}_{cb}", 'clusters': [ca, cb],
                    'domains': [da, db], 'weight': 0.5,
                    'shared_concepts': [], 'connections': [],
                }
                bridges.append(bridge)
                mod = Module(
                    id=f"bridge_{ca}_{cb}", source_files=[],
                    concepts=[], content=f"# Bridge: {da} ↔ {db}\n",
                    tier=3, module_type='bridge',
                )
                modules.append(mod)
                dag.add_node(mod.id, tier=3, module_type='bridge')
                return

    elif action == 'remove_bridge':
        # Remove weakest bridge
        bridge_mods = [m for m in modules if m.module_type == 'bridge']
        if len(bridge_mods) > 2:
            weakest = min(bridge_mods, key=lambda m: len(m.content))
            if weakest in modules:
                modules.remove(weakest)
            if weakest.id in dag:
                dag.remove_node(weakest.id)
            bridges[:] = [b for b in bridges if b['id'] != weakest.id]


# ── Output Generation ────────────────────────────────────────────────────────

def generate_outputs(output_dir: str, nodes: list[FileNode], clusters: list[Cluster],
                     modules: list[Module], bridges: list[dict], dag: nx.DiGraph,
                     topo_order: list[str], topology: dict, final_score: float,
                     score_detail: dict, opt_log: list[dict], verbose: bool = False):
    """Write all output files. [B20] No truncation of tier 2 listing."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'core').mkdir(exist_ok=True)
    (out / 'modules').mkdir(exist_ok=True)
    (out / 'bridges').mkdir(exist_ok=True)

    # ── index.md ──
    lines = ['# Unified Context Index\n']
    lines.append(f'Convergence score: **{final_score}** | '
                 f'Modules: {len(modules)} | Clusters: {len(clusters)}\n')
    lines.append('## Tier 0 — Core Abstractions (always load)\n')
    for m in sorted(modules, key=lambda m: m.depth):
        if m.tier == 0:
            lines.append(f'- [`core/{m.id}.md`](core/{m.id}.md) — {", ".join(m.concepts[:5])}')
    lines.append('\n## Tier 1 — Cluster Bases (load on domain match)\n')
    for m in modules:
        if m.tier == 1:
            lines.append(f'- [`modules/{m.id}.md`](modules/{m.id}.md)')
    lines.append('\n## Tier 2 — Specific Modules (load on demand)\n')
    for c in clusters:
        domain_dir = out / 'modules' / c.domain
        domain_dir.mkdir(exist_ok=True)
        domain_modules = [m for m in modules if m.tier == 2 and m.cluster_id == c.id]
        if domain_modules:
            lines.append(f'\n### {c.domain.title()} ({len(domain_modules)} modules)')
            # [B20] Show all, don't truncate
            for m in domain_modules:
                lines.append(f'- `modules/{c.domain}/{m.id}.md`')
    lines.append('\n## Tier 3 — Bridges (load on cross-domain query)\n')
    for b in bridges:
        d = b['domains']
        lines.append(f'- [`bridges/{b["id"]}.md`](bridges/{b["id"]}.md) — {d[0]} ↔ {d[1]}')
    lines.append('\n## DAG Traversal Order\n')
    lines.append('```')
    for idx, nid in enumerate(topo_order[:50]):
        m = next((m for m in modules if m.id == nid), None)
        if m:
            lines.append(f'{idx:3d}. [{m.module_type:13s}] {nid}')
    if len(topo_order) > 50:
        lines.append(f'    ... and {len(topo_order) - 50} more')
    lines.append('```\n')
    (out / 'index.md').write_text('\n'.join(lines))

    # ── Module files ──
    for m in modules:
        if m.tier == 0:
            (out / 'core' / f'{m.id}.md').write_text(m.content)
        elif m.tier == 1:
            (out / 'modules' / f'{m.id}.md').write_text(m.content)
        elif m.tier == 2:
            domain = 'general'
            for c in clusters:
                if m.cluster_id == c.id:
                    domain = c.domain
                    break
            domain_dir = out / 'modules' / domain
            domain_dir.mkdir(exist_ok=True)
            (domain_dir / f'{m.id}.md').write_text(m.content)
        elif m.module_type == 'bridge':
            (out / 'bridges' / f'{m.id}.md').write_text(m.content)

    # ── analysis.json ──
    analysis = {
        'metadata': {
            'total_files': len(nodes),
            'total_lines': sum(n.line_count for n in nodes),
            'total_bytes': sum(n.byte_size for n in nodes),
            'clusters': len(clusters),
            'modules': len(modules),
            'bridges': len(bridges),
        },
        'topology': topology,
        'nodes': [{
            'id': n.id, 'name': n.name, 'path': n.path,
            'domain': n.domain, 'role': n.role,
            'pagerank': n.pagerank, 'cluster': n.cluster,
            'concepts': [c.term for c in n.concepts[:10]],
            'line_count': n.line_count,
        } for n in sorted(nodes, key=lambda n: n.pagerank, reverse=True)],
        'clusters': [{
            'id': c.id, 'domain': c.domain, 'hub': c.hub,
            'members': c.members, 'size': len(c.members),
            'internal_edges': c.internal_edges, 'external_edges': c.external_edges,
        } for c in clusters],
        'bridges': bridges,
        'dag': {
            'nodes': len(dag.nodes), 'edges': len(dag.edges),
            'depth': max((m.depth for m in modules), default=0),
            'topo_order': topo_order[:50],
        },
        'optimization': {
            'final_score': final_score, 'scores': score_detail,
            'iterations': len(opt_log) - 1,
        },
    }
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=2))

    # ── report.md ──
    report = ['# Unification Report\n']
    report.append('## Summary\n')
    report.append(f'- **Input**: {len(nodes)} files, {sum(n.line_count for n in nodes):,} lines')
    report.append(f'- **Output**: {len(modules)} modules in {len(clusters)} clusters')
    report.append(f'- **Bridges**: {len(bridges)}')
    report.append(f'- **Final Score**: {final_score}\n')
    report.append('## Topology Health\n')
    report.append('| Metric | Value | Target | Pass |')
    report.append('|--------|-------|--------|------|')
    report.append(f'| η (density) | {topology["eta"]} | ≥ {topology["eta_target"]} | '
                  f'{"✓" if topology["eta_pass"] else "✗"} |')
    report.append(f'| φ (isolation) | {topology["phi"]} | < {topology["phi_target"]} | '
                  f'{"✓" if topology["phi_pass"] else "✗"} |')
    report.append(f'| κ (clustering) | {topology["kappa"]} | > {topology["kappa_target"]} | '
                  f'{"✓" if topology["kappa_pass"] else "✗"} |')
    report.append('\n## Score Breakdown\n')
    report.append('| Dimension | Score | Weight | Contribution |')
    report.append('|-----------|-------|--------|--------------|')
    weights = {'parsimony': 0.15, 'redundancy': 0.20, 'connectivity': 0.15,
               'dag_validity': 0.10, 'bridge_coverage': 0.10, 'load_efficiency': 0.10,
               'modularity': 0.10, 'depth_ratio': 0.10}
    for dim, w in weights.items():
        s = score_detail.get(dim, 0)
        report.append(f'| {dim} | {s:.3f} | {w:.2f} | {s * w:.4f} |')
    report.append(f'\n## Clusters\n')
    node_map = {n.id: n for n in nodes}
    for c in clusters:
        hub_name = node_map[c.hub].name if c.hub in node_map else c.hub
        report.append(f'### Cluster {c.id}: {c.domain.title()} ({len(c.members)} members)')
        report.append(f'- Hub: `{hub_name}`')
        report.append(f'- Internal edges: {c.internal_edges}, External: {c.external_edges}')
        report.append(f'- Members: {", ".join(c.members[:10])}'
                      f'{"..." if len(c.members) > 10 else ""}\n')
    report.append('## Hub Nodes (by PageRank)\n')
    hubs = sorted([n for n in nodes if n.role in ('hub', 'bridge')],
                  key=lambda n: n.pagerank, reverse=True)
    for h in hubs[:10]:
        report.append(f'- `{h.name}` — PR={h.pagerank:.4f}, role={h.role}, domain={h.domain}')
    report.append('\n## Optimization Log\n')
    report.append('```')
    for entry in opt_log[:30]:
        report.append(f'  iter={entry["iteration"]:3d}  score={entry["score"]:.4f}  '
                      f'action={entry["action"]}')
    if len(opt_log) > 30:
        report.append(f'  ... {len(opt_log) - 30} more iterations')
    report.append('```\n')
    (out / 'report.md').write_text('\n'.join(report))

    if verbose:
        print(f"  Output written to {output_dir}", file=sys.stderr)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Unify v2: Recursive context unification engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('target', help='Target directory to unify')
    parser.add_argument('--output', default='./unified', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.95, help='Convergence threshold')
    parser.add_argument('--max-iter', type=int, default=50, help='Max MCTS iterations')
    parser.add_argument('--max-clusters', type=int, default=12, help='Max clusters')
    parser.add_argument('--block-size', type=int, default=500, help='Max lines per module')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='Bytes per chunk for large files')  # [B22]
    parser.add_argument('--analyze-only', action='store_true', help='Stop after analysis')
    parser.add_argument('--verbose', action='store_true', help='Print progress to stderr')

    args = parser.parse_args()

    if args.verbose:
        print(f"Unify v2: Starting pipeline on {args.target}", file=sys.stderr)

    # Φ0: Ingest
    if args.verbose:
        print("Phase Φ0: Ingest", file=sys.stderr)
    nodes = ingest(args.target, chunk_size=args.chunk_size, verbose=args.verbose)

    # Φ1: Analyze
    if args.verbose:
        print("Phase Φ1: Analyze", file=sys.stderr)
    G, edges_list = build_graph(nodes, verbose=args.verbose)
    clusters = cluster_graph(G, nodes, max_clusters=args.max_clusters, verbose=args.verbose)
    topology = compute_topology(G, nodes)

    if args.analyze_only:
        # [B21] Write to output dir + stdout
        analysis = {
            'topology': topology,
            'nodes': [{'id': n.id, 'domain': n.domain, 'role': n.role,
                       'pagerank': n.pagerank,
                       'concepts': [c.term for c in n.concepts[:10]]}
                      for n in sorted(nodes, key=lambda n: n.pagerank, reverse=True)],
            'clusters': [{'id': c.id, 'domain': c.domain, 'hub': c.hub,
                          'size': len(c.members)} for c in clusters],
        }
        out_json = json.dumps(analysis, indent=2)
        print(out_json)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'analysis.json').write_text(out_json)
        return

    # Φ2: Decompose
    if args.verbose:
        print("Phase Φ2: Decompose", file=sys.stderr)
    modules, bridges = decompose(nodes, clusters, G,
                                 block_size=args.block_size, verbose=args.verbose)

    # Φ3: Unify
    if args.verbose:
        print("Phase Φ3: Unify (DAG)", file=sys.stderr)
    dag, topo_order = build_dag(modules, nodes, G, verbose=args.verbose)

    # Φ4: Optimize
    if args.verbose:
        print("Phase Φ4: Optimize (MCTS-UCB1)", file=sys.stderr)
    final_score, score_detail, opt_log = mcts_optimize(
        modules, bridges, dag, G, nodes, clusters,
        threshold=args.threshold, max_iter=args.max_iter, verbose=args.verbose,
    )

    # Generate outputs
    if args.verbose:
        print("Generating outputs...", file=sys.stderr)
    generate_outputs(
        args.output, nodes, clusters, modules, bridges, dag,
        topo_order, topology, final_score, score_detail, opt_log,
        verbose=args.verbose,
    )

    print(json.dumps({
        'status': 'complete',
        'input_files': len(nodes),
        'output_modules': len(modules),
        'clusters': len(clusters),
        'bridges': len(bridges),
        'final_score': final_score,
        'scores': score_detail,
        'topology': topology,
        'output_dir': args.output,
    }, indent=2))


if __name__ == '__main__':
    main()
