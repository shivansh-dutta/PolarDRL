"""
Graph loaders for the reproduction datasets (Zhu et al. 2021, Table 1 & Sec 7).

Karate is a networkx builtin. Dolphins and Netscience are fetched from KONECT
via datasets/download.sh into datasets/raw/<name>/out.*. Diseasome is NOT on
KONECT under the moreno_disease slug the vault's dataset README originally
guessed (verified 404 as of 2026-07-09) -- it's fetched from the Network Data
Repository (networkrepository.com/bio-diseasome, MatrixMarket .mtx format)
instead, which hosts the same Goh et al. Human Disease Network data (verified
516 nodes / 1188 edges, matching Zhu et al. 2021 Table 1 exactly).

Per the paper ("we implement our experiments on its largest components"),
every loader returns the largest connected component, relabeled to 0..n-1.
"""
from __future__ import annotations

import random
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

DATASETS_ROOT = Path(__file__).resolve().parent.parent.parent / "datasets"
RAW_ROOT = DATASETS_ROOT / "raw"

# KONECT slug for each KONECT-hosted network name.
_KONECT_SLUGS = {
    "dolphins": "dolphins",
    "netscience": "dimacs10-netscience",
}


def _largest_component(graph: nx.Graph) -> nx.Graph:
    """Largest connected component, relabeled to a contiguous 0..n-1 range."""
    components = nx.connected_components(graph)
    largest = max(components, key=len)
    subgraph = graph.subgraph(largest).copy()
    return nx.convert_node_labels_to_integers(subgraph, ordering="sorted")


def load_karate() -> sp.csr_matrix:
    """
    Zachary's karate club (networkx builtin) -- expected 34 nodes, 78 edges.

    nx.karate_club_graph() stores each edge's original interaction-count as a
    'weight' attribute; nx.to_scipy_sparse_array uses that weight by default,
    which would silently turn this into a weighted graph. The paper (Table 1:
    "34 nodes, 78 edges", no weight column) treats Karate as unweighted, same
    as every other loader here -- drop the weight attribute before conversion
    so all edges get adjacency value 1.0.
    """
    graph = _largest_component(nx.karate_club_graph())
    for _u, _v, data in graph.edges(data=True):
        data.clear()
    return nx.to_scipy_sparse_array(graph, format="csr", dtype=float)


def load_konect(name: str) -> sp.csr_matrix:
    """
    Load a KONECT edge-list network from datasets/raw/<name>/, fetched by
    datasets/download.sh. Skips '%' comment lines; treats the file as an
    undirected simple graph (self-loops and duplicate edges collapsed).
    """
    if name not in _KONECT_SLUGS:
        raise ValueError(f"Unknown KONECT dataset {name!r}; expected one of {sorted(_KONECT_SLUGS)}")

    network_dir = RAW_ROOT / name
    edge_files = sorted(network_dir.glob("out.*")) if network_dir.is_dir() else []
    if not edge_files:
        raise FileNotFoundError(
            f"No edge list found under {network_dir} -- run datasets/download.sh first."
        )

    graph = nx.Graph()
    with edge_files[0].open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                graph.add_edge(u, v)

    graph = _largest_component(graph)
    return nx.to_scipy_sparse_array(graph, format="csr", dtype=float)


def load_dolphins() -> sp.csr_matrix:
    """Expected 62 nodes, 159 edges (Zhu et al. 2021, Sec 7)."""
    return load_konect("dolphins")


def load_netscience() -> sp.csr_matrix:
    """Expected 379 nodes, 914 edges (Zhu et al. 2021, Sec 7)."""
    return load_konect("netscience")


def load_diseasome() -> sp.csr_matrix:
    """
    Human Disease Network (Goh et al.), fetched from networkrepository.com's
    bio-diseasome .mtx file via datasets/download.sh (NOT KONECT -- see
    module docstring). Expected 516 nodes, 1188 edges (Zhu et al. 2021,
    Table 1) -- matches this source exactly.
    """
    network_dir = RAW_ROOT / "diseasome"
    mtx_files = sorted(network_dir.glob("*.mtx")) if network_dir.is_dir() else []
    if not mtx_files:
        raise FileNotFoundError(
            f"No .mtx file found under {network_dir} -- run datasets/download.sh first."
        )

    graph = nx.Graph()
    with mtx_files[0].open() as f:
        lines = (line.strip() for line in f)
        lines = (line for line in lines if line and not line.startswith("%"))
        header = next(lines)  # "n n m" -- dimensions and nonzero count
        _n_rows, _n_cols, _nnz = (int(tok) for tok in header.split())
        for line in lines:
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                graph.add_edge(u, v)

    graph = _largest_component(graph)
    return nx.to_scipy_sparse_array(graph, format="csr", dtype=float)


def load_grqc() -> sp.csr_matrix:
    """
    SNAP ca-GrQc (Arxiv General Relativity collaboration network), fetched
    via datasets/download.sh. Smallest network in Zhu et al. 2021 Table 1 --
    expected 4,158 nodes, 13,422 edges after taking the largest connected
    component (verified 2026-07-09: exact match).
    """
    edge_file = RAW_ROOT / "grqc" / "ca-GrQc.txt"
    if not edge_file.exists():
        raise FileNotFoundError(
            f"{edge_file} not found -- run datasets/download.sh first."
        )

    graph = nx.Graph()
    with edge_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                graph.add_edge(u, v)

    graph = _largest_component(graph)
    return nx.to_scipy_sparse_array(graph, format="csr", dtype=float)


# Name -> loader registry for scripts iterating over all four small networks.
LOADERS = {
    "Karate": load_karate,
    "Dolphins": load_dolphins,
    "Netscience": load_netscience,
    "Diseasome": load_diseasome,
}


def load_all() -> dict[str, sp.csr_matrix]:
    """Load all four small networks used in the Figure 1 reproduction."""
    return {name: loader() for name, loader in LOADERS.items()}


def sample_candidate_edges(
    adjacency: sp.spmatrix, n_candidates: int = 30, seed: int | None = None
) -> list[tuple[int, int]]:
    """
    Sample n_candidates non-existing undirected edges uniformly at random.

    Zhu et al. 2021 (Sec 7) fixes |E_C| = 30 for the four small networks but
    does not specify how the candidate set is chosen -- this is our documented
    reproduction choice: a uniform random sample with a fixed seed.
    """
    n = adjacency.shape[0]
    existing = set()
    coo = adjacency.tocoo()
    for u, v in zip(coo.row, coo.col):
        if u != v:
            existing.add((min(u, v), max(u, v)))

    rng = random.Random(seed)
    non_edges: list[tuple[int, int]] = []
    # Sample without materializing all O(n^2) non-edges for larger graphs.
    max_possible = n * (n - 1) // 2 - len(existing)
    if max_possible < n_candidates:
        raise ValueError(
            f"Graph has only {max_possible} non-existing edges, fewer than requested {n_candidates}"
        )

    seen: set[tuple[int, int]] = set()
    while len(non_edges) < n_candidates:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair in existing or pair in seen:
            continue
        seen.add(pair)
        non_edges.append(pair)

    return sorted(non_edges)
