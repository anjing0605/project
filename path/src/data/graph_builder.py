from __future__ import annotations

from typing import Any, Dict, Tuple

import networkx as nx

try:
    from torch_geometric.utils import to_networkx
except ImportError:  # pragma: no cover
    to_networkx = None


class GraphBuilder:
    @staticmethod
    def pyg_to_networkx(data: Any, to_undirected: bool = True) -> nx.Graph:
        if to_networkx is None:
            raise ImportError(
                "torch_geometric is required for pyg_to_networkx. Please install torch-geometric."
            )
        G = to_networkx(data, to_undirected=to_undirected)
        if not isinstance(G, nx.Graph):
            G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    @staticmethod
    def compute_edge_betweenness(G: nx.Graph) -> Dict[Tuple[int, int], float]:
        raw = nx.edge_betweenness_centrality(G, normalized=True)
        edge_bc: Dict[Tuple[int, int], float] = {}
        for (u, v), score in raw.items():
            edge_bc[(u, v)] = float(score)
            edge_bc[(v, u)] = float(score)
        return edge_bc
