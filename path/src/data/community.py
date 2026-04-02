from __future__ import annotations

import numpy as np
import networkx as nx

try:
    import community as community_louvain  # python-louvain
except ImportError:  # pragma: no cover
    community_louvain = None


class CommunityDetector:
    @staticmethod
    def detect_louvain(G: nx.Graph) -> np.ndarray:
        if community_louvain is None:
            # fallback: connected components / greedy modularity communities
            communities = list(nx.community.greedy_modularity_communities(G))
            comm = np.full(G.number_of_nodes(), -1, dtype=int)
            for cid, nodes in enumerate(communities):
                for n in nodes:
                    comm[int(n)] = cid
            if (comm < 0).any():
                raise RuntimeError("Failed to assign community ids for all nodes.")
            return comm

        partition = community_louvain.best_partition(G)
        comm = np.full(G.number_of_nodes(), -1, dtype=int)
        for node, cid in partition.items():
            comm[int(node)] = int(cid)
        if (comm < 0).any():
            raise RuntimeError("Failed to assign community ids for all nodes.")
        return comm

    @staticmethod
    def detect_by_labels(y) -> np.ndarray:
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        return np.asarray(y, dtype=int).reshape(-1)
