from __future__ import annotations

from typing import Literal

import numpy as np

from src.core.types import GraphDataBundle
from src.data.community import CommunityDetector
from src.data.dataset_loader import PlanetoidDatasetLoader
from src.data.graph_builder import GraphBuilder


class GraphPreprocessor:
    @staticmethod
    def build_graph_bundle(
        name: str,
        root: str,
        importance_path: str,
        community_mode: Literal["louvain", "label"] = "louvain",
    ) -> GraphDataBundle:
        data = PlanetoidDatasetLoader.load_planetoid(name=name, root=root)
        G = GraphBuilder.pyg_to_networkx(data, to_undirected=True)

        importance = PlanetoidDatasetLoader.load_importance(
            importance_path, num_nodes=int(data.num_nodes)
        )

        if community_mode == "louvain":
            community = CommunityDetector.detect_louvain(G)
        elif community_mode == "label":
            community = CommunityDetector.detect_by_labels(data.y)
        else:
            raise ValueError(f"Unsupported community_mode: {community_mode}")

        edge_bc = GraphBuilder.compute_edge_betweenness(G)

        return GraphDataBundle(
            name=name,
            num_nodes=int(data.num_nodes),
            edge_index=data.edge_index,
            x=data.x,
            y=data.y,
            nx_graph=G,
            importance=np.asarray(importance, dtype=float),
            community=np.asarray(community, dtype=int),
            edge_bc=edge_bc,
            metadata={
                "community_mode": community_mode,
                "num_edges": int(G.number_of_edges()),
                "num_features": int(data.x.shape[1]) if hasattr(data.x, "shape") else None,
            },
        )
