from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from path.src.core.types import GraphDataBundle
from path.src.data.dataset_loader import GraphDatasetLoader
from path.src.data.graph_builder import GraphBuilder
from path.src.data.community import CommunityDetector


class GraphPreprocessor:
    """
    One-stop preprocessing entrypoint:
        dataset -> nx graph -> importance(old_id) -> community -> edge betweenness

    Supported dataset families:
        - Planetoid: Cora / Citeseer / Pubmed
        - Amazon: Computers / Photo
        - Coauthor: CS / Physics

    Importance alignment:
        - direct old_id alignment
        - recovered new_id -> old_id alignment via node_features.csv

    Output:
        GraphDataBundle with graph-space importance vector.
    """

    @staticmethod
    def build_graph_bundle(
        name: str,
        root: str,
        importance_path: str,
        community_mode: Literal["louvain", "label"] = "louvain",
        node_features_path: Optional[str] = None,
        old_id_col_in_node_features: Optional[str] = None,
        strict_importance_alignment: bool = True,
        importance_fill_value: float = 0.0,
        verbose: bool = True,
    ) -> GraphDataBundle:
        """
        Build a unified graph bundle for downstream modules.

        Args:
            name:
                dataset name or alias
                Supported: Cora / Citeseer / Pubmed / Computers / Photo / CS / Physics
            root:
                dataset root directory
            importance_path:
                path to gnn score csv
            community_mode:
                'louvain' or 'label'
            node_features_path:
                optional node_features.csv path.
                Required when importance file is stored in continuous new_id space.
            old_id_col_in_node_features:
                explicit old-id column name in node_features.csv
            strict_importance_alignment:
                if True, missing old_id scores raise error
                if False, missing old_id scores are filled with importance_fill_value
            importance_fill_value:
                fill value used only when strict_importance_alignment=False
            verbose:
                whether to print diagnostics

        Returns:
            GraphDataBundle
        """
        # --------------------------------------------------
        # 1) load graph dataset
        # --------------------------------------------------
        data = GraphDatasetLoader.load_dataset(name=name, root=root)

        # --------------------------------------------------
        # 2) build networkx graph
        # --------------------------------------------------
        G = GraphBuilder.pyg_to_networkx(data, to_undirected=True)

        # --------------------------------------------------
        # 3) load importance aligned to graph old_id space
        # --------------------------------------------------
        importance, importance_info = GraphDatasetLoader.load_importance_aligned(
            score_path=importance_path,
            num_nodes=int(data.num_nodes),
            node_features_path=node_features_path,
            old_id_col_in_node_features=old_id_col_in_node_features,
            strict=strict_importance_alignment,
            fill_value=importance_fill_value,
            verbose=verbose,
            return_info=True,
        )

        if len(importance) != int(data.num_nodes):
            raise ValueError(
                f"Importance length mismatch. Got {len(importance)}, "
                f"but graph has {data.num_nodes} nodes."
            )

        if np.isnan(importance).any():
            bad = np.where(np.isnan(importance))[0].tolist()
            raise ValueError(
                f"Importance vector still contains NaN after alignment. "
                f"Examples: {bad[:20]}"
            )

        # --------------------------------------------------
        # 4) detect community
        # --------------------------------------------------
        if community_mode == "louvain":
            community = CommunityDetector.detect_louvain(G)

        elif community_mode == "label":
            if not hasattr(data, "y") or data.y is None:
                raise ValueError(
                    f"community_mode='label' requires dataset labels, "
                    f"but dataset '{name}' does not provide usable y."
                )
            community = CommunityDetector.detect_by_labels(data.y)

        else:
            raise ValueError(f"Unsupported community_mode: {community_mode}")

        if len(community) != int(data.num_nodes):
            raise ValueError(
                f"Community length mismatch. Got {len(community)}, "
                f"but graph has {data.num_nodes} nodes."
            )

        # --------------------------------------------------
        # 5) compute edge betweenness
        # --------------------------------------------------
        edge_bc = GraphBuilder.compute_edge_betweenness(G)

        # --------------------------------------------------
        # 6) collect metadata
        # --------------------------------------------------
        metadata = {
            "dataset_name": name,
            "community_mode": community_mode,
            "num_nodes": int(data.num_nodes),
            "num_edges": int(G.number_of_edges()),
            "num_features": int(data.x.shape[1]) if hasattr(data.x, "shape") else None,
            "importance_path": str(importance_path),
            "node_features_path": str(node_features_path) if node_features_path is not None else None,
            "old_id_col_in_node_features": old_id_col_in_node_features,
            "strict_importance_alignment": bool(strict_importance_alignment),
            "importance_fill_value": float(importance_fill_value),
            "importance_alignment": importance_info,
        }

        if verbose:
            print("=" * 80)
            print("[GraphPreprocessor] build_graph_bundle finished")
            print(f"  dataset              = {name}")
            print(f"  num_nodes            = {int(data.num_nodes)}")
            print(f"  num_edges            = {int(G.number_of_edges())}")
            print(f"  num_features         = {metadata['num_features']}")
            print(f"  community_mode       = {community_mode}")
            print(f"  importance_mode      = {importance_info.get('mode')}")
            print(f"  importance_missing   = {importance_info.get('missing_count', 0)}")
            print("=" * 80)

        # --------------------------------------------------
        # 7) pack bundle
        # --------------------------------------------------
        return GraphDataBundle(
            name=name,
            num_nodes=int(data.num_nodes),
            edge_index=data.edge_index,
            x=data.x,
            y=data.y if hasattr(data, "y") else None,
            nx_graph=G,
            importance=np.asarray(importance, dtype=float),
            community=np.asarray(community, dtype=int),
            edge_bc=edge_bc,
            metadata=metadata,
        )