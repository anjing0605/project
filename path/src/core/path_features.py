from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class PathFeatureExtractor:
    @staticmethod
    def path_to_edges(path: List[int]) -> List[Tuple[int, int]]:
        if len(path) < 2:
            return []
        return list(zip(path[:-1], path[1:]))

    @staticmethod
    def avg_node_importance(path: List[int], importance: np.ndarray) -> float:
        if len(path) == 0:
            return 0.0
        vals = [float(importance[v]) for v in path]
        return float(np.mean(vals))

    @staticmethod
    def internal_node_importance(path: List[int], importance: np.ndarray) -> float:
        if len(path) <= 2:
            return 0.0
        vals = [float(importance[v]) for v in path[1:-1]]
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def avg_edge_betweenness(
        edges: List[Tuple[int, int]], edge_bc: Dict[Tuple[int, int], float]
    ) -> float:
        if len(edges) == 0:
            return 0.0
        vals = [float(edge_bc.get((u, v), edge_bc.get((v, u), 0.0))) for u, v in edges]
        return float(np.mean(vals))

    @staticmethod
    def cross_community_ratio(path: List[int], community: np.ndarray) -> float:
        edges = PathFeatureExtractor.path_to_edges(path)
        if len(edges) == 0:
            return 0.0
        cross = 0
        for u, v in edges:
            cross += int(community[u] != community[v])
        return float(cross / len(edges))

    @staticmethod
    def extract_features(
        path: List[int],
        importance: np.ndarray,
        community: np.ndarray,
        edge_bc: Dict[Tuple[int, int], float],
    ) -> Dict[str, float]:
        edges = PathFeatureExtractor.path_to_edges(path)
        return {
            "avg_node_importance": PathFeatureExtractor.avg_node_importance(path, importance),
            "internal_node_importance": PathFeatureExtractor.internal_node_importance(path, importance),
            "avg_edge_bc": PathFeatureExtractor.avg_edge_betweenness(edges, edge_bc),
            "cross_comm_ratio": PathFeatureExtractor.cross_community_ratio(path, community),
            "path_length": float(len(path)),
            "num_edges": float(len(edges)),
        }
