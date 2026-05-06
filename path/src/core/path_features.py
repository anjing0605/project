'''
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
        if not path:
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
        edges: List[Tuple[int, int]],
        edge_bc: Dict[Tuple[int, int], float],
    ) -> float:
        if not edges:
            return 0.0
        vals = [float(edge_bc.get((u, v), edge_bc.get((v, u), 0.0))) for u, v in edges]
        return float(np.mean(vals))

    @staticmethod
    def cross_community_ratio(path: List[int], community: np.ndarray) -> float:
        edges = PathFeatureExtractor.path_to_edges(path)
        if not edges:
            return 0.0
        cross = sum(int(community[u] != community[v]) for u, v in edges)
        return float(cross) / float(len(edges))

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
            "path_length": float(len(edges)),
        }
'''
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx


class PathFeatureExtractor:

    @staticmethod
    def path_to_edges(path: List[int]) -> List[Tuple[int, int]]:
        if len(path) < 2:
            return []
        return list(zip(path[:-1], path[1:]))

    @staticmethod
    def avg_node_importance(path: List[int], importance: np.ndarray) -> float:
        if not path:
            return 0.0
        return float(np.mean([importance[v] for v in path]))

    @staticmethod
    def internal_node_importance(path: List[int], importance: np.ndarray) -> float:
        if len(path) <= 2:
            return 0.0
        return float(np.mean([importance[v] for v in path[1:-1]]))

    @staticmethod
    def avg_edge_betweenness(
        edges: List[Tuple[int, int]],
        edge_bc: Dict[Tuple[int, int], float],
    ) -> float:
        if not edges:
            return 0.0
        vals = [edge_bc.get((u, v), edge_bc.get((v, u), 0.0)) for u, v in edges]
        return float(np.mean(vals))

    @staticmethod
    def cross_community_ratio(path: List[int], community: np.ndarray) -> float:
        edges = PathFeatureExtractor.path_to_edges(path)
        if not edges:
            return 0.0
        cross = sum(community[u] != community[v] for u, v in edges)
        return float(cross) / len(edges)

    @staticmethod
    def extract_features(
            path: List[int],
            importance: np.ndarray,
            community: np.ndarray,
            edge_bc: Dict[Tuple[int, int], float],
            shortest_len: int | None = None,
            source: int | None = None,
            target: int | None = None,
    ) -> Dict[str, float]:

        edges = PathFeatureExtractor.path_to_edges(path)
        path_len = len(edges)

        if not path:
            return {
                "avg_node_importance": 0.0,
                "internal_node_importance": 0.0,
                "avg_edge_bc": 0.0,
                "cross_comm_ratio": 0.0,
                "path_length": 0.0,
                "path_length_int": 0,
                "shortest_len": 0.0,
                "same_community": 0.0,
                "pair_score": 0.0,
                "stretch_ratio": 0.0,
                "efficiency_ratio": 0.0,
                "max_edge_bc": 0.0,
                "node_importance_entropy": 0.0,
                "community_diversity": 0.0,
            }

        if source is None:
            source = int(path[0])
        if target is None:
            target = int(path[-1])
        if shortest_len is None:
            shortest_len = path_len

        avg_imp = PathFeatureExtractor.avg_node_importance(path, importance)
        internal_imp = PathFeatureExtractor.internal_node_importance(path, importance)
        avg_bc = PathFeatureExtractor.avg_edge_betweenness(edges, edge_bc)
        cross_ratio = PathFeatureExtractor.cross_community_ratio(path, community)

        same_comm = float(community[source] == community[target])
        pair_score = float(importance[source] + importance[target])

        stretch_ratio = float(path_len / max(int(shortest_len), 1))
        efficiency_ratio = float(int(shortest_len) / max(path_len, 1))

        max_bc = float(
            max(
                [edge_bc.get((u, v), edge_bc.get((v, u), 0.0)) for u, v in edges],
                default=0.0,
            )
        )

        scores = np.asarray([float(importance[v]) for v in path], dtype=float)

        # 防止 importance 中存在负值或全 0 时 entropy 异常
        scores = scores - scores.min() + 1e-8
        probs = scores / max(float(scores.sum()), 1e-12)
        entropy = float(-np.sum(probs * np.log(probs)))

        comms = [int(community[v]) for v in path]
        comm_div = float(len(set(comms)) / max(len(comms), 1))

        return {
            "avg_node_importance": avg_imp,
            "internal_node_importance": internal_imp,
            "avg_edge_bc": avg_bc,
            "cross_comm_ratio": cross_ratio,
            "path_length": float(path_len),

            "path_length_int": int(path_len),
            "shortest_len": float(shortest_len),
            "same_community": same_comm,
            "pair_score": pair_score,

            "stretch_ratio": stretch_ratio,
            "efficiency_ratio": efficiency_ratio,
            "max_edge_bc": max_bc,
            "node_importance_entropy": entropy,
            "community_diversity": comm_div,
        }