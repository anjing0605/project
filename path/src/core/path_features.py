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
        shortest_len: int,                  # ⭐ 新增（必须传入）
        source: int,
        target: int,
    ) -> Dict[str, float]:

        edges = PathFeatureExtractor.path_to_edges(path)
        path_len = len(edges)

        # =============================
        # 原有特征（保留）
        # =============================
        avg_imp = PathFeatureExtractor.avg_node_importance(path, importance)
        internal_imp = PathFeatureExtractor.internal_node_importance(path, importance)
        avg_bc = PathFeatureExtractor.avg_edge_betweenness(edges, edge_bc)
        cross_ratio = PathFeatureExtractor.cross_community_ratio(path, community)

        # =============================
        # ⭐ 新增：结构核心特征（必须）
        # =============================
        same_comm = float(community[source] == community[target])
        pair_score = float(importance[source] + importance[target])

        # =============================
        # ⭐ 新增：长度相关（关键）
        # =============================
        stretch_ratio = float(path_len / max(shortest_len, 1))
        efficiency_ratio = float(shortest_len / max(path_len, 1))

        # =============================
        # ⭐ 新增：edge bridge 强度
        # =============================
        max_bc = float(max([edge_bc.get((u, v), edge_bc.get((v, u), 0.0)) for u, v in edges], default=0.0))

        # =============================
        # ⭐ 新增：node entropy（论文常用）
        # =============================
        scores = np.array([importance[v] for v in path]) + 1e-8
        probs = scores / scores.sum()
        entropy = float(-np.sum(probs * np.log(probs)))

        # =============================
        # ⭐ 新增：community diversity
        # =============================
        comms = [community[v] for v in path]
        comm_div = float(len(set(comms)) / len(comms))

        return {
            # ===== 原有 =====
            "avg_node_importance": avg_imp,
            "internal_node_importance": internal_imp,
            "avg_edge_bc": avg_bc,
            "cross_comm_ratio": cross_ratio,
            "path_length": float(path_len),

            # ===== 必须补齐 =====
            "path_length_int": int(path_len),
            "shortest_len": float(shortest_len),
            "same_community": same_comm,
            "pair_score": pair_score,

            # ===== 新增增强 =====
            "stretch_ratio": stretch_ratio,
            "efficiency_ratio": efficiency_ratio,
            "max_edge_bc": max_bc,
            "node_importance_entropy": entropy,
            "community_diversity": comm_div,
        }
