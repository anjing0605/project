from __future__ import annotations

from typing import Dict, List

import networkx as nx

from src.core.path_features import PathFeatureExtractor


class FragilityEvaluator:
    def __init__(self, lambda_E: float = 0.4, lambda_LCC: float = 0.4, lambda_ASP: float = 0.2):
        self.lambda_E = float(lambda_E)
        self.lambda_LCC = float(lambda_LCC)
        self.lambda_ASP = float(lambda_ASP)

    @staticmethod
    def global_efficiency(G: nx.Graph) -> float:
        if G.number_of_nodes() <= 1:
            return 0.0
        return float(nx.global_efficiency(G))

    @staticmethod
    def avg_shortest_path_of_lcc(G: nx.Graph) -> float:
        if G.number_of_nodes() <= 1 or G.number_of_edges() == 0:
            return 0.0
        if nx.is_connected(G):
            return float(nx.average_shortest_path_length(G))
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc).copy()
        if H.number_of_nodes() <= 1:
            return 0.0
        return float(nx.average_shortest_path_length(H))

    @staticmethod
    def lcc_ratio(G: nx.Graph, num_nodes: int) -> float:
        if num_nodes <= 0 or G.number_of_nodes() == 0:
            return 0.0
        if G.number_of_nodes() == 1:
            return 1.0 / num_nodes
        largest_cc = max(nx.connected_components(G), key=len)
        return float(len(largest_cc) / num_nodes)

    @staticmethod
    def remove_path_edges(G: nx.Graph, path: List[int]) -> nx.Graph:
        H = G.copy()
        H.remove_edges_from(PathFeatureExtractor.path_to_edges(path))
        return H

    def compute_base_metrics(self, G: nx.Graph) -> Dict[str, float]:
        return {
            "global_efficiency": self.global_efficiency(G),
            "avg_shortest_path_lcc": self.avg_shortest_path_of_lcc(G),
            "lcc_ratio": self.lcc_ratio(G, G.number_of_nodes()),
        }

    def compute_fragility(
        self,
        G: nx.Graph,
        path: List[int],
        base_metrics: Dict[str, float],
        num_nodes: int,
    ) -> Dict[str, float]:
        H = self.remove_path_edges(G, path)
        E0 = float(base_metrics["global_efficiency"])
        ASP0 = float(base_metrics["avg_shortest_path_lcc"])
        LCC0 = float(base_metrics["lcc_ratio"])

        E1 = self.global_efficiency(H)
        ASP1 = self.avg_shortest_path_of_lcc(H)
        LCC1 = self.lcc_ratio(H, num_nodes)

        delta_E = max(0.0, E0 - E1)
        delta_LCC = max(0.0, LCC0 - LCC1)
        delta_ASP = max(0.0, ASP1 - ASP0)

        fragility_score = (
            self.lambda_E * delta_E
            + self.lambda_LCC * delta_LCC
            + self.lambda_ASP * delta_ASP
        )
        return {
            "delta_E": float(delta_E),
            "delta_LCC": float(delta_LCC),
            "delta_ASP": float(delta_ASP),
            "fragility_score": float(fragility_score),
        }
