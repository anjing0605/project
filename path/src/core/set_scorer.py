from __future__ import annotations

from typing import Dict, List, Tuple, Any

import networkx as nx

from path.src.core.fragility import FragilityEvaluator


class MarginalGainPathSelector:
    """
    Greedy path-set selector:
    1. start from empty selected set S
    2. each round choose path p maximizing marginal gain:
           gain(p | S) = Fragility(S U {p}) - Fragility(S)
    3. meanwhile enforce redundancy constraints
    """

    def __init__(
        self,
        lambda_E: float = 0.4,
        lambda_LCC: float = 0.4,
        lambda_ASP: float = 0.2,
        edge_overlap_threshold: float = 0.5,
        max_shared_internal_nodes: int = 1,
        alpha_pred: float = 0.35,
        alpha_gain: float = 0.65,
    ) -> None:
        self.evaluator = FragilityEvaluator(
            lambda_E=lambda_E,
            lambda_LCC=lambda_LCC,
            lambda_ASP=lambda_ASP,
        )
        self.edge_overlap_threshold = float(edge_overlap_threshold)
        self.max_shared_internal_nodes = int(max_shared_internal_nodes)

        # rerank score = alpha_pred * pred_score + alpha_gain * marginal_gain
        self.alpha_pred = float(alpha_pred)
        self.alpha_gain = float(alpha_gain)

    @staticmethod
    def path_to_edges(path: List[int]) -> List[Tuple[int, int]]:
        return [tuple(sorted((int(u), int(v)))) for u, v in zip(path[:-1], path[1:])]

    @staticmethod
    def edge_jaccard(path_a: List[int], path_b: List[int]) -> float:
        ea = set(MarginalGainPathSelector.path_to_edges(path_a))
        eb = set(MarginalGainPathSelector.path_to_edges(path_b))
        inter = len(ea & eb)
        union = len(ea | eb)
        return inter / max(union, 1)

    @staticmethod
    def shared_internal_nodes(path_a: List[int], path_b: List[int]) -> int:
        na = set(int(x) for x in path_a[1:-1])
        nb = set(int(x) for x in path_b[1:-1])
        return len(na & nb)

    @staticmethod
    def union_edge_set(records: List[Any]) -> set[Tuple[int, int]]:
        out = set()
        for r in records:
            out.update(MarginalGainPathSelector.path_to_edges(r.nodes))
        return out

    def evaluate_edge_set_fragility(
        self,
        G: nx.Graph,
        removed_edges: List[Tuple[int, int]],
        base_metrics: Dict[str, float],
        num_nodes: int,
    ) -> Dict[str, float]:
        H = G.copy()
        H.remove_edges_from(removed_edges)

        E0 = float(base_metrics["global_efficiency"])
        L0 = float(base_metrics["lcc_ratio"])
        A0 = float(base_metrics["avg_shortest_path_lcc"])

        E1 = self.evaluator.global_efficiency_approx(H)
        L1 = self.evaluator.lcc_ratio(H, num_nodes=num_nodes)
        A1 = self.evaluator.avg_shortest_path_of_lcc_approx(H)

        delta_E = max(E0 - E1, 0.0)
        delta_LCC = max(L0 - L1, 0.0)
        delta_ASP = max(A1 - A0, 0.0)

        fragility_score = (
            self.evaluator.lambda_E * delta_E
            + self.evaluator.lambda_LCC * delta_LCC
            + self.evaluator.lambda_ASP * delta_ASP
        )

        return {
            "delta_E": float(delta_E),
            "delta_LCC": float(delta_LCC),
            "delta_ASP": float(delta_ASP),
            "fragility_score": float(fragility_score),
        }

    def is_too_redundant(self, candidate, selected: List[Any]) -> bool:
        for s in selected:
            ej = self.edge_jaccard(candidate.nodes, s.nodes)
            if ej >= self.edge_overlap_threshold:
                return True

            shared = self.shared_internal_nodes(candidate.nodes, s.nodes)
            if shared > self.max_shared_internal_nodes:
                return True
        return False

    def select(
        self,
        G: nx.Graph,
        candidates: List[Any],
        top_q: int,
    ) -> List[Any]:
        if not candidates:
            return []

        base_metrics = self.evaluator.compute_base_metrics(G)
        num_nodes = G.number_of_nodes()

        selected: List[Any] = []
        current_union_edges: set[Tuple[int, int]] = set()
        current_metrics = {
            "delta_E": 0.0,
            "delta_LCC": 0.0,
            "delta_ASP": 0.0,
            "fragility_score": 0.0,
        }

        remaining = list(candidates)

        for step in range(int(top_q)):
            best_idx = -1
            best_rerank_score = float("-inf")
            best_gain = float("-inf")
            best_metrics = None

            for i, cand in enumerate(remaining):
                if self.is_too_redundant(cand, selected):
                    continue

                cand_edges = set(self.path_to_edges(cand.nodes))
                merged_edges = sorted(current_union_edges | cand_edges)

                merged_metrics = self.evaluate_edge_set_fragility(
                    G=G,
                    removed_edges=merged_edges,
                    base_metrics=base_metrics,
                    num_nodes=num_nodes,
                )

                marginal_gain = (
                    merged_metrics["fragility_score"] - current_metrics["fragility_score"]
                )

                pred_score = float(cand.score) if getattr(cand, "score", None) is not None else 0.0
                rerank_score = (
                    self.alpha_pred * pred_score
                    + self.alpha_gain * float(marginal_gain)
                )

                if rerank_score > best_rerank_score:
                    best_idx = i
                    best_rerank_score = float(rerank_score)
                    best_gain = float(marginal_gain)
                    best_metrics = dict(merged_metrics)

            if best_idx < 0:
                break

            chosen = remaining.pop(best_idx)

            if getattr(chosen, "metadata", None) is None:
                chosen.metadata = {}

            chosen.metadata["selector"] = "greedy_marginal_gain"
            chosen.metadata["marginal_gain"] = float(best_gain)
            chosen.metadata["set_fragility_after_select"] = float(best_metrics["fragility_score"])
            chosen.metadata["selection_step"] = int(step + 1)

            selected.append(chosen)
            current_union_edges = self.union_edge_set(selected)
            current_metrics = dict(best_metrics)

        return selected