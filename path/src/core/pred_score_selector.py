from __future__ import annotations

from typing import Any, List, Tuple
'''
让 rank_set 真正回答这个问题：

学习到的路径评分作为先验，再做集合去冗余，是否优于纯排序？

它不会再偷用真实 fragility，因此实验逻辑更干净。
'''

class PredScoreSelector:
    """
    Use predicted score as the value term and redundancy as the penalty term.

    Objective:
        F(S) = sum_{P in S} pred_score(P) - lambda_red * Redundancy(S)

    This selector does NOT use true fragility or true marginal gain.
    It is used to evaluate:
        rank + set selection
    """

    def __init__(
        self,
        lambda_red: float = 0.2,
        edge_overlap_threshold: float = 0.8,
        max_shared_internal_nodes: int = 5,
        top_q: int = 10,
    ) -> None:
        self.lambda_red = float(lambda_red)
        self.edge_overlap_threshold = float(edge_overlap_threshold)
        self.max_shared_internal_nodes = int(max_shared_internal_nodes)
        self.top_q = int(top_q)

    @staticmethod
    def path_to_edges(path: List[int]) -> List[Tuple[int, int]]:
        return [tuple(sorted((u, v))) for u, v in zip(path[:-1], path[1:])]

    @staticmethod
    def edge_jaccard(a: List[int], b: List[int]) -> float:
        ea = set(PredScoreSelector.path_to_edges(a))
        eb = set(PredScoreSelector.path_to_edges(b))
        inter = len(ea & eb)
        union = len(ea | eb)
        return inter / max(union, 1)

    @staticmethod
    def shared_internal_nodes(a: List[int], b: List[int]) -> int:
        return len(set(a[1:-1]) & set(b[1:-1]))

    def _redundancy_penalty(self, cand: Any, selected: List[Any]) -> float:
        if not selected:
            return 0.0
        ej = 0.0
        sn = 0.0
        for s in selected:
            ej += self.edge_jaccard(cand.nodes, s.nodes)
            sn += self.shared_internal_nodes(cand.nodes, s.nodes)
        ej /= len(selected)
        sn /= len(selected)
        return ej + 0.1 * sn

    def _hard_redundant(self, cand: Any, selected: List[Any]) -> bool:
        for s in selected:
            if self.edge_jaccard(cand.nodes, s.nodes) >= self.edge_overlap_threshold:
                return True
            if self.shared_internal_nodes(cand.nodes, s.nodes) > self.max_shared_internal_nodes:
                return True
        return False

    def select(self, candidates: List[Any]) -> List[Any]:
        """
        True greedy MMR selection.

        At each step, choose:
            argmax_p [ pred_score(p) - lambda_red * redundancy(p, selected) ]

        not simply sorting once by pred_score.
        """
        if not candidates:
            return []

        selected: List[Any] = []
        remaining: List[Any] = list(candidates)

        while remaining and len(selected) < self.top_q:
            best = None
            best_utility = -float("inf")
            best_pred = 0.0
            best_red = 0.0

            for cand in remaining:
                if self._hard_redundant(cand, selected):
                    continue

                pred_score = float(getattr(cand, "score", 0.0))
                red_pen = self._redundancy_penalty(cand, selected)
                utility = pred_score - self.lambda_red * red_pen

                if utility > best_utility:
                    best = cand
                    best_utility = utility
                    best_pred = pred_score
                    best_red = red_pen

            if best is None:
                break

            if getattr(best, "metadata", None) is None:
                best.metadata = {}

            best.metadata.update({
                "selector": "pred_score_selector",
                "pred_score_used": float(best_pred),
                "redundancy_penalty": float(best_red),
                "selection_utility": float(best_utility),
                "selection_step": len(selected) + 1,
                "lambda_red": float(self.lambda_red),
            })

            selected.append(best)
            remaining.remove(best)

        return selected