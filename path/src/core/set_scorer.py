from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import heapq
import networkx as nx

from path.src.core.fragility import FragilityEvaluator


class SubmodularPathSelector:
    """
    Submodular (or near-submodular) greedy selector with (1-1/e) guarantee
    under monotone submodular assumption.

    Objective:
        F(S) = Fragility(S) - lambda_red * Redundancy(S)

    - Greedy maximization with lazy (CELF) acceleration.
    - pred_score is used only as a PRIOR for queue ordering (not in F).
    """

    def __init__(
        self,
        lambda_E: float = 0.4,
        lambda_LCC: float = 0.4,
        lambda_ASP: float = 0.2,
        lambda_red: float = 0.2,                 # 冗余惩罚权重
        edge_overlap_threshold: float = 0.7,     # 仅作为“极端冗余”硬过滤（可选）
        max_shared_internal_nodes: int = 3,      # 同上
        min_marginal_gain: float = 1e-8,         # 单调性阈值（严格版）
        allow_negative_gain: float = 0.0,        # 若<0 则允许轻微负增益
        use_lazy: bool = True,                  # CELF
    ) -> None:
        self.evaluator = FragilityEvaluator(
            lambda_E=lambda_E,
            lambda_LCC=lambda_LCC,
            lambda_ASP=lambda_ASP,
        )
        self.lambda_red = float(lambda_red)
        self.edge_overlap_threshold = float(edge_overlap_threshold)
        self.max_shared_internal_nodes = int(max_shared_internal_nodes)
        self.min_marginal_gain = float(min_marginal_gain)
        self.allow_negative_gain = float(allow_negative_gain)
        self.use_lazy = bool(use_lazy)

    # ---------- utils ----------
    @staticmethod
    def path_to_edges(path: List[int]) -> List[Tuple[int, int]]:
        return [tuple(sorted((u, v))) for u, v in zip(path[:-1], path[1:])]

    @staticmethod
    def edge_jaccard(a: List[int], b: List[int]) -> float:
        ea = set(SubmodularPathSelector.path_to_edges(a))
        eb = set(SubmodularPathSelector.path_to_edges(b))
        inter = len(ea & eb)
        union = len(ea | eb)
        return inter / max(union, 1)

    @staticmethod
    def shared_internal_nodes(a: List[int], b: List[int]) -> int:
        return len(set(a[1:-1]) & set(b[1:-1]))

    @staticmethod
    def path_to_internal_nodes(path: List[int]) -> set[int]:
        if path is None or len(path) <= 2:
            return set()
        return {int(n) for n in path[1:-1]}

    @staticmethod
    def union_internal_node_set(records: List[Any]) -> set[int]:
        s: set[int] = set()
        for r in records:
            s.update(SubmodularPathSelector.path_to_internal_nodes(r.nodes))
        return s

    # ---------- objective ----------
    def _fragility(
            self,
            G: nx.Graph,
            removed_nodes: set[int],
            base_metrics: Dict[str, float],
            num_nodes: int,
    ) -> float:
        H = G.copy()
        H.remove_nodes_from([int(n) for n in removed_nodes if H.has_node(int(n))])

        E0 = base_metrics["global_efficiency"]
        L0 = base_metrics["lcc_ratio"]
        A0 = base_metrics["avg_shortest_path_lcc"]

        E1 = self.evaluator.global_efficiency_approx(H)
        L1 = self.evaluator.lcc_ratio(H, num_nodes=num_nodes)
        A1 = self.evaluator.avg_shortest_path_of_lcc_approx(H)

        dE = max(E0 - E1, 0.0)
        dL = max(L0 - L1, 0.0)
        dA = max(A1 - A0, 0.0)

        return (
            self.evaluator.lambda_E * dE
            + self.evaluator.lambda_LCC * dL
            + self.evaluator.lambda_ASP * dA
        )

    def _redundancy_penalty(self, cand: Any, selected: List[Any]) -> float:
        if not selected:
            return 0.0
        # 平均重叠作为惩罚（可替换为更复杂形式）
        ej = 0.0
        sn = 0.0
        for s in selected:
            ej += self.edge_jaccard(cand.nodes, s.nodes)
            sn += self.shared_internal_nodes(cand.nodes, s.nodes)
        ej /= len(selected)
        sn /= len(selected)
        return ej + 0.1 * sn  # 可调比例

    def _hard_redundant(self, cand: Any, selected: List[Any]) -> bool:
        for s in selected:
            if self.edge_jaccard(cand.nodes, s.nodes) >= self.edge_overlap_threshold:
                return True
            if self.shared_internal_nodes(cand.nodes, s.nodes) > self.max_shared_internal_nodes:
                return True
        return False

    def _marginal_gain(
            self,
            G: nx.Graph,
            base_metrics: Dict[str, float],
            num_nodes: int,
            current_nodes: set[int],
            current_frag: float,
            selected: List[Any],
            cand: Any,
    ) -> Tuple[float, float, float, float]:
        """
        返回：
            utility_gain: 加入冗余惩罚后的选择效用
            raw_gain:     纯结构破坏边际增益
            frag:         加入该路径后的集合 fragility
            red:          当前候选路径相对已选集合的冗余
        """
        cand_nodes = self.path_to_internal_nodes(cand.nodes)
        merged_nodes = current_nodes | cand_nodes

        frag = self._fragility(G, merged_nodes, base_metrics, num_nodes)

        raw_gain = float(frag - current_frag)
        red = float(self._redundancy_penalty(cand, selected))

        utility_gain = raw_gain - self.lambda_red * red

        return utility_gain, raw_gain, frag, red

    # ---------- main ----------
    def select(
        self,
        G: nx.Graph,
        candidates: List[Any],
        top_q: int,
        shared_base_metrics: Optional[Dict[str, float]] = None,
    ) -> List[Any]:

        if not candidates or top_q <= 0:
            return []

        if shared_base_metrics is None:
            base = self.evaluator.compute_base_metrics(G)
        else:
            base = shared_base_metrics

        num_nodes = G.number_of_nodes()

        selected: List[Any] = []
        current_nodes: set[int] = set()
        current_frag = 0.0

        # ---------- Lazy Greedy (CELF) ----------
        # heap 存 (-upper_bound_gain, idx, cand, last_updated_round)
        heap = []
        for i, c in enumerate(candidates):
            # 用 pred_score 作为初始上界（仅排序，不进目标函数）
            ub = float(getattr(c, "score", 0.0))
            heap.append((-ub, i, c, -1))
        heapq.heapify(heap)

        round_id = 0

        while heap and len(selected) < top_q:
            neg_ub, i, cand, last_round = heapq.heappop(heap)

            if self._hard_redundant(cand, selected):
                continue

            # 若不是本轮更新过的，重新计算真实 gain 并放回
            if self.use_lazy and last_round != round_id:
                gain, _, _, _ = self._marginal_gain(
                    G, base, num_nodes, current_nodes, current_frag, selected, cand
                )
                heapq.heappush(heap, (-gain, i, cand, round_id))
                continue

            # 计算真实 gain（若不用 lazy 或已更新）
            gain, raw_gain, frag_new, red_new = self._marginal_gain(
                G, base, num_nodes, current_nodes, current_frag, selected, cand
            )

            # 单调性/放松阈值
            if self.allow_negative_gain < 0:
                gain_threshold = self.allow_negative_gain
            else:
                gain_threshold = self.min_marginal_gain

            if gain <= gain_threshold:
                continue

            # 接受
            if getattr(cand, "metadata", None) is None:
                cand.metadata = {}

            cand.metadata.update({
                "selector": "submodular_greedy",
                "marginal_gain": float(gain),
                "raw_marginal_gain": float(raw_gain),
                "set_fragility_after_select": float(frag_new),
                "selection_step": len(selected) + 1,
                "redundancy_penalty": float(red_new),
                "lambda_red": float(self.lambda_red),
            })

            selected.append(cand)
            current_nodes = self.union_internal_node_set(selected)
            current_frag = float(frag_new)

            round_id += 1

        # 如果正边际收益路径不足，则用预测分补齐，保证 top-k 评估完整。
        if len(selected) < top_q:
            selected_keys = {tuple(r.nodes) for r in selected}

            # 第一轮 fallback：仍然遵守硬重叠约束
            for cand in candidates:
                if len(selected) >= top_q:
                    break

                key = tuple(cand.nodes)
                if key in selected_keys:
                    continue

                if self._hard_redundant(cand, selected):
                    continue

                if getattr(cand, "metadata", None) is None:
                    cand.metadata = {}

                cand.metadata.update({
                    "selector": "submodular_greedy_fallback",
                    "selection_step": len(selected) + 1,
                    "fallback": True,
                    "fallback_relaxed": False,
                    "pred_score_used": float(getattr(cand, "score", 0.0)),
                })

                selected.append(cand)
                selected_keys.add(key)

            # 第二轮 fallback：如果还不够，再放宽重叠约束，只保证数量完整。
            for cand in candidates:
                if len(selected) >= top_q:
                    break

                key = tuple(cand.nodes)
                if key in selected_keys:
                    continue

                if getattr(cand, "metadata", None) is None:
                    cand.metadata = {}

                cand.metadata.update({
                    "selector": "submodular_greedy_fallback",
                    "selection_step": len(selected) + 1,
                    "fallback": True,
                    "fallback_relaxed": True,
                    "pred_score_used": float(getattr(cand, "score", 0.0)),
                })

                selected.append(cand)
                selected_keys.add(key)
        return selected