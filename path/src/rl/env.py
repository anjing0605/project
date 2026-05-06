from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import torch

from path.src.core.path_features import PathFeatureExtractor
from path.src.core.types import GraphDataBundle, TaskPair
from path.src.core.fragility import FragilityEvaluator
from path.src.rl.reward import RewardCalculator

REWARD_MODE_ALIASES = {
    "reachability": "reachability",
    "stage_a": "reachability",

    "fragility": "fragility_finetune",
    "surrogate": "fragility_finetune",
    "single": "fragility_finetune",
    "stage_b": "fragility_finetune",
    "fragility_finetune": "fragility_finetune",

    "hybrid": "fragility_topk_align",
    "topk": "fragility_topk_align",
    "set": "fragility_topk_align",
    "stage_c": "fragility_topk_align",
    "fragility_topk_align": "fragility_topk_align",
}


def normalize_reward_mode(reward_mode: str) -> str:
    key = str(reward_mode).strip()
    if key not in REWARD_MODE_ALIASES:
        raise ValueError(
            f"Unsupported reward_mode: {reward_mode}. "
            f"Expected one of {sorted(REWARD_MODE_ALIASES.keys())}."
        )
    return REWARD_MODE_ALIASES[key]
@dataclass
class OnlineTopKContext:
    """
    Maintain an online top-k path set during Stage 2 training, so that
    terminal reward can approximate the final evaluator's set-level behavior.
    """
    top_k: int = 10
    overlap_threshold: float = 0.6
    selected_paths: List[List[int]] = field(default_factory=list)
    selected_scores: List[float] = field(default_factory=list)
    damage_cache: Dict[Tuple[Tuple[int, int], ...], Dict[str, float]] = field(default_factory=dict)

    @staticmethod
    def _canon_edge(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def path_to_edge_set(path: List[int]) -> set[Tuple[int, int]]:
        if path is None or len(path) < 2:
            return set()
        return {
            OnlineTopKContext._canon_edge(path[i], path[i + 1])
            for i in range(len(path) - 1)
        }

    @staticmethod
    def path_to_internal_node_set(path: List[int]) -> set[int]:
        if path is None or len(path) <= 2:
            return set()
        return {int(n) for n in path[1:-1]}
    @staticmethod
    def node_overlap_ratio(path_a: List[int], path_b: List[int]) -> float:
        na = OnlineTopKContext.path_to_internal_node_set(path_a)
        nb = OnlineTopKContext.path_to_internal_node_set(path_b)
        if not na and not nb:
            return 0.0
        union = na | nb
        inter = na & nb
        return float(len(inter)) / float(len(union)) if union else 0.0

    def max_node_overlap_with_selected(self, path: List[int]) -> float:
        if not self.selected_paths:
            return 0.0
        return max(self.node_overlap_ratio(path, p) for p in self.selected_paths)

    def selected_internal_node_union(self) -> set[int]:
        out: set[int] = set()
        for p in self.selected_paths:
            out |= self.path_to_internal_node_set(p)
        return out
    @staticmethod
    def edge_overlap_ratio(path_a: List[int], path_b: List[int]) -> float:
        ea = OnlineTopKContext.path_to_edge_set(path_a)
        eb = OnlineTopKContext.path_to_edge_set(path_b)
        if not ea and not eb:
            return 0.0
        inter = len(ea & eb)
        union = len(ea | eb)
        return float(inter) / float(union) if union > 0 else 0.0

    def max_overlap_with_selected(self, path: List[int]) -> float:
        if not self.selected_paths:
            return 0.0
        return max(self.edge_overlap_ratio(path, p) for p in self.selected_paths)

    def reset(self) -> None:
        self.selected_paths.clear()
        self.selected_scores.clear()
        self.damage_cache.clear()


class CriticalPathEnv:
    """
    RL environment for critical-path search.

    Supported reward modes:
        - "reachability"         -> stage_a
        - "fragility_finetune"   -> stage_b
        - "fragility_topk_align" -> stage_c
    """

    def __init__(
        self,
        bundle: GraphDataBundle,
        node_embeddings: torch.Tensor,
        max_hops: int = 8,
        reward_mode: str = "reachability",
        reward_kwargs: Dict[str, Any] | None = None,
    ):
        self.bundle = bundle
        self.G: nx.Graph = bundle.nx_graph
        self.importance = np.asarray(bundle.importance, dtype=float)
        self.community = np.asarray(bundle.community, dtype=int)
        self.edge_bc = bundle.edge_bc
        self.node_embeddings = node_embeddings
        self.max_hops = int(max_hops)
        self.reward_mode = normalize_reward_mode(reward_mode)
        self.reward_kwargs = reward_kwargs or {}

        self.fragility_evaluator = FragilityEvaluator(
            lambda_E=float(self.reward_kwargs.get("lambda_E", 0.55)),
            lambda_LCC=float(self.reward_kwargs.get("lambda_LCC", 0.0)),
            lambda_ASP=float(self.reward_kwargs.get("lambda_ASP", 0.30)),
            efficiency_num_pairs=int(self.reward_kwargs.get("efficiency_num_pairs", 2000)),
            asp_num_sources=int(self.reward_kwargs.get("asp_num_sources", 64)),
            random_seed=int(self.reward_kwargs.get("random_seed", 42)),
            use_bridge_shortcut=bool(self.reward_kwargs.get("use_bridge_shortcut", False)),
        )
        self.base_metrics = self.fragility_evaluator.compute_base_metrics(self.G)
        self.fragility_cache: Dict[Any, Any] = {}

        self.topk_context = OnlineTopKContext(
            top_k=int(self.reward_kwargs.get("top_k", 10)),
            overlap_threshold=float(self.reward_kwargs.get("overlap_threshold", 0.6)),
        )

        self.task: TaskPair | None = None
        self.curr_node: int | None = None
        self.target_node: int | None = None
        self.path: List[int] = []
        self.visited: set[int] = set()
        self.num_steps: int = 0
        self.done: bool = False

    def set_reward_mode(
            self,
            reward_mode: str,
            reward_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        self.reward_mode = normalize_reward_mode(reward_mode)
        if reward_kwargs is not None:
            self.reward_kwargs = reward_kwargs

        self.fragility_evaluator = FragilityEvaluator(
            lambda_E=float(self.reward_kwargs.get("lambda_E", 0.55)),
            lambda_LCC=float(self.reward_kwargs.get("lambda_LCC", 0.0)),
            lambda_ASP=float(self.reward_kwargs.get("lambda_ASP", 0.30)),
            efficiency_num_pairs=int(self.reward_kwargs.get("efficiency_num_pairs", 2000)),
            asp_num_sources=int(self.reward_kwargs.get("asp_num_sources", 64)),
            random_seed=int(self.reward_kwargs.get("random_seed", 42)),
            use_bridge_shortcut=bool(self.reward_kwargs.get("use_bridge_shortcut", False)),
        )
        self.base_metrics = self.fragility_evaluator.compute_base_metrics(self.G)

        self.topk_context = OnlineTopKContext(
            top_k=int(self.reward_kwargs.get("top_k", 10)),
            overlap_threshold=float(self.reward_kwargs.get("overlap_threshold", 0.6)),
        )

    def reset_topk_context(self) -> None:
        self.topk_context.reset()

    def reset(self, task: TaskPair) -> Dict[str, Any]:
        self.task = task
        self.curr_node = int(task.source)
        self.target_node = int(task.target)
        self.path = [self.curr_node]
        self.visited = {self.curr_node}
        self.num_steps = 0
        self.done = False
        return self.build_state()

    def get_valid_actions(self) -> List[int]:
        if self.done or self.curr_node is None:
            return []

        neighbors = list(self.G.neighbors(self.curr_node))
        valid = [
            int(u)
            for u in neighbors
            if (int(u) not in self.visited) or (int(u) == self.target_node)
        ]

        # Stage C: 避免复用已经进入 top-k context 的 internal nodes
        if (
                self.reward_mode == "fragility_topk_align"
                and bool(self.reward_kwargs.get("avoid_selected_internal_nodes", False))
        ):
            selected_internal = self.topk_context.selected_internal_node_union()
            valid = [
                int(u)
                for u in valid
                if int(u) == self.target_node or int(u) not in selected_internal
            ]

        return sorted(valid)

    def _safe_shortest_dist(self, u: int, v: int) -> float:
        try:
            return float(nx.shortest_path_length(self.G, u, v))
        except nx.NetworkXNoPath:
            return float(self.bundle.num_nodes)

    def build_state(self) -> Dict[str, Any]:
        assert self.curr_node is not None
        assert self.target_node is not None

        valid_actions = self.get_valid_actions()

        if len(self.path) >= 2:
            edges = PathFeatureExtractor.path_to_edges(self.path)
            avg_node_importance = PathFeatureExtractor.avg_node_importance(
                self.path, self.importance
            )
            avg_edge_bc = PathFeatureExtractor.avg_edge_betweenness(
                edges, self.edge_bc
            )
            cross_comm_ratio = PathFeatureExtractor.cross_community_ratio(
                self.path, self.community
            )
        else:
            avg_node_importance = float(self.importance[self.curr_node])
            avg_edge_bc = 0.0
            cross_comm_ratio = 0.0

        dist_to_target = self._safe_shortest_dist(self.curr_node, self.target_node)

        selected_internal_nodes = self.topk_context.selected_internal_node_union()

        state = {
            "curr_node": int(self.curr_node),
            "target_node": int(self.target_node),
            "curr_emb": self.node_embeddings[self.curr_node].detach().cpu(),
            "target_emb": self.node_embeddings[self.target_node].detach().cpu(),
            "path_length": int(len(self.path)),
            "avg_node_importance": float(avg_node_importance),
            "avg_edge_bc": float(avg_edge_bc),
            "cross_comm_ratio": float(cross_comm_ratio),
            "dist_to_target": float(dist_to_target),
            "valid_actions": list(valid_actions),
            "path_nodes": list(self.path),

            # Stage C 让 policy 看见已有 top-k 集合，避免重复走同一批 internal nodes
            "selected_internal_nodes": list(selected_internal_nodes),
            "selected_internal_count": int(len(selected_internal_nodes)),
        }
        return state

    @staticmethod
    def _path_to_internal_node_set(path: List[int]) -> set[int]:
        if path is None or len(path) <= 2:
            return set()
        return {int(n) for n in path[1:-1]}

    def _compute_removed_graph_damage_from_node_union(
            self,
            node_union: set[int],
    ) -> Dict[str, float]:
        key = tuple(sorted(int(n) for n in node_union))
        if key in self.topk_context.damage_cache:
            return self.topk_context.damage_cache[key]

        G_removed = self.G.copy()
        if node_union:
            G_removed.remove_nodes_from(
                [n for n in node_union if G_removed.has_node(n)]
            )

        new_E = self.fragility_evaluator.global_efficiency_approx(G_removed)
        new_ASP = self.fragility_evaluator.avg_shortest_path_of_lcc_approx(G_removed)
        new_LCC = self.fragility_evaluator.lcc_ratio(
            G_removed,
            self.G.number_of_nodes(),
        )

        base_E = float(self.base_metrics.get("global_efficiency", 0.0))
        base_ASP = float(self.base_metrics.get("avg_shortest_path_lcc", 0.0))
        base_LCC = float(self.base_metrics.get("lcc_ratio", 0.0))

        out = {
            "delta_E": float(max(0.0, base_E - float(new_E))),
            "delta_ASP": float(max(0.0, float(new_ASP) - base_ASP)),
            "delta_LCC": float(max(0.0, base_LCC - float(new_LCC))),
        }

        self.topk_context.damage_cache[key] = out
        return out

    def _set_score_from_damage(self, damage: Dict[str, float]) -> float:
        return (
                float(self.reward_kwargs.get("lambda_E", 0.55)) * float(damage.get("delta_E", 0.0))
                + float(self.reward_kwargs.get("lambda_ASP", 0.30)) * float(damage.get("delta_ASP", 0.0))
                + float(self.reward_kwargs.get("lambda_LCC", 0.0)) * float(damage.get("delta_LCC", 0.0))
        )

    def _selected_internal_node_union(self) -> set[int]:
        union_nodes: set[int] = set()
        for p in self.topk_context.selected_paths:
            union_nodes |= self._path_to_internal_node_set(p)
        return union_nodes

    def _marginal_topk_reward_info(self, path: List[int]) -> Dict[str, float]:
        """
        Compute set-level damage / overlap statistics for Stage C.
        The actual terminal reward is computed by RewardCalculator.terminal_topk_reward().
        """
        current_union = self._selected_internal_node_union()
        current_damage = self._compute_removed_graph_damage_from_node_union(current_union)
        current_score = self._set_score_from_damage(current_damage)

        candidate_nodes = self._path_to_internal_node_set(path)
        new_nodes = candidate_nodes - current_union
        new_union = current_union | candidate_nodes

        new_damage = self._compute_removed_graph_damage_from_node_union(new_union)
        new_score = self._set_score_from_damage(new_damage)

        single_path_damage = self.fragility_evaluator.compute_fragility(
            G=self.G,
            path=path,
            base_metrics=self.base_metrics,
            num_nodes=self.G.number_of_nodes(),
        )

        max_edge_overlap = float(self.topk_context.max_overlap_with_selected(path))
        max_node_overlap = float(self.topk_context.max_node_overlap_with_selected(path))

        info = {
            "delta_E_union": float(new_damage["delta_E"]),
            "delta_ASP_union": float(new_damage["delta_ASP"]),
            "delta_LCC_union": float(new_damage["delta_LCC"]),
            "set_score_before": float(current_score),
            "set_score_after": float(new_score),

            "candidate_internal_node_count": float(len(candidate_nodes)),
            "new_internal_nodes": float(len(new_nodes)),
            "selected_internal_nodes_before": float(len(current_union)),

            "max_overlap": float(max_edge_overlap),
            "max_edge_overlap": float(max_edge_overlap),
            "max_node_overlap": float(max_node_overlap),

            "single_delta_E": float(single_path_damage.get("delta_E", 0.0)),
            "single_delta_ASP": float(single_path_damage.get("delta_ASP", 0.0)),
            "single_delta_LCC": float(single_path_damage.get("delta_LCC", 0.0)),
        }
        return info

    def _maybe_commit_topk_path(self, path: List[int], reward_info: Dict[str, float]) -> None:
        hard_edge_overlap_threshold = float(
            self.reward_kwargs.get(
                "hard_edge_overlap_threshold",
                self.reward_kwargs.get("overlap_threshold", 0.6),
            )
        )

        min_commit_score = float(
            self.reward_kwargs.get("min_commit_selection_score", 0.0)
        )
        min_new_internal_nodes = int(
            self.reward_kwargs.get("min_new_internal_nodes_for_commit", 1)
        )
        hard_node_overlap_threshold = float(
            self.reward_kwargs.get("hard_node_overlap_threshold", 0.55)
        )

        new_internal_nodes = int(reward_info.get("new_internal_nodes", 0))
        max_node_overlap = float(reward_info.get("max_node_overlap", 0.0))
        max_edge_overlap = float(
            reward_info.get(
                "max_edge_overlap",
                reward_info.get("max_overlap", 0.0),
            )
        )

        if self.topk_context.selected_paths:
            if new_internal_nodes < min_new_internal_nodes:
                return
            if max_node_overlap > hard_node_overlap_threshold:
                return
            if max_edge_overlap > hard_edge_overlap_threshold:
                return

        candidate_score = float(
            reward_info.get("selection_score", reward_info.get("marginal_gain", 0.0))
        )
        if candidate_score <= min_commit_score:
            return

        if len(self.topk_context.selected_paths) < self.topk_context.top_k:
            self.topk_context.selected_paths.append(list(path))
            self.topk_context.selected_scores.append(candidate_score)
            return

        min_idx = int(np.argmin(self.topk_context.selected_scores))
        if candidate_score > self.topk_context.selected_scores[min_idx]:
            self.topk_context.selected_paths[min_idx] = list(path)
            self.topk_context.selected_scores[min_idx] = candidate_score

    def _controlled_stretch_bonus(self) -> tuple[float, Dict[str, float]]:
        if self.task is None or self.target_node is None:
            return 0.0, {}

        shortest_len = int(getattr(self.task, "shortest_len", 0))
        if shortest_len <= 0:
            shortest_len = int(self._safe_shortest_dist(self.path[0], self.target_node))

        shortest_len = max(1, shortest_len)
        path_hops = max(1, int(len(self.path) - 1))
        stretch_ratio = float(path_hops) / float(shortest_len)

        weight = float(self.reward_kwargs.get("stretch_bonus_weight", 0.0))
        stretch_min = float(self.reward_kwargs.get("stretch_min", 1.05))
        stretch_max = float(self.reward_kwargs.get("stretch_max", 1.60))
        hard_max = float(self.reward_kwargs.get("stretch_hard_max", 2.00))
        over_penalty_weight = float(self.reward_kwargs.get("stretch_over_penalty_weight", 0.0))

        bonus = 0.0

        if stretch_min <= stretch_ratio <= stretch_max:
            mid = 0.5 * (stretch_min + stretch_max)
            half_width = max(1e-6, 0.5 * (stretch_max - stretch_min))
            bonus = weight * max(0.0, 1.0 - abs(stretch_ratio - mid) / half_width)

        if stretch_ratio > hard_max:
            bonus -= over_penalty_weight * (stretch_ratio - hard_max)

        return float(bonus), {
            "shortest_len": float(shortest_len),
            "path_hops": float(path_hops),
            "stretch_ratio": float(stretch_ratio),
            "stretch_bonus": float(bonus),
        }
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Environment is done. Please call reset() first.")

        assert self.curr_node is not None
        assert self.target_node is not None

        info: Dict[str, Any] = {
            "invalid_action": 0,
            "reached_target": 0,
            "terminal_mode": self.reward_mode,
            "fragility": None,
        }

        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            self.done = True
            info["invalid_action"] = 1
            reward = -1.0
            return self.build_state(), reward, self.done, info

        curr = self.curr_node
        nxt = int(action)

        # -------------------------
        # Unified step reward
        # -------------------------
        reward = RewardCalculator.compute_step_reward(
            G=self.G,
            curr=curr,
            nxt=nxt,
            target=self.target_node,
            visited=self.visited,
            importance=self.importance,
            community=self.community,
            edge_bc=self.edge_bc,
            stage=self.reward_mode,          # legacy mode names are supported in RewardCalculator
            reward_kwargs=self.reward_kwargs,
        )

        # -------------------------
        # Transition
        # -------------------------
        self.curr_node = nxt
        self.path.append(nxt)
        self.num_steps += 1

        repeated = nxt in self.visited
        self.visited.add(nxt)

        reached_target = nxt == self.target_node
        exceed_hops = self.num_steps >= self.max_hops
        dead_end = (not reached_target) and (len(self.get_valid_actions()) == 0)

        self.done = bool(reached_target or exceed_hops or dead_end)

        if reached_target:
            info["reached_target"] = 1

        # -------------------------
        # Terminal reward
        # -------------------------
        if self.done:
            # Stage A: reachability
            if self.reward_mode == "reachability":
                terminal_reward = (
                    RewardCalculator.get_stage_weights(self.reward_mode, self.reward_kwargs)["reach_bonus"]
                    if reached_target
                    else RewardCalculator.terminal_fail_reward(self.reward_mode, self.reward_kwargs)
                )
                reward += float(terminal_reward)

                if reached_target:
                    info["fragility"] = {
                        "terminal_reward": float(terminal_reward),
                    }
                else:
                    info["fragility"] = {
                        "terminal_reward": float(terminal_reward),
                    }

            # Stage B: single-path fragility shaping
            elif self.reward_mode == "fragility_finetune":
                if reached_target:
                    path_key = tuple(self.path)

                    if path_key in self.fragility_cache:
                        frag = self.fragility_cache[path_key]
                        terminal_reward = float(frag["terminal_reward"])
                    else:
                        terminal_reward, frag = RewardCalculator.terminal_single_path_reward(
                            G=self.G,
                            path=self.path,
                            success=True,
                            base_metrics=self.base_metrics,
                            edge_bc=self.edge_bc,
                            fragility_evaluator=self.fragility_evaluator,
                            stage=self.reward_mode,
                            reward_kwargs=self.reward_kwargs,
                            lambda_E=float(self.reward_kwargs.get("lambda_E", 0.55)),
                            lambda_ASP=float(self.reward_kwargs.get("lambda_ASP", 0.30)),
                            lambda_LCC=float(self.reward_kwargs.get("lambda_LCC", 0.0)),
                        )
                        self.fragility_cache[path_key] = frag

                    reward += float(terminal_reward)
                    info["fragility"] = frag

                else:
                    terminal_reward = RewardCalculator.terminal_fail_reward(
                        self.reward_mode, self.reward_kwargs
                    )
                    reward += float(terminal_reward)
                    info["fragility"] = {
                        "delta_E": 0.0,
                        "delta_LCC": 0.0,
                        "delta_ASP": 0.0,
                        "avg_edge_bc": 0.0,
                        "single_path_score": 0.0,
                        "terminal_reward": float(terminal_reward),
                    }

            # Stage C: set-level top-k alignment
            elif self.reward_mode == "fragility_topk_align":
                if reached_target:
                    topk_info = self._marginal_topk_reward_info(self.path)

                    single_path_damage = {
                        "delta_E": float(topk_info["single_delta_E"]),
                        "delta_ASP": float(topk_info["single_delta_ASP"]),
                        "delta_LCC": float(topk_info["single_delta_LCC"]),
                    }

                    terminal_reward, frag = RewardCalculator.terminal_topk_reward(
                        path=self.path,
                        success=True,
                        single_path_damage=single_path_damage,
                        current_set_score=float(topk_info["set_score_before"]),
                        new_set_score=float(topk_info["set_score_after"]),
                        max_overlap=float(topk_info["max_overlap"]),
                        max_node_overlap=float(topk_info["max_node_overlap"]),
                        new_internal_nodes=int(topk_info["new_internal_nodes"]),
                        internal_node_count=int(topk_info["candidate_internal_node_count"]),
                        overlap_threshold=float(self.reward_kwargs.get("overlap_threshold", 0.6)),
                        edge_bc=self.edge_bc,
                        stage=self.reward_mode,
                        reward_kwargs=self.reward_kwargs,
                        lambda_E=float(self.reward_kwargs.get("lambda_E", 0.55)),
                        lambda_ASP=float(self.reward_kwargs.get("lambda_ASP", 0.30)),
                        lambda_LCC=float(self.reward_kwargs.get("lambda_LCC", 0.0)),
                    )

                    frag = {
                        **topk_info,
                        **frag,
                    }

                    reward += float(terminal_reward)
                    info["fragility"] = frag
                    self._maybe_commit_topk_path(self.path, frag)

                else:
                    terminal_reward, frag = RewardCalculator.terminal_topk_reward(
                        path=self.path,
                        success=False,
                        single_path_damage={},
                        current_set_score=0.0,
                        new_set_score=0.0,
                        max_overlap=0.0,
                        max_node_overlap=0.0,
                        new_internal_nodes=0,
                        internal_node_count=0,
                        overlap_threshold=float(self.reward_kwargs.get("overlap_threshold", 0.6)),
                        edge_bc=self.edge_bc,
                        stage=self.reward_mode,
                        reward_kwargs=self.reward_kwargs,
                        lambda_E=float(self.reward_kwargs.get("lambda_E", 0.55)),
                        lambda_ASP=float(self.reward_kwargs.get("lambda_ASP", 0.30)),
                        lambda_LCC=float(self.reward_kwargs.get("lambda_LCC", 0.0)),
                    )
                    reward += float(terminal_reward)
                    info["fragility"] = frag

            else:
                raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")

        if self.done and reached_target and self.reward_mode != "reachability":
            stretch_bonus, stretch_info = self._controlled_stretch_bonus()
            reward += float(stretch_bonus)

            if info.get("fragility") is None:
                info["fragility"] = {}
            info["fragility"].update(stretch_info)
            info["fragility"]["terminal_reward"] = float(
                info["fragility"].get("terminal_reward", 0.0) + stretch_bonus
            )
        info["repeated"] = int(repeated)
        info["path_length"] = len(self.path)

        next_state = self.build_state()
        return next_state, float(reward), self.done, info