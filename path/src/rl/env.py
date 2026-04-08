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
        self.reward_mode = reward_mode
        self.reward_kwargs = reward_kwargs or {}

        if self.reward_mode not in {
            "reachability",
            "fragility_finetune",
            "fragility_topk_align",
        }:
            raise ValueError(
                f"Unsupported reward_mode: {self.reward_mode}. "
                f"Expected one of ['reachability', 'fragility_finetune', 'fragility_topk_align']."
            )

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
        if reward_mode not in {
            "reachability",
            "fragility_finetune",
            "fragility_topk_align",
        }:
            raise ValueError(
                f"Unsupported reward_mode: {reward_mode}. "
                f"Expected one of ['reachability', 'fragility_finetune', 'fragility_topk_align']."
            )

        self.reward_mode = reward_mode
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
        valid = [u for u in neighbors if u not in self.visited]

        if not valid:
            valid = neighbors

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
        }
        return state

    def _compute_removed_graph_damage_from_edge_union(
        self,
        edge_union: set[Tuple[int, int]],
    ) -> Dict[str, float]:
        key = tuple(sorted(edge_union))
        if key in self.topk_context.damage_cache:
            return self.topk_context.damage_cache[key]

        G_removed = self.G.copy()
        if edge_union:
            G_removed.remove_edges_from(list(edge_union))

        if hasattr(self.fragility_evaluator, "global_efficiency_approx"):
            new_E = self.fragility_evaluator.global_efficiency_approx(G_removed)
        else:
            new_E = self.fragility_evaluator.global_efficiency(G_removed)

        if hasattr(self.fragility_evaluator, "avg_shortest_path_of_lcc_approx"):
            new_ASP = self.fragility_evaluator.avg_shortest_path_of_lcc_approx(G_removed)
        else:
            new_ASP = self.fragility_evaluator.avg_shortest_path_of_lcc(G_removed)

        new_LCC = self.fragility_evaluator.lcc_ratio(
            G_removed,
            self.G.number_of_nodes(),
        )

        base_E = float(self.base_metrics.get("global_efficiency", 0.0))
        base_ASP = float(self.base_metrics.get("avg_shortest_path_lcc", 0.0))
        base_LCC = float(self.base_metrics.get("lcc_ratio", 0.0))

        delta_E = max(0.0, base_E - float(new_E))
        delta_ASP = max(0.0, float(new_ASP) - base_ASP)
        delta_LCC = max(0.0, base_LCC - float(new_LCC))

        out = {
            "delta_E": float(delta_E),
            "delta_ASP": float(delta_ASP),
            "delta_LCC": float(delta_LCC),
        }
        self.topk_context.damage_cache[key] = out
        return out

    def _set_score_from_damage(self, damage: Dict[str, float]) -> float:
        return (
            float(self.reward_kwargs.get("lambda_E", 0.55)) * float(damage.get("delta_E", 0.0))
            + float(self.reward_kwargs.get("lambda_ASP", 0.30)) * float(damage.get("delta_ASP", 0.0))
            + float(self.reward_kwargs.get("lambda_LCC", 0.0)) * float(damage.get("delta_LCC", 0.0))
        )

    def _selected_edge_union(self) -> set[Tuple[int, int]]:
        union_edges: set[Tuple[int, int]] = set()
        for p in self.topk_context.selected_paths:
            union_edges |= self.topk_context.path_to_edge_set(p)
        return union_edges

    def _marginal_topk_reward_info(self, path: List[int]) -> Dict[str, float]:
        """
        Compute set-level damage / overlap statistics for Stage C.
        The actual terminal reward is computed by RewardCalculator.terminal_topk_reward().
        """
        current_union = self._selected_edge_union()
        current_damage = self._compute_removed_graph_damage_from_edge_union(current_union)
        current_score = self._set_score_from_damage(current_damage)

        candidate_edges = self.topk_context.path_to_edge_set(path)
        new_union = current_union | candidate_edges
        new_damage = self._compute_removed_graph_damage_from_edge_union(new_union)
        new_score = self._set_score_from_damage(new_damage)

        single_path_damage = self.fragility_evaluator.compute_fragility(
            G=self.G,
            path=path,
            base_metrics=self.base_metrics,
            num_nodes=self.G.number_of_nodes(),
        )
        max_overlap = float(self.topk_context.max_overlap_with_selected(path))

        info = {
            "delta_E_union": float(new_damage["delta_E"]),
            "delta_ASP_union": float(new_damage["delta_ASP"]),
            "delta_LCC_union": float(new_damage["delta_LCC"]),
            "set_score_before": float(current_score),
            "set_score_after": float(new_score),
            "max_overlap": float(max_overlap),
            "single_delta_E": float(single_path_damage.get("delta_E", 0.0)),
            "single_delta_ASP": float(single_path_damage.get("delta_ASP", 0.0)),
            "single_delta_LCC": float(single_path_damage.get("delta_LCC", 0.0)),
        }
        return info

    def _maybe_commit_topk_path(self, path: List[int], reward_info: Dict[str, float]) -> None:
        candidate_score = float(reward_info.get("marginal_gain", 0.0))
        if candidate_score <= 0:
            return

        if len(self.topk_context.selected_paths) < self.topk_context.top_k:
            self.topk_context.selected_paths.append(list(path))
            self.topk_context.selected_scores.append(candidate_score)
            return

        min_idx = int(np.argmin(self.topk_context.selected_scores))
        if candidate_score > self.topk_context.selected_scores[min_idx]:
            self.topk_context.selected_paths[min_idx] = list(path)
            self.topk_context.selected_scores[min_idx] = candidate_score

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
        dead_end = len(list(self.G.neighbors(nxt))) == 0

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

        info["repeated"] = int(repeated)
        info["path_length"] = len(self.path)

        next_state = self.build_state()
        return next_state, float(reward), self.done, info