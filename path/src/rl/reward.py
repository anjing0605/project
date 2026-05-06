from __future__ import annotations

from typing import Dict, List, Tuple, Any
import networkx as nx
import numpy as np

from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor


class RewardCalculator:
    """
    Unified curriculum reward calculator.

    Stage aliases:
        - "reachability"         -> "stage_a"
        - "fragility_finetune"   -> "stage_b"
        - "fragility_topk_align" -> "stage_c"
    """

    # -----------------------------
    # stage config
    # -----------------------------
    DEFAULT_STAGE_WEIGHTS: Dict[str, Dict[str, float]] = {
        # Stage A: reachability bootstrapping
        "stage_a": {
            "distance_weight": 1.00,
            "importance_weight": 0.00,
            "edge_bc_weight": 0.00,
            "cross_comm_weight": 0.00,
            "step_cost": 0.03,
            "repeat_penalty": 0.60,
            "reach_bonus": 4.00,
            "fail_penalty": 1.00,
            "single_frag_weight": 0.00,
            "marginal_gain_weight": 0.00,
            "overlap_penalty_weight": 0.00,
            "gain_temp": 0.10,   # for stage_c only, kept here for uniformity
        },

        # Stage B: single-path fragility shaping
        "stage_b": {
            "distance_weight": 0.60,
            "importance_weight": 0.10,
            "edge_bc_weight": 0.08,
            "cross_comm_weight": 0.04,
            "step_cost": 0.03,
            "repeat_penalty": 0.50,
            "reach_bonus": 2.00,
            "fail_penalty": 1.00,
            "single_frag_weight": 1.50,
            "marginal_gain_weight": 0.00,
            "overlap_penalty_weight": 0.00,
            "gain_temp": 0.10,
        },

        "stage_c": {
            "distance_weight": 0.35,
            "importance_weight": 0.03,
            "edge_bc_weight": 0.18,
            "cross_comm_weight": 0.12,
            "step_cost": 0.05,
            "repeat_penalty": 0.60,
            "reach_bonus": 1.20,
            "fail_penalty": 1.20,

            "single_frag_weight": 0.25,
            "marginal_gain_weight": 5.00,
            "budget_eff_weight": 3.00,
            "node_cost_weight": 0.02,

            "overlap_penalty_weight": 1.00,
            "node_overlap_penalty_weight": 2.00,
            "negative_gain_penalty_weight": 1.00,
            "low_new_node_penalty_weight": 0.50,
            "min_new_internal_nodes": 1.00,
            "gain_temp": 0.10,

            "selection_single_path_weight": 0.10,
            "new_node_bonus_weight": 0.02,
            "new_node_bonus_cap": 3.00,
        },
    }

    LEGACY_STAGE_ALIAS = {
        # stage A
        "reachability": "stage_a",
        "stage_a": "stage_a",

        # stage B
        "fragility": "stage_b",
        "surrogate": "stage_b",
        "single": "stage_b",
        "fragility_finetune": "stage_b",
        "stage_b": "stage_b",

        # stage C
        "hybrid": "stage_c",
        "topk": "stage_c",
        "set": "stage_c",
        "fragility_topk_align": "stage_c",
        "stage_c": "stage_c",
    }

    # -----------------------------
    # basic helpers
    # -----------------------------
    @staticmethod
    def normalize_stage_name(stage: str) -> str:
        if stage not in RewardCalculator.LEGACY_STAGE_ALIAS:
            raise ValueError(
                f"Unsupported reward stage/mode: {stage}. "
                f"Expected one of {sorted(RewardCalculator.LEGACY_STAGE_ALIAS.keys())}"
            )
        return RewardCalculator.LEGACY_STAGE_ALIAS[stage]

    @staticmethod
    def get_stage_weights(
            stage: str,
            overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        stage = RewardCalculator.normalize_stage_name(stage)
        cfg = dict(RewardCalculator.DEFAULT_STAGE_WEIGHTS[stage])

        if overrides is None:
            return cfg

        key_alias = {
            "node_bonus_weight": "importance_weight",
            "bridge_bonus_weight": "edge_bc_weight",
            "cross_comm_bonus_weight": "cross_comm_weight",
            "fragility_weight": "single_frag_weight",
        }

        for k, v in overrides.items():
            kk = key_alias.get(k, k)

            # surrogate_weights 是 dict，不能 float(v)
            # normalize_fragility / normalize_surrogate 这类布尔控制项也不要混入 reward 权重
            if kk not in cfg:
                continue

            if isinstance(v, (int, float, bool, np.integer, np.floating)):
                cfg[kk] = float(v)

        return cfg

    @staticmethod
    def shortest_distance(G: nx.Graph, u: int, v: int) -> int:
        try:
            return int(nx.shortest_path_length(G, u, v))
        except nx.NetworkXNoPath:
            return 10**6

    @staticmethod
    def _canon_edge(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def _edge_bc_value(
        curr: int,
        nxt: int,
        edge_bc: Dict[Tuple[int, int], float],
    ) -> float:
        e = RewardCalculator._canon_edge(curr, nxt)
        return float(edge_bc.get(e, edge_bc.get((curr, nxt), edge_bc.get((nxt, curr), 0.0))))

    @staticmethod
    def _single_path_score_from_fragility(
        frag: Dict[str, float],
        lambda_E: float,
        lambda_ASP: float,
        lambda_LCC: float,
    ) -> float:
        return (
            float(lambda_E) * float(frag.get("delta_E", 0.0))
            + float(lambda_ASP) * float(frag.get("delta_ASP", 0.0))
            + float(lambda_LCC) * float(frag.get("delta_LCC", 0.0))
        )

    @staticmethod
    def _compress_gain(gain: float, gain_temp: float) -> float:
        gain = max(0.0, float(gain))
        gain_temp = max(1e-6, float(gain_temp))
        return float(np.tanh(gain / gain_temp))

    # -----------------------------
    # unified step reward
    # -----------------------------
    @staticmethod
    def compute_step_reward(
        G: nx.Graph,
        curr: int,
        nxt: int,
        target: int,
        visited: set,
        importance: np.ndarray,
        community: np.ndarray,
        edge_bc: Dict[Tuple[int, int], float],
        stage: str,
        reward_kwargs: Dict[str, Any] | None = None,
    ) -> float:
        cfg = RewardCalculator.get_stage_weights(stage, reward_kwargs)

        d_curr = RewardCalculator.shortest_distance(G, curr, target)
        d_next = RewardCalculator.shortest_distance(G, nxt, target)
        delta_d = float(d_curr - d_next)

        node_imp = float(importance[nxt])
        edge_bc_val = RewardCalculator._edge_bc_value(curr, nxt, edge_bc)
        cross_comm = float(int(community[curr] != community[nxt]))
        repeated = float(int(nxt in visited))

        reward = 0.0
        reward += cfg["distance_weight"] * delta_d
        reward += cfg["importance_weight"] * node_imp
        reward += cfg["edge_bc_weight"] * edge_bc_val
        reward += cfg["cross_comm_weight"] * cross_comm
        reward -= cfg["step_cost"]
        reward -= cfg["repeat_penalty"] * repeated

        return float(reward)

    # -----------------------------
    # unified terminal reward: fail
    # -----------------------------
    @staticmethod
    def terminal_fail_reward(
        stage: str,
        reward_kwargs: Dict[str, Any] | None = None,
    ) -> float:
        cfg = RewardCalculator.get_stage_weights(stage, reward_kwargs)
        fail_penalty = float(cfg.get("fail_penalty", 1.0))
        return -abs(fail_penalty)

    # -----------------------------
    # unified terminal reward: single-path
    # used by stage_a / stage_b
    # -----------------------------
    @staticmethod
    def terminal_single_path_reward(
        G: nx.Graph,
        path: List[int],
        success: bool,
        base_metrics: Dict[str, float],
        edge_bc: Dict[Tuple[int, int], float],
        fragility_evaluator: FragilityEvaluator,
        stage: str,
        reward_kwargs: Dict[str, Any] | None = None,
        lambda_E: float = 0.55,
        lambda_ASP: float = 0.30,
        lambda_LCC: float = 0.00,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = RewardCalculator.get_stage_weights(stage, reward_kwargs)

        if not success:
            fail_reward = RewardCalculator.terminal_fail_reward(stage, reward_kwargs)
            return fail_reward, {
                "delta_E": 0.0,
                "delta_LCC": 0.0,
                "delta_ASP": 0.0,
                "avg_edge_bc": 0.0,
                "single_path_score": 0.0,
                "terminal_reward": float(fail_reward),
            }

        frag = fragility_evaluator.compute_fragility(
            G=G,
            path=path,
            base_metrics=base_metrics,
            num_nodes=G.number_of_nodes(),
        )
        frag = dict(frag)

        edges = PathFeatureExtractor.path_to_edges(path)
        avg_bc = PathFeatureExtractor.avg_edge_betweenness(edges, edge_bc)
        frag["avg_edge_bc"] = float(avg_bc)

        single_path_score = RewardCalculator._single_path_score_from_fragility(
            frag=frag,
            lambda_E=lambda_E,
            lambda_ASP=lambda_ASP,
            lambda_LCC=lambda_LCC,
        )
        frag["single_path_score"] = float(single_path_score)

        terminal_reward = (
            float(cfg["reach_bonus"])
            + float(cfg["single_frag_weight"]) * float(single_path_score)
        )
        frag["terminal_reward"] = float(terminal_reward)

        return float(terminal_reward), frag

    # -----------------------------
    # unified terminal reward: set-level top-k
    # used by stage_c
    # current_set_score / new_set_score are computed in env
    # -----------------------------
    @staticmethod
    def terminal_topk_reward(
            path: List[int],
            success: bool,
            single_path_damage: Dict[str, float],
            current_set_score: float,
            new_set_score: float,
            max_overlap: float,
            overlap_threshold: float,
            edge_bc: Dict[Tuple[int, int], float],
            stage: str,
            reward_kwargs: Dict[str, Any] | None = None,
            lambda_E: float = 0.55,
            lambda_ASP: float = 0.30,
            lambda_LCC: float = 0.00,
            max_node_overlap: float = 0.0,
            new_internal_nodes: int = 0,
            internal_node_count: int = 0,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = RewardCalculator.get_stage_weights(stage, reward_kwargs)

        if not success:
            fail_reward = RewardCalculator.terminal_fail_reward(stage, reward_kwargs)
            return fail_reward, {
                "single_delta_E": 0.0,
                "single_delta_LCC": 0.0,
                "single_delta_ASP": 0.0,
                "single_path_score": 0.0,
                "set_score_before": 0.0,
                "set_score_after": 0.0,
                "marginal_gain": 0.0,
                "compressed_marginal_gain": 0.0,
                "max_overlap": 0.0,
                "max_edge_overlap": 0.0,
                "max_node_overlap": 0.0,
                "new_internal_nodes": 0.0,
                "internal_node_count": 0.0,
                "overlap_penalty": 0.0,
                "node_overlap_penalty": 0.0,
                "negative_gain_penalty": 0.0,
                "low_new_node_penalty": 0.0,
                "selection_score": 0.0,
                "avg_edge_bc": 0.0,
                "terminal_reward": float(fail_reward),
            }

        single_path_damage = dict(single_path_damage)
        single_path_score = RewardCalculator._single_path_score_from_fragility(
            frag=single_path_damage,
            lambda_E=lambda_E,
            lambda_ASP=lambda_ASP,
            lambda_LCC=lambda_LCC,
        )

        marginal_gain = float(new_set_score - current_set_score)
        compressed_gain = RewardCalculator._compress_gain(
            gain=marginal_gain,
            gain_temp=cfg["gain_temp"],
        )
        new_node_count = max(1, int(new_internal_nodes))
        gain_per_new_node = float(marginal_gain) / float(new_node_count)
        compressed_gain_per_node = RewardCalculator._compress_gain(
            gain=gain_per_new_node,
            gain_temp=cfg["gain_temp"],
        )

        node_cost_penalty = (
                float(cfg.get("node_cost_weight", 0.0))
                * float(new_node_count)
        )

        edge_overlap_penalty = (
                float(cfg["overlap_penalty_weight"])
                * max(0.0, float(max_overlap) - float(overlap_threshold))
        )

        node_overlap_penalty = (
                float(cfg.get("node_overlap_penalty_weight", 0.0))
                * max(0.0, float(max_node_overlap) - float(overlap_threshold))
        )

        negative_gain_penalty = (
                float(cfg.get("negative_gain_penalty_weight", 0.0))
                * max(0.0, -float(marginal_gain))
        )

        min_new_internal_nodes = int(cfg.get("min_new_internal_nodes", 0))
        low_new_node_penalty = (
                float(cfg.get("low_new_node_penalty_weight", 0.0))
                * max(0, min_new_internal_nodes - int(new_internal_nodes))
        )

        avg_bc = 0.0
        if len(path) >= 2:
            edges = PathFeatureExtractor.path_to_edges(path)
            avg_bc = PathFeatureExtractor.avg_edge_betweenness(edges, edge_bc)

        selection_single_path_weight = float(
            cfg.get("selection_single_path_weight", 0.10)
        )

        new_node_bonus_weight = float(
            cfg.get("new_node_bonus_weight", 0.0)
        )
        new_node_bonus_cap = max(
            1,
            int(cfg.get("new_node_bonus_cap", 5))
        )
        new_node_bonus = (
                new_node_bonus_weight
                * min(int(new_internal_nodes), new_node_bonus_cap)
                / float(new_node_bonus_cap)
        )

        selection_score = (
                float(marginal_gain)
                + float(cfg.get("budget_eff_weight", 0.0)) * float(gain_per_new_node)
                + selection_single_path_weight * float(single_path_score)
                + float(new_node_bonus)
                - float(node_cost_penalty)
                - float(edge_overlap_penalty)
                - float(node_overlap_penalty)
                - float(negative_gain_penalty)
                - float(low_new_node_penalty)
        )

        terminal_reward = (
                float(cfg["reach_bonus"])
                + float(cfg["single_frag_weight"]) * float(single_path_score)
                + float(cfg["marginal_gain_weight"]) * float(compressed_gain)
                + float(cfg.get("budget_eff_weight", 0.0)) * float(compressed_gain_per_node)
                + float(new_node_bonus)
                - float(node_cost_penalty)
                - float(edge_overlap_penalty)
                - float(node_overlap_penalty)
                - float(negative_gain_penalty)
                - float(low_new_node_penalty)
        )

        info = {
            "single_delta_E": float(single_path_damage.get("delta_E", 0.0)),
            "single_delta_LCC": float(single_path_damage.get("delta_LCC", 0.0)),
            "single_delta_ASP": float(single_path_damage.get("delta_ASP", 0.0)),
            "single_path_score": float(single_path_score),
            "set_score_before": float(current_set_score),
            "set_score_after": float(new_set_score),
            "marginal_gain": float(marginal_gain),
            "compressed_marginal_gain": float(compressed_gain),
            "max_overlap": float(max_overlap),
            "max_edge_overlap": float(max_overlap),
            "max_node_overlap": float(max_node_overlap),
            "new_internal_nodes": float(new_internal_nodes),
            "internal_node_count": float(internal_node_count),
            "overlap_penalty": float(edge_overlap_penalty),
            "node_overlap_penalty": float(node_overlap_penalty),
            "negative_gain_penalty": float(negative_gain_penalty),
            "low_new_node_penalty": float(low_new_node_penalty),
            "selection_score": float(selection_score),
            "avg_edge_bc": float(avg_bc),
            "terminal_reward": float(terminal_reward),
            "new_node_bonus": float(new_node_bonus),
            "selection_single_path_weight": float(selection_single_path_weight),
            "gain_per_new_node": float(gain_per_new_node),
            "compressed_gain_per_node": float(compressed_gain_per_node),
            "node_cost_penalty": float(node_cost_penalty),
            "budget_eff_weight": float(cfg.get("budget_eff_weight", 0.0)),
            "node_cost_weight": float(cfg.get("node_cost_weight", 0.0)),
        }
        return float(terminal_reward), info
    # -----------------------------
    # backward-compatible wrappers
    # -----------------------------
    @staticmethod
    def reachability_step_reward(
        G: nx.Graph,
        curr: int,
        nxt: int,
        target: int,
        visited: set,
        step_cost: float = 0.05,
        repeat_penalty: float = 0.5,
        distance_shaping_weight: float = 0.25,
    ) -> float:
        reward_kwargs = {
            "step_cost": step_cost,
            "repeat_penalty": repeat_penalty,
            "distance_weight": distance_shaping_weight,
            "importance_weight": 0.0,
            "edge_bc_weight": 0.0,
            "cross_comm_weight": 0.0,
        }
        dummy_importance = np.zeros(max(curr, nxt, target) + 1, dtype=float)
        dummy_community = np.zeros(max(curr, nxt, target) + 1, dtype=int)
        dummy_edge_bc: Dict[Tuple[int, int], float] = {}

        return RewardCalculator.compute_step_reward(
            G=G,
            curr=curr,
            nxt=nxt,
            target=target,
            visited=visited,
            importance=dummy_importance,
            community=dummy_community,
            edge_bc=dummy_edge_bc,
            stage="stage_a",
            reward_kwargs=reward_kwargs,
        )

    @staticmethod
    def fragility_step_reward(
        G: nx.Graph,
        curr: int,
        nxt: int,
        target: int,
        visited: set,
        community: np.ndarray,
        edge_bc: Dict[Tuple[int, int], float],
        step_cost: float = 0.05,
        repeat_penalty: float = 0.5,
        distance_shaping_weight: float = 0.15,
        edge_bc_weight: float = 0.10,
        cross_comm_weight: float = 0.05,
    ) -> float:
        dummy_importance = np.zeros(len(community), dtype=float)
        reward_kwargs = {
            "step_cost": step_cost,
            "repeat_penalty": repeat_penalty,
            "distance_weight": distance_shaping_weight,
            "importance_weight": 0.0,
            "edge_bc_weight": edge_bc_weight,
            "cross_comm_weight": cross_comm_weight,
        }
        return RewardCalculator.compute_step_reward(
            G=G,
            curr=curr,
            nxt=nxt,
            target=target,
            visited=visited,
            importance=dummy_importance,
            community=community,
            edge_bc=edge_bc,
            stage="stage_b",
            reward_kwargs=reward_kwargs,
        )

    @staticmethod
    def terminal_reachability_reward(
        success: bool,
        reach_bonus: float = 2.0,
        fail_penalty: float = 1.0,
    ) -> float:
        return float(reach_bonus) if success else -float(fail_penalty)

    @staticmethod
    def terminal_fragility_reward(
        G: nx.Graph,
        path: List[int],
        success: bool,
        base_metrics: Dict[str, float],
        edge_bc: Dict[Tuple[int, int], float],
        fragility_evaluator: FragilityEvaluator,
        lambda_E: float = 0.55,
        lambda_ASP: float = 0.30,
        lambda_LCC: float = 0.00,
        reach_bonus: float = 1.5,
        fail_penalty: float = 1.0,
        single_frag_weight: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        reward_kwargs = {
            "reach_bonus": reach_bonus,
            "fail_penalty": fail_penalty,
            "single_frag_weight": single_frag_weight,
        }
        return RewardCalculator.terminal_single_path_reward(
            G=G,
            path=path,
            success=success,
            base_metrics=base_metrics,
            edge_bc=edge_bc,
            fragility_evaluator=fragility_evaluator,
            stage="stage_b",
            reward_kwargs=reward_kwargs,
            lambda_E=lambda_E,
            lambda_ASP=lambda_ASP,
            lambda_LCC=lambda_LCC,
        )