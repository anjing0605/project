from __future__ import annotations

from typing import Dict, List, Tuple, Any

from path.src.core.deduplicate import PathDeduplicator
from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.types import PathRecord, TaskPair


class RLPathInferencer:
    @staticmethod
    def sample_paths(
        agent,
        env,
        tasks: List[TaskPair],
        num_samples_per_task: int = 10,
        keep_failed: bool = False,
        deterministic: bool = False,
    ) -> List[PathRecord]:
        """
        Sample raw paths from trained PPO policy.
        """
        records: List[PathRecord] = []

        for task in tasks:
            for _ in range(int(num_samples_per_task)):
                ep_result, _ = agent.rollout_episode(
                    env,
                    task,
                    deterministic=deterministic,
                )
                if ep_result.path.success or keep_failed:
                    records.append(ep_result.path)

        return records

    @staticmethod
    def _compute_rescore(
        feats: Dict[str, float],
        frag: Dict[str, float],
        score_weights: Dict[str, float] | None = None,
    ) -> float:
        score_weights = score_weights or {
            "single_path_score": 0.75,
            "avg_node_importance": 0.15,
            "avg_edge_bc": 0.10,
            "cross_comm_ratio": 0.05,
            "path_length": 0.05,
        }

        score = 0.0
        score += float(score_weights.get("single_path_score", 0.0)) * float(
            frag.get("single_path_score", 0.0)
        )
        score += float(score_weights.get("avg_node_importance", 0.0)) * float(
            feats.get("avg_node_importance", 0.0)
        )
        score += float(score_weights.get("avg_edge_bc", 0.0)) * float(
            feats.get("avg_edge_bc", 0.0)
        )
        score += float(score_weights.get("cross_comm_ratio", 0.0)) * float(
            feats.get("cross_comm_ratio", 0.0)
        )
        score -= float(score_weights.get("path_length", 0.0)) * float(
            feats.get("path_length", 0.0)
        )

        return float(score)


    @staticmethod
    def rescore_and_select(
            bundle,
            path_records: List[PathRecord],
            top_q: int = 10,
            overlap_threshold: float = 0.6,
            fragility_weights: Dict[str, float] | None = None,
            score_weights: Dict[str, float] | None = None,
            require_success: bool = True,
            min_new_internal_nodes: int = 1,
            max_node_overlap: float = 0.50,
            edge_overlap_penalty_weight: float = 0.25,
            node_overlap_penalty_weight: float = 0.50,
            single_path_weight: float = 0.10,
            set_gain_weight: float = 1.00,
            budget_eff_weight: float = 1.50,
            node_cost_weight: float = 0.02,
            hard_edge_overlap: bool = True,
            min_marginal_gain: float = 0.0,
            min_selection_score: float = 0.0,
            fill_to_top_q: bool = False,
            relaxed_max_node_overlap: float = 0.35,
            relaxed_edge_overlap: float = 0.50,
    ) -> List[PathRecord]:
        fragility_weights = fragility_weights or {
            "lambda_E": 0.55,
            "lambda_LCC": 0.0,
            "lambda_ASP": 0.45,
        }

        evaluator = FragilityEvaluator(**fragility_weights)
        base_metrics = evaluator.compute_base_metrics(bundle.nx_graph)

        rescored: List[PathRecord] = []
        seen = set()
        frag_cache: Dict[Tuple[int, ...], Dict[str, float]] = {}

        for r in path_records:
            if require_success and not bool(getattr(r, "success", False)):
                continue

            if not getattr(r, "nodes", None) or len(r.nodes) < 2:
                continue

            key = tuple(int(n) for n in r.nodes)
            if key in seen:
                continue
            seen.add(key)

            feats = PathFeatureExtractor.extract_features(
                path=r.nodes,
                importance=bundle.importance,
                community=bundle.community,
                edge_bc=bundle.edge_bc,
            )

            if key in frag_cache:
                frag = dict(frag_cache[key])
            else:
                frag = evaluator.compute_fragility(
                    G=bundle.nx_graph,
                    path=r.nodes,
                    base_metrics=base_metrics,
                    num_nodes=bundle.num_nodes,
                )
                frag_cache[key] = dict(frag)

            avg_edge_bc = float(feats.get("avg_edge_bc", 0.0))
            frag["avg_edge_bc"] = avg_edge_bc
            frag["single_path_score"] = (
                float(fragility_weights.get("lambda_E", 0.55)) * float(frag.get("delta_E", 0.0))
                + float(fragility_weights.get("lambda_ASP", 0.45)) * float(frag.get("delta_ASP", 0.0))
                + float(fragility_weights.get("lambda_LCC", 0.0)) * float(frag.get("delta_LCC", 0.0))
            )

            final_score = RLPathInferencer._compute_rescore(
                feats=feats,
                frag=frag,
                score_weights=score_weights,
            )

            merged_features = {
                **{k: float(v) for k, v in feats.items()},
                **{k: float(v) for k, v in frag.items()},
            }

            rescored.append(
                PathRecord(
                    nodes=[int(x) for x in r.nodes],
                    edges=PathFeatureExtractor.path_to_edges(r.nodes),
                    source=int(r.source),
                    target=int(r.target),
                    success=True,
                    method="rl",
                    score=float(final_score),
                    features=merged_features,
                    fragility={k: float(v) for k, v in frag.items()},
                    metadata={
                        **dict(getattr(r, "metadata", {}) or {}),
                        "raw_method": getattr(r, "method", "rl"),
                        "rescore_mode": "single_path_posthoc",
                    },
                )
            )

        def internal_nodes(path: List[int]) -> set[int]:
            if path is None or len(path) <= 2:
                return set()
            return {int(n) for n in path[1:-1]}

        def node_jaccard(a: set[int], b: set[int]) -> float:
            if not a and not b:
                return 0.0
            u = a | b
            return float(len(a & b)) / float(len(u)) if u else 0.0

        set_score_cache: Dict[Tuple[int, ...], Tuple[float, Dict[str, float]]] = {}

        def set_damage_score(node_set: set[int]) -> Tuple[float, Dict[str, float]]:
            key = tuple(sorted(int(n) for n in node_set))
            if key in set_score_cache:
                return set_score_cache[key]

            H = bundle.nx_graph.copy()
            if node_set:
                H.remove_nodes_from([n for n in node_set if H.has_node(n)])

            new_E = evaluator.global_efficiency_approx(H)
            new_ASP = evaluator.avg_shortest_path_of_lcc_approx(H)
            new_LCC = evaluator.lcc_ratio(H, bundle.num_nodes)

            base_E = float(base_metrics.get("global_efficiency", 0.0))
            base_ASP = float(base_metrics.get("avg_shortest_path_lcc", 0.0))
            base_LCC = float(base_metrics.get("lcc_ratio", 0.0))

            damage = {
                "delta_E": float(max(0.0, base_E - float(new_E))),
                "delta_ASP": float(max(0.0, float(new_ASP) - base_ASP)),
                "delta_LCC": float(max(0.0, base_LCC - float(new_LCC))),
            }

            score = (
                    float(fragility_weights.get("lambda_E", 0.55)) * float(damage["delta_E"])
                    + float(fragility_weights.get("lambda_ASP", 0.45)) * float(damage["delta_ASP"])
                    + float(fragility_weights.get("lambda_LCC", 0.0)) * float(damage["delta_LCC"])
            )

            set_score_cache[key] = (float(score), damage)
            return set_score_cache[key]

        selected: List[PathRecord] = []
        selected_internal: set[int] = set()
        remaining = list(rescored)

        current_set_score, _ = set_damage_score(selected_internal)

        while remaining and len(selected) < int(top_q):
            best_idx = None
            best_obj = -1e18
            best_payload = None

            for i, rec in enumerate(remaining):
                cand_internal = internal_nodes(rec.nodes)
                new_nodes = cand_internal - selected_internal

                if selected and len(new_nodes) < min_new_internal_nodes:
                    continue

                max_edge_ov = 0.0
                max_node_ov = 0.0

                for s in selected:
                    max_edge_ov = max(
                        max_edge_ov,
                        PathDeduplicator.edge_overlap(rec.nodes, s.nodes),
                    )
                    max_node_ov = max(
                        max_node_ov,
                        node_jaccard(cand_internal, internal_nodes(s.nodes)),
                    )

                if selected and max_node_ov > max_node_overlap:
                    continue
                if selected and hard_edge_overlap and max_edge_ov > float(overlap_threshold):
                    continue

                next_internal = selected_internal | cand_internal
                next_set_score, next_damage = set_damage_score(next_internal)
                marginal_gain = float(next_set_score - current_set_score)

                single_score = float((rec.fragility or {}).get("single_path_score", 0.0))

                new_node_count = max(1, len(new_nodes))
                gain_per_new_node = float(marginal_gain) / float(new_node_count)

                obj = (
                        set_gain_weight * float(marginal_gain)
                        + budget_eff_weight * float(gain_per_new_node)
                        + single_path_weight * float(single_score)
                        - node_cost_weight * float(new_node_count)
                        - edge_overlap_penalty_weight * max(0.0, max_edge_ov - float(overlap_threshold))
                        - node_overlap_penalty_weight * max(0.0, max_node_ov - float(max_node_overlap))
                )

                if obj > best_obj:
                    best_idx = i
                    best_obj = obj
                    best_payload = {
                        "set_score_before": float(current_set_score),
                        "set_score_after": float(next_set_score),
                        "marginal_gain": float(marginal_gain),
                        "gain_per_new_node": float(gain_per_new_node),
                        "selection_score": float(obj),
                        "node_cost": float(node_cost_weight * new_node_count),
                        "new_internal_nodes": float(len(new_nodes)),
                        "candidate_internal_node_count": float(len(cand_internal)),
                        "max_edge_overlap": float(max_edge_ov),
                        "max_node_overlap": float(max_node_ov),
                        "delta_E_union": float(next_damage["delta_E"]),
                        "delta_ASP_union": float(next_damage["delta_ASP"]),
                        "delta_LCC_union": float(next_damage["delta_LCC"]),
                    }

            if best_idx is None or best_payload is None:
                break

            # 不要因为后续路径 marginal_gain 微负就直接终止。
            # TopKAlign 的目标是构造完整路径集合，而不是只保留第一条正边际路径。
            if float(best_obj) <= float(min_selection_score) and not fill_to_top_q:
                break

            if (
                    selected
                    and float(best_payload.get("marginal_gain", 0.0)) <= float(min_marginal_gain)
                    and not fill_to_top_q
            ):
                break

            rec = remaining.pop(best_idx)
            cand_internal = internal_nodes(rec.nodes)

            frag = dict(rec.fragility or {})
            frag.update(best_payload or {})

            features = dict(rec.features or {})
            features.update(best_payload or {})

            selected.append(
                PathRecord(
                    nodes=list(rec.nodes),
                    edges=list(rec.edges),
                    source=int(rec.source),
                    target=int(rec.target),
                    success=True,
                    method="rl",
                    score=float(best_obj),
                    features=features,
                    fragility=frag,
                    metadata={
                        **dict(rec.metadata or {}),
                        "selection_mode": "set_level_greedy_marginal",
                        "rank": len(selected) + 1,
                    },
                )
            )

            selected_internal |= cand_internal
            current_set_score = float((best_payload or {}).get("set_score_after", current_set_score))

        # 兜底补齐：如果严格集合贪心无法凑满 top_q，
        # 用单路径 fragility + 软重叠约束继续补齐。
        # 这样不会虚增 RL 效果，因为最终 evaluator 仍会真实计算 top-k damage。
        if fill_to_top_q and len(selected) < int(top_q) and remaining:
            remaining.sort(
                key=lambda r: (
                    float((r.fragility or {}).get("single_path_score", 0.0)),
                    float(r.score or 0.0),
                ),
                reverse=True,
            )

            for rec in list(remaining):
                if len(selected) >= int(top_q):
                    break

                cand_internal = internal_nodes(rec.nodes)
                new_nodes = cand_internal - selected_internal

                max_edge_ov = 0.0
                max_node_ov = 0.0

                for s in selected:
                    max_edge_ov = max(
                        max_edge_ov,
                        PathDeduplicator.edge_overlap(rec.nodes, s.nodes),
                    )
                    max_node_ov = max(
                        max_node_ov,
                        node_jaccard(cand_internal, internal_nodes(s.nodes)),
                    )

                if selected and len(new_nodes) == 0:
                    continue
                if selected and max_node_ov > float(relaxed_max_node_overlap):
                    continue
                if selected and max_edge_ov > float(relaxed_edge_overlap):
                    continue

                next_internal = selected_internal | cand_internal
                next_set_score, next_damage = set_damage_score(next_internal)
                marginal_gain = float(next_set_score - current_set_score)

                single_score = float((rec.fragility or {}).get("single_path_score", 0.0))
                obj = float(marginal_gain + single_path_weight * single_score)

                payload = {
                    "set_score_before": float(current_set_score),
                    "set_score_after": float(next_set_score),
                    "marginal_gain": float(marginal_gain),
                    "selection_score": float(obj),
                    "new_internal_nodes": float(len(new_nodes)),
                    "candidate_internal_node_count": float(len(cand_internal)),
                    "max_edge_overlap": float(max_edge_ov),
                    "max_node_overlap": float(max_node_ov),
                    "delta_E_union": float(next_damage["delta_E"]),
                    "delta_ASP_union": float(next_damage["delta_ASP"]),
                    "delta_LCC_union": float(next_damage["delta_LCC"]),
                    "fallback_selected": 1.0,
                }

                frag = dict(rec.fragility or {})
                frag.update(payload)

                features = dict(rec.features or {})
                features.update(payload)

                selected.append(
                    PathRecord(
                        nodes=list(rec.nodes),
                        edges=list(rec.edges),
                        source=int(rec.source),
                        target=int(rec.target),
                        success=True,
                        method="rl",
                        score=float(obj),
                        features=features,
                        fragility=frag,
                        metadata={
                            **dict(rec.metadata or {}),
                            "selection_mode": "fallback_single_score_diverse_fill",
                            "rank": len(selected) + 1,
                        },
                    )
                )

                selected_internal |= cand_internal
                current_set_score = float(next_set_score)
        return selected