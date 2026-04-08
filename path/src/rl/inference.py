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
        deterministic: bool = True,
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

        rescored.sort(
            key=lambda x: (
                float(x.score or 0.0),
                float((x.fragility or {}).get("single_path_score", 0.0)),
                -len(x.nodes),
            ),
            reverse=True,
        )

        deduped = PathDeduplicator.greedy_deduplicate(
            rescored,
            overlap_threshold=float(overlap_threshold),
            top_q=int(top_q),
        )
        return deduped