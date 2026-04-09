from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

from path.src.core.deduplicate import PathDeduplicator
from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.path_generator import PathGenerator
from path.src.core.path_scorer import RulePathScorer
from path.src.core.types import GraphDataBundle, PathRecord, TaskPair


class RuleBasedCriticalPath:
    """
    Rule-based critical path identification.

    Final score(P) =
        a * avg_node_importance
      + b * avg_edge_bc
      + c * cross_comm_ratio
      + d * fragility_score
      - e * path_length

    Notes
    -----
    1. Cheap-stage coarse screening:
       first use cheap structural features only, then compute fragility only
       for top_m_for_fragility candidates within each task.

    2. Shared base metrics:
       if shared_base_metrics is provided, reuse it across methods so that
       all methods are evaluated on the same original graph baseline.

    3. Candidate coverage statistics:
       if candidate_stats is provided, this function will fill it in-place
       with task-level and global candidate-pool statistics.
    """

    DEFAULT_WEIGHTS = {
        "avg_node_importance": 0.12,
        "avg_edge_bc": 0.10,
        "cross_comm_ratio": 0.03,
        "fragility_score": 0.65,
        "path_length": 0.10,
    }

    # cheap-stage coarse screening weights
    DEFAULT_CHEAP_WEIGHTS = {
        "avg_node_importance": 0.45,
        "avg_edge_bc": 0.30,
        "cross_comm_ratio": 0.15,
        "path_length": 0.10,  # penalty
    }

    @staticmethod
    def _safe_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def _cheap_score_from_raw_features(
        features: Dict[str, float],
        cheap_weights: Dict[str, float],
    ) -> float:
        """
        Cheap-stage score without fragility.
        Since each task usually has very few candidates (e.g. k=3),
        raw-feature coarse ranking is sufficient.
        """
        return float(
            cheap_weights["avg_node_importance"] * float(features.get("avg_node_importance", 0.0))
            + cheap_weights["avg_edge_bc"] * float(features.get("avg_edge_bc", 0.0))
            + cheap_weights["cross_comm_ratio"] * float(features.get("cross_comm_ratio", 0.0))
            - cheap_weights["path_length"] * float(features.get("path_length", 0.0))
        )

    @classmethod
    def _build_candidate_stats(
        cls,
        *,
        tasks: List[TaskPair],
        total_paths_generated: int,
        total_paths_after_coarse: int,
        task_candidate_rows: List[Dict[str, Any]],
        path_length_hist: Dict[int, int],
        all_candidate_avg_node_importance: List[float],
        all_candidate_avg_edge_bc: List[float],
        all_candidate_cross_comm_ratio: List[float],
        all_candidate_fragility_scores: List[float],
        total_cache_hits: int,
        total_cache_misses: int,
    ) -> Dict[str, Any]:
        num_tasks = int(len(tasks))

        return {
            "num_tasks": num_tasks,
            "total_candidates": int(total_paths_generated),
            "total_candidates_after_coarse": int(total_paths_after_coarse),
            "mean_candidates_per_task": (
                float(total_paths_generated / num_tasks) if num_tasks > 0 else 0.0
            ),
            "mean_candidates_after_coarse_per_task": (
                float(total_paths_after_coarse / num_tasks) if num_tasks > 0 else 0.0
            ),
            "path_length_distribution": {
                str(k): int(v) for k, v in sorted(path_length_hist.items())
            },
            "task_rows": task_candidate_rows,
            "feature_ranges": {
                "avg_node_importance": cls._safe_stats(all_candidate_avg_node_importance),
                "avg_edge_bc": cls._safe_stats(all_candidate_avg_edge_bc),
                "cross_comm_ratio": cls._safe_stats(all_candidate_cross_comm_ratio),
                "fragility_score": cls._safe_stats(all_candidate_fragility_scores),
            },
            "fragility_cache": {
                "hits": int(total_cache_hits),
                "misses": int(total_cache_misses),
                "size": int(total_cache_misses),
            },
        }
    @staticmethod
    def _edge_set_from_path_record(record: PathRecord) -> set[tuple[int, int]]:
        return {tuple(sorted(e)) for e in record.edges}

    @staticmethod
    def _edge_jaccard_overlap(
        a: set[tuple[int, int]],
        b: set[tuple[int, int]],
    ) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _single_path_score(
        record: PathRecord,
        weights: Dict[str, float],
    ) -> float:
        """
        单路径分数：使用 RulePathScorer 已经写回的 normalized features
        """
        feats = record.features or {}

        s = 0.0
        s += float(weights.get("avg_node_importance", 0.0)) * float(feats.get("norm_avg_node_importance", 0.0))
        s += float(weights.get("avg_edge_bc", 0.0)) * float(feats.get("norm_avg_edge_bc", 0.0))
        s += float(weights.get("cross_comm_ratio", 0.0)) * float(feats.get("norm_cross_comm_ratio", 0.0))
        s += float(weights.get("fragility_score", 0.0)) * float(feats.get("norm_fragility_score", 0.0))
        s -= float(weights.get("path_length", 0.0)) * float(feats.get("norm_path_length", 0.0))
        return float(s)

    @classmethod
    def _greedy_set_level_select(
        cls,
        scored_records: List[PathRecord],
        top_q: int,
        overlap_threshold: float,
        lambda_marginal: float = 0.70,
        lambda_single: float = 0.30,
        overlap_penalty: float = 0.20,
    ) -> List[PathRecord]:
        """
        集合级贪心选择：
        每一步选择“对当前已选集合带来最大边际增益”的路径，而不是只看单路径分数。

        定义：
            set_score = lambda_single * single_score
                        + lambda_marginal * marginal_gain
                        - overlap_penalty * max_overlap

        这里的 marginal_gain 用“相对当前已选边集新增了多少边”近似。
        这不是最终 RL/set-level 最优，但足够作为规则法的论文级升级版本。
        """
        if not scored_records or top_q <= 0:
            return []

        remaining = list(scored_records)
        selected: List[PathRecord] = []
        selected_edge_union: set[tuple[int, int]] = set()

        while remaining and len(selected) < top_q:
            best_idx = None
            best_value = float("-inf")
            best_augmented_score = None

            for idx, record in enumerate(remaining):
                edge_set = cls._edge_set_from_path_record(record)
                single_score = float(record.score if record.score is not None else 0.0)

                # 相对当前集合的新增边比例
                if len(edge_set) == 0:
                    marginal_gain = 0.0
                else:
                    new_edges = edge_set - selected_edge_union
                    marginal_gain = float(len(new_edges) / len(edge_set))

                # 与已选集合中最相似路径的 overlap
                if not selected:
                    max_overlap = 0.0
                else:
                    max_overlap = max(
                        cls._edge_jaccard_overlap(edge_set, cls._edge_set_from_path_record(prev))
                        for prev in selected
                    )

                # 超过阈值不必直接硬踢掉，先允许进入打分；但给明显惩罚
                hard_penalty = 1.0 if max_overlap > overlap_threshold else 0.0

                set_level_value = (
                    lambda_single * single_score
                    + lambda_marginal * marginal_gain
                    - overlap_penalty * max_overlap
                    - 0.50 * hard_penalty
                )

                if set_level_value > best_value:
                    best_value = set_level_value
                    best_idx = idx
                    best_augmented_score = {
                        "single_score": single_score,
                        "marginal_gain": marginal_gain,
                        "max_overlap": max_overlap,
                        "set_level_value": set_level_value,
                    }

            chosen = remaining.pop(best_idx)
            chosen.metadata = dict(chosen.metadata or {})
            chosen.metadata["set_level"] = best_augmented_score

            selected.append(chosen)
            selected_edge_union |= cls._edge_set_from_path_record(chosen)

        return selected
    @classmethod
    def run(
        cls,
        bundle: GraphDataBundle,
        tasks: List[TaskPair],
        path_k: int = 3,
        max_hops: int = 8,
        delta: int = 2,
        weights: Dict[str, float] | None = None,
        top_q: int = 10,
        overlap_threshold: float = 0.6,
        fragility_weights: Dict[str, float] | None = None,
        top_m_for_fragility: int = 3,
        fragility_gate: float = 0.50,
        gate_penalty: float = 0.08,
        shared_base_metrics: Dict[str, float] | None = None,
        candidate_stats: Dict[str, Any] | None = None,
    ) -> List[PathRecord]:
        weights = weights or dict(cls.DEFAULT_WEIGHTS)
        fragility_weights = fragility_weights or {
            "lambda_E": 0.4,
            "lambda_LCC": 0.4,
            "lambda_ASP": 0.2,
        }
        cheap_weights = dict(cls.DEFAULT_CHEAP_WEIGHTS)

        print(
            f"[rule] start, num_tasks={len(tasks)}, path_k={path_k}, "
            f"top_m_for_fragility={top_m_for_fragility}",
            flush=True,
        )

        evaluator = FragilityEvaluator(**fragility_weights)

        if shared_base_metrics is None:
            t0 = time.perf_counter()
            base_metrics = evaluator.compute_base_metrics(bundle.nx_graph)
            print(
                f"[rule] base_metrics computed locally in {time.perf_counter() - t0:.2f}s: "
                f"{base_metrics}",
                flush=True,
            )
        else:
            base_metrics = dict(shared_base_metrics)
            print(f"[rule] using shared_base_metrics: {base_metrics}", flush=True)

        # fragility cache
        frag_cache: Dict[Tuple[int, ...], Dict[str, float]] = {}

        candidates: List[PathRecord] = []

        total_gen_time = 0.0
        total_feat_time = 0.0
        total_frag_time = 0.0
        total_paths_generated = 0
        total_paths_after_coarse = 0
        total_cache_hits = 0
        total_cache_misses = 0

        task_candidate_rows: List[Dict[str, Any]] = []
        path_length_hist: Dict[int, int] = {}

        all_candidate_avg_node_importance: List[float] = []
        all_candidate_avg_edge_bc: List[float] = []
        all_candidate_cross_comm_ratio: List[float] = []
        all_candidate_fragility_scores: List[float] = []

        for task_idx, task in enumerate(tasks):
            if task_idx % 10 == 0 or task_idx == len(tasks) - 1:
                print(
                    f"[rule][task] {task_idx + 1}/{len(tasks)} "
                    f"source={task.source} target={task.target} shortest_len={task.shortest_len}",
                    flush=True,
                )

            # 1) generate candidate paths
            t_gen0 = time.perf_counter()
            try:
                paths = PathGenerator.k_shortest_simple_paths(
                    bundle.nx_graph,
                    source=task.source,
                    target=task.target,
                    k=path_k,
                    max_hops=max_hops,
                    delta=delta,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            t_gen = time.perf_counter() - t_gen0
            total_gen_time += t_gen
            total_paths_generated += len(paths)

            print(
                f"[rule][task {task_idx + 1}] generated {len(paths)} paths in {t_gen:.2f}s",
                flush=True,
            )

            if not paths:
                continue

            task_row: Dict[str, Any] = {
                "source": int(task.source),
                "target": int(task.target),
                "shortest_len": int(task.shortest_len),
                "same_community": bool(task.same_community),
                "pair_score": float(task.pair_score),
                "num_candidates": int(len(paths)),
            }

            # 2) cheap feature extraction
            task_cheap_records: List[PathRecord] = []
            for p_idx, path in enumerate(paths):
                t_feat0 = time.perf_counter()
                feats = PathFeatureExtractor.extract_features(
                    path=path,
                    importance=bundle.importance,
                    community=bundle.community,
                    edge_bc=bundle.edge_bc,
                )
                t_feat = time.perf_counter() - t_feat0
                total_feat_time += t_feat

                cheap_score = cls._cheap_score_from_raw_features(feats, cheap_weights)
                path_len_edges = int(len(path) - 1)

                path_length_hist[path_len_edges] = path_length_hist.get(path_len_edges, 0) + 1
                all_candidate_avg_node_importance.append(
                    float(feats.get("avg_node_importance", 0.0))
                )
                all_candidate_avg_edge_bc.append(
                    float(feats.get("avg_edge_bc", 0.0))
                )
                all_candidate_cross_comm_ratio.append(
                    float(feats.get("cross_comm_ratio", 0.0))
                )

                record = PathRecord(
                    nodes=path,
                    edges=PathFeatureExtractor.path_to_edges(path),
                    source=task.source,
                    target=task.target,
                    success=True,
                    method="rule",
                    score=cheap_score,  # temporary cheap score
                    features=feats,
                    fragility=None,
                    metadata={
                        "same_community": task.same_community,
                        "pair_score": task.pair_score,
                        "shortest_len": task.shortest_len,
                        "cheap_score": cheap_score,
                    },
                )
                task_cheap_records.append(record)

                print(
                    f"[rule][cheap] task={task_idx + 1}/{len(tasks)} "
                    f"p={p_idx + 1}/{len(paths)} len={len(path)} "
                    f"feat={t_feat:.2f}s cheap_score={cheap_score:.6f}",
                    flush=True,
                )

            # 3) coarse screening within task
            task_cheap_records.sort(
                key=lambda r: r.score if r.score is not None else -1e18,
                reverse=True,
            )
            selected_for_frag = task_cheap_records[: max(1, int(top_m_for_fragility))]
            total_paths_after_coarse += len(selected_for_frag)

            print(
                f"[rule][coarse] task={task_idx + 1}/{len(tasks)} "
                f"keep {len(selected_for_frag)}/{len(task_cheap_records)} for fragility",
                flush=True,
            )

            task_row["num_candidates_after_coarse"] = int(len(selected_for_frag))
            task_row["mean_candidate_length"] = float(
                sum(len(r.nodes) - 1 for r in task_cheap_records) / len(task_cheap_records)
            ) if task_cheap_records else 0.0
            task_candidate_rows.append(task_row)

            # 4) compute fragility only for kept candidates
            for keep_idx, record in enumerate(selected_for_frag):
                path = record.nodes
                path_key = tuple(path)

                t_frag0 = time.perf_counter()
                if path_key in frag_cache:
                    frag = frag_cache[path_key]
                    cache_hit = True
                    total_cache_hits += 1
                else:
                    frag = evaluator.compute_fragility(
                        G=bundle.nx_graph,
                        path=path,
                        base_metrics=base_metrics,
                        num_nodes=bundle.num_nodes,
                    )
                    frag_cache[path_key] = frag
                    cache_hit = False
                    total_cache_misses += 1

                all_candidate_fragility_scores.append(float(frag.get("fragility_score", 0.0)))

                t_frag = time.perf_counter() - t_frag0
                total_frag_time += t_frag

                print(
                    f"[rule][frag] task={task_idx + 1}/{len(tasks)} "
                    f"keep={keep_idx + 1}/{len(selected_for_frag)} "
                    f"len={len(path)} frag={t_frag:.2f}s cache_hit={cache_hit}",
                    flush=True,
                )

                feats = dict(record.features or {})
                feats.update(frag)

                final_record = replace(
                    record,
                    score=None,   # final normalized score will be assigned later
                    features=feats,
                    fragility=frag,
                )
                candidates.append(final_record)

        print(f"[rule] total_paths_generated = {total_paths_generated}", flush=True)
        print(f"[rule] total_paths_after_coarse = {total_paths_after_coarse}", flush=True)
        print(f"[rule] total candidates = {len(candidates)}", flush=True)
        print(f"[rule] total_gen_time = {total_gen_time:.2f}s", flush=True)
        print(f"[rule] total_feat_time = {total_feat_time:.2f}s", flush=True)
        print(f"[rule] total_frag_time = {total_frag_time:.2f}s", flush=True)
        print(
            f"[rule] frag_cache size = {len(frag_cache)}, "
            f"hits = {total_cache_hits}, misses = {total_cache_misses}",
            flush=True,
        )

        if candidate_stats is not None:
            stats = cls._build_candidate_stats(
                tasks=tasks,
                total_paths_generated=total_paths_generated,
                total_paths_after_coarse=total_paths_after_coarse,
                task_candidate_rows=task_candidate_rows,
                path_length_hist=path_length_hist,
                all_candidate_avg_node_importance=all_candidate_avg_node_importance,
                all_candidate_avg_edge_bc=all_candidate_avg_edge_bc,
                all_candidate_cross_comm_ratio=all_candidate_cross_comm_ratio,
                all_candidate_fragility_scores=all_candidate_fragility_scores,
                total_cache_hits=total_cache_hits,
                total_cache_misses=total_cache_misses,
            )
            candidate_stats.clear()
            candidate_stats.update(stats)

        if not candidates:
            return []

        # 第一步：先做单路径归一化评分
        scored = RulePathScorer.rank_paths(
            path_records=candidates,
            weights=weights,
            fragility_gate=fragility_gate,
            gate_penalty=gate_penalty,
        )

        # 第二步：集合级贪心选择
        selected = cls._greedy_set_level_select(
            scored_records=scored,
            top_q=top_q,
            overlap_threshold=overlap_threshold,
            lambda_marginal=0.70,
            lambda_single=0.30,
            overlap_penalty=0.20,
        )

        print(f"[rule] final selected = {len(selected)}", flush=True)
        return selected