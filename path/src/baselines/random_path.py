from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.path_generator import PathGenerator
from path.src.core.types import GraphDataBundle, PathRecord, TaskPair


class RandomPathBaseline:
    """Sample random simple paths for each task and keep the strongest one."""

    DEFAULT_CHEAP_WEIGHTS = {
        "avg_node_importance": 0.45,
        "avg_edge_bc": 0.30,
        "cross_comm_ratio": 0.15,
        "path_length": 0.10,   # 惩罚项，后面减掉
    }

    @staticmethod
    def _cheap_score_from_raw_features(
        features: Dict[str, float],
        cheap_weights: Dict[str, float],
    ) -> float:
        return float(
            cheap_weights["avg_node_importance"] * float(features.get("avg_node_importance", 0.0))
            + cheap_weights["avg_edge_bc"] * float(features.get("avg_edge_bc", 0.0))
            + cheap_weights["cross_comm_ratio"] * float(features.get("cross_comm_ratio", 0.0))
            - cheap_weights["path_length"] * float(features.get("path_length", 0.0))
        )

    @staticmethod
    def run(
        bundle: GraphDataBundle,
        tasks: List[TaskPair],
        max_hops: int = 8,
        num_trials: int = 30,
        num_samples: int = 5,
        seed: int = 42,
        fragility_weights: Optional[Dict[str, float]] = None,
        select_by: str = "fragility_score",
        top_m_for_fragility: int = 1,   # 新增：每个 task 粗筛后保留多少条做 fragility
    ) -> List[PathRecord]:
        fragility_weights = fragility_weights or {
            "lambda_E": 0.4,
            "lambda_LCC": 0.4,
            "lambda_ASP": 0.2,
        }

        print(
            f"[random] start, num_tasks={len(tasks)}, max_hops={max_hops}, "
            f"num_trials={num_trials}, num_samples={num_samples}, "
            f"top_m_for_fragility={top_m_for_fragility}, seed={seed}",
            flush=True,
        )

        evaluator = FragilityEvaluator(**fragility_weights)

        t0 = time.perf_counter()
        base_metrics = evaluator.compute_base_metrics(bundle.nx_graph)
        print(
            f"[random] base_metrics computed in {time.perf_counter() - t0:.2f}s: {base_metrics}",
            flush=True,
        )

        rng = random.Random(seed)
        cheap_weights = dict(RandomPathBaseline.DEFAULT_CHEAP_WEIGHTS)

        # fragility cache
        frag_cache: Dict[Tuple[int, ...], Dict[str, float]] = {}

        records: List[PathRecord] = []

        total_sample_time = 0.0
        total_feat_time = 0.0
        total_frag_time = 0.0

        total_sampled_paths = 0
        total_valid_paths = 0
        total_after_coarse = 0
        total_cache_hits = 0
        total_cache_misses = 0
        total_tasks_with_no_valid_path = 0

        for task_idx, task in enumerate(tasks):
            if task_idx % 10 == 0 or task_idx == len(tasks) - 1:
                print(
                    f"[random][task] {task_idx + 1}/{len(tasks)} "
                    f"source={task.source} target={task.target} shortest_len={task.shortest_len}",
                    flush=True,
                )

            task_cheap_records: List[PathRecord] = []

            # ===== 1. 随机采样 num_samples 条路径 =====
            for sample_idx in range(num_samples):
                t_samp0 = time.perf_counter()
                path = PathGenerator.random_simple_path(
                    G=bundle.nx_graph,
                    source=task.source,
                    target=task.target,
                    max_hops=max_hops,
                    num_trials=num_trials,
                    rng=rng,
                )
                t_samp = time.perf_counter() - t_samp0
                total_sample_time += t_samp
                total_sampled_paths += 1

                if path is None:
                    print(
                        f"[random][sample] task={task_idx + 1}/{len(tasks)} "
                        f"s={sample_idx + 1}/{num_samples} path=None sample_time={t_samp:.2f}s",
                        flush=True,
                    )
                    continue

                total_valid_paths += 1

                # ===== 2. 只提 cheap features，不算 fragility =====
                t_feat0 = time.perf_counter()
                feats = PathFeatureExtractor.extract_features(
                    path=path,
                    importance=bundle.importance,
                    community=bundle.community,
                    edge_bc=bundle.edge_bc,
                )
                t_feat = time.perf_counter() - t_feat0
                total_feat_time += t_feat

                cheap_score = RandomPathBaseline._cheap_score_from_raw_features(feats, cheap_weights)

                record = PathRecord(
                    nodes=path,
                    edges=PathFeatureExtractor.path_to_edges(path),
                    source=task.source,
                    target=task.target,
                    success=True,
                    method="random",
                    score=cheap_score,  # 先临时存 cheap score
                    features=feats,
                    fragility=None,
                    metadata={
                        "same_community": task.same_community,
                        "pair_score": task.pair_score,
                        "shortest_len": task.shortest_len,
                        "num_random_samples": num_samples,
                        "num_trials": num_trials,
                        "cheap_score": cheap_score,
                    },
                )
                task_cheap_records.append(record)

                print(
                    f"[random][cheap] task={task_idx + 1}/{len(tasks)} "
                    f"s={sample_idx + 1}/{num_samples} len={len(path)} "
                    f"sample={t_samp:.2f}s feat={t_feat:.2f}s cheap_score={cheap_score:.6f}",
                    flush=True,
                )

            if not task_cheap_records:
                total_tasks_with_no_valid_path += 1
                continue

            # ===== 3. 先粗筛，再精算 =====
            task_cheap_records.sort(
                key=lambda r: r.score if r.score is not None else -1e18,
                reverse=True,
            )
            selected_for_frag = task_cheap_records[: max(1, int(top_m_for_fragility))]
            total_after_coarse += len(selected_for_frag)

            print(
                f"[random][coarse] task={task_idx + 1}/{len(tasks)} "
                f"keep {len(selected_for_frag)}/{len(task_cheap_records)} for fragility",
                flush=True,
            )

            # ===== 4. 只对粗筛保留的随机路径计算 fragility =====
            best_record: Optional[PathRecord] = None

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
                t_frag = time.perf_counter() - t_frag0
                total_frag_time += t_frag

                print(
                    f"[random][frag] task={task_idx + 1}/{len(tasks)} "
                    f"keep={keep_idx + 1}/{len(selected_for_frag)} "
                    f"len={len(path)} frag={t_frag:.2f}s cache_hit={cache_hit}",
                    flush=True,
                )

                feats = dict(record.features)
                feats.update(frag)

                final_record = PathRecord(
                    nodes=record.nodes,
                    edges=record.edges,
                    source=record.source,
                    target=record.target,
                    success=True,
                    method="random",
                    score=float(feats.get(select_by, 0.0)),
                    features=feats,
                    fragility=frag,
                    metadata=record.metadata,
                )

                if best_record is None or (final_record.score or 0.0) > (best_record.score or 0.0):
                    best_record = final_record

            if best_record is not None:
                records.append(best_record)

        print(f"[random] total_sampled_paths = {total_sampled_paths}", flush=True)
        print(f"[random] total_valid_paths = {total_valid_paths}", flush=True)
        print(f"[random] total_after_coarse = {total_after_coarse}", flush=True)
        print(f"[random] total records = {len(records)}", flush=True)
        print(f"[random] total_tasks_with_no_valid_path = {total_tasks_with_no_valid_path}", flush=True)
        print(f"[random] total_sample_time = {total_sample_time:.2f}s", flush=True)
        print(f"[random] total_feat_time = {total_feat_time:.2f}s", flush=True)
        print(f"[random] total_frag_time = {total_frag_time:.2f}s", flush=True)
        print(
            f"[random] frag_cache size = {len(frag_cache)}, "
            f"hits = {total_cache_hits}, misses = {total_cache_misses}",
            flush=True,
        )

        t_sort0 = time.perf_counter()
        records.sort(key=lambda r: r.score if r.score is not None else -1e18, reverse=True)
        print(f"[random] sort done in {time.perf_counter() - t_sort0:.2f}s", flush=True)

        return records