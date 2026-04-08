from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import networkx as nx

from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.path_generator import PathGenerator
from path.src.core.types import GraphDataBundle, PathRecord, TaskPair


class NodeScorePathBaseline:
    """Among k-shortest candidate paths, choose the one with highest node-importance score."""

    @staticmethod
    def run(
        bundle: GraphDataBundle,
        tasks: List[TaskPair],
        k_candidates: int = 5,
        max_hops: int = 8,
        delta: int = 2,
        use_internal_only: bool = False,
        fragility_weights: Optional[Dict[str, float]] = None,
        sort_by: str = "fragility_score",
        top_m_for_fragility: int = 1,   # 新增：粗筛后保留多少条做 fragility
    ) -> List[PathRecord]:
        fragility_weights = fragility_weights or {
            "lambda_E": 0.4,
            "lambda_LCC": 0.4,
            "lambda_ASP": 0.2,
        }

        selection_metric = "internal_node_importance" if use_internal_only else "avg_node_importance"

        print(
            f"[node_score] start, num_tasks={len(tasks)}, "
            f"k_candidates={k_candidates}, max_hops={max_hops}, delta={delta}, "
            f"use_internal_only={use_internal_only}, "
            f"top_m_for_fragility={top_m_for_fragility}",
            flush=True,
        )

        evaluator = FragilityEvaluator(**fragility_weights)

        t0 = time.perf_counter()
        base_metrics = evaluator.compute_base_metrics(bundle.nx_graph)
        print(
            f"[node_score] base_metrics computed in {time.perf_counter() - t0:.2f}s: {base_metrics}",
            flush=True,
        )

        frag_cache: Dict[Tuple[int, ...], Dict[str, float]] = {}

        records: List[PathRecord] = []

        total_gen_time = 0.0
        total_feat_time = 0.0
        total_frag_time = 0.0

        total_candidates_generated = 0
        total_after_coarse = 0
        total_no_path = 0
        total_cache_hits = 0
        total_cache_misses = 0

        for task_idx, task in enumerate(tasks):
            if task_idx % 10 == 0 or task_idx == len(tasks) - 1:
                print(
                    f"[node_score][task] {task_idx + 1}/{len(tasks)} "
                    f"source={task.source} target={task.target} shortest_len={task.shortest_len}",
                    flush=True,
                )

            # ===== 1. 生成候选路径 =====
            t_gen0 = time.perf_counter()
            try:
                candidates = PathGenerator.k_shortest_simple_paths(
                    G=bundle.nx_graph,
                    source=task.source,
                    target=task.target,
                    k=k_candidates,
                    max_hops=max_hops,
                    delta=delta,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                total_no_path += 1
                continue
            t_gen = time.perf_counter() - t_gen0
            total_gen_time += t_gen
            total_candidates_generated += len(candidates)

            print(
                f"[node_score][task {task_idx + 1}] generated {len(candidates)} candidates in {t_gen:.2f}s",
                flush=True,
            )

            if not candidates:
                total_no_path += 1
                continue

            # ===== 2. 先提 cheap features，并按 node_score 粗筛 =====
            cheap_records: List[PathRecord] = []
            for cand_idx, path in enumerate(candidates):
                t_feat0 = time.perf_counter()
                feats = PathFeatureExtractor.extract_features(
                    path=path,
                    importance=bundle.importance,
                    community=bundle.community,
                    edge_bc=bundle.edge_bc,
                )
                t_feat = time.perf_counter() - t_feat0
                total_feat_time += t_feat

                node_score = float(
                    feats.get("internal_node_importance", 0.0)
                    if use_internal_only
                    else feats.get("avg_node_importance", 0.0)
                )

                record = PathRecord(
                    nodes=path,
                    edges=PathFeatureExtractor.path_to_edges(path),
                    source=task.source,
                    target=task.target,
                    success=True,
                    method="node_score",
                    score=node_score,   # 先临时存粗筛指标
                    features=feats,
                    fragility=None,
                    metadata={
                        "same_community": task.same_community,
                        "pair_score": task.pair_score,
                        "shortest_len": task.shortest_len,
                        "selection_metric": selection_metric,
                        "coarse_node_score": node_score,
                    },
                )
                cheap_records.append(record)

                print(
                    f"[node_score][cheap] task={task_idx + 1}/{len(tasks)} "
                    f"cand={cand_idx + 1}/{len(candidates)} len={len(path)} "
                    f"feat={t_feat:.2f}s {selection_metric}={node_score:.6f}",
                    flush=True,
                )

            cheap_records.sort(
                key=lambda r: r.score if r.score is not None else -1e18,
                reverse=True,
            )
            selected_for_frag = cheap_records[: max(1, int(top_m_for_fragility))]
            total_after_coarse += len(selected_for_frag)

            print(
                f"[node_score][coarse] task={task_idx + 1}/{len(tasks)} "
                f"keep {len(selected_for_frag)}/{len(cheap_records)} for fragility",
                flush=True,
            )

            # ===== 3. 只对粗筛保留的路径算 fragility，再从中选 best =====
            best_record: Optional[PathRecord] = None
            best_node_score = float("-inf")

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
                    f"[node_score][frag] task={task_idx + 1}/{len(tasks)} "
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
                    method="node_score",
                    score=float(feats.get(sort_by, 0.0)),
                    features=feats,
                    fragility=frag,
                    metadata=record.metadata,
                )

                node_score = float(
                    feats.get("internal_node_importance", 0.0)
                    if use_internal_only
                    else feats.get("avg_node_importance", 0.0)
                )
                if node_score > best_node_score:
                    best_node_score = node_score
                    best_record = final_record

            if best_record is not None:
                records.append(best_record)

        print(f"[node_score] total_candidates_generated = {total_candidates_generated}", flush=True)
        print(f"[node_score] total_after_coarse = {total_after_coarse}", flush=True)
        print(f"[node_score] total records = {len(records)}", flush=True)
        print(f"[node_score] total_no_path = {total_no_path}", flush=True)
        print(f"[node_score] total_gen_time = {total_gen_time:.2f}s", flush=True)
        print(f"[node_score] total_feat_time = {total_feat_time:.2f}s", flush=True)
        print(f"[node_score] total_frag_time = {total_frag_time:.2f}s", flush=True)
        print(
            f"[node_score] frag_cache size = {len(frag_cache)}, "
            f"hits = {total_cache_hits}, misses = {total_cache_misses}",
            flush=True,
        )

        t_sort0 = time.perf_counter()
        records.sort(key=lambda r: r.score if r.score is not None else -1e18, reverse=True)
        print(f"[node_score] sort done in {time.perf_counter() - t_sort0:.2f}s", flush=True)

        return records