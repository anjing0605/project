from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import networkx as nx

from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.path_generator import PathGenerator
from path.src.core.types import GraphDataBundle, PathRecord, TaskPair


class BetweennessPathBaseline:
    """Among k-shortest candidate paths, choose the one with highest average edge betweenness."""

    @staticmethod
    def run(
        bundle: GraphDataBundle,
        tasks: List[TaskPair],
        k_candidates: int = 5,
        max_hops: int = 8,
        delta: int = 2,
        fragility_weights: Optional[Dict[str, float]] = None,
        shared_base_metrics: Optional[Dict[str, float]] = None,
        sort_by: str = "fragility_score",
        top_m_for_fragility: int = 1,   # 新增：粗筛后保留多少条做 fragility
    ) -> List[PathRecord]:
        fragility_weights = fragility_weights or {
            "lambda_E": 0.4,
            "lambda_LCC": 0.4,
            "lambda_ASP": 0.2,
        }

        print(
            f"[betweenness] start, num_tasks={len(tasks)}, "
            f"k_candidates={k_candidates}, max_hops={max_hops}, "
            f"delta={delta}, top_m_for_fragility={top_m_for_fragility}",
            flush=True,
        )

        evaluator = FragilityEvaluator(**fragility_weights)

        if shared_base_metrics is None:
            t0 = time.perf_counter()
            base_metrics = evaluator.compute_base_metrics(bundle.nx_graph)
            print(
                f"[betweenness] base_metrics computed locally in {time.perf_counter() - t0:.2f}s: {base_metrics}",
                flush=True,
            )
        else:
            base_metrics = dict(shared_base_metrics)
            print(f"[betweenness] using shared_base_metrics: {base_metrics}", flush=True)

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
                    f"[betweenness][task] {task_idx + 1}/{len(tasks)} "
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
                f"[betweenness][task {task_idx + 1}] generated {len(candidates)} candidates in {t_gen:.2f}s",
                flush=True,
            )

            if not candidates:
                total_no_path += 1
                continue

            # ===== 2. 先提 cheap features，并按 avg_edge_bc 粗筛 =====
            cheap_records: List[PathRecord] = []
            for cand_idx, path in enumerate(candidates):
                t_feat0 = time.perf_counter()
                feats = PathFeatureExtractor.extract_features(
                    path=path,
                    importance=bundle.importance,
                    community=bundle.community,
                    edge_bc=bundle.edge_bc,
                    shortest_len=task.shortest_len,
                    source=task.source,
                    target=task.target,
                )
                t_feat = time.perf_counter() - t_feat0
                total_feat_time += t_feat

                avg_edge_bc = float(feats.get("avg_edge_bc", 0.0))

                record = PathRecord(
                    nodes=path,
                    edges=PathFeatureExtractor.path_to_edges(path),
                    source=task.source,
                    target=task.target,
                    success=True,
                    method="betweenness",
                    score=avg_edge_bc,   # 先临时存粗筛指标
                    features=feats,
                    fragility=None,
                    metadata={
                        "same_community": task.same_community,
                        "pair_score": task.pair_score,
                        "shortest_len": task.shortest_len,
                        "selection_metric": "avg_edge_bc",
                        "coarse_avg_edge_bc": avg_edge_bc,
                    },
                )
                cheap_records.append(record)

                print(
                    f"[betweenness][cheap] task={task_idx + 1}/{len(tasks)} "
                    f"cand={cand_idx + 1}/{len(candidates)} len={len(path)} "
                    f"feat={t_feat:.2f}s avg_edge_bc={avg_edge_bc:.6f}",
                    flush=True,
                )

            cheap_records.sort(
                key=lambda r: r.score if r.score is not None else -1e18,
                reverse=True,
            )
            selected_for_frag = cheap_records[: max(1, int(top_m_for_fragility))]
            total_after_coarse += len(selected_for_frag)

            print(
                f"[betweenness][coarse] task={task_idx + 1}/{len(tasks)} "
                f"keep {len(selected_for_frag)}/{len(cheap_records)} for fragility",
                flush=True,
            )

            # ===== 3. 只对粗筛保留的路径算 fragility，再从中选 best =====
            best_record: Optional[PathRecord] = None
            best_edge_bc = float("-inf")

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
                    f"[betweenness][frag] task={task_idx + 1}/{len(tasks)} "
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
                    method="betweenness",
                    score=float(feats.get(sort_by, 0.0)),
                    features=feats,
                    fragility=frag,
                    metadata=record.metadata,
                )

                avg_edge_bc = float(feats.get("avg_edge_bc", 0.0))
                if avg_edge_bc > best_edge_bc:
                    best_edge_bc = avg_edge_bc
                    best_record = final_record

            if best_record is not None:
                records.append(best_record)

        print(f"[betweenness] total_candidates_generated = {total_candidates_generated}", flush=True)
        print(f"[betweenness] total_after_coarse = {total_after_coarse}", flush=True)
        print(f"[betweenness] total records = {len(records)}", flush=True)
        print(f"[betweenness] total_no_path = {total_no_path}", flush=True)
        print(f"[betweenness] total_gen_time = {total_gen_time:.2f}s", flush=True)
        print(f"[betweenness] total_feat_time = {total_feat_time:.2f}s", flush=True)
        print(f"[betweenness] total_frag_time = {total_frag_time:.2f}s", flush=True)
        print(
            f"[betweenness] frag_cache size = {len(frag_cache)}, "
            f"hits = {total_cache_hits}, misses = {total_cache_misses}",
            flush=True,
        )

        t_sort0 = time.perf_counter()
        records.sort(key=lambda r: r.score if r.score is not None else -1e18, reverse=True)
        print(f"[betweenness] sort done in {time.perf_counter() - t_sort0:.2f}s", flush=True)

        return records