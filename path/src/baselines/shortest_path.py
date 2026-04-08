from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import networkx as nx

from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.path_generator import PathGenerator
from path.src.core.types import GraphDataBundle, PathRecord, TaskPair


class ShortestPathBaseline:
    """Use the shortest path between each task pair as the candidate path."""

    @staticmethod
    def run(
        bundle: GraphDataBundle,
        tasks: List[TaskPair],
        fragility_weights: Optional[Dict[str, float]] = None,
        shared_base_metrics: Optional[Dict[str, float]] = None,
        sort_by: str = "fragility_score",
    ) -> List[PathRecord]:
        fragility_weights = fragility_weights or {
            "lambda_E": 0.4,
            "lambda_LCC": 0.4,
            "lambda_ASP": 0.2,
        }

        print(f"[shortest] start, num_tasks={len(tasks)}", flush=True)

        evaluator = FragilityEvaluator(**fragility_weights)

        if shared_base_metrics is None:
            t0 = time.perf_counter()
            base_metrics = evaluator.compute_base_metrics(bundle.nx_graph)
            print(
                f"[shortest] base_metrics computed locally in {time.perf_counter() - t0:.2f}s: {base_metrics}",
                flush=True,
            )
        else:
            base_metrics = dict(shared_base_metrics)
            print(f"[shortest] using shared_base_metrics: {base_metrics}", flush=True)

        # 新增：局部缓存
        frag_cache: Dict[Tuple[int, ...], Dict[str, float]] = {}

        records: List[PathRecord] = []

        total_path_time = 0.0
        total_feat_time = 0.0
        total_frag_time = 0.0
        total_cache_hits = 0
        total_cache_misses = 0
        total_no_path = 0

        for task_idx, task in enumerate(tasks):
            if task_idx % 10 == 0 or task_idx == len(tasks) - 1:
                print(
                    f"[shortest][task] {task_idx + 1}/{len(tasks)} "
                    f"source={task.source} target={task.target} shortest_len={task.shortest_len}",
                    flush=True,
                )

            # ===== shortest path generation =====
            t_path0 = time.perf_counter()
            try:
                path = PathGenerator.shortest_path(bundle.nx_graph, task.source, task.target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                total_no_path += 1
                continue
            t_path = time.perf_counter() - t_path0
            total_path_time += t_path

            # ===== cheap features =====
            t_feat0 = time.perf_counter()
            feats = PathFeatureExtractor.extract_features(
                path=path,
                importance=bundle.importance,
                community=bundle.community,
                edge_bc=bundle.edge_bc,
            )
            t_feat = time.perf_counter() - t_feat0
            total_feat_time += t_feat

            # ===== fragility with cache =====
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
                f"[shortest][path] task={task_idx + 1}/{len(tasks)} "
                f"len={len(path)} path={t_path:.2f}s feat={t_feat:.2f}s "
                f"frag={t_frag:.2f}s cache_hit={cache_hit}",
                flush=True,
            )

            feats.update(frag)

            records.append(
                PathRecord(
                    nodes=path,
                    edges=PathFeatureExtractor.path_to_edges(path),
                    source=task.source,
                    target=task.target,
                    success=True,
                    method="shortest",
                    score=float(feats.get(sort_by, 0.0)),
                    features=feats,
                    fragility=frag,
                    metadata={
                        "same_community": task.same_community,
                        "pair_score": task.pair_score,
                        "shortest_len": task.shortest_len,
                    },
                )
            )

        print(f"[shortest] total records = {len(records)}", flush=True)
        print(f"[shortest] total_no_path = {total_no_path}", flush=True)
        print(f"[shortest] total_path_time = {total_path_time:.2f}s", flush=True)
        print(f"[shortest] total_feat_time = {total_feat_time:.2f}s", flush=True)
        print(f"[shortest] total_frag_time = {total_frag_time:.2f}s", flush=True)
        print(
            f"[shortest] frag_cache size = {len(frag_cache)}, "
            f"hits = {total_cache_hits}, misses = {total_cache_misses}",
            flush=True,
        )

        t_sort0 = time.perf_counter()
        records.sort(key=lambda r: r.score if r.score is not None else -1e18, reverse=True)
        print(f"[shortest] sort done in {time.perf_counter() - t_sort0:.2f}s", flush=True)

        return records