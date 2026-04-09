from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from path.src.core.types import GraphDataBundle, TaskPair
from path.src.core.path_generator import PathGenerator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.fragility import FragilityEvaluator


def _safe_mean(xs: List[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def _safe_std(xs: List[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def summarize_candidate_coverage(
    bundle: GraphDataBundle,
    tasks: List[TaskPair],
    path_k: int,
    max_hops: int,
    delta: int,
    shared_base_metrics: Optional[Dict[str, float]] = None,
    with_fragility: bool = True,
) -> Dict[str, Any]:
    """
    Build task-level and global candidate coverage statistics.

    Returns:
        {
            "num_tasks": ...,
            "candidate_summary": ...,
            "path_length_distribution": ...,
            "by_shortest_len": ...,
            "feature_ranges": ...,
            "task_rows": [...]
        }
    """
    if shared_base_metrics is None and with_fragility:
        shared_base_metrics = FragilityEvaluator.compute_base_metrics(bundle.nx_graph)

    task_rows: List[Dict[str, Any]] = []
    global_path_lengths: List[int] = []

    feat_avg_node = []
    feat_avg_edge_bc = []
    feat_cross_comm = []
    feat_path_length = []
    feat_fragility = []

    by_shortest_len_candidates = defaultdict(list)
    path_length_counter = Counter()

    total_candidates = 0
    total_valid_candidates = 0

    for task in tasks:
        candidates = PathGenerator.k_shortest_simple_paths(
            G=bundle.nx_graph,
            source=task.source,
            target=task.target,
            k=path_k,
            max_hops=max_hops,
            delta=delta,
        )

        num_candidates = len(candidates)
        total_candidates += num_candidates
        total_valid_candidates += num_candidates

        cand_lengths = []
        cand_feat_fragility = []

        for path in candidates:
            plen_edges = len(path) - 1
            cand_lengths.append(plen_edges)
            global_path_lengths.append(plen_edges)
            path_length_counter[plen_edges] += 1

            feats = PathFeatureExtractor.extract_features(
                path=path,
                importance=bundle.importance,
                community=bundle.community,
                edge_bc=bundle.edge_bc,
            )

            feat_avg_node.append(float(feats["avg_node_importance"]))
            feat_avg_edge_bc.append(float(feats["avg_edge_bc"]))
            feat_cross_comm.append(float(feats["cross_comm_ratio"]))
            feat_path_length.append(float(feats["path_length"]))

            if with_fragility:
                frag = FragilityEvaluator.compute_fragility(
                    G=bundle.nx_graph,
                    path=path,
                    base_metrics=shared_base_metrics,
                    num_nodes=bundle.num_nodes,
                )
                feat_fragility.append(float(frag["fragility_score"]))
                cand_feat_fragility.append(float(frag["fragility_score"]))

        by_shortest_len_candidates[task.shortest_len].append(num_candidates)

        task_rows.append(
            {
                "source": task.source,
                "target": task.target,
                "shortest_len": task.shortest_len,
                "same_community": bool(task.same_community),
                "pair_score": float(task.pair_score),
                "num_candidates": num_candidates,
                "mean_candidate_length": _safe_mean(cand_lengths),
                "min_candidate_length": min(cand_lengths) if cand_lengths else None,
                "max_candidate_length": max(cand_lengths) if cand_lengths else None,
                "mean_candidate_fragility": _safe_mean(cand_feat_fragility)
                if with_fragility else None,
            }
        )

    by_shortest_len = {}
    for slen, cnts in sorted(by_shortest_len_candidates.items()):
        by_shortest_len[str(slen)] = {
            "num_tasks": len(cnts),
            "mean_candidates": _safe_mean(cnts),
            "std_candidates": _safe_std(cnts),
            "min_candidates": min(cnts) if cnts else 0,
            "max_candidates": max(cnts) if cnts else 0,
        }

    feature_ranges = {
        "avg_node_importance": {
            "mean": _safe_mean(feat_avg_node),
            "std": _safe_std(feat_avg_node),
            "min": min(feat_avg_node) if feat_avg_node else 0.0,
            "max": max(feat_avg_node) if feat_avg_node else 0.0,
        },
        "avg_edge_bc": {
            "mean": _safe_mean(feat_avg_edge_bc),
            "std": _safe_std(feat_avg_edge_bc),
            "min": min(feat_avg_edge_bc) if feat_avg_edge_bc else 0.0,
            "max": max(feat_avg_edge_bc) if feat_avg_edge_bc else 0.0,
        },
        "cross_comm_ratio": {
            "mean": _safe_mean(feat_cross_comm),
            "std": _safe_std(feat_cross_comm),
            "min": min(feat_cross_comm) if feat_cross_comm else 0.0,
            "max": max(feat_cross_comm) if feat_cross_comm else 0.0,
        },
        "path_length": {
            "mean": _safe_mean(feat_path_length),
            "std": _safe_std(feat_path_length),
            "min": min(feat_path_length) if feat_path_length else 0.0,
            "max": max(feat_path_length) if feat_path_length else 0.0,
        },
    }

    if with_fragility:
        feature_ranges["fragility_score"] = {
            "mean": _safe_mean(feat_fragility),
            "std": _safe_std(feat_fragility),
            "min": min(feat_fragility) if feat_fragility else 0.0,
            "max": max(feat_fragility) if feat_fragility else 0.0,
        }

    return {
        "dataset": bundle.name,
        "num_tasks": len(tasks),
        "candidate_summary": {
            "total_candidates": total_candidates,
            "total_valid_candidates": total_valid_candidates,
            "mean_candidates_per_task": (
                total_valid_candidates / len(tasks) if tasks else 0.0
            ),
            "min_candidates_per_task": min(r["num_candidates"] for r in task_rows) if task_rows else 0,
            "max_candidates_per_task": max(r["num_candidates"] for r in task_rows) if task_rows else 0,
        },
        "path_length_distribution": {
            str(k): int(v) for k, v in sorted(path_length_counter.items())
        },
        "by_shortest_len": by_shortest_len,
        "feature_ranges": feature_ranges,
        "task_rows": task_rows,
    }