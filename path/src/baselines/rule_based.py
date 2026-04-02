from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np
import networkx as nx

from src.core.deduplicate import PathDeduplicator
from src.core.fragility import FragilityEvaluator
from src.core.path_features import PathFeatureExtractor
from src.core.path_generator import PathGenerator
from src.core.types import GraphDataBundle, PathRecord, TaskPair


class RuleBasedCriticalPath:
    """
    Rule-based critical path identification.
    Score(P) = a * avg_node_importance + b * avg_edge_bc + c * cross_comm_ratio
               + d * fragility_score - e * path_length
    Features are min-max normalized within the candidate pool.
    """

    DEFAULT_WEIGHTS = {
        "avg_node_importance": 0.20,
        "avg_edge_bc": 0.15,
        "cross_comm_ratio": 0.10,
        "fragility_score": 0.45,
        "path_length": 0.10,
    }

    @staticmethod
    def _minmax_normalize(records: List[PathRecord], feature_names: List[str]) -> List[PathRecord]:
        if not records:
            return records
        values = {name: np.array([r.features.get(name, 0.0) for r in records], dtype=float) for name in feature_names}
        stats = {}
        for name, arr in values.items():
            stats[name] = (float(arr.min()), float(arr.max()))

        normalized: List[PathRecord] = []
        for r in records:
            feats = dict(r.features)
            for name in feature_names:
                lo, hi = stats[name]
                raw = float(feats.get(name, 0.0))
                feats[f"norm_{name}"] = 0.0 if hi <= lo else (raw - lo) / (hi - lo)
            normalized.append(replace(r, features=feats))
        return normalized

    @staticmethod
    def _score_record(record: PathRecord, weights: Dict[str, float]) -> float:
        f = record.features
        score = (
            weights["avg_node_importance"] * f["norm_avg_node_importance"]
            + weights["avg_edge_bc"] * f["norm_avg_edge_bc"]
            + weights["cross_comm_ratio"] * f["norm_cross_comm_ratio"]
            + weights["fragility_score"] * f["norm_fragility_score"]
            - weights["path_length"] * f["norm_path_length"]
        )
        return float(score)

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
    ) -> List[PathRecord]:
        weights = weights or dict(cls.DEFAULT_WEIGHTS)
        fragility_weights = fragility_weights or {"lambda_E": 0.4, "lambda_LCC": 0.4, "lambda_ASP": 0.2}
        evaluator = FragilityEvaluator(**fragility_weights)
        base_metrics = evaluator.compute_base_metrics(bundle.nx_graph)
        candidates: List[PathRecord] = []

        for task in tasks:
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

            for path in paths:
                feats = PathFeatureExtractor.extract_features(
                    path=path,
                    importance=bundle.importance,
                    community=bundle.community,
                    edge_bc=bundle.edge_bc,
                )
                frag = evaluator.compute_fragility(
                    G=bundle.nx_graph,
                    path=path,
                    base_metrics=base_metrics,
                    num_nodes=bundle.num_nodes,
                )
                feats.update(frag)
                record = PathRecord(
                    nodes=path,
                    edges=PathFeatureExtractor.path_to_edges(path),
                    source=task.source,
                    target=task.target,
                    success=True,
                    method="rule",
                    features=feats,
                    fragility=frag,
                    metadata={
                        "same_community": task.same_community,
                        "pair_score": task.pair_score,
                        "shortest_len": task.shortest_len,
                    },
                )
                candidates.append(record)

        if not candidates:
            return []

        candidates = cls._minmax_normalize(
            candidates,
            feature_names=[
                "avg_node_importance",
                "avg_edge_bc",
                "cross_comm_ratio",
                "fragility_score",
                "path_length",
            ],
        )

        scored: List[PathRecord] = []
        for record in candidates:
            score = cls._score_record(record, weights=weights)
            scored.append(replace(record, score=score))

        scored.sort(key=lambda r: r.score if r.score is not None else -1e18, reverse=True)
        selected = PathDeduplicator.greedy_deduplicate(
            scored, overlap_threshold=overlap_threshold, top_q=top_q
        )
        return selected
