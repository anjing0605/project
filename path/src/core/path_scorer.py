from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from path.src.core.types import PathRecord


class RulePathScorer:
    """
    Rule-based path scorer for critical path identification.

    Design goals:
    1. Make fragility_score the dominant signal.
    2. Keep node importance / edge betweenness as auxiliary priors.
    3. Strongly weaken cross-community bonus to avoid "good-looking but not destructive" paths.
    4. Apply an explicit fragility gate so that paths with mediocre true fragility
       cannot rank too high merely because proxy features look nice.
    """

    # 推荐默认权重：脆弱性主导版
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "avg_node_importance": 0.12,
        "avg_edge_bc": 0.10,
        "cross_comm_ratio": 0.03,
        "fragility_score": 0.65,
        "path_length": 0.10,   # treated as penalty
    }

    # fragility gating
    DEFAULT_FRAGILITY_GATE: float = 0.50
    DEFAULT_GATE_PENALTY: float = 0.08

    # 用于归一化的字段；其中 path_length 是负向项
    POSITIVE_KEYS: Tuple[str, ...] = (
        "avg_node_importance",
        "avg_edge_bc",
        "cross_comm_ratio",
        "fragility_score",
    )

    NEGATIVE_KEYS: Tuple[str, ...] = (
        "path_length",
    )

    @staticmethod
    def _safe_float(x, default: float = 0.0) -> float:
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    @classmethod
    def _collect_raw_feature_dicts(
        cls,
        path_records: Sequence[PathRecord],
    ) -> List[Dict[str, float]]:
        """
        Merge record.features and record.fragility into a flat feature dict.
        fragility_score is expected in record.fragility or record.features.
        """
        merged: List[Dict[str, float]] = []

        for record in path_records:
            feat = dict(record.features or {})
            frag = dict(record.fragility or {})

            out: Dict[str, float] = {}

            # positive keys
            for k in cls.POSITIVE_KEYS:
                if k in feat:
                    out[k] = cls._safe_float(feat.get(k))
                elif k in frag:
                    out[k] = cls._safe_float(frag.get(k))
                else:
                    out[k] = 0.0

            # negative keys
            for k in cls.NEGATIVE_KEYS:
                if k in feat:
                    out[k] = cls._safe_float(feat.get(k))
                elif k in frag:
                    out[k] = cls._safe_float(frag.get(k))
                else:
                    out[k] = 0.0

            merged.append(out)

        return merged

    @staticmethod
    def normalize_feature_dicts(
        records: List[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        """
        Min-max normalize each feature across all candidate paths.

        Output keys are prefixed with `norm_`, e.g.:
            norm_avg_node_importance
            norm_avg_edge_bc
            norm_cross_comm_ratio
            norm_fragility_score
            norm_path_length
        """
        if not records:
            return []

        keys = sorted(records[0].keys())
        mins: Dict[str, float] = {}
        maxs: Dict[str, float] = {}

        for k in keys:
            vals = [float(r.get(k, 0.0)) for r in records]
            mins[k] = min(vals)
            maxs[k] = max(vals)

        normed: List[Dict[str, float]] = []
        for r in records:
            nr: Dict[str, float] = {}
            for k in keys:
                v = float(r.get(k, 0.0))
                lo = mins[k]
                hi = maxs[k]

                if hi - lo < 1e-12:
                    nr[f"norm_{k}"] = 0.0
                else:
                    nr[f"norm_{k}"] = (v - lo) / (hi - lo)
            normed.append(nr)

        return normed

    @classmethod
    def score_path(
        cls,
        features: Dict[str, float],
        weights: Dict[str, float] | None = None,
        fragility_gate: float | None = None,
        gate_penalty: float | None = None,
    ) -> float:
        """
        Compute final rule score from normalized features.

        Expected input keys:
            norm_avg_node_importance
            norm_avg_edge_bc
            norm_cross_comm_ratio
            norm_fragility_score
            norm_path_length
        """
        w = dict(cls.DEFAULT_WEIGHTS)
        if weights is not None:
            w.update(weights)

        fragility_gate = (
            cls.DEFAULT_FRAGILITY_GATE if fragility_gate is None else float(fragility_gate)
        )
        gate_penalty = (
            cls.DEFAULT_GATE_PENALTY if gate_penalty is None else float(gate_penalty)
        )

        norm_avg_node_importance = cls._safe_float(
            features.get("norm_avg_node_importance", 0.0)
        )
        norm_avg_edge_bc = cls._safe_float(
            features.get("norm_avg_edge_bc", 0.0)
        )
        norm_cross_comm_ratio = cls._safe_float(
            features.get("norm_cross_comm_ratio", 0.0)
        )
        norm_fragility_score = cls._safe_float(
            features.get("norm_fragility_score", 0.0)
        )
        norm_path_length = cls._safe_float(
            features.get("norm_path_length", 0.0)
        )

        # 基础线性项：以 fragility 为主导
        score = (
            w["fragility_score"] * norm_fragility_score
            + w["avg_node_importance"] * norm_avg_node_importance
            + w["avg_edge_bc"] * norm_avg_edge_bc
            + w["cross_comm_ratio"] * norm_cross_comm_ratio
            - w["path_length"] * norm_path_length
        )

        # 显式 gate：真实 fragility 不够时，防止 proxy 特征把路径顶太高
        if norm_fragility_score < fragility_gate:
            score -= gate_penalty

        return float(score)

    @classmethod
    def rank_paths(
        cls,
        path_records: List[PathRecord],
        weights: Dict[str, float] | None = None,
        fragility_gate: float | None = None,
        gate_penalty: float | None = None,
    ) -> List[PathRecord]:
        """
        Attach normalized features and final score to each PathRecord, then sort descending.
        """
        if not path_records:
            return []

        raw_feature_dicts = cls._collect_raw_feature_dicts(path_records)
        norm_feature_dicts = cls.normalize_feature_dicts(raw_feature_dicts)

        ranked: List[PathRecord] = []
        for record, raw_feat, norm_feat in zip(path_records, raw_feature_dicts, norm_feature_dicts):
            score = cls.score_path(
                features=norm_feat,
                weights=weights,
                fragility_gate=fragility_gate,
                gate_penalty=gate_penalty,
            )

            merged_features = dict(record.features or {})
            merged_features.update(raw_feat)
            merged_features.update(norm_feat)

            metadata = dict(record.metadata or {})
            metadata["scorer"] = "RulePathScorer"
            metadata["fragility_gate"] = (
                cls.DEFAULT_FRAGILITY_GATE if fragility_gate is None else float(fragility_gate)
            )
            metadata["gate_penalty"] = (
                cls.DEFAULT_GATE_PENALTY if gate_penalty is None else float(gate_penalty)
            )
            metadata["score_weights"] = dict(cls.DEFAULT_WEIGHTS if weights is None else {**cls.DEFAULT_WEIGHTS, **weights})

            ranked.append(
                replace(
                    record,
                    score=float(score),
                    features=merged_features,
                    metadata=metadata,
                )
            )

        ranked.sort(
            key=lambda r: (
                -cls._safe_float(r.score, 0.0),
                -cls._safe_float((r.fragility or {}).get("fragility_score", 0.0), 0.0),
                cls._safe_float((r.features or {}).get("path_length", 0.0), 0.0),
            )
        )
        return ranked