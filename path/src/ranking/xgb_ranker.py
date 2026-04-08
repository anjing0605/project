from __future__ import annotations

import json
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from path.src.core.path_features import PathFeatureExtractor
from path.src.core.types import PathRecord

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from sklearn.ensemble import GradientBoostingRegressor #pip install scikit-learn
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None


class XGBPathRanker:
    """
    Learn a path score from handcrafted features and rank candidate paths per task pair.

    Primary backend: xgboost.XGBRegressor
    Fallback backend: sklearn.GradientBoostingRegressor
    """

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "objective": "reg:squarederror",
    }

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        params: Optional[Dict[str, object]] = None,
    ) -> None:
        self.feature_cols = feature_cols or []
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.model = None
        self.backend = None

    def _build_model(self):
        if XGBRegressor is not None:
            self.backend = "xgboost"
            return XGBRegressor(**self.params)
        if GradientBoostingRegressor is not None:
            self.backend = "sklearn_gbr"
            return GradientBoostingRegressor(random_state=int(self.params.get("random_state", 42)))
        raise ImportError("Neither xgboost nor scikit-learn GradientBoostingRegressor is available.")

    def fit(self, train_df: pd.DataFrame) -> None:
        if train_df.empty:
            raise ValueError("train_df is empty; cannot train ranker.")
        if not self.feature_cols:
            raise ValueError("feature_cols is empty; please pass explicit feature columns.")
        X = train_df[self.feature_cols].to_numpy(dtype=float)
        y = train_df["y_fragility"].to_numpy(dtype=float)
        self.model = self._build_model()
        self.model.fit(X, y)

    def predict_score(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Ranker has not been fitted.")
        X = test_df[self.feature_cols].to_numpy(dtype=float)
        pred = self.model.predict(X)
        return np.asarray(pred, dtype=float)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df.copy()
        out["pred_score"] = self.predict_score(df)
        out = out.sort_values(
            by=["source", "target", "pred_score", "y_fragility"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)
        return out

    def rank_by_task(self, df: pd.DataFrame) -> Dict[Tuple[int, int], List[dict]]:
        scored = self.score_dataframe(df)
        grouped: Dict[Tuple[int, int], List[dict]] = {}
        for (src, tgt), g in scored.groupby(["source", "target"], sort=True):
            grouped[(int(src), int(tgt))] = g.to_dict(orient="records")
        return grouped

    @staticmethod
    def dataframe_to_pathrecords(df: pd.DataFrame, top_per_task: int = 1) -> List[PathRecord]:
        if df.empty:
            return []
        records: List[PathRecord] = []
        for (src, tgt), g in df.groupby(["source", "target"], sort=True):
            g = g.sort_values(by=["pred_score", "y_fragility"], ascending=[False, False]).head(top_per_task)
            for _, row in g.iterrows():
                nodes = json.loads(row["path_nodes"])
                features = {
                    c: float(row[c])
                    for c in row.index
                    if c not in {"source", "target", "path_nodes", "pred_score", "method"}
                    and pd.api.types.is_number(row[c])
                }
                fragility = {
                    "delta_E": float(row.get("delta_E", 0.0)),
                    "delta_LCC": float(row.get("delta_LCC", 0.0)),
                    "delta_ASP": float(row.get("delta_ASP", 0.0)),
                    "fragility_score": float(row.get("fragility_score", 0.0)),
                }
                rec = PathRecord(
                    nodes=[int(n) for n in nodes],
                    edges=PathFeatureExtractor.path_to_edges(nodes),
                    source=int(src),
                    target=int(tgt),
                    success=True,
                    method="xgb_rank",
                    score=float(row["pred_score"]),
                    features=features,
                    fragility=fragility,
                    metadata={
                        "candidate_rank": int(row.get("candidate_rank", 0)),
                        "backend": row.get("backend", "xgboost"),
                    },
                )
                records.append(rec)
        records.sort(key=lambda r: r.score if r.score is not None else -1e18, reverse=True)
        return records
