from __future__ import annotations

import json
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from path.src.core.path_features import PathFeatureExtractor
from path.src.core.types import PathRecord

try:
    from xgboost import XGBRanker
except ImportError:  # pragma: no cover
    XGBRanker = None


class XGBPathRanker:
    """
    Learn a path score from handcrafted features and rank candidate paths per task pair.
    Backend: xgboost.XGBRanker (Learning-to-Rank)
    """

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 5,  # 加深以捕获特征交叉
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,  # L1 正则化
        "reg_lambda": 1.0,  # L2 正则化
        "random_state": 42,
        "objective": "rank:ndcg",  # 使用 NDCG 排序损失
        "eval_metric": "ndcg@10"  # 重点关注 Top-10 排序质量
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

        if XGBRanker is None:
            raise ImportError("xgboost is not installed. Please install it using 'pip install xgboost'.")

        self.model = XGBRanker(**self.params)
        self.is_trained = False
        self.backend = "xgboost"

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> None:
        """
        训练排序模型。注意：DataFrame 必须按 Query 分组，这里通过 source 和 target 分组。
        """
        if train_df.empty:
            raise ValueError("train_df is empty; cannot train ranker.")
        if not self.feature_cols:
            raise ValueError("feature_cols is empty; please pass explicit feature columns.")

        # 1. 强制按 Query Group (即 source, target) 排序，这是 XGBRanker 的硬性要求
        train_df = train_df.sort_values(by=["source", "target"]).reset_index(drop=True)

        # 2. 计算训练集的 Group 数组
        group_train = train_df.groupby(["source", "target"], sort=False).size().values

        # 3. 确定标签列：如果之前在 dataset.py 做了相关度分档就用 relevance，否则退化用 y_fragility
        target_col = "relevance" if "relevance" in train_df.columns else "y"

        X_train = train_df[self.feature_cols].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=float)

        # 4. 如果有验证集，同样处理
        eval_set = None
        eval_group = None
        if val_df is not None and not val_df.empty:
            val_df = val_df.sort_values(by=["source", "target"]).reset_index(drop=True)
            group_val = val_df.groupby(["source", "target"], sort=False).size().values
            X_val = val_df[self.feature_cols].to_numpy(dtype=float)
            y_val = val_df[target_col].to_numpy(dtype=float)
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_group = [group_train, group_val]

        # 5. 开始训练 (必须传入 group)
        self.model.fit(
            X_train, y_train,
            group=group_train,
            eval_set=eval_set,
            eval_group=eval_group,
            verbose=10
        )
        self.is_trained = True

    def predict_score(self, test_df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Ranker has not been fitted.")
        X = test_df[self.feature_cols].to_numpy(dtype=float)
        pred = self.model.predict(X)
        return np.asarray(pred, dtype=float)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df.copy()
        out["pred_score"] = self.predict_score(df)
        tie_col = "y" if "y" in out.columns else "y_fragility"

        out = out.sort_values(
            by=["source", "target", "pred_score", tie_col],
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
            tie_col = "y" if "y" in g.columns else "y_fragility"
            g = g.sort_values(
                by=["pred_score", tie_col],
                ascending=[False, False],
            ).head(top_per_task)
            for _, row in g.iterrows():
                nodes = json.loads(row["path_nodes"])
                features = {
                    c: float(row[c])
                    for c in row.index
                    if c not in {
                        "source",
                        "target",
                        "path_nodes",
                        "pred_score",
                        "method",
                        "backend",
                        "relevance",
                        "y",
                        "y_single",
                        "y_marginal",
                        "y_fragility",
                        "y_gain",
                        "delta_E",
                        "delta_LCC",
                        "delta_ASP",
                        "fragility_score",
                    }
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
                        "pred_score": float(row.get("pred_score", 0.0)),
                        "y": float(row.get("y", 0.0)),
                        "y_single": float(row.get("y_single", row.get("y_fragility", 0.0))),
                        "y_marginal": float(row.get("y_marginal", row.get("y_gain", 0.0))),
                        "relevance": float(row.get("relevance", 0.0)),
                    },
                )
                records.append(rec)
        records.sort(key=lambda r: r.score if r.score is not None else -1e18, reverse=True)
        return records