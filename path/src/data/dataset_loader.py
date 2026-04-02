from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from torch_geometric.datasets import Planetoid
except ImportError:  # pragma: no cover
    Planetoid = None


class PlanetoidDatasetLoader:
    """Load Planetoid datasets and aligned node-importance files."""

    SUPPORTED = {"Cora", "Citeseer", "Pubmed"}

    @staticmethod
    def load_planetoid(name: str, root: str) -> Any:
        if Planetoid is None:
            raise ImportError(
                "torch_geometric is required to load Planetoid datasets. "
                "Please install torch-geometric first."
            )
        if name not in PlanetoidDatasetLoader.SUPPORTED:
            raise ValueError(
                f"Unsupported dataset: {name}. "
                f"Expected one of {sorted(PlanetoidDatasetLoader.SUPPORTED)}"
            )
        dataset = Planetoid(root=root, name=name)
        return dataset[0]

    @staticmethod
    def load_importance(path: str, num_nodes: int | None = None) -> np.ndarray:
        """
        Supported formats:
            - .npy: 1D score array
            - .csv: supports columns [node, gnn_score], [node_id, importance],
                    [importance], [gnn_score]
            - .txt/.tsv: single-column numeric file
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Importance file not found: {path}")

        suffix = p.suffix.lower()

        if suffix == ".npy":
            arr = np.load(p).reshape(-1).astype(float)

        elif suffix == ".csv":
            df = pd.read_csv(p)

            if "node" in df.columns and "gnn_score" in df.columns:
                if num_nodes is None:
                    raise ValueError("num_nodes must be provided when CSV contains 'node'.")
                arr = PlanetoidDatasetLoader._align_indexed_scores(
                    node_ids=df["node"].to_numpy(),
                    scores=df["gnn_score"].to_numpy(),
                    num_nodes=num_nodes,
                    node_col_name="node",
                    score_col_name="gnn_score",
                )
            elif "node_id" in df.columns and "importance" in df.columns:
                if num_nodes is None:
                    raise ValueError(
                        "num_nodes must be provided when CSV contains 'node_id'."
                    )
                arr = PlanetoidDatasetLoader._align_indexed_scores(
                    node_ids=df["node_id"].to_numpy(),
                    scores=df["importance"].to_numpy(),
                    num_nodes=num_nodes,
                    node_col_name="node_id",
                    score_col_name="importance",
                )
            elif "importance" in df.columns:
                arr = df["importance"].to_numpy(dtype=float)
            elif "gnn_score" in df.columns:
                arr = df["gnn_score"].to_numpy(dtype=float)
            else:
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if len(numeric_cols) != 1:
                    raise ValueError(
                        f"Ambiguous numeric columns in CSV: {numeric_cols}. "
                        "Please provide an explicit score column such as 'gnn_score' or 'importance'."
                    )
                arr = df[numeric_cols[0]].to_numpy(dtype=float)

        elif suffix in {".txt", ".tsv"}:
            try:
                arr = np.loadtxt(p, delimiter=None).reshape(-1).astype(float)
            except Exception:
                arr = np.loadtxt(p, delimiter=",").reshape(-1).astype(float)
        else:
            raise ValueError(f"Unsupported importance format: {suffix}")

        if np.isnan(arr).any():
            raise ValueError("Importance array contains NaN values.")
        if num_nodes is not None and len(arr) != num_nodes:
            raise ValueError(
                f"Importance length mismatch: got {len(arr)}, expected {num_nodes}."
            )
        return arr

    @staticmethod
    def _align_indexed_scores(
        node_ids: np.ndarray,
        scores: np.ndarray,
        num_nodes: int,
        node_col_name: str,
        score_col_name: str,
    ) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=int).reshape(-1)
        scores = np.asarray(scores, dtype=float).reshape(-1)

        if len(node_ids) != len(scores):
            raise ValueError(
                f"Length mismatch between {node_col_name} and {score_col_name}."
            )
        if len(np.unique(node_ids)) != len(node_ids):
            duplicated = pd.Series(node_ids)[pd.Series(node_ids).duplicated()].tolist()
            raise ValueError(f"Duplicate {node_col_name} values found: {duplicated[:10]}")
        if (node_ids < 0).any() or (node_ids >= num_nodes).any():
            invalid = node_ids[(node_ids < 0) | (node_ids >= num_nodes)].tolist()
            raise ValueError(f"{node_col_name} out of range: {invalid[:10]}")

        arr = np.full(num_nodes, np.nan, dtype=float)
        arr[node_ids] = scores
        if np.isnan(arr).any():
            missing = np.where(np.isnan(arr))[0].tolist()
            raise ValueError(f"Missing scores for node ids: {missing[:20]}")
        return arr
