from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from torch_geometric.datasets import Planetoid, Amazon, Coauthor
except ImportError:
    Planetoid = None
    Amazon = None
    Coauthor = None


class GraphDatasetLoader:
    """
    Load graph datasets from PyG and load importance scores aligned to old_id graph space.

    Supported dataset families:
        - Planetoid: Cora / Citeseer / Pubmed
        - Amazon: Computers / Photo
        - Coauthor: CS / Physics

    Importance loading modes:
        1) direct alignment:
           score file node ids already equal graph old_id space

        2) recovered alignment:
           score file node ids are continuous new_id, while node_features.csv
           stores old_id in row order corresponding to continuous new_id order.

           Supported continuous new_id forms:
               - 0,1,2,...,N-1
               - 1,2,3,...,N

           Example:
               node_features.csv old_id column rows = [1, 5, 6]
               gnn_score node ids = [1, 2, 3]
               then mapping is:
                   1 -> 1
                   2 -> 5
                   3 -> 6
    """

    SUPPORTED = {
        "Cora",
        "Citeseer",
        "Pubmed",
        "Computers",
        "Photo",
        "CS",
        "Physics",
    }

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_dataset_name(name: str) -> str:
        """
        Normalize user dataset name to canonical PyG-style name.
        """
        aliases = {
            # Planetoid
            "cora": "Cora",
            "citeseer": "Citeseer",
            "citeseer_struct": "Citeseer",
            "citeseer_graph": "Citeseer",
            "citeseer_planetoid": "Citeseer",
            "pubmed": "Pubmed",
            "pubmed_struct": "Pubmed",
            "pubmed_graph": "Pubmed",

            # Amazon
            "computers": "Computers",
            "computer": "Computers",
            "photo": "Photo",

            # Coauthor
            "cs": "CS",
            "physics": "Physics",
        }

        key = name.strip().lower()
        if key in aliases:
            return aliases[key]

        raise ValueError(
            f"Unsupported dataset alias: {name}. "
            f"Supported datasets are: {sorted(GraphDatasetLoader.SUPPORTED)}"
        )

    @staticmethod
    def load_dataset(name: str, root: str) -> Any:
        """
        Load graph dataset from PyG.

        Args:
            name: dataset name or alias
            root: root directory for dataset cache

        Returns:
            PyG Data object
        """
        canonical_name = GraphDatasetLoader._normalize_dataset_name(name)

        if Planetoid is None or Amazon is None or Coauthor is None:
            raise ImportError(
                "torch_geometric is required to load datasets. "
                "Please install torch-geometric first."
            )

        # Planetoid family
        if canonical_name in {"Cora", "Citeseer", "Pubmed"}:
            dataset = Planetoid(root=root, name=canonical_name)
            return dataset[0]

        # Amazon family
        if canonical_name in {"Computers", "Photo"}:
            dataset = Amazon(root=root, name=canonical_name)
            return dataset[0]

        # Coauthor family
        if canonical_name in {"CS", "Physics"}:
            dataset = Coauthor(root=root, name=canonical_name)
            return dataset[0]

        raise ValueError(
            f"Unsupported dataset: {name} -> normalized as {canonical_name}"
        )

    # ------------------------------------------------------------------
    # Score-file column detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_score_columns(df: pd.DataFrame) -> tuple[str, str]:
        candidate_node_cols = ["node", "node_idx", "node_id", "new_id", "old_id"]
        candidate_score_cols = ["gnn_score", "importance", "score"]

        node_col = None
        score_col = None

        for c in candidate_node_cols:
            if c in df.columns:
                node_col = c
                break

        for c in candidate_score_cols:
            if c in df.columns:
                score_col = c
                break

        if node_col is None:
            raise ValueError(
                f"Cannot detect node column in score file. "
                f"Columns = {list(df.columns)}"
            )

        if score_col is None:
            raise ValueError(
                f"Cannot detect score column in score file. "
                f"Columns = {list(df.columns)}"
            )

        return node_col, score_col

    @staticmethod
    def _detect_old_id_col(node_feat_df: pd.DataFrame) -> str:
        """
        Detect old-id column from node_features.csv.
        Prefer explicit names first.
        """
        candidates = ["old_id", "node", "node_idx", "node_id"]

        for c in candidates:
            if c in node_feat_df.columns:
                return c

        raise ValueError(
            "Cannot detect old-id column from node_features.csv. "
            f"Columns = {list(node_feat_df.columns)}"
        )

    # ------------------------------------------------------------------
    # Mapping recovery helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_contiguous_new_id_start(score_node_values: list[int]) -> Optional[int]:
        """
        Detect whether score nodes form a valid contiguous new_id range.

        Supported forms:
            - 0,1,2,...,N-1  -> start = 0
            - 1,2,3,...,N    -> start = 1

        Returns:
            start value if valid contiguous new_id range, otherwise None.
        """
        if len(score_node_values) == 0:
            return None

        vals = sorted(score_node_values)
        n = len(vals)

        if vals == list(range(0, n)):
            return 0

        if vals == list(range(1, n + 1)):
            return 1

        return None

    @staticmethod
    def _build_new_to_old_from_node_features(
        node_features_path: str,
        score_len: int,
        new_id_start: int,
        old_id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Rebuild mapping:
            new_id -> old_id

        Assumption:
            row order of node_features.csv is exactly the continuous new_id order.

        If new_id_start = 0:
            row 0 -> new_id 0
            row 1 -> new_id 1
            ...

        If new_id_start = 1:
            row 0 -> new_id 1
            row 1 -> new_id 2
            ...
        """
        p = Path(node_features_path)
        if not p.exists():
            raise FileNotFoundError(f"node_features file not found: {node_features_path}")

        feat_df = pd.read_csv(p)
        feat_df = feat_df.reset_index(drop=True)

        if old_id_col is None:
            old_id_col = GraphDatasetLoader._detect_old_id_col(feat_df)

        if old_id_col not in feat_df.columns:
            raise ValueError(
                f"old_id_col='{old_id_col}' not found in node_features.csv. "
                f"Columns = {list(feat_df.columns)}"
            )

        if len(feat_df) != score_len:
            raise ValueError(
                f"node_features rows != score rows, cannot rebuild new_id->old_id safely. "
                f"node_features={len(feat_df)}, score={score_len}"
            )

        map_df = pd.DataFrame({
            "new_id": np.arange(new_id_start, new_id_start + len(feat_df), dtype=int),
            "old_id": feat_df[old_id_col].astype(int).to_numpy()
        })

        if map_df["new_id"].duplicated().any():
            dup = map_df.loc[map_df["new_id"].duplicated(), "new_id"].tolist()
            raise ValueError(f"Duplicate new_id found in rebuilt mapping: {dup[:20]}")

        if map_df["old_id"].duplicated().any():
            dup = map_df.loc[map_df["old_id"].duplicated(), "old_id"].tolist()
            raise ValueError(f"Duplicate old_id found in rebuilt mapping: {dup[:20]}")

        return map_df

    # ------------------------------------------------------------------
    # Importance loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_importance_aligned(
        score_path: str,
        num_nodes: int,
        node_features_path: Optional[str] = None,
        old_id_col_in_node_features: Optional[str] = None,
        strict: bool = True,
        fill_value: float = 0.0,
        verbose: bool = True,
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """
        Load node importance scores and always return an importance vector in old_id graph space.

        Supported cases:
        1) score file already stores old_id directly
           -> direct alignment

        2) score file stores continuous new_id, and node_features.csv stores old_id
           in the same row order as the new_id ordering
           -> rebuild new_id -> old_id, then project scores back to old_id space

        Continuous new_id supported:
            - 0..N-1
            - 1..N

        Args:
            score_path:
                Path to gnn score csv, e.g. gnn_node_scores_mean.csv
            num_nodes:
                Number of nodes in the target graph (old_id space)
            node_features_path:
                Optional path to node_features.csv. Required when score file is in new_id space.
            old_id_col_in_node_features:
                Explicit old-id column name in node_features.csv, if not auto-detected.
            strict:
                If True, raise error on any missing old_id score.
                If False, fill missing entries with fill_value.
            fill_value:
                Value used to fill missing old_id nodes when strict=False.
            verbose:
                Whether to print alignment diagnostics.
            return_info:
                If True, return (importance, info)

        Returns:
            importance: np.ndarray of shape [num_nodes], aligned to old_id graph space
            or
            (importance, info)
        """
        score_p = Path(score_path)
        if not score_p.exists():
            raise FileNotFoundError(f"Importance score file not found: {score_path}")

        score_df = pd.read_csv(score_p)
        if score_df.empty:
            raise ValueError(f"Score file is empty: {score_path}")

        node_col, score_col = GraphDatasetLoader._detect_score_columns(score_df)

        score_df = score_df[[node_col, score_col]].copy()
        score_df[node_col] = score_df[node_col].astype(int)
        score_df[score_col] = score_df[score_col].astype(float)

        if score_df[node_col].duplicated().any():
            dup = score_df.loc[score_df[node_col].duplicated(), node_col].tolist()
            raise ValueError(f"Duplicate node ids found in score file: {dup[:20]}")

        if score_df[score_col].isna().any():
            raise ValueError("Score file contains NaN values.")

        score_nodes = sorted(score_df[node_col].tolist())
        direct_expected = list(range(num_nodes))

        # --------------------------------------------------
        # Case A: direct old_id alignment
        # Only accept graph-space old_id = 0..num_nodes-1
        # --------------------------------------------------
        if len(score_df) == num_nodes and score_nodes == direct_expected:
            importance = np.zeros(num_nodes, dtype=float)
            importance[score_df[node_col].to_numpy()] = score_df[score_col].to_numpy()

            info = {
                "mode": "direct",
                "score_path": str(score_path),
                "node_features_path": None,
                "score_node_col": node_col,
                "score_col": score_col,
                "num_nodes": int(num_nodes),
                "loaded_rows": int(len(score_df)),
                "missing_old_ids": [],
                "missing_count": 0,
                "filled_missing": False,
                "fill_value": None,
                "new_id_start": None,
            }

            if verbose:
                print("[importance] direct alignment mode")
                print(f"  score_path = {score_path}")
                print(f"  num_nodes  = {num_nodes}")
                print(f"  detected node col = {node_col}, score col = {score_col}")

            return (importance, info) if return_info else importance

        # --------------------------------------------------
        # Case B: recovered via node_features.csv
        # --------------------------------------------------
        if node_features_path is None:
            raise ValueError(
                "Score file cannot be aligned directly to old_id space, and "
                "node_features_path is not provided for new_id -> old_id recovery.\n"
                f"score rows={len(score_df)}, graph num_nodes={num_nodes}, "
                f"score node range=[{min(score_nodes)}, {max(score_nodes)}]"
            )

        new_id_start = GraphDatasetLoader._detect_contiguous_new_id_start(score_nodes)
        if new_id_start is None:
            raise ValueError(
                "Score file is not direct old_id alignment, and its node ids are not a valid "
                "continuous new_id range either.\n"
                "Supported new_id ranges are 0..N-1 or 1..N.\n"
                f"Got range=[{min(score_nodes)}, {max(score_nodes)}], rows={len(score_df)}"
            )

        map_df = GraphDatasetLoader._build_new_to_old_from_node_features(
            node_features_path=node_features_path,
            score_len=len(score_df),
            new_id_start=new_id_start,
            old_id_col=old_id_col_in_node_features,
        )

        merged = pd.merge(
            score_df,
            map_df,
            left_on=node_col,
            right_on="new_id",
            how="left"
        )

        if merged["old_id"].isna().any():
            bad = merged.loc[merged["old_id"].isna(), node_col].tolist()
            raise ValueError(
                f"Some new_id entries in score file cannot be mapped to old_id. "
                f"Examples: {bad[:20]}"
            )

        merged["old_id"] = merged["old_id"].astype(int)

        out_of_range = merged.loc[
            (merged["old_id"] < 0) | (merged["old_id"] >= num_nodes), "old_id"
        ].tolist()
        if len(out_of_range) > 0:
            raise ValueError(
                f"Recovered old_id out of graph range [0, {num_nodes-1}]. "
                f"Examples: {out_of_range[:20]}"
            )

        if merged["old_id"].duplicated().any():
            dup = merged.loc[merged["old_id"].duplicated(), "old_id"].tolist()
            raise ValueError(
                f"Recovered old_id duplicated after mapping. Examples: {dup[:20]}"
            )

        importance = np.full(num_nodes, np.nan, dtype=float)
        importance[merged["old_id"].to_numpy()] = merged[score_col].to_numpy()

        missing = np.where(np.isnan(importance))[0]
        missing_list = missing.tolist()

        if len(missing) > 0 and strict:
            raise ValueError(
                f"Recovered importance misses {len(missing)} old_id nodes. "
                f"Examples: {missing[:20].tolist()}"
            )

        filled_missing = False
        if len(missing) > 0 and not strict:
            if verbose:
                print(
                    f"[warning] recovered importance misses {len(missing)} old_id nodes; "
                    f"fill {fill_value} for examples: {missing[:20].tolist()}"
                )
            importance[np.isnan(importance)] = fill_value
            filled_missing = True

        info = {
            "mode": "recovered",
            "score_path": str(score_path),
            "node_features_path": str(node_features_path) if node_features_path is not None else None,
            "score_node_col": node_col,
            "score_col": score_col,
            "num_nodes": int(num_nodes),
            "loaded_rows": int(len(score_df)),
            "recovered_rows": int(len(merged)),
            "missing_old_ids": missing_list,
            "missing_count": int(len(missing_list)),
            "filled_missing": filled_missing,
            "fill_value": fill_value if filled_missing else None,
            "new_id_start": int(new_id_start),
        }

        if verbose:
            print("[importance] recovered from new_id -> old_id mode")
            print(f"  score_path         = {score_path}")
            print(f"  node_features_path = {node_features_path}")
            print(f"  graph num_nodes    = {num_nodes}")
            print(f"  recovered rows     = {len(merged)}")
            print(f"  new_id_start       = {new_id_start}")
            print(f"  missing old_ids    = {len(missing_list)}")

        return (importance, info) if return_info else importance