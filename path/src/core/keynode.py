from __future__ import annotations

from typing import List
import numpy as np


class KeyNodeSelector:
    """
    Select key nodes from node importance scores.
    """

    @staticmethod
    def select_topk_nodes(importance: np.ndarray, k: int) -> List[int]:
        """
        Select top-k node ids by descending importance.

        Args:
            importance: shape [N], node importance scores
            k: number of key nodes

        Returns:
            key_nodes: list of node ids sorted by descending importance
        """
        importance = np.asarray(importance, dtype=float).reshape(-1)

        if importance.ndim != 1:
            raise ValueError("importance must be a 1D array")

        n = len(importance)
        if n == 0:
            raise ValueError("importance is empty")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        k = min(k, n)

        if np.isnan(importance).any():
            raise ValueError("importance contains NaN values")

        # 降序；若分数相同，则节点编号小的在前
        order = np.lexsort((np.arange(n), -importance))
        key_nodes = order[:k].tolist()
        return key_nodes

    @staticmethod
    def select_top_ratio_nodes(importance: np.ndarray, ratio: float) -> List[int]:
        """
        Select top-ratio node ids by descending importance.

        Args:
            importance: shape [N]
            ratio: e.g. 0.05 means top 5%

        Returns:
            key_nodes
        """
        importance = np.asarray(importance, dtype=float).reshape(-1)

        if ratio <= 0 or ratio > 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        k = max(1, int(np.ceil(len(importance) * ratio)))
        return KeyNodeSelector.select_topk_nodes(importance, k)