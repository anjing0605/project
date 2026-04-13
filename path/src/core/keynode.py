from __future__ import annotations

from typing import Dict, List
import numpy as np


class KeyNodeSelector:
    """
    Select key nodes from node importance scores.
    """

    @staticmethod
    def select_topk_nodes(importance: np.ndarray, k: int) -> List[int]:
        """
        Select top-k node ids by descending importance.
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
        """
        importance = np.asarray(importance, dtype=float).reshape(-1)

        if ratio <= 0 or ratio > 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        k = max(1, int(np.ceil(len(importance) * ratio)))
        return KeyNodeSelector.select_topk_nodes(importance, k)

    @staticmethod
    def select_stratified_nodes(
        importance: np.ndarray,
        total_k: int,
        high_ratio: float = 0.4,
        mid_ratio: float = 0.4,
        low_ratio: float = 0.2,
    ) -> Dict[str, List[int]]:
        """
        Stratified key-node selection by importance ranking.

        Split all nodes into three buckets by ranking:
            - high: top 20%
            - mid : 20% ~ 60%
            - low : 60% ~ 100%

        Then sample a fixed number from each bucket according to
        total_k * {high_ratio, mid_ratio, low_ratio}.

        Returns:
            {
                "high": [...],
                "mid": [...],
                "low": [...],
                "all": [...]
            }
        """
        importance = np.asarray(importance, dtype=float).reshape(-1)

        if importance.ndim != 1:
            raise ValueError("importance must be a 1D array")
        if len(importance) == 0:
            raise ValueError("importance is empty")
        if np.isnan(importance).any():
            raise ValueError("importance contains NaN values")
        if total_k <= 0:
            raise ValueError(f"total_k must be positive, got {total_k}")

        ratios = np.array([high_ratio, mid_ratio, low_ratio], dtype=float)
        if np.any(ratios < 0):
            raise ValueError("ratios must be non-negative")
        if ratios.sum() <= 0:
            raise ValueError("sum of ratios must be positive")

        ratios = ratios / ratios.sum()

        n = len(importance)
        total_k = min(total_k, n)

        # importance descending; tie-break by smaller node id
        order = np.lexsort((np.arange(n), -importance))

        high_end = max(1, int(np.ceil(n * 0.20)))
        mid_end = max(high_end + 1, int(np.ceil(n * 0.60)))

        high_pool = order[:high_end].tolist()
        mid_pool = order[high_end:mid_end].tolist()
        low_pool = order[mid_end:].tolist()

        raw = ratios * total_k
        counts = np.floor(raw).astype(int)
        remainders = raw - counts

        while counts.sum() < total_k:
            idx = int(np.argmax(remainders))
            counts[idx] += 1
            remainders[idx] = -1.0

        high_k, mid_k, low_k = counts.tolist()

        def take(pool: List[int], k: int) -> List[int]:
            if k <= 0:
                return []
            return pool[: min(k, len(pool))]

        high_nodes = take(high_pool, high_k)
        mid_nodes = take(mid_pool, mid_k)
        low_nodes = take(low_pool, low_k)

        selected_all = []
        seen = set()
        for bucket in [high_nodes, mid_nodes, low_nodes]:
            for x in bucket:
                if x not in seen:
                    seen.add(x)
                    selected_all.append(x)

        # 如果由于 bucket 太短导致数量不够，则从全局 ranking 补齐
        if len(selected_all) < total_k:
            for x in order.tolist():
                if x not in seen:
                    seen.add(x)
                    selected_all.append(x)
                if len(selected_all) >= total_k:
                    break

        return {
            "high": high_nodes,
            "mid": mid_nodes,
            "low": low_nodes,
            "all": selected_all,
        }