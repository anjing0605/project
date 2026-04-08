from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from path.src.core.types import TaskPair


class TaskBucketizer:
    @staticmethod
    def bucket_by_shortest_len(tasks: List[TaskPair]) -> Dict[str, List[TaskPair]]:
        buckets = defaultdict(list)
        for t in tasks:
            if t.shortest_len == 2:
                buckets["len2"].append(t)
            elif t.shortest_len == 3:
                buckets["len3"].append(t)
            else:
                buckets["len4p"].append(t)
        return dict(buckets)

    @staticmethod
    def mix_buckets(
        bucket_dict: Dict[str, List[TaskPair]],
        ratios: Dict[str, float],
        total_size: int,
        seed: int = 42,
    ) -> List[TaskPair]:
        import random

        rng = random.Random(seed)
        out: List[TaskPair] = []

        for name, ratio in ratios.items():
            pool = bucket_dict.get(name, [])
            if not pool:
                continue
            k = max(1, int(total_size * ratio))
            if k <= len(pool):
                out.extend(rng.sample(pool, k))
            else:
                out.extend(rng.choices(pool, k=k))
        rng.shuffle(out)
        return out[:total_size]