from __future__ import annotations

from typing import List
import random
import numpy as np
import networkx as nx

from path.src.core.types import TaskPair


class TaskPairBuilder:
    """
    Build and sample (source, target) task pairs from key nodes.
    """

    @staticmethod
    def build_task_pairs(
        G: nx.Graph,
        key_nodes: List[int],
        community: np.ndarray,
        importance: np.ndarray,
        min_shortest_len: int = 2,
    ) -> List[TaskPair]:
        """
        Build all legal task pairs from key nodes.

        Rules:
            - source != target
            - must be connected
            - shortest path length >= min_shortest_len

        pair_score:
            here we use a simple score:
                pair_score = importance[s] + importance[t]

        Args:
            G: networkx graph
            key_nodes: list of key node ids
            community: shape [N], community label per node
            importance: shape [N], node importance
            min_shortest_len: minimum shortest path length

        Returns:
            task_pairs: list[TaskPair]
        """
        importance = np.asarray(importance, dtype=float).reshape(-1)
        community = np.asarray(community).reshape(-1)

        if len(importance) != G.number_of_nodes():
            raise ValueError(
                f"importance length {len(importance)} != num_nodes {G.number_of_nodes()}"
            )
        if len(community) != G.number_of_nodes():
            raise ValueError(
                f"community length {len(community)} != num_nodes {G.number_of_nodes()}"
            )

        key_nodes = list(dict.fromkeys(key_nodes))  # 去重并保持顺序
        tasks: List[TaskPair] = []

        for i, s in enumerate(key_nodes):
            if s not in G:
                continue
            for t in key_nodes[i + 1:]:
                if t not in G:
                    continue
                if s == t:
                    continue

                try:
                    d = nx.shortest_path_length(G, s, t)
                except nx.NetworkXNoPath:
                    continue

                if d < min_shortest_len:
                    continue

                same_community = bool(community[s] == community[t])
                pair_score = float(importance[s] + importance[t])

                tasks.append(
                    TaskPair(
                        source=int(s),
                        target=int(t),
                        shortest_len=int(d),
                        same_community=same_community,
                        pair_score=pair_score,
                    )
                )

        # 按 pair_score 降序；若分数相同，按 shortest_len 降序
        tasks.sort(
            key=lambda x: (x.pair_score, x.shortest_len, -x.source, -x.target),
            reverse=True
        )
        return tasks

    @staticmethod
    def sample_task_pairs(
        task_pairs: List[TaskPair],
        num_samples: int,
        mode: str = "mixed",
        random_seed: int = 42,
    ) -> List[TaskPair]:
        """
        Sample task pairs for experiments/training.

        Modes:
            - mixed: directly sample from all tasks
            - high_importance: prefer high pair_score
            - cross_community: only keep cross-community pairs
            - long_distance: prefer larger shortest_len

        Args:
            task_pairs: all candidate task pairs
            num_samples: number of sampled tasks
            mode: sampling mode
            random_seed: random seed

        Returns:
            sampled task list
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        if not task_pairs:
            return []

        rng = random.Random(random_seed)

        if mode == "mixed":
            pool = task_pairs

        elif mode == "high_importance":
            # 直接取前 num_samples 个高分任务对
            return task_pairs[: min(num_samples, len(task_pairs))]

        elif mode == "cross_community":
            pool = [t for t in task_pairs if not t.same_community]

        elif mode == "long_distance":
            sorted_pairs = sorted(task_pairs, key=lambda x: x.shortest_len, reverse=True)
            return sorted_pairs[: min(num_samples, len(sorted_pairs))]

        else:
            raise ValueError(
                f"Unsupported mode: {mode}. "
                f"Expected one of ['mixed', 'high_importance', 'cross_community', 'long_distance']"
            )

        if not pool:
            return []

        if len(pool) <= num_samples:
            return pool

        return rng.sample(pool, num_samples)