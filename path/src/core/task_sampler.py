from __future__ import annotations

from typing import Dict, List, Tuple
import random
import numpy as np
import networkx as nx

from path.src.core.types import TaskPair


class TaskPairBuilder:
    """
    Build and sample (source, target) task pairs from key nodes.
    """

    @staticmethod
    def _score_level_from_pair_score(
        pair_score: float,
        q_high: float,
        q_mid: float,
    ) -> str:
        if pair_score >= q_high:
            return "high"
        elif pair_score >= q_mid:
            return "mid"
        else:
            return "low"

    @staticmethod
    def _distance_level_from_shortest_len(d: int) -> str:
        if d <= 3:
            return "short"
        elif d <= 5:
            return "mid"
        else:
            return "long"

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
            pair_score = importance[s] + importance[t]

        Additional metadata will be stored in task.metadata:
            - community_type: "same" / "cross"
            - score_level: "high" / "mid" / "low"
            - distance_level: "short" / "mid" / "long"
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

        raw_rows: List[Tuple[int, int, int, bool, float]] = []

        for i, s in enumerate(key_nodes):
            if s not in G:
                continue
            for t in key_nodes[i + 1:]:
                if t not in G or s == t:
                    continue

                try:
                    d = nx.shortest_path_length(G, s, t)
                except nx.NetworkXNoPath:
                    continue

                if d < min_shortest_len:
                    continue

                same_community = bool(community[s] == community[t])
                pair_score = float(importance[s] + importance[t])

                raw_rows.append((int(s), int(t), int(d), same_community, pair_score))

        if not raw_rows:
            return []

        pair_scores = np.array([x[4] for x in raw_rows], dtype=float)
        q_high = float(np.quantile(pair_scores, 0.80))
        q_mid = float(np.quantile(pair_scores, 0.40))

        tasks: List[TaskPair] = []

        for s, t, d, same_community, pair_score in raw_rows:
            community_type = "same" if same_community else "cross"
            score_level = TaskPairBuilder._score_level_from_pair_score(
                pair_score=pair_score,
                q_high=q_high,
                q_mid=q_mid,
            )
            distance_level = TaskPairBuilder._distance_level_from_shortest_len(d)

            tasks.append(
                TaskPair(
                    source=s,
                    target=t,
                    shortest_len=d,
                    same_community=same_community,
                    pair_score=pair_score,
                    metadata={
                        "community_type": community_type,
                        "score_level": score_level,
                        "distance_level": distance_level,
                    },
                )
            )

        # 不再单纯按高分排序，而是保留完整任务池；
        # 排序仅作为 deterministic 输出。
        tasks.sort(
            key=lambda x: (
                x.pair_score,
                x.shortest_len,
                int(not x.same_community),  # cross community 略优先
                -x.source,
                -x.target,
            ),
            reverse=True,
        )
        return tasks

    @staticmethod
    def _group_tasks(
        task_pairs: List[TaskPair],
    ) -> Dict[Tuple[str, str, str], List[TaskPair]]:
        """
        Group tasks by:
            (community_type, score_level, distance_level)
        """
        grouped: Dict[Tuple[str, str, str], List[TaskPair]] = {}

        for t in task_pairs:
            meta = getattr(t, "metadata", None) or {}
            community_type = meta.get("community_type", "cross" if not t.same_community else "same")
            score_level = meta.get("score_level", "mid")
            distance_level = meta.get("distance_level", "mid")

            key = (community_type, score_level, distance_level)
            grouped.setdefault(key, []).append(t)

        return grouped

    @staticmethod
    def sample_task_pairs(
        task_pairs: List[TaskPair],
        num_samples: int,
        mode: str = "paper_default",
        random_seed: int = 42,
    ) -> List[TaskPair]:
        """
        Sample task pairs for experiments/training.

        Modes:
            - mixed: directly sample from all tasks
            - high_importance: top pair_score first
            - cross_community: only keep cross-community pairs
            - long_distance: prefer larger shortest_len
            - stratified: stratified by community_type / score_level / distance_level
            - paper_default:
                70% cross-community + 30% same-community
                and balanced across score / distance levels as much as possible
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        if not task_pairs:
            return []

        rng = random.Random(random_seed)

        if mode == "mixed":
            pool = list(task_pairs)
            if len(pool) <= num_samples:
                return pool
            return rng.sample(pool, num_samples)

        elif mode == "high_importance":
            sorted_pairs = sorted(task_pairs, key=lambda x: x.pair_score, reverse=True)
            return sorted_pairs[: min(num_samples, len(sorted_pairs))]

        elif mode == "cross_community":
            pool = [t for t in task_pairs if not t.same_community]
            if len(pool) <= num_samples:
                return pool
            return rng.sample(pool, num_samples)

        elif mode == "long_distance":
            sorted_pairs = sorted(task_pairs, key=lambda x: x.shortest_len, reverse=True)
            return sorted_pairs[: min(num_samples, len(sorted_pairs))]

        elif mode in ("stratified", "paper_default"):
            grouped = TaskPairBuilder._group_tasks(task_pairs)

            if mode == "stratified":
                # 所有 bucket 尽量均匀抽
                active_keys = [k for k, v in grouped.items() if v]
                if not active_keys:
                    return []
                quota = max(1, num_samples // len(active_keys))

                sampled = []
                used = set()

                for k in active_keys:
                    pool = grouped[k]
                    pool = sorted(
                        pool,
                        key=lambda x: (x.pair_score, x.shortest_len),
                        reverse=True,
                    )
                    take_n = min(quota, len(pool))
                    for t in pool[:take_n]:
                        key_id = (t.source, t.target)
                        if key_id not in used:
                            used.add(key_id)
                            sampled.append(t)

                if len(sampled) < num_samples:
                    remain = []
                    for pool in grouped.values():
                        for t in pool:
                            key_id = (t.source, t.target)
                            if key_id not in used:
                                remain.append(t)

                    remain = sorted(
                        remain,
                        key=lambda x: (x.pair_score, x.shortest_len),
                        reverse=True,
                    )
                    sampled.extend(remain[: max(0, num_samples - len(sampled))])

                return sampled[:num_samples]

            else:
                # paper_default:
                # 70% cross, 30% same
                cross = [t for t in task_pairs if not t.same_community]
                same = [t for t in task_pairs if t.same_community]

                n_cross = int(round(num_samples * 0.7))
                n_same = num_samples - n_cross

                def stratified_take(pool: List[TaskPair], k: int) -> List[TaskPair]:
                    if k <= 0 or not pool:
                        return []

                    grouped_local: Dict[Tuple[str, str], List[TaskPair]] = {}
                    for t in pool:
                        meta = getattr(t, "metadata", None) or {}
                        score_level = meta.get("score_level", "mid")
                        distance_level = meta.get("distance_level", "mid")
                        grouped_local.setdefault((score_level, distance_level), []).append(t)

                    keys = [kk for kk, vv in grouped_local.items() if vv]
                    if not keys:
                        return []

                    quota = max(1, k // len(keys))
                    out = []
                    used = set()

                    for kk in keys:
                        bucket = sorted(
                            grouped_local[kk],
                            key=lambda x: (x.pair_score, x.shortest_len),
                            reverse=True,
                        )
                        take_n = min(quota, len(bucket))
                        for t in bucket[:take_n]:
                            key_id = (t.source, t.target)
                            if key_id not in used:
                                used.add(key_id)
                                out.append(t)

                    if len(out) < k:
                        remain = []
                        for bucket in grouped_local.values():
                            for t in bucket:
                                key_id = (t.source, t.target)
                                if key_id not in used:
                                    remain.append(t)

                        remain = sorted(
                            remain,
                            key=lambda x: (x.pair_score, x.shortest_len),
                            reverse=True,
                        )
                        out.extend(remain[: max(0, k - len(out))])

                    return out[:k]

                sampled_cross = stratified_take(cross, n_cross)
                sampled_same = stratified_take(same, n_same)

                sampled = sampled_cross + sampled_same

                if len(sampled) < num_samples:
                    used = {(t.source, t.target) for t in sampled}
                    remain = [
                        t for t in sorted(
                            task_pairs,
                            key=lambda x: (x.pair_score, x.shortest_len),
                            reverse=True,
                        )
                        if (t.source, t.target) not in used
                    ]
                    sampled.extend(remain[: max(0, num_samples - len(sampled))])

                return sampled[:num_samples]

        else:
            raise ValueError(
                f"Unsupported mode: {mode}. "
                f"Expected one of ['mixed', 'high_importance', 'cross_community', "
                f"'long_distance', 'stratified', 'paper_default']"
            )