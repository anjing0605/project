from __future__ import annotations

from typing import List, Optional
import networkx as nx


class PathGenerator:
    @staticmethod
    def shortest_path(G: nx.Graph, source: int, target: int) -> List[int]:
        return nx.shortest_path(G, source=source, target=target)

    @staticmethod
    def k_shortest_simple_paths(
        G: nx.Graph,
        source: int,
        target: int,
        k: int,
        max_hops: int = 8,
        delta: int = 2,
    ) -> List[List[int]]:
        if k <= 0:
            return []

        shortest = nx.shortest_path_length(G, source=source, target=target)
        max_len = min(max_hops, shortest + delta)

        paths: List[List[int]] = []
        generator = nx.shortest_simple_paths(G, source=source, target=target)

        for path in generator:
            hops = len(path) - 1

            # shortest_simple_paths 已按路径代价非降序输出，
            # 一旦 hop 超过 max_len，后续一般不会更短，直接停止。
            if hops > max_len:
                break

            paths.append(path)
            if len(paths) >= k:
                break

        return paths

    @staticmethod
    def _internal_node_set(path: List[int]) -> set[int]:
        if path is None or len(path) <= 2:
            return set()
        return set(int(x) for x in path[1:-1])

    @staticmethod
    def _jaccard_overlap(a: set[int], b: set[int]) -> float:
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return float(len(a & b) / len(union))

    @classmethod
    def diversify_paths(
        cls,
        paths: List[List[int]],
        top_k: int,
        max_internal_overlap: float = 0.60,
    ) -> List[List[int]]:
        """
        Diversity-aware path filtering.

        Keep paths whose INTERNAL-NODE overlap with already selected paths
        is not too large.

        Notes
        -----
        - endpoints are ignored
        - order of input paths matters:
          paths earlier in the list have higher priority
        """
        if top_k <= 0 or not paths:
            return []

        selected: List[List[int]] = []
        selected_internal_sets: List[set[int]] = []

        for path in paths:
            internal = cls._internal_node_set(path)

            keep = True
            for prev_internal in selected_internal_sets:
                overlap = cls._jaccard_overlap(internal, prev_internal)
                if overlap > max_internal_overlap:
                    keep = False
                    break

            if keep:
                selected.append(path)
                selected_internal_sets.append(internal)

            if len(selected) >= top_k:
                break

        return selected

    @classmethod
    def diversified_k_shortest_simple_paths(
        cls,
        G: nx.Graph,
        source: int,
        target: int,
        raw_k: int,
        final_k: int,
        max_hops: int = 8,
        delta: int = 2,
        max_internal_overlap: float = 0.60,
    ) -> List[List[int]]:
        """
        Two-stage candidate generation:
        1) generate more near-shortest simple paths
        2) perform diversity-aware filtering
        """
        raw_paths = cls.k_shortest_simple_paths(
            G=G,
            source=source,
            target=target,
            k=raw_k,
            max_hops=max_hops,
            delta=delta,
        )

        return cls.diversify_paths(
            paths=raw_paths,
            top_k=final_k,
            max_internal_overlap=max_internal_overlap,
        )

    @staticmethod
    def random_simple_path(
        G: nx.Graph,
        source: int,
        target: int,
        max_hops: int = 8,
        num_trials: int = 20,
        rng=None,
    ) -> Optional[List[int]]:
        import random

        rng = rng or random.Random()
        for _ in range(num_trials):
            curr = source
            path = [curr]
            visited = {curr}
            for _step in range(max_hops):
                if curr == target:
                    return path
                nbrs = [n for n in G.neighbors(curr) if n not in visited]
                if not nbrs:
                    break
                curr = rng.choice(nbrs)
                path.append(curr)
                visited.add(curr)
                if curr == target:
                    return path
        return None