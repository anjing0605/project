from __future__ import annotations

from itertools import islice
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
        shortest = nx.shortest_path_length(G, source=source, target=target)
        max_len = min(max_hops, shortest + delta)
        paths: List[List[int]] = []
        generator = nx.shortest_simple_paths(G, source=source, target=target)
        for path in islice(generator, 0, None):
            hops = len(path) - 1
            if hops <= max_len:
                paths.append(path)
            if len(paths) >= k:
                break
        return paths

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
