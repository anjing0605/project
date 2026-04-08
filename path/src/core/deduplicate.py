from __future__ import annotations

from typing import List, Set, Tuple

from path.src.core.path_features import PathFeatureExtractor
from path.src.core.types import PathRecord


class PathDeduplicator:
    @staticmethod
    def _edge_set(path: List[int]) -> Set[Tuple[int, int]]:
        edges = PathFeatureExtractor.path_to_edges(path)
        return {tuple(sorted(e)) for e in edges}

    @staticmethod
    def edge_overlap(path_a: List[int], path_b: List[int]) -> float:
        A = PathDeduplicator._edge_set(path_a)
        B = PathDeduplicator._edge_set(path_b)
        if not A and not B:
            return 1.0
        union = A | B
        inter = A & B
        return 0.0 if len(union) == 0 else float(len(inter) / len(union))

    @staticmethod
    def greedy_deduplicate(
        path_records: List[PathRecord],
        overlap_threshold: float = 0.6,
        top_q: int = 10,
    ) -> List[PathRecord]:
        selected: List[PathRecord] = []
        for rec in path_records:
            keep = True
            for s in selected:
                if PathDeduplicator.edge_overlap(rec.nodes, s.nodes) > overlap_threshold:
                    keep = False
                    break
            if keep:
                selected.append(rec)
            if len(selected) >= top_q:
                break
        return selected
