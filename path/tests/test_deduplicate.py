from __future__ import annotations

from path.src.core.deduplicate import PathDeduplicator
from path.src.core.types import PathRecord


def make_record(nodes, score):
    return PathRecord(
        nodes=nodes,
        edges=list(zip(nodes[:-1], nodes[1:])),
        source=nodes[0],
        target=nodes[-1],
        success=True,
        method="test",
        score=score,
        features={},
        fragility={},
        metadata={},
    )


def main():
    p1 = make_record([0, 1, 2, 3], 0.9)
    p2 = make_record([0, 1, 2, 4], 0.8)
    p3 = make_record([5, 6, 7], 0.7)

    overlap_12 = PathDeduplicator.edge_overlap(p1.nodes, p2.nodes)
    overlap_13 = PathDeduplicator.edge_overlap(p1.nodes, p3.nodes)

    print("overlap(p1, p2) =", overlap_12)
    print("overlap(p1, p3) =", overlap_13)

    selected = PathDeduplicator.greedy_deduplicate(
        [p1, p2, p3],
        overlap_threshold=0.6,
        top_q=10,
    )

    print("selected num =", len(selected))
    for p in selected:
        print("selected:", p.nodes, "score=", p.score)


if __name__ == "__main__":
    main()