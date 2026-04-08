from __future__ import annotations

import networkx as nx

from path.src.core.fragility import FragilityEvaluator


def main():
    G = nx.Graph()
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 3),
            (3, 4),
        ]
    )

    base_metrics = FragilityEvaluator.compute_base_metrics(G)
    print("base_metrics =", base_metrics)

    path = [1, 2, 3]

    evaluator = FragilityEvaluator(lambda_E=0.4, lambda_LCC=0.4, lambda_ASP=0.2)
    out = evaluator.compute_fragility(
        G=G,
        path=path,
        base_metrics=base_metrics,
        num_nodes=G.number_of_nodes(),
    )

    print("path =", path)
    print("fragility =", out)


if __name__ == "__main__":
    main()