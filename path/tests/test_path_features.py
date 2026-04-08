from __future__ import annotations

import numpy as np

from path.src.core.path_features import PathFeatureExtractor

'''
python -m path.tests.test
python -m path.tests.test_path_features
python -m path.tests.test_fragility
python -m path.tests.test_deduplicate
python -m path.tests.run_rule_stage1
python -m path.tests.test_env'''
def main():
    path = [0, 1, 2, 3]
    importance = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
    community = np.array([0, 0, 1, 1], dtype=int)
    edge_bc = {
        (0, 1): 0.2,
        (1, 2): 0.9,
        (2, 3): 0.3,
    }

    edges = PathFeatureExtractor.path_to_edges(path)
    print("edges =", edges)

    print("avg_node_importance =", PathFeatureExtractor.avg_node_importance(path, importance))
    print("internal_node_importance =", PathFeatureExtractor.internal_node_importance(path, importance))
    print("avg_edge_betweenness =", PathFeatureExtractor.avg_edge_betweenness(edges, edge_bc))
    print("cross_community_ratio =", PathFeatureExtractor.cross_community_ratio(path, community))

    feats = PathFeatureExtractor.extract_features(
        path=path,
        importance=importance,
        community=community,
        edge_bc=edge_bc,
    )
    print("features =", feats)


if __name__ == "__main__":
    main()