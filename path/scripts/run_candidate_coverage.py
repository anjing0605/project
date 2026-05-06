from __future__ import annotations

import json
from pathlib import Path

from path.src.data.preprocess import GraphPreprocessor
from path.src.core.keynode import KeyNodeSelector
from path.src.core.task_sampler import TaskPairBuilder
from path.src.core.fragility import FragilityEvaluator
from path.src.analysis.candidate_coverage import summarize_candidate_coverage
#候选路径覆盖率分析”。

def main() -> None:
    bundle = GraphPreprocessor.build_graph_bundle(
        name="Cora",
        root="D:/project/keynode/project/public_datasets/Planetoid",
        importance_path="D:/project/keynode/project/public_process/extract/Cora_struct/gnn_node_scores_mean.csv",
        community_mode="louvain",
    )

    key_nodes = KeyNodeSelector.select_topk_nodes(bundle.importance, 30)
    tasks = TaskPairBuilder.build_task_pairs(
        G=bundle.nx_graph,
        key_nodes=key_nodes,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=2,
    )

    fragility_evaluator = FragilityEvaluator()
    shared_base_metrics = fragility_evaluator.compute_base_metrics(bundle.nx_graph)

    stats = summarize_candidate_coverage(
        bundle=bundle,
        tasks=tasks,
        path_k=3,
        max_hops=8,
        delta=2,
        shared_base_metrics=shared_base_metrics,
        with_fragility=True,
    )

    out_path = Path("outputs/metrics/candidate_coverage_cora.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("saved:", out_path)
    print("mean_candidates_per_task =", stats["candidate_summary"]["mean_candidates_per_task"])
    print("path_length_distribution =", stats["path_length_distribution"])


if __name__ == "__main__":
    main()