from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data.preprocess import GraphPreprocessor
from src.core.keynode import KeyNodeSelector
from src.core.task_sampler import TaskPairBuilder
from src.core.fragility import FragilityEvaluator
from src.core.evaluator import MethodEvaluator
from src.baselines.rule_based import RuleBasedCriticalPath


def serialize_path_record(p) -> Dict:
    return {
        "source": p.source,
        "target": p.target,
        "nodes": p.nodes,
        "length": len(p.nodes) - 1,
        "score": p.score,
        "method": p.method,
        "features": p.features,
        "fragility": p.fragility,
        "metadata": getattr(p, "metadata", None),
    }


def get_weight_sets() -> Dict[str, Dict[str, float]]:
    """
    Minimal ablation sets:
      A: current
      B: remove cross_comm_ratio
      C: fragility 0.55
      D: balanced 0.45
    """
    return {
        "A_current": {
            "avg_node_importance": 0.12,
            "avg_edge_bc": 0.10,
            "cross_comm_ratio": 0.03,
            "fragility_score": 0.65,
            "path_length": 0.10,
        },
        "B_no_cross_comm": {
            "avg_node_importance": 0.13,
            "avg_edge_bc": 0.12,
            "cross_comm_ratio": 0.00,
            "fragility_score": 0.65,
            "path_length": 0.10,
        },
        "C_fragility_055": {
            "avg_node_importance": 0.16,
            "avg_edge_bc": 0.12,
            "cross_comm_ratio": 0.07,
            "fragility_score": 0.55,
            "path_length": 0.10,
        },
        "D_balanced_045": {
            "avg_node_importance": 0.20,
            "avg_edge_bc": 0.15,
            "cross_comm_ratio": 0.10,
            "fragility_score": 0.45,
            "path_length": 0.10,
        },
    }


def main() -> None:
    dataset_name = "Cora"
    bundle = GraphPreprocessor.build_graph_bundle(
        name=dataset_name,
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

    shared_base_metrics = FragilityEvaluator.compute_base_metrics(bundle.nx_graph)
    evaluator = MethodEvaluator(lambda_E=0.4, lambda_LCC=0.4, lambda_ASP=0.2)

    all_results = {
        "dataset": dataset_name,
        "num_tasks": len(tasks),
        "shared_base_metrics": shared_base_metrics,
        "ablation": {},
    }

    weight_sets = get_weight_sets()

    for name, weights in weight_sets.items():
        print(f"\n========== running ablation: {name} ==========")
        print("weights =", weights)

        selected_paths = RuleBasedCriticalPath.run(
            bundle=bundle,
            tasks=tasks,
            path_k=3,
            max_hops=8,
            delta=2,
            weights=weights,
            top_q=10,
            shared_base_metrics=shared_base_metrics,
        )

        metrics = evaluator.evaluate_topk_damage(
            G=bundle.nx_graph,
            selected_paths=selected_paths,
            k_list=[1, 3, 5, 10],
            shared_base_metrics=shared_base_metrics,
        )

        all_results["ablation"][name] = {
            "weights": weights,
            "metrics": metrics,
            "top_paths": [serialize_path_record(p) for p in selected_paths],
        }

        print("top1 score =", selected_paths[0].score if selected_paths else None)
        print("fragility_score_curve =", metrics["fragility_score_curve"])

    out_path = Path("outputs/metrics/rule_ablation_cora.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nsaved:", out_path)


if __name__ == "__main__":
    main()