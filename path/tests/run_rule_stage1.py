from path.src.data.preprocess import GraphPreprocessor
from path.src.core.keynode import KeyNodeSelector
from path.src.core.task_sampler import TaskPairBuilder
from path.src.baselines.rule_based import RuleBasedCriticalPath
from path.src.core.evaluator import MethodEvaluator


def run_one_dataset(
    dataset_name,
    root,
    importance_path,
    node_features_path,
    old_id_col="node",
    strict=True,
    topk=30,
    path_k=3,
    max_hops=8,
    delta=2,
    top_q=10,
):
    bundle = GraphPreprocessor.build_graph_bundle(
        name=dataset_name,
        root=root,
        importance_path=importance_path,
        community_mode="louvain",
        node_features_path=node_features_path,
        old_id_col_in_node_features=old_id_col,
        strict_importance_alignment=strict,
        importance_fill_value=0.0,
        verbose=True,
    )

    print(f"\n========== {dataset_name} ==========")
    print("bundle loaded.")
    print("num_nodes =", bundle.num_nodes)
    print("num_edges =", bundle.nx_graph.number_of_edges())

    # 1) 选择关键节点
    key_nodes = KeyNodeSelector.select_topk_nodes(bundle.importance, topk)
    print("num key nodes =", len(key_nodes))
    print("top 10 key nodes =", key_nodes[:10])

    # 2) 构造任务对
    tasks = TaskPairBuilder.build_task_pairs(
        G=bundle.nx_graph,
        key_nodes=key_nodes,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=2,
    )
    print("num task pairs =", len(tasks))

    # 3) 跑规则法关键路径识别
    weights = {
        "avg_node_importance": 0.20,
        "avg_edge_bc": 0.15,
        "cross_comm_ratio": 0.10,
        "fragility_score": 0.45,
        "path_length": 0.10,
    }

    selected_paths = RuleBasedCriticalPath.run(
        bundle=bundle,
        tasks=tasks,
        path_k=path_k,
        max_hops=max_hops,
        delta=delta,
        weights=weights,
        top_q=top_q,
    )

    print("num selected critical paths =", len(selected_paths))
    for i, p in enumerate(selected_paths[:5]):
        print(f"[Path {i}]")
        print("nodes =", p.nodes)
        print("score =", p.score)
        print("features =", p.features)
        print("fragility =", p.fragility)
        print()

    # 4) 删除验证
    metrics = MethodEvaluator.evaluate_topk_damage(
        G=bundle.nx_graph,
        selected_paths=selected_paths,
        k_list=[1, 3, 5, 10],
    )

    print("damage metrics:")
    print(metrics)

    return bundle, selected_paths, metrics


if __name__ == "__main__":
    planetoid_root = r"D:\project\keynode\project\public_datasets\Planetoid"

    # 先只跑 Cora，确认全流程稳定
    run_one_dataset(
        dataset_name="Cora",
        root=planetoid_root,
        importance_path=r"D:\project\keynode\project\public_process\extract\Cora_struct\gnn_node_scores_mean.csv",
        node_features_path=r"D:\project\keynode\project\public_process\extract\Cora_struct\node_features.csv",
        old_id_col="node",
        strict=True,
        topk=30,
        path_k=3,
        max_hops=8,
        delta=2,
        top_q=10,
    )