from path.src.data.preprocess import GraphPreprocessor


def test_one(
    dataset_name,
    root,
    importance_path,
    node_features_path,
    old_id_col="node",
    strict=True,
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
    print(f"{dataset_name}: success")
    print("importance shape =", bundle.importance.shape)
    print("importance min =", bundle.importance.min())
    print("importance max =", bundle.importance.max())
    print("importance alignment =", bundle.metadata["importance_alignment"])
    print()


if __name__ == "__main__":
    # Planetoid
    planetoid_root = r"D:\project\keynode\project\public_datasets\Planetoid"

    test_one(
        "Cora",
        planetoid_root,
        r"D:\project\keynode\project\public_process\extract\Cora_struct\gnn_node_scores_mean.csv",
        r"D:\project\keynode\project\public_process\extract\Cora_struct\node_features.csv",
        old_id_col="node",
        strict=True,
    )

    test_one(
        "Citeseer",
        planetoid_root,
        r"D:\project\keynode\project\public_process\extract\CiteSeer_struct\gnn_node_scores_mean.csv",
        r"D:\project\keynode\project\public_process\extract\CiteSeer_struct\node_features.csv",
        old_id_col="node",
        strict=False,
    )

    test_one(
        "Pubmed",
        planetoid_root,
        r"D:\project\keynode\project\public_process\extract\PubMed_struct\gnn_node_scores_mean.csv",
        r"D:\project\keynode\project\public_process\extract\PubMed_struct\node_features.csv",
        old_id_col="node",
        strict=True,
    )