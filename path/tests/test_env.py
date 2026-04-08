from __future__ import annotations

from pathlib import Path

from path.src.data.preprocess import GraphPreprocessor
from path.src.models.gnn_encoder import FrozenGNNEncoder
from path.src.core.keynode import KeyNodeSelector
from path.src.core.task_sampler import TaskPairBuilder
from path.src.rl.env import CriticalPathEnv


def main():
    project_root = Path(r"D:\project\keynode\project")
    planetoid_root = project_root / "public_datasets" / "Planetoid"

    bundle = GraphPreprocessor.build_graph_bundle(
        name="Cora",
        root=str(planetoid_root),
        importance_path=str(project_root / "public_process" / "extract" / "Cora_struct" / "gnn_node_scores_mean.csv"),
        community_mode="louvain",
        node_features_path=str(project_root / "public_process" / "extract" / "Cora_struct" / "node_features.csv"),
        old_id_col_in_node_features="node",
        strict_importance_alignment=True,
        importance_fill_value=0.0,
        verbose=True,
    )

    key_nodes = KeyNodeSelector.select_topk_nodes(bundle.importance, 30)
    tasks = TaskPairBuilder.build_task_pairs(
        G=bundle.nx_graph,
        key_nodes=key_nodes,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=2,
    )
    if not tasks:
        raise RuntimeError("No tasks constructed for env test.")

    emb_ckpt = project_root / "path" / "outputs" / "checkpoints" / "debug_node_embeddings.pt"
    embeddings = FrozenGNNEncoder.fit_or_load(
        bundle=bundle,
        ckpt_path=str(emb_ckpt),
        hidden_dim=128,
        out_dim=64,
        epochs=10,
        lr=1e-2,
    )

    env = CriticalPathEnv(
        bundle=bundle,
        node_embeddings=embeddings,
        max_hops=8,
        reward_mode="surrogate",
        reward_kwargs={},
    )

    state = env.reset(tasks[0])
    print("reset ok")
    print("curr_node =", state["curr_node"])
    print("target_node =", state["target_node"])
    print("valid_actions[:10] =", state["valid_actions"][:10])

    valid_actions = env.get_valid_actions()
    if not valid_actions:
        raise RuntimeError("No valid actions available after reset.")

    next_state, reward, done, info = env.step(valid_actions[0])
    print("step ok")
    print("reward =", reward)
    print("done =", done)
    print("info =", info)
    print("next curr_node =", next_state["curr_node"])


if __name__ == "__main__":
    main()