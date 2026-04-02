import os
import copy
import numpy as np
import torch
import pandas as pd
from torch_geometric.loader import DataLoader

from train_gat import (
    SlimRunDataset,
    #SlimMultiSAGE,
    SlimMultiGAT,
    train_one_epoch,
    evaluate,
)

# ============================================================
# 1️⃣ 基础默认配置
# ============================================================

BASE_CONFIG = {
    "w_g_reg":  2.5,
    "w_g_rank": 3.5,
    "w_b_reg":  0.1,
    "w_b_rank": 0.5,
    "w_mono": 0.05,
    "w_topk": 4.0,

    "use_struct": True,
    "use_perf": False,
    "use_semantic": True,

    "use_graph": True,
    "multi_beta": True,
    "num_layers": 2,
}

# ============================================================
# 2️⃣ 功能递进式配置
# ============================================================

PROGRESSIVE_SETTINGS = {

    # L0: 纯MLP（无图）
    "L0_MLP": {
        "use_graph": False,
        "num_layers": 0,
        "w_g_rank": 0.0,
        "multi_beta": False,
    },

    # L1: 1层GraphSAGE + 回归
    "L1_1Layer_GNN": {
        "use_graph": True,
        "num_layers": 1,
        "w_g_rank": 0.0,
        "multi_beta": False,
    },

    # L2: 2层GraphSAGE + 回归
    "L2_2Layer_GNN": {
        "use_graph": True,
        "num_layers": 2,
        "w_g_rank": 0.0,
        "multi_beta": False,
    },

    # L3: + 排序损失
    "L3_GNN_Rank": {
        "use_graph": True,
        "num_layers": 2,
        "w_g_rank": 3.5,
        "multi_beta": False,
    },

    # L4: + Multi-beta
    "L4_MultiBeta": {
        "use_graph": True,
        "num_layers": 2,
        "w_g_rank": 3.5,
        "multi_beta": True,
        "w_b_reg": 0.0,
        "w_b_rank": 0.0,
        "w_mono": 0.0,
        "w_topk": 0.0,
    },

    # L5: 完整模型
    "L5_Full": {
        "use_graph": True,
        "num_layers": 2,
        "w_g_rank": 3.5,
        "multi_beta": True,
        "w_b_reg": 0.1,
        "w_b_rank": 0.5,
        "w_mono": 0.05,
        "w_topk": 4.0,
    },
}
MODAL_ABLATION = {

    "Struct_only": {
        "use_struct": True,
        "use_semantic": False,
    },

    "Semantic_only": {
        "use_struct": False,
        "use_semantic": True,
    },

    "Struct+Semantic": {
        "use_struct": True,
        "use_semantic": True,
    },
}
# ============================================================
# 3️⃣ 单配置训练函数
# ============================================================

def run_single_config(data_root, dataset_name, config, device, seed):

    # ---- 固定随机种子 ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = SlimRunDataset(
        root_dir=data_root,
        dataset_name=dataset_name,
        use_struct=config["use_struct"],
        use_perf=config["use_perf"],
        use_semantic=config["use_semantic"],
        seed=seed
    )

    data = dataset[0]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---- multi-beta 控制 ----
    if config["multi_beta"]:
        num_beta = data.y_sir_multi.size(1)
    else:
        num_beta = 1
    struct_dim = 5 if config["use_struct"] else 0
    semantic_dim = 0
    if config["use_semantic"]:
        s = data.modal_slices["semantic"]
        semantic_dim = s[1] - s[0]

    model = SlimMultiGAT(
        perf_dim=4,
        struct_dim=struct_dim,
        semantic_dim=semantic_dim,
        hidden_dim=128,
        modal_dim=128,
        fusion_heads=4,
        config={
            "num_beta": num_beta,
            "use_perf": config["use_perf"],
            "use_struct": config["use_struct"],
            "use_semantic":config["use_semantic"],
            "use_graph": config["use_graph"],
            "multi_beta": config["multi_beta"],      # ← 必须加
            "num_layers": config["num_layers"],      # ← 建议加
        }
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4
    )

    best_sp = -1
    best_state = None
    patience = 200
    counter = 0
    max_epoch = 30000

    print("Training...")

    for epoch in range(1, max_epoch + 1):

        loss, _ = train_one_epoch(
            model, loader, optimizer, device, config
        )

        metrics = evaluate(model, loader, device, split="val")
        sp = metrics["spearman"]

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Loss {loss:.4f} | "
                f"SP {metrics['spearman']:.4f} | "
                f"KD {metrics['kendall']:.4f}"
            )

        if sp > best_sp:
            best_sp = sp
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    test_metrics = evaluate(model, loader, device, split="test")

    return test_metrics


# ============================================================
# 4️⃣ 主函数
# ============================================================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE = "scratch/project/public_process/extract"

    DATASETS = [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Computers",
        "Photo",
        "CS",
        "Physics"
    ]

    seeds = [0,1,2,3,4]

    # =====================================================
    # 遍历所有数据集
    # =====================================================

    for dataset_name in DATASETS:

        print(f"\n================ DATASET: {dataset_name} ================\n")

        data_root = f"{BASE}/{dataset_name}_struct"

        # =====================================================
        # Progressive Ablation
        # =====================================================

        final_results = {}

        for stage_name, override in PROGRESSIVE_SETTINGS.items():

            print(f"\n------ {stage_name} ------")

            seed_metrics = []

            for seed in seeds:

                print(f"Seed {seed}")

                config = copy.deepcopy(BASE_CONFIG)
                config.update(override)

                metrics = run_single_config(
                    data_root,
                    dataset_name,
                    config,
                    device,
                    seed
                )

                seed_metrics.append(metrics)

            df_seed = pd.DataFrame(seed_metrics)

            mean_vals = df_seed.mean()
            std_vals = df_seed.std()

            final_results[stage_name] = {
                "SP_mean": mean_vals["spearman"],
                "SP_std": std_vals["spearman"],
                "KD_mean": mean_vals["kendall"],
                "KD_std": std_vals["kendall"],
            }

        df_final = pd.DataFrame(final_results).T

        print("\nProgressive Ablation Result:")
        print(df_final)

        df_final.to_csv(
            os.path.join(data_root, "progressive_ablation_multi_seed.csv")
        )

        # =====================================================
        # Modality Ablation
        # =====================================================

        modal_results = {}

        for modal_name, override in MODAL_ABLATION.items():

            print(f"\n------ {modal_name} ------")

            seed_metrics = []

            for seed in seeds:

                print(f"Seed {seed}")

                config = copy.deepcopy(BASE_CONFIG)
                config.update(override)

                metrics = run_single_config(
                    data_root,
                    dataset_name,
                    config,
                    device,
                    seed
                )

                seed_metrics.append(metrics)

            df_seed = pd.DataFrame(seed_metrics)

            mean_vals = df_seed.mean()
            std_vals = df_seed.std()

            modal_results[modal_name] = {
                "SP_mean": mean_vals["spearman"],
                "SP_std": std_vals["spearman"],
                "KD_mean": mean_vals["kendall"],
                "KD_std": std_vals["kendall"],
            }

        df_modal = pd.DataFrame(modal_results).T

        print("\nModal Ablation Result:")
        print(df_modal)

        df_modal.to_csv(
            os.path.join(data_root, "modal_ablation_multi_seed.csv")
        )

    print("\nAll datasets finished.")