from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import yaml


PROJECT_ROOT = Path(r"D:\project\keynode\project")
OUTPUT_CONFIG_DIR = PROJECT_ROOT / "path" / "configs"


def norm(p: Path) -> str:
    return str(p).replace("\\", "/")


# -------------------------------------------------------------------
# 数据集注册表
# 说明：
# - name: 传给 GraphPreprocessor.build_graph_bundle(...) 的真实数据集名
# - root: 对应 PyG 数据族目录
# - importance_path / node_features_path: 你自己的对齐输入
# - strict_importance_alignment: Citeseer=False，其余目前按 True
# -------------------------------------------------------------------
DATASET_INFO = {
    "Cora": {
        "family": "Planetoid",
        "name": "Cora",
        "root": norm(PROJECT_ROOT / "public_datasets" / "Planetoid"),
        "importance_path": norm(PROJECT_ROOT / "public_process" / "extract" / "Cora_struct" / "gnn_node_scores_mean.csv"),
        "node_features_path": norm(PROJECT_ROOT / "public_process" / "extract" / "Cora_struct" / "node_features.csv"),
        "old_id_col_in_node_features": "node",
        "strict_importance_alignment": True,
        "tag": "cora",
    },
    "Citeseer": {
        "family": "Planetoid",
        "name": "Citeseer",
        "root": norm(PROJECT_ROOT / "public_datasets" / "Planetoid"),
        "importance_path": norm(PROJECT_ROOT / "public_process" / "extract" / "CiteSeer_struct" / "gnn_node_scores_mean.csv"),
        "node_features_path": norm(PROJECT_ROOT / "public_process" / "extract" / "CiteSeer_struct" / "node_features.csv"),
        "old_id_col_in_node_features": "node",
        "strict_importance_alignment": False,
        "tag": "citeseer",
    },
    "Pubmed": {
        "family": "Planetoid",
        "name": "Pubmed",
        "root": norm(PROJECT_ROOT / "public_datasets" / "Planetoid"),
        "importance_path": norm(PROJECT_ROOT / "public_process" / "extract" / "PubMed_struct" / "gnn_node_scores_mean.csv"),
        "node_features_path": norm(PROJECT_ROOT / "public_process" / "extract" / "PubMed_struct" / "node_features.csv"),
        "old_id_col_in_node_features": "node",
        "strict_importance_alignment": True,
        "tag": "pubmed",
    },
    "Computers": {
        "family": "Amazon",
        "name": "Computers",
        "root": norm(PROJECT_ROOT / "public_datasets" / "Amazon"),
        "importance_path": norm(PROJECT_ROOT / "public_process" / "extract" / "Computers_struct" / "gnn_node_scores_mean.csv"),
        "node_features_path": norm(PROJECT_ROOT / "public_process" / "extract" / "Computers_struct" / "node_features.csv"),
        "old_id_col_in_node_features": "node",
        "strict_importance_alignment": True,
        "tag": "computers",
    },
    "Photo": {
        "family": "Amazon",
        "name": "Photo",
        "root": norm(PROJECT_ROOT / "public_datasets" / "Amazon"),
        "importance_path": norm(PROJECT_ROOT / "public_process" / "extract" / "Photo_struct" / "gnn_node_scores_mean.csv"),
        "node_features_path": norm(PROJECT_ROOT / "public_process" / "extract" / "Photo_struct" / "node_features.csv"),
        "old_id_col_in_node_features": "node",
        "strict_importance_alignment": True,
        "tag": "photo",
    },
    "CS": {
        "family": "Coauthor",
        "name": "CS",
        "root": norm(PROJECT_ROOT / "public_datasets" / "Coauthor"),
        "importance_path": norm(PROJECT_ROOT / "public_process" / "extract" / "CS_struct" / "gnn_node_scores_mean.csv"),
        "node_features_path": norm(PROJECT_ROOT / "public_process" / "extract" / "CS_struct" / "node_features.csv"),
        "old_id_col_in_node_features": "node",
        "strict_importance_alignment": True,
        "tag": "cs",
    },
    "Physics": {
        "family": "Coauthor",
        "name": "Physics",
        "root": norm(PROJECT_ROOT / "public_datasets" / "Coauthor"),
        "importance_path": norm(PROJECT_ROOT / "public_process" / "extract" / "Physics_struct" / "gnn_node_scores_mean.csv"),
        "node_features_path": norm(PROJECT_ROOT / "public_process" / "extract" / "Physics_struct" / "node_features.csv"),
        "old_id_col_in_node_features": "node",
        "strict_importance_alignment": True,
        "tag": "physics",
    },
}


# -------------------------------------------------------------------
# 共享配置模板
# -------------------------------------------------------------------
COMMON_KEYNODE = {
    "mode": "topk",
    "k": 30,
    "min_shortest_len": 2,
    "max_pairs": 200,
}

COMMON_PATHS = {
    "k_shortest": 3,
    "max_hops": 8,
    "delta": 2,
    "top_q": 10,
    "overlap_threshold": 0.6,
    "top_m_for_fragility": 1,
}

COMMON_FRAGILITY = {
    "lambda_E": 0.4,
    "lambda_LCC": 0.4,
    "lambda_ASP": 0.2,
}

COMMON_BASELINES = {
    "k_candidates": 5,
    "random_num_trials": 30,
    "random_num_samples": 5,
    "random_seed": 42,
    "use_internal_node_importance": False,
}

COMMON_RULE_WEIGHTS = {
    "avg_node_importance": 0.12,
    "avg_edge_bc": 0.10,
    "cross_comm_ratio": 0.03,
    "fragility_score": 0.65,
    "path_length": 0.10,
}
COMMON_SCORER = {
    "fragility_gate": 0.50,
    "gate_penalty": 0.08,
}
COMMON_TENSORBOARD = {
    "enabled": True,
    "log_root": "outputs/tb",
}

COMMON_RL_REWARD = {
    "lambda_E": 0.4,
    "lambda_LCC": 0.4,
    "lambda_ASP": 0.2,
    "node_bonus_weight": 0.10,
    "bridge_bonus_weight": 0.10,
    "cross_comm_bonus_weight": 0.05,
    "step_cost": 0.05,
    "reach_bonus": 1.0,
    "fail_penalty": 1.0,
    "repeat_penalty": 0.3,
    "hybrid_alpha": 0.7,
    "normalize_fragility": True,
    "normalize_surrogate": True,
    "surrogate_weights": {
        "avg_node_importance": 0.35,
        "avg_edge_bc": 0.25,
        "cross_comm_ratio": 0.15,
        "path_length": 0.25,
    },
}


def build_dataset_block(info: dict) -> dict:
    return {
        "family": info["family"],
        "name": info["name"],
        "root": info["root"],
        "importance_path": info["importance_path"],
        "node_features_path": info["node_features_path"],
        "old_id_col_in_node_features": info.get("old_id_col_in_node_features", "node"),
        "community_mode": "louvain",
        "strict_importance_alignment": info["strict_importance_alignment"],
        "importance_fill_value": 0.0,
        "verbose": True,
    }


def build_rule_yaml(info: dict) -> dict:
    tag = info["tag"]
    return {
        "dataset": build_dataset_block(info),
        "keynode": deepcopy(COMMON_KEYNODE),
        "paths": deepcopy(COMMON_PATHS),
        "rule_weights": deepcopy(COMMON_RULE_WEIGHTS),
        "scorer": deepcopy(COMMON_SCORER),
        "fragility": deepcopy(COMMON_FRAGILITY),
        "baselines": deepcopy(COMMON_BASELINES),
        "tensorboard": {
            **deepcopy(COMMON_TENSORBOARD),
            "run_name": f"{tag}_rule",
        },
        "output": {
            "k_list": [1, 3, 5, 10],
            "top_n_summary": 10,
            "metrics_json": f"outputs/metrics/{tag}_rule_metrics.json",
            "paths_json": f"outputs/paths/{tag}_rule_paths.json",
        },
    }


def build_rank_yaml(info: dict) -> dict:
    tag = info["tag"]
    return {
        "dataset": build_dataset_block(info),
        "keynode": deepcopy(COMMON_KEYNODE),
        "paths": deepcopy(COMMON_PATHS),
        "fragility": deepcopy(COMMON_FRAGILITY),
        "ranking": {
            "train_ratio": 0.8,
            "top_per_task": 1,
            "random_state": 42,

            # ===== 新增：dataset 构造阶段的 fragility 加速配置 =====
            "fragility_mode": "hybrid",          # exact | cached | approx | hybrid
            "exact_every_n_tasks": 20,
            "exact_top_ranks": 1,
            "exact_max_path_len": 4,
            "progress_every": 10,

            # ===== 新增：compare_methods / evaluate_topk_damage 配置 =====
            "eval_mode": "hybrid",               # exact | approx | hybrid
            "eval_early_stop": True,
            "eval_tol": 1e-4,
            "eval_debug": True,
            "max_shared_internal_nodes": 1,
            "alpha_pred": 0.35,
            "alpha_gain": 0.65,


            "feature_cols": [
                "shortest_len",
                "same_community",
                "pair_score",
                "candidate_rank",
                "path_length_int",
                "avg_node_importance",
                "internal_node_importance",
                "avg_edge_bc",
                "cross_comm_ratio",
                "path_length",
                "num_edges",
            ],
            "xgb_params": {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "random_state": 42,
                "objective": "reg:squarederror",
            },
        },
        "tensorboard": {
            **deepcopy(COMMON_TENSORBOARD),
            "run_name": f"{tag}_rank",
        },

        # ===== 可选：统一 debug 开关 =====
        "debug": {
            "enabled": True,
        },
        "scorer": deepcopy(COMMON_SCORER),

        "output": {
            "k_list": [1, 3, 5, 10],
            "top_n_summary": 10,
            "metrics_json": f"outputs/metrics/{tag}_rank_metrics.json",
            "paths_json": f"outputs/paths/{tag}_rank_paths.json",
            "dataset_csv": f"outputs/metrics/{tag}_rank_dataset.csv",
            "scored_test_csv": f"outputs/metrics/{tag}_rank_scored_test.csv",

            # ===== 新增：fragility 缓存文件 =====
            "fragility_cache_json": f"path/outputs/cache/{tag}_rank_fragility_cache.json",
        },
    }

def build_rl_yaml(info: dict) -> dict:
    tag = info["tag"]
    return {
        "seed": 42,
        "dataset": build_dataset_block(info),
        "keynode": deepcopy(COMMON_KEYNODE),
        "paths": deepcopy(COMMON_PATHS),
        "fragility": deepcopy(COMMON_FRAGILITY),
        "baselines": deepcopy(COMMON_BASELINES),
        "rl": {
            # 推荐默认主配置用 hybrid
            "reward_mode": "hybrid",
            "eval_reward_mode": "fragility",

            "embedding_dim": 64,
            "gnn_hidden_dim": 128,
            "gnn_epochs": 120,
            "gnn_lr": 1e-2,

            "max_hops": 8,
            "hidden_dim": 128,
            "lr": 1e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ppo_epochs": 4,
            "epochs": 80,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "importance_bias": 0.0,

            "num_samples_per_task": 10,

            "reward": deepcopy(COMMON_RL_REWARD),
        },
        "tensorboard": {
            **deepcopy(COMMON_TENSORBOARD),
            "run_name": f"{tag}_rl_hybrid",
        },
        "scorer": deepcopy(COMMON_SCORER),
        "output": {
            "embedding_ckpt": f"outputs/checkpoints/{tag}_node_embeddings.pt",
            "rl_ckpt": f"outputs/checkpoints/{tag}_ppo_agent.pt",
            "train_log_json": f"outputs/logs/{tag}_rl_train_log.json",
            "train_summary_json": f"outputs/logs/{tag}_rl_train_summary.json",
            "rl_eval_metrics_json": f"outputs/metrics/{tag}_rl_eval_metrics.json",
            "rl_eval_paths_json": f"outputs/paths/{tag}_rl_eval_paths.json",
            "k_list": [1, 3, 5, 10],
        },
    }


def build_rl_surrogate_yaml(info: dict) -> dict:
    tag = info["tag"]
    cfg = build_rl_yaml(info)
    cfg["rl"]["reward_mode"] = "surrogate"
    cfg["rl"]["epochs"] = 60
    cfg["rl"]["reward"]["hybrid_alpha"] = 1.0
    cfg["tensorboard"]["run_name"] = f"{tag}_rl_surrogate"
    cfg["output"]["rl_ckpt"] = f"outputs/checkpoints/{tag}_ppo_agent_surrogate.pt"
    cfg["output"]["train_log_json"] = f"outputs/logs/{tag}_rl_train_log_surrogate.json"
    cfg["output"]["train_summary_json"] = f"outputs/logs/{tag}_rl_train_summary_surrogate.json"
    cfg["output"]["rl_eval_metrics_json"] = f"outputs/metrics/{tag}_rl_eval_metrics_surrogate.json"
    cfg["output"]["rl_eval_paths_json"] = f"outputs/paths/{tag}_rl_eval_paths_surrogate.json"
    return cfg


def build_rl_fragility_yaml(info: dict) -> dict:
    tag = info["tag"]
    cfg = build_rl_yaml(info)
    cfg["rl"]["reward_mode"] = "fragility"
    cfg["rl"]["eval_reward_mode"] = "fragility"
    cfg["rl"]["epochs"] = 100
    cfg["rl"]["reward"]["hybrid_alpha"] = 0.0
    cfg["tensorboard"]["run_name"] = f"{tag}_rl_fragility"
    cfg["output"]["rl_ckpt"] = f"outputs/checkpoints/{tag}_ppo_agent_fragility.pt"
    cfg["output"]["train_log_json"] = f"outputs/logs/{tag}_rl_train_log_fragility.json"
    cfg["output"]["train_summary_json"] = f"outputs/logs/{tag}_rl_train_summary_fragility.json"
    cfg["output"]["rl_eval_metrics_json"] = f"outputs/metrics/{tag}_rl_eval_metrics_fragility.json"
    cfg["output"]["rl_eval_paths_json"] = f"outputs/paths/{tag}_rl_eval_paths_fragility.json"
    return cfg


def dump_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    OUTPUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_key, info in DATASET_INFO.items():
        tag = info["tag"]

        rule_yaml = build_rule_yaml(info)
        rank_yaml = build_rank_yaml(info)
        rl_yaml = build_rl_yaml(info)
        rl_surrogate_yaml = build_rl_surrogate_yaml(info)
        rl_fragility_yaml = build_rl_fragility_yaml(info)

        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rule.yaml", rule_yaml)
        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rank.yaml", rank_yaml)
        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rl.yaml", rl_yaml)
        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rl_surrogate.yaml", rl_surrogate_yaml)
        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rl_fragility.yaml", rl_fragility_yaml)

        print(f"[OK] generated configs for {dataset_key}")

    print(f"[DONE] all configs saved to: {OUTPUT_CONFIG_DIR}")


if __name__ == "__main__":
    main()