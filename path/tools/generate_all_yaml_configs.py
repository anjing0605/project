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
    "task_sampling_mode": "hybrid",
    "random_task_ratio": 0.10,
    "random_seed": 42,
}

COMMON_PATHS = {
    "k_shortest": 20,
    "final_k": 15,
    "raw_k_multiplier": 6,
    "raw_k_min_extra": 30,
    "max_hops": 8,
    "delta": 4,
    "max_internal_overlap": 0.90,
    "fallback_relax_overlap": 0.98,
    "fallback_extra_hops": 3,
    "top_q": 10,
    "overlap_threshold": 0.6,
    "top_m_for_fragility": 5,
}

COMMON_FRAGILITY = {
    # rule / rank / RL 统一使用同一套结构破坏权重，避免评价目标不一致
    "lambda_E": 0.55,
    "lambda_LCC": 0.0,
    "lambda_ASP": 0.45,
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
    # rule / rank / RL 统一使用同一套结构破坏权重
    "lambda_E": 0.55,
    "lambda_LCC": 0.0,
    "lambda_ASP": 0.45,

    # Stage 1：只学可达性
    "stage1_distance_weight": 1.00,
    "stage1_reach_bonus": 4.00,

    # Stage 2 / Stage C：保留可达性，但优化中心改成集合边际收益
    "stage2_distance_weight": 0.35,
    "stage2_reach_bonus": 1.20,

    # 稠密结构过程奖励
    "importance_weight": 0.03,
    "edge_bc_weight": 0.18,
    "cross_comm_weight": 0.12,

    # 兼容旧字段名
    "node_bonus_weight": 0.03,
    "bridge_bonus_weight": 0.18,
    "cross_comm_bonus_weight": 0.12,

    # 通用惩罚：提高 step/repeat/fail，压制 RL 过长路径
    "step_cost": 0.05,
    "fail_penalty": 1.20,
    "repeat_penalty": 0.60,

    # TopKAlign：从“单路径高分”改成“集合边际增益 + 单位新增节点效率”
    "single_frag_weight": 0.25,
    "marginal_gain_weight": 5.00,
    "budget_eff_weight": 3.00,
    "node_cost_weight": 0.02,

    # 重叠 / 负边际惩罚显著加强，目标是降低 node overlap
    "overlap_penalty_weight": 1.00,
    "node_overlap_penalty_weight": 2.00,
    "negative_gain_penalty_weight": 1.00,
    "low_new_node_penalty_weight": 0.50,

    "min_new_internal_nodes": 1,
    "gain_temp": 0.10,

    # top-k 集合对齐参数
    "top_k": 10,
    "overlap_threshold": 0.35,
    "min_new_internal_nodes_for_commit": 1,
    "hard_node_overlap_threshold": 0.35,
    "hard_edge_overlap_threshold": 0.50,

    # Stage C 训练阶段直接避开已选 internal nodes
    "avoid_selected_internal_nodes": True,

    # 近似评估参数
    "efficiency_num_pairs": 2000,
    "asp_num_sources": 64,
    "random_seed": 42,

    # 关闭 shortcut，避免 reward 目标偏离真实 fragility
    "use_bridge_shortcut": False,

    # 旧字段保留
    "hybrid_alpha": 0.0,
    "normalize_fragility": True,
    "normalize_surrogate": True,
    "surrogate_weights": {
        "avg_node_importance": 0.35,
        "avg_edge_bc": 0.25,
        "cross_comm_ratio": 0.15,
        "path_length": 0.25,
    },

    # 不再鼓励明显绕路；当前 RL 已经偏长
    "stretch_bonus_weight": 0.00,
    "stretch_min": 1.05,
    "stretch_max": 1.40,
    "stretch_hard_max": 1.70,
    "stretch_over_penalty_weight": 1.00,

    # 训练阶段 selection score 用；新增节点只作极弱辅助，不再鼓励长路径
    "selection_single_path_weight": 0.10,
    "new_node_bonus_weight": 0.02,
    "new_node_bonus_cap": 3,

    # RL 推理后处理：禁止负边际补齐，强调 fixed-node-budget 效率
    "inference_set_gain_weight": 1.00,
    "inference_budget_eff_weight": 1.50,
    "inference_node_cost_weight": 0.02,
    "inference_edge_overlap_penalty_weight": 1.00,
    "inference_node_overlap_penalty_weight": 2.00,
    "inference_single_path_weight": 0.05,
    "inference_hard_edge_overlap": True,
    "inference_min_marginal_gain": 0.0,
    "inference_min_selection_score": 0.0,
    "inference_fill_to_top_q": False,
    "inference_relaxed_max_node_overlap": 0.35,
    "inference_relaxed_edge_overlap": 0.50,
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
            "node_budget_list": [5, 10, 20, 30, 50],
            "mechanism_topk": 10,
            "top_n_summary": 10,
            "metrics_json": f"outputs/metrics/{tag}_rule_metrics.json",
            "paths_json": f"outputs/paths/{tag}_rule_paths.json",
        },

    }


from copy import deepcopy

def build_rank_yaml(info: dict) -> dict:
    tag = info["tag"]

    # ===== 支持从 info 控制 rank 模式 =====
    # pure_rank:
    #   学单路径 fragility，只做普通排序 baseline
    # rank_set + pred_score_selector:
    #   学集合边际收益，用预测分 + 重叠惩罚选路径
    # rank_set + submodular_greedy:
    #   学集合边际收益，用真实边际结构破坏贪心选路径
    rank_mode = info.get("rank_mode", "pure_rank")   # pure_rank | rank_set
    selector = info.get("selector", "pred_score_selector")  # pred_score_selector | submodular_greedy

    if rank_mode == "pure_rank":
        label_mode = "single"
        suffix = "pure_rank"
    elif rank_mode == "rank_set":
        label_mode = "marginal"
        suffix = f"{rank_mode}_{selector}"
    else:
        raise ValueError(f"Unsupported rank_mode: {rank_mode}")

    # ===== 三种模式使用不同 selector 参数 =====
    # 注意：
    # pure_rank 不做集合选择，只作为单路径排序 baseline
    # pred_score_selector 是当前最稳的 rank-set 版本
    # submodular_greedy 是理论最优版本，但必须配合 set_scorer.py 的 fallback 修复
    if rank_mode == "pure_rank":
        selector_params = {
            "lambda_red": 0.0,
            "allow_negative_gain": -0.02,
            "max_shared_internal_nodes": 8,
            "min_marginal_gain": 1.0e-8,
            "pure_global_sort": False,
        }
    elif selector == "pred_score_selector":
        selector_params = {
            "lambda_red": 0.10,
            "allow_negative_gain": -0.02,
            "max_shared_internal_nodes": 5,
            "min_marginal_gain": 1.0e-8,
            "per_task_set_top_q": 3,
            "pure_global_sort": False,
        }
    elif selector == "submodular_greedy":
        selector_params = {
            # submodular 本身已经计算真实边际增益，
            # lambda_red 不宜太大，否则容易提前停止。
            "lambda_red": 0.05,
            "allow_negative_gain": -0.02,
            "max_shared_internal_nodes": 8,
            "min_marginal_gain": 1.0e-8,
            "per_task_set_top_q": 3,
            "pure_global_sort": False,
        }
    else:
        raise ValueError(f"Unsupported selector: {selector}")

    cfg = {
        "dataset": build_dataset_block(info),
        "keynode": deepcopy(COMMON_KEYNODE),

        # ===== rank 阶段候选路径配置 =====
        # 目标：给 rank_set 足够大的候选池，否则 submodular_greedy 后期无路可选。
        "paths": {
            **deepcopy(COMMON_PATHS),

            "k_shortest": 20,
            "final_k": 15,
            "max_hops": 10,
            "delta": 4,

            # 扩大 raw candidate pool
            "raw_k_multiplier": 6,
            "raw_k_min_extra": 30,

            # 候选路径生成阶段不要过早过滤，否则 rank_set 没有足够多样路径
            "max_internal_overlap": 0.90,
            "fallback_relax_overlap": 0.98,
            "fallback_extra_hops": 3,

            # 供路径集合去冗余使用
            "top_q": 10,
            "overlap_threshold": 0.80,

            # 只对候选路径前若干条精算 fragility，控制构造数据集成本
            "top_m_for_fragility": 5,
        },

        "fragility": deepcopy(COMMON_FRAGILITY),

        "ranking": {
            # ===== 三种 rank 模式核心字段 =====
            "mode": rank_mode,
            "selector": selector,
            "label_mode": label_mode,

            # ===== 输出规模 =====
            "global_top_q": 100,
            "top_per_task": 10,
            "train_ratio": 0.8,
            "random_state": 42,

            # ===== dataset 构造阶段 fragility 配置 =====
            # 构造训练集时可以用 hybrid 加速。
            # 最终评估必须用 exact，否则论文表格不够干净。
            "fragility_mode": "hybrid",
            "exact_every_n_tasks": 10,
            "exact_top_ranks": 3,
            "exact_max_path_len": 5,
            "progress_every": 20,

            # ===== 最终 top-k damage 评估配置 =====
            # 这里建议统一 exact，避免 selector 内部和 evaluator 指标不一致。
            "eval_mode": "exact",
            "eval_early_stop": False,
            "eval_tol": 1.0e-4,
            "eval_debug": False,

            # ===== selector 参数 =====
            **selector_params,

            # ===== 如果旧代码仍读取这些字段，保留，避免 KeyError =====
            "alpha_pred": 1.0,
            "alpha_gain": 0.0,
            "normalize_pred_score": False,
            "normalize_marginal_gain": False,

            # ===== 特征列 =====
            "feature_cols": [
                "shortest_len",
                "same_community",
                "pair_score",
                "path_length_int",
                "avg_node_importance",
                "internal_node_importance",
                "avg_edge_bc",
                "cross_comm_ratio",
                "path_length",
            ],

            # ===== XGBoost LambdaMART / Ranker 参数 =====
            "xgb_params": {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "objective": "rank:ndcg",
                "eval_metric": "ndcg@10",
            },
        },

        "tensorboard": {
            **deepcopy(COMMON_TENSORBOARD),
            "run_name": f"{tag}_rank_{suffix}",
        },

        "debug": {
            "enabled": False,
        },

        "scorer": deepcopy(COMMON_SCORER),

        "output": {
            "k_list": [1, 3, 5, 10],
            "top_n_summary": 10,

            # 输出文件名带 suffix，避免三种 rank 模式互相覆盖
            "metrics_json": f"outputs/metrics/{tag}_rank_{suffix}_metrics.json",
            "paths_json": f"outputs/paths/{tag}_rank_{suffix}_paths.json",
            "dataset_csv": f"outputs/metrics/{tag}_rank_{suffix}_dataset.csv",
            "scored_test_csv": f"outputs/metrics/{tag}_rank_{suffix}_scored_test.csv",

            "fragility_cache_json": f"outputs/cache/{tag}_rank_{suffix}_fragility_cache.json",
        },
    }

    return cfg
def build_rl_yaml(info: dict) -> dict:
    tag = info["tag"]
    return {
        "seed": 42,
        "dataset": build_dataset_block(info),
        "keynode": deepcopy(COMMON_KEYNODE),
        "paths": {
            **deepcopy(COMMON_PATHS),

            # RL 后处理必须比原来更严格，否则 fixed-node-budget 下会输给 rule
            "top_q": 10,
            "overlap_threshold": 0.35,
            "min_new_internal_nodes": 1,
            "max_node_overlap": 0.35,
        },

        # RL 阶段使用与 reward/evaluator 一致的 fragility 权重；
        # 不再沿用 rule/rank 的 COMMON_FRAGILITY，避免 build_train_config_from_yaml 覆盖 reward 中的 lambda。
        "fragility": deepcopy(COMMON_FRAGILITY),
        "baselines": deepcopy(COMMON_BASELINES),
        "rl": {
            # 最终 top-k 对齐版本
            "reward_mode": "fragility_topk_align",
            "eval_reward_mode": "fragility_finetune",
            # 先运行 *_rl_fragility.yaml，再运行 *_rl.yaml；
            # TopKAlign 从 SingleFrag checkpoint 继续微调
            "stage2_init_ckpt": f"outputs/checkpoints/{tag}_ppo_agent_fragility.pt",

            "embedding_dim": 64,
            "gnn_hidden_dim": 128,
            "gnn_epochs": 120,
            "gnn_lr": 1e-2,

            "max_hops": 7,
            "hidden_dim": 128,
            "lr": 1e-4,
            "policy_lr": 1e-4,
            "value_lr": 1e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ppo_epochs": 4,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "importance_bias": 0.0,

            "stage1_max_epochs": 120,
            "stage1_eval_every": 5,
            "stage1_gate_threshold": 0.20,
            "stage1_target_threshold": 0.30,
            "stage1_val_ratio": 0.20,
            "stage1_gate_patience": 3,

            "stage1_len2_pass_threshold": 0.40,
            "stage1_len23_mix_threshold": 0.30,
            "stage1_len3_pass_threshold": 0.20,
            "stage2_epochs": 50,
            "epochs": 100,
            "num_samples_per_epoch": 120,

            "stage2_min_arrival_for_ckpt": 0.20,
            "num_samples_per_task": 120,
            "deterministic_eval": False,
            "eval_action_temperature": 1.3,


            "reward": deepcopy(COMMON_RL_REWARD),
        },


        "tensorboard": {
            **deepcopy(COMMON_TENSORBOARD),
            "run_name": f"{tag}_rl_topk_align",
        },

        "scorer": deepcopy(COMMON_SCORER),

        "output": {
            "embedding_ckpt": f"outputs/checkpoints/{tag}_node_embeddings.pt",

            # Stage 1 checkpoint
            "rl_ckpt_stage1": f"outputs/checkpoints/{tag}_ppo_agent_stage1.pt",

            # Stage 2 / final checkpoint
            "rl_ckpt": f"outputs/checkpoints/{tag}_ppo_agent.pt",

            "train_log_json": f"outputs/logs/{tag}_rl_train_log.json",
            "train_summary_json": f"outputs/logs/{tag}_rl_train_summary.json",

            "rl_eval_metrics_json": f"outputs/metrics/{tag}_rl_eval_metrics.json",
            "rl_eval_paths_json": f"outputs/paths/{tag}_rl_eval_paths.json",

            "k_list": [1, 3, 5, 10],
            "eval_mode": "exact",
            "fixed_node_budget_list": [5, 10, 20, 30, 50],
        },
    }

def build_rl_surrogate_yaml(info: dict) -> dict:
    tag = info["tag"]
    cfg = build_rl_yaml(info)

    cfg["rl"]["reward_mode"] = "fragility_finetune"
    cfg["rl"]["eval_reward_mode"] = "fragility_finetune"
    cfg["rl"].pop("stage2_init_ckpt", None)

    cfg["rl"]["stage2_epochs"] = 20
    cfg["rl"]["epochs"] = 60
    cfg["rl"]["stage2_min_arrival_for_ckpt"] = 0.25

    r = cfg["rl"]["reward"]

    # 可达性保留得更强
    r["stage2_distance_weight"] = 0.35
    r["stage2_reach_bonus"] = 1.20

    # 弱结构奖励
    r["single_frag_weight"] = 0.30
    r["marginal_gain_weight"] = 0.00
    r["overlap_penalty_weight"] = 0.00
    r["node_overlap_penalty_weight"] = 0.00
    r["negative_gain_penalty_weight"] = 0.00
    r["low_new_node_penalty_weight"] = 0.00

    # 稠密结构引导
    r["importance_weight"] = 0.03
    r["edge_bc_weight"] = 0.15
    r["cross_comm_weight"] = 0.10
    r["node_bonus_weight"] = 0.03
    r["bridge_bonus_weight"] = 0.15
    r["cross_comm_bonus_weight"] = 0.10

    # 不做硬动作屏蔽
    r["avoid_selected_internal_nodes"] = False

    # 轻微鼓励非最短但不过长路径
    r["stretch_bonus_weight"] = 0.15
    r["stretch_min"] = 1.05
    r["stretch_max"] = 1.50
    r["stretch_hard_max"] = 2.00
    r["stretch_over_penalty_weight"] = 0.50
    # surrogate 只作为弱奖励消融，推理仍按统一 budget-aware 选择器
    cfg["rl"]["num_samples_per_task"] = 120
    cfg["rl"]["eval_action_temperature"] = 1.3

    r["inference_edge_overlap_penalty_weight"] = 1.00
    r["inference_node_overlap_penalty_weight"] = 2.00
    r["inference_single_path_weight"] = 0.05
    r["inference_set_gain_weight"] = 1.00
    r["inference_budget_eff_weight"] = 1.50
    r["inference_node_cost_weight"] = 0.02
    r["inference_hard_edge_overlap"] = True
    r["inference_min_marginal_gain"] = 0.0
    r["inference_min_selection_score"] = 0.0
    r["inference_fill_to_top_q"] = False
    r["inference_relaxed_max_node_overlap"] = 0.35
    r["inference_relaxed_edge_overlap"] = 0.50
    cfg["tensorboard"]["run_name"] = f"{tag}_rl_surrogate_like"

    cfg["output"]["rl_ckpt_stage1"] = f"outputs/checkpoints/{tag}_ppo_agent_surrogate_stage1.pt"
    cfg["output"]["rl_ckpt"] = f"outputs/checkpoints/{tag}_ppo_agent_surrogate.pt"
    cfg["output"]["train_log_json"] = f"outputs/logs/{tag}_rl_train_log_surrogate.json"
    cfg["output"]["train_summary_json"] = f"outputs/logs/{tag}_rl_train_summary_surrogate.json"
    cfg["output"]["rl_eval_metrics_json"] = f"outputs/metrics/{tag}_rl_eval_metrics_surrogate.json"
    cfg["output"]["rl_eval_paths_json"] = f"outputs/paths/{tag}_rl_eval_paths_surrogate.json"

    return cfg

def build_rl_fragility_yaml(info: dict) -> dict:
    tag = info["tag"]
    cfg = build_rl_yaml(info)

    cfg["rl"]["reward_mode"] = "fragility_finetune"
    cfg["rl"]["eval_reward_mode"] = "fragility_finetune"
    cfg["rl"].pop("stage2_init_ckpt", None)

    cfg["rl"]["stage2_epochs"] = 40
    cfg["rl"]["epochs"] = 100
    cfg["rl"]["stage2_min_arrival_for_ckpt"] = 0.25

    r = cfg["rl"]["reward"]

    # 保留较强可达性
    r["stage2_distance_weight"] = 0.35
    r["stage2_reach_bonus"] = 1.50

    # 单路径 fragility 微调
    r["single_frag_weight"] = 0.80
    r["marginal_gain_weight"] = 0.00
    r["overlap_penalty_weight"] = 0.00
    r["node_overlap_penalty_weight"] = 0.00
    r["negative_gain_penalty_weight"] = 0.00
    r["low_new_node_penalty_weight"] = 0.00

    # 强化桥接结构偏好
    r["importance_weight"] = 0.03
    r["edge_bc_weight"] = 0.18
    r["cross_comm_weight"] = 0.12
    r["node_bonus_weight"] = 0.03
    r["bridge_bonus_weight"] = 0.18
    r["cross_comm_bonus_weight"] = 0.12

    # 训练阶段不做 top-k hard mask
    r["avoid_selected_internal_nodes"] = False

    # 控制式绕行奖励
    r["stretch_bonus_weight"] = 0.20
    r["stretch_min"] = 1.05
    r["stretch_max"] = 1.60
    r["stretch_hard_max"] = 2.00
    r["stretch_over_penalty_weight"] = 0.50
    # SingleFrag 是 TopKAlign 的初始化来源，推理也使用统一 budget-aware 选择器
    cfg["rl"]["num_samples_per_task"] = 120
    cfg["rl"]["eval_action_temperature"] = 1.3

    r["inference_edge_overlap_penalty_weight"] = 1.00
    r["inference_node_overlap_penalty_weight"] = 2.00
    r["inference_single_path_weight"] = 0.05
    r["inference_set_gain_weight"] = 1.00
    r["inference_budget_eff_weight"] = 1.50
    r["inference_node_cost_weight"] = 0.02
    r["inference_hard_edge_overlap"] = True
    r["inference_min_marginal_gain"] = 0.0
    r["inference_min_selection_score"] = 0.0
    r["inference_fill_to_top_q"] = False
    r["inference_relaxed_max_node_overlap"] = 0.35
    r["inference_relaxed_edge_overlap"] = 0.50
    cfg["tensorboard"]["run_name"] = f"{tag}_rl_fragility"

    cfg["output"]["rl_ckpt_stage1"] = f"outputs/checkpoints/{tag}_ppo_agent_fragility_stage1.pt"
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
        dump_yaml(OUTPUT_CONFIG_DIR/ f"{tag}_rank_pure.yaml", build_rank_yaml({**info, "rank_mode": "pure_rank"}))
        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rank_set_pred.yaml", build_rank_yaml({**info, "rank_mode": "rank_set", "selector": "pred_score_selector"}))
        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rank_set_submod.yaml",build_rank_yaml({**info, "rank_mode": "rank_set", "selector": "submodular_greedy"}))
        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rl.yaml", rl_yaml)
        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rl_surrogate.yaml", rl_surrogate_yaml)
        dump_yaml(OUTPUT_CONFIG_DIR / f"{tag}_rl_fragility.yaml", rl_fragility_yaml)

        print(f"[OK] generated configs for {dataset_key}")

    print(f"[DONE] all configs saved to: {OUTPUT_CONFIG_DIR}")


if __name__ == "__main__":
    main()