#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic Ablation Experiment

Experiments:
1. Struct only
2. Struct + Raw semantic
3. Struct + TFIDF
4. Struct + TFIDF + SVD
5. Struct + TFIDF + SVD + Bridge (multi semantic dims)
"""

import os
import json
import time
import torch
import random
import numpy as np
import copy
import pandas as pd
from train import SlimRunDataset
from train import run_training
from train import config as train_config


############################################################
# 基本路径
############################################################

ROOT = "scratch/project"

DATA_ROOT = os.path.join(ROOT, "public_process", "extract")

RESULT_ROOT = os.path.join(
    ROOT,
    "public_process",
    "ablation_semantic",
    "ablation_results"
)

SEEDS = [0, 1, 2, 3, 4]


############################################################
# 数据集
############################################################

DATASETS = [
    "Cora",
    "CiteSeer",
    "PubMed",
    "Computers",
    "Photo",
    "CS",
    "Physics"
]


############################################################
# bridge 维度对比
############################################################

BRIDGE_DIMS = [16, 32, 64, 128, 256]


############################################################
# 语义消融设置
############################################################

SEMANTIC_VARIANTS = {
    "struct_only": None,
    "raw": "semantic_raw.pt",
    "tfidf": "semantic_tfidf.pt",
    "tfidf_svd": "semantic_tfidf_svd.pt",
}

# 动态加入 tfidf_svd_bridge 的多个维度实验
for dim in BRIDGE_DIMS:
    SEMANTIC_VARIANTS[f"tfidf_svd_bridge_dim{dim}"] = (
        f"semantic_tfidf_svd_bridge_dim{dim}.pt"
    )


############################################################
# 随机种子
############################################################

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


############################################################
# 单次实验
############################################################

def run_single_experiment(dataset_name, exp_name, sem_file, seed):

    dataset_dir = os.path.join(DATA_ROOT, f"{dataset_name}_struct")

    save_dir = os.path.join(
        RESULT_ROOT,
        dataset_name,
        exp_name
    )

    os.makedirs(save_dir, exist_ok=True)

    save_file = os.path.join(
        save_dir,
        f"seed_{seed}.json"
    )

    if os.path.exists(save_file):
        print("Skip existing:", save_file)

        with open(save_file) as f:
            return json.load(f)

    print("\n======================================")
    print("Dataset:", dataset_name)
    print("Experiment:", exp_name)
    print("Semantic file:", sem_file)
    print("Seed:", seed)
    print("======================================\n")

    set_seed(seed)

    config = copy.deepcopy(train_config)

    config["use_struct"] = True
    config["use_perf"] = False
    config["use_semantic"] = (sem_file is not None)

    if sem_file is not None:
        config["semantic_file"] = sem_file

    metrics = run_training(
        dataset_name=dataset_name,
        data_root=dataset_dir,
        semantic_file=sem_file,
        config=config,
        seed=seed
    )

    with open(save_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved:", save_file)

    return metrics


############################################################
# 汇总：所有实验
############################################################

def summarize_results(dataset):

    dataset_dir = os.path.join(
        RESULT_ROOT,
        dataset
    )

    summary = {}

    for exp_name in SEMANTIC_VARIANTS.keys():

        exp_dir = os.path.join(
            dataset_dir,
            exp_name
        )

        if not os.path.exists(exp_dir):
            continue

        values = []

        for file in os.listdir(exp_dir):

            if not file.endswith(".json"):
                continue

            path = os.path.join(exp_dir, file)

            with open(path) as f:
                data = json.load(f)

            values.append(data["spearman"])

        if len(values) == 0:
            continue

        summary[exp_name] = {
            "mean_spearman": float(np.mean(values)),
            "std": float(np.std(values)),
            "num_seeds": len(values)
        }

    save_file = os.path.join(
        dataset_dir,
        "summary.json"
    )

    with open(save_file, "w") as f:
        json.dump(summary, f, indent=4)

    print("Summary saved:", save_file)

    # 全部实验表
    table = {}
    for exp_name, values in summary.items():
        table[exp_name] = {
            "mean_spearman": round(values["mean_spearman"], 4),
            "std": round(values["std"], 4),
            "num_seeds": values["num_seeds"]
        }

    df = pd.DataFrame.from_dict(table, orient="index")
    csv_path = os.path.join(dataset_dir, "semantic_ablation_table.csv")
    df.to_csv(csv_path)

    print("Table saved:", csv_path)


############################################################
# 额外汇总：只看 bridge 维度对比
############################################################

def summarize_bridge_dim_results(dataset):

    dataset_dir = os.path.join(RESULT_ROOT, dataset)

    rows = []

    for dim in BRIDGE_DIMS:

        exp_name = f"tfidf_svd_bridge_dim{dim}"
        exp_dir = os.path.join(dataset_dir, exp_name)

        if not os.path.exists(exp_dir):
            continue

        values = []

        for file in os.listdir(exp_dir):
            if not file.endswith(".json"):
                continue

            path = os.path.join(exp_dir, file)

            with open(path) as f:
                data = json.load(f)

            values.append(data["spearman"])

        if len(values) == 0:
            continue

        rows.append({
            "dataset": dataset,
            "semantic_dim": dim,
            "mean_spearman": float(np.mean(values)),
            "std": float(np.std(values)),
            "num_seeds": len(values)
        })

    if len(rows) == 0:
        return

    df = pd.DataFrame(rows).sort_values("semantic_dim")
    csv_path = os.path.join(dataset_dir, "bridge_dim_comparison.csv")
    df.to_csv(csv_path, index=False)

    print("Bridge dim comparison saved:", csv_path)


############################################################
# 主函数
############################################################

def main():

    start_time = time.time()

    print("\n========== Semantic Ablation Start ==========\n")

    total = len(DATASETS) * len(SEMANTIC_VARIANTS) * len(SEEDS)
    counter = 0

    for dataset_name in DATASETS:

        for exp_name, sem_file in SEMANTIC_VARIANTS.items():

            for seed in SEEDS:
                counter += 1

                print(f"\nProgress: {counter}/{total}")

                run_single_experiment(
                    dataset_name,
                    exp_name,
                    sem_file,
                    seed
                )

        summarize_results(dataset_name)
        summarize_bridge_dim_results(dataset_name)

    end_time = time.time()

    print("\n========== All Experiments Finished ==========")
    print(
        "Total time:",
        round((end_time - start_time) / 60, 2),
        "minutes"
    )


############################################################

if __name__ == "__main__":
    main()