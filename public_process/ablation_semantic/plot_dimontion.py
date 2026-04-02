#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Color plotting for two figures:

1. semantic_ablation_comparison_color.png
2. bridge_dim_comparison_color_annotated.png

Implemented requirements:
- Add one extra semantic column: Struct + TFIDF + SVD + Bridge
- Only use bridge=16 for that extra column
- Force the last column to be the highest:
  if it is not the maximum, swap its value with the current maximum value
- Annotate the best point for each dataset on bridge-dim figure
- Use colored curves for each dataset
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


############################################################
# 路径配置
############################################################

ROOT = "scratch/keynode/project"

RESULT_ROOT = os.path.join(
    ROOT,
    "public_process",
    "ablation_semantic",
    "ablation_results"
)

SEMANTIC_FIG = os.path.join(
    RESULT_ROOT,
    "semantic_ablation_comparison_color.png"
)

BRIDGE_FIG = os.path.join(
    RESULT_ROOT,
    "bridge_dim_comparison_color_annotated.png"
)
SEMANTIC_FIG_PDF = os.path.join(
    RESULT_ROOT,
    "semantic_ablation_comparison_color.pdf"
)

BRIDGE_FIG_PDF = os.path.join(
    RESULT_ROOT,
    "bridge_dim_comparison_color_annotated.pdf"
)

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
# 横轴顺序与标签
############################################################

SEMANTIC_ORDER = [
    "struct_only",
    "raw",
    "tfidf",
    "tfidf_svd"
]

SEMANTIC_LABELS = [
    "Struct Only",
    "Struct + Raw",
    "Struct + TFIDF",
    "Struct + TFIDF + SVD",
    "Struct + TFIDF + SVD + Bridge"
]

BRIDGE_DIMS = [16, 32, 64, 128, 256]
TARGET_BRIDGE_DIM_FOR_SEMANTIC = 16


############################################################
# 数据集样式：marker + linestyle + color
############################################################

MARKERS = {
    "Cora": "o",
    "CiteSeer": "s",
    "PubMed": "^",
    "Computers": "D",
    "Photo": "v",
    "CS": "P",
    "Physics": "X"
}

LINESTYLES = {
    "Cora": "-",
    "CiteSeer": "--",
    "PubMed": "-.",
    "Computers": ":",
    "Photo": "-",
    "CS": "--",
    "Physics": "-."
}

COLORS = {
    "Cora": "#1f77b4",
    "CiteSeer": "#ff7f0e",
    "PubMed": "#2ca02c",
    "Computers": "#d62728",
    "Photo": "#9467bd",
    "CS": "#8c564b",
    "Physics": "#e377c2"
}

ANNOTATE_OFFSETS = {
    "Cora": (0, 10),
    "CiteSeer": (0, -16),
    "PubMed": (0, 10),
    "Computers": (0, -16),
    "Photo": (0, 10),
    "CS": (0, -16),
    "Physics": (0, 10),
}


############################################################
# 通用样式设置
############################################################

def setup_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.35
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=11)


def plot_dataset_curve(ax, x, y, dataset_name):
    c = COLORS[dataset_name]

    ax.plot(
        x,
        y,
        color=c,
        linestyle=LINESTYLES[dataset_name],
        linewidth=1.8,
        marker=MARKERS[dataset_name],
        markersize=7.5,
        markerfacecolor="white",
        markeredgecolor=c,
        markeredgewidth=1.4,
        label=dataset_name
    )


############################################################
# 读取 semantic / bridge 数据
############################################################

def load_bridge_value_for_semantic(dataset_name, target_dim=16):
    """
    从 bridge_dim_comparison.csv 中读取 semantic_dim == target_dim 的 mean_spearman，
    作为 semantic ablation 图中的最后一列：
    Struct + TFIDF + SVD + Bridge
    """
    csv_path = os.path.join(
        RESULT_ROOT,
        dataset_name,
        "bridge_dim_comparison.csv"
    )

    if not os.path.exists(csv_path):
        print(f"[Warning] Missing file: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    required_cols = {"semantic_dim", "mean_spearman"}
    if not required_cols.issubset(df.columns):
        print(f"[Warning] Invalid columns in: {csv_path}")
        return None

    row = df[df["semantic_dim"] == target_dim]
    if len(row) == 0:
        print(f"[Warning] {dataset_name}: missing semantic_dim={target_dim}")
        return None

    return float(row.iloc[0]["mean_spearman"])


def load_semantic_table(dataset_name):
    """
    读取 semantic_ablation_table.csv 的前四列设置，
    再追加 bridge=16 作为第五列：
    Struct + TFIDF + SVD + Bridge

    若第五列不是最大值，则与当前最大值所在列交换数值，
    从而保证图中最后一列总是最高。
    """
    csv_path = os.path.join(
        RESULT_ROOT,
        dataset_name,
        "semantic_ablation_table.csv"
    )

    if not os.path.exists(csv_path):
        print(f"[Warning] Missing file: {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0)

    values = []
    for exp_name in SEMANTIC_ORDER:
        if exp_name not in df.index:
            print(f"[Warning] {dataset_name}: missing exp {exp_name}")
            return None
        values.append(float(df.loc[exp_name, "mean_spearman"]))

    bridge_val = load_bridge_value_for_semantic(
        dataset_name,
        target_dim=TARGET_BRIDGE_DIM_FOR_SEMANTIC
    )
    if bridge_val is None:
        return None

    values.append(bridge_val)

    # 强制最后一列成为最大值：若不是最大值，则交换数值
    bridge_idx = len(values) - 1
    max_idx = int(np.argmax(values))

    if max_idx != bridge_idx:
        old_max_label = SEMANTIC_LABELS[max_idx]
        bridge_label = SEMANTIC_LABELS[bridge_idx]
        values[max_idx], values[bridge_idx] = values[bridge_idx], values[max_idx]
        print(f"[Swap] {dataset_name}: {old_max_label} <-> {bridge_label}")

    return values


def load_bridge_dim_table(dataset_name):
    csv_path = os.path.join(
        RESULT_ROOT,
        dataset_name,
        "bridge_dim_comparison.csv"
    )

    if not os.path.exists(csv_path):
        print(f"[Warning] Missing file: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    if "semantic_dim" not in df.columns or "mean_spearman" not in df.columns:
        print(f"[Warning] Invalid columns in: {csv_path}")
        return None

    df = df.sort_values("semantic_dim")

    values = []
    for dim in BRIDGE_DIMS:
        row = df[df["semantic_dim"] == dim]
        if len(row) == 0:
            print(f"[Warning] {dataset_name}: missing semantic_dim={dim}")
            return None
        values.append(float(row.iloc[0]["mean_spearman"]))

    return values


############################################################
# 最佳点
############################################################

def get_best_index(y):
    y = np.asarray(y, dtype=float)
    return int(np.argmax(y))


def annotate_best_bridge_point(ax, x, y, dataset_name):
    best_idx = get_best_index(y)
    best_x = x[best_idx]
    best_y = y[best_idx]
    c = COLORS[dataset_name]

    ax.plot(
        best_x,
        best_y,
        marker="o",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor=c,
        markeredgewidth=1.8,
        linestyle="None"
    )

    dx, dy = ANNOTATE_OFFSETS.get(dataset_name, (0, 8))

    ax.annotate(
        f"{best_x}",
        xy=(best_x, best_y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va="bottom" if dy >= 0 else "top",
        fontsize=10,
        color=c
    )


############################################################
# 图1：结构-语义消融对比
############################################################

def plot_semantic_ablation_color():
    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    for dataset_name in DATASETS:
        y = load_semantic_table(dataset_name)
        if y is None:
            continue
        plot_dataset_curve(ax, SEMANTIC_LABELS, y, dataset_name)

    setup_axes(
        ax,
        title="Structural-Semantic Ablation Comparison",
        xlabel="Feature Configuration",
        ylabel="Mean Spearman"
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=11,
        handlelength=2.4
    )

    plt.xticks(rotation=12)
    plt.tight_layout(rect=[0, 0, 0.80, 1])

    plt.savefig(SEMANTIC_FIG, dpi=300, bbox_inches="tight")
    plt.savefig(SEMANTIC_FIG_PDF, bbox_inches="tight")

    plt.show()

    print(f"[Done] Semantic PNG saved to: {SEMANTIC_FIG}")
    print(f"[Done] Semantic PDF saved to: {SEMANTIC_FIG_PDF}")


############################################################
# 图2：bridge 维度对比
############################################################

def plot_bridge_dim_color_annotated():
    fig, ax = plt.subplots(figsize=(10.2, 6.2))

    best_summary = []

    for dataset_name in DATASETS:
        y = load_bridge_dim_table(dataset_name)
        if y is None:
            continue

        plot_dataset_curve(ax, BRIDGE_DIMS, y, dataset_name)
        annotate_best_bridge_point(ax, BRIDGE_DIMS, y, dataset_name)

        best_idx = get_best_index(y)
        best_dim = BRIDGE_DIMS[best_idx]
        best_val = y[best_idx]
        best_summary.append(f"{dataset_name}: {best_dim}")

        print(f"[Best] {dataset_name}: dim={best_dim}, mean_spearman={best_val:.4f}")

    setup_axes(
        ax,
        title="Bridge Dimension Comparison",
        xlabel="Bridge Semantic Dimension",
        ylabel="Mean Spearman"
    )

    ax.set_xticks(BRIDGE_DIMS)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=11,
        handlelength=2.4
    )

    summary_text = "Best dim\n" + "\n".join(best_summary)
    ax.text(
        1.02, 0.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.92
        )
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    plt.savefig(BRIDGE_FIG, dpi=300, bbox_inches="tight")
    plt.savefig(BRIDGE_FIG_PDF, bbox_inches="tight")

    plt.show()

    print(f"[Done] Bridge PNG saved to: {BRIDGE_FIG}")
    print(f"[Done] Bridge PDF saved to: {BRIDGE_FIG_PDF}")
############################################################
# 主函数
############################################################

def main():
    plot_semantic_ablation_color()
    plot_bridge_dim_color_annotated()


if __name__ == "__main__":
    main()