#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

############################################################
# 路径配置
############################################################

BASE_DIR = "scratch/project/public_process/extract"
OUT_DIR = "scratch/project/public_process/sir/sir_png"

os.makedirs(OUT_DIR, exist_ok=True)

datasets = [
    "Cora",
    "CiteSeer",
    "PubMed",
    "Computers",
    "Photo",
    "CS",
    "Physics"
]

############################################################
# SCI论文风格配置
############################################################

NATURE_COLORS = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2"
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False
})

############################################################
# 计算 βc
############################################################

def compute_beta_c(edge_path, mu=0.2):

    edge_df = pd.read_csv(edge_path)

    G = nx.Graph()
    G.add_edges_from(edge_df.values)

    degrees = np.array([d for _, d in G.degree()])

    mean_k = degrees.mean()
    mean_k2 = (degrees ** 2).mean()

    beta_c = mu * mean_k / (mean_k2 - mean_k + 1e-8)

    return beta_c, G


############################################################
# 加载数据
############################################################

def load_spread_data(dataset):

    label_path = os.path.join(
        BASE_DIR,
        f"{dataset}_struct",
        "sir_node_labels.csv"
    )

    phase_path = os.path.join(
        BASE_DIR,
        f"{dataset}_struct",
        "sir_phase_curve.csv"
    )

    df = pd.read_csv(label_path)
    phase = pd.read_csv(phase_path)

    spread_cols = [c for c in df.columns if "spread_beta" in c]

    spread_matrix = df[spread_cols].values
    influence = df["soft_label"].values

    beta_list = phase["beta"].values
    avg_curve = phase["avg_spread"].values

    return spread_matrix, influence, beta_list, avg_curve


############################################################
# Top节点传播曲线
############################################################

def plot_top_nodes(ax, spread_matrix, influence, beta):

    top_nodes = np.argsort(influence)[-10:]

    for i,node in enumerate(top_nodes):

        ax.plot(
            beta,
            spread_matrix[node],
            marker="o",
            linewidth=1.5,
            color=NATURE_COLORS[i%len(NATURE_COLORS)]
        )

    ax.set_xlabel("Infection Rate β")
    ax.set_ylabel("Infected Ratio")
    ax.grid(alpha=0.3)


############################################################
# 95%置信区间
############################################################

def plot_confidence_interval(ax, spread_matrix, beta):

    mean_curve = spread_matrix.mean(axis=0)
    std_curve = spread_matrix.std(axis=0)

    ci = 1.96 * std_curve / np.sqrt(spread_matrix.shape[0])

    ax.plot(
        beta,
        mean_curve,
        color="#E64B35",
        linewidth=2,
        label="Mean Spread"
    )

    ax.fill_between(
        beta,
        mean_curve - ci,
        mean_curve + ci,
        color="#4DBBD5",
        alpha=0.35,
        label="95% CI"
    )

    ax.set_xlabel("β")
    ax.set_ylabel("Spread")

    ax.legend(frameon=False)
    ax.grid(alpha=0.3)


############################################################
# Slope + AUC示意
############################################################

def plot_slope_auc(ax, spread_matrix, influence, beta, beta_c):

    node = np.argmax(influence)
    y = spread_matrix[node]

    ax.plot(beta,y,marker="o",color="#3C5488")

    idx = np.argmin(np.abs(beta - beta_c))
    idx = np.clip(idx,1,len(beta)-2)

    ax.plot(
        [beta[idx-1], beta[idx+1]],
        [y[idx-1], y[idx+1]],
        linestyle="--",
        color="black",
        label="Slope"
    )

    # AUC 阴影
    ax.fill_between(
        beta,
        y,
        alpha=0.25,
        color="#F39B7F",
        label="AUC"
    )

    ax.axvline(
        beta_c,
        linestyle=":",
        color="black"
    )


    ax.set_xlabel("β")
    ax.set_ylabel("Spread")

    ax.legend(frameon=False)
    ax.grid(alpha=0.3)


############################################################
# Influence排名
############################################################

def plot_influence_rank(ax, influence):

    rank = np.sort(influence)[::-1]

    ax.plot(rank, color="#3C5488", linewidth=2)

    ax.set_xlabel("Node Rank")
    ax.set_ylabel("Influence")

    ax.grid(alpha=0.3)


############################################################
# 网络可视化
############################################################

def plot_top_network(ax, G, influence):

    pos = nx.spring_layout(G, seed=42, k=0.15)

    top_nodes = set(np.argsort(influence)[-10:])

    node_colors = []

    for n in G.nodes():

        if n in top_nodes:
            node_colors.append("#E64B35")
        else:
            node_colors.append("#BBBBBB")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=35,
        node_color=node_colors,
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.15,
        width=0.4,
        ax=ax
    )

    ax.set_title("Top Nodes Highlighted")
    ax.axis("off")


############################################################
# 相变曲线
############################################################

def plot_phase_transition(ax, beta, avg_curve, beta_c):

    ax.plot(
        beta,
        avg_curve,
        marker="o",
        color="#E64B35",
        linewidth=2
    )

    ax.axvline(
        beta_c,
        linestyle="--",
        color="#3C5488"
    )



    ax.set_xlabel("β")
    ax.set_ylabel("Average Spread")

    ax.grid(alpha=0.3)


############################################################
# 六子图
############################################################

def plot_full_figure(dataset):

    print("Processing:", dataset)

    spread_matrix, influence, beta, avg_curve = load_spread_data(dataset)

    edge_path = os.path.join(
        BASE_DIR,
        f"{dataset}_struct",
        "edgelist.csv"
    )

    beta_c, G = compute_beta_c(edge_path)

    fig, axes = plt.subplots(2,3,figsize=(10,6))

    ############################################################
    # (a) Phase Transition
    ############################################################
    plot_phase_transition(axes[0,0], beta, avg_curve, beta_c)
    axes[0,0].set_title("(a) Phase Transition")

    ############################################################
    # (b) Response Curves
    ############################################################
    plot_top_nodes(axes[0,1], spread_matrix, influence, beta)
    axes[0,1].set_title("(b) Top Node Response Curves")

    ############################################################
    # (c) Spread Statistics
    ############################################################
    plot_confidence_interval(axes[0,2], spread_matrix, beta)
    axes[0,2].set_title("(c) Spread Statistics (95% CI)")

    ############################################################
    # (d) Slope + AUC
    ############################################################
    plot_slope_auc(axes[1,0], spread_matrix, influence, beta, beta_c)
    axes[1,0].set_title("(d) Slope and AUC Illustration")

    ############################################################
    # (e) Influence Ranking
    ############################################################
    plot_influence_rank(axes[1,1], influence)
    axes[1,1].set_title("(e) Influence Ranking")

    ############################################################
    # (f) Network Visualization
    ############################################################
    plot_top_network(axes[1,2], G, influence)
    axes[1,2].set_title("(f) Network Visualization")

    plt.tight_layout()

    save_png = os.path.join(
        OUT_DIR,
        f"{dataset}_SCI_figure.png"
    )

    save_pdf = os.path.join(
        OUT_DIR,
        f"{dataset}_SCI_figure.pdf"
    )

    plt.savefig(save_png)
    plt.savefig(save_pdf)

    plt.close()

    print("Saved:", save_png)

############################################################
# 主程序
############################################################

def main():

    for dataset in datasets:

        plot_full_figure(dataset)

    print("\nAll SCI figures generated.")


if __name__ == "__main__":

    main()