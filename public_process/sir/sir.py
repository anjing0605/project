#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from multiprocessing import get_context

# =========================================================
# 路径与数据集
# =========================================================
BASE_DIR = "scratch/project/public_process/extract"
datasets = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "CS", "Physics"]

# =========================================================
# SIR 全局参数
# =========================================================
MC = 80
MU = 0.2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# =========================================================
# SIR 单节点 Monte-Carlo（向量化版本）
# =========================================================
def run_sir_mc(args):
    node_idx, adj_list, beta, N = args
    rng = np.random.default_rng(SEED + node_idx)
    infected_ratio = []
    for _ in range(MC):
        S = np.ones(N, dtype=bool)
        I = np.zeros(N, dtype=bool)
        R = np.zeros(N, dtype=bool)

        I[node_idx] = True
        S[node_idx] = False

        while I.any():
            new_I = np.zeros(N, dtype=bool)
            for i in np.where(I)[0]:
                neighbors = adj_list[i]
                susceptible_neighbors = neighbors[S[neighbors]]
                transmit = rng.random(len(susceptible_neighbors)) < beta
                new_I[susceptible_neighbors[transmit]] = True

            recover = rng.random(I.sum()) < MU
            recovered_nodes = np.where(I)[0][recover]

            R[recovered_nodes] = True
            I[recovered_nodes] = False
            I |= new_I
            S[new_I] = False

        infected_ratio.append((I | R).sum() / N)
    return np.mean(infected_ratio)

# =========================================================
# 主循环
# =========================================================
def main():
    for name in datasets:
        print(f"\n================ {name} ================")
        out_dir = os.path.join(BASE_DIR, f"{name}_struct")
        edge_file = os.path.join(out_dir, "edgelist.csv")

        edge_df = pd.read_csv(edge_file)
        G = nx.Graph()
        G.add_edges_from(edge_df.values)

        nodes = sorted(G.nodes())
        N = len(nodes)
        print("节点数:", N, "边数:", G.number_of_edges())

        # 邻接表转索引形式
        node2idx = {n: i for i, n in enumerate(nodes)}
        adj_list = [np.array([node2idx[nbr] for nbr in G.neighbors(n)], dtype=int) for n in nodes]

        # 理论相变阈值
        degrees = np.array([d for _, d in G.degree()])
        mean_k = degrees.mean()
        mean_k2 = (degrees ** 2).mean()
        beta_c = MU * mean_k / (mean_k2 - mean_k + 1e-8)
        beta_list = np.linspace(0.5 * beta_c, 1.8 * beta_c, 12)
        print("βc =", beta_c)

        # 传播响应矩阵
        spread_matrix = np.zeros((N, len(beta_list)))

        ctx = get_context("spawn")
        with ctx.Pool(cpu_count()) as pool:
            for b_idx, beta in enumerate(beta_list):
                print(f"\nβ = {beta:.6f}")
                args_list = [(i, adj_list, beta, N) for i in range(N)]
                results = list(tqdm(pool.imap(run_sir_mc, args_list), total=N))
                spread_matrix[:, b_idx] = results

        # 平滑
        spread_matrix = gaussian_filter1d(spread_matrix, sigma=1, axis=1)

        # 临界斜率 + AUC
        crit_idx = np.argmin(np.abs(beta_list - beta_c))
        crit_idx = np.clip(crit_idx, 1, len(beta_list) - 2)
        crit_slope = spread_matrix[:, crit_idx + 1] - spread_matrix[:, crit_idx - 1]
        auc = np.trapezoid(spread_matrix, beta_list, axis=1)
        influence = 0.75 * crit_slope + 0.25 * auc
        influence = (influence - influence.min()) / (influence.max() - influence.min() + 1e-8)

        # 保存节点标签
        sir_df = pd.DataFrame({"node": nodes, "soft_label": influence})
        for i in range(len(beta_list)):
            sir_df[f"spread_beta_{i}"] = spread_matrix[:, i]
        sir_df.to_csv(os.path.join(out_dir, "sir_node_labels.csv"), index=False)
        print("SIR 标签已保存")

        # 保存相变曲线
        avg_curve = spread_matrix.mean(axis=0)
        curve_df = pd.DataFrame({"beta": beta_list, "avg_spread": avg_curve})
        curve_df.to_csv(os.path.join(out_dir, "sir_phase_curve.csv"), index=False)
        print("相变曲线已保存")

        # 绘图
        plt.figure(figsize=(6, 4))
        plt.plot(beta_list, avg_curve, marker='o')
        plt.axvline(beta_c, linestyle='--', label='beta_c')
        plt.xlabel("Infection Rate β")
        plt.ylabel("Average Infected Ratio")
        plt.title(f"SIR Phase Transition - {name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sir_phase_curve.png"), dpi=300)
        plt.close()
        print("相变曲线图已保存 (PNG)")

    print("\n=========== 所有数据集处理完成 ===========")
if __name__ == "__main__":
    main()
