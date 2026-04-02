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
from multiprocessing import cpu_count, get_context

# =========================================================
# 路径
# =========================================================
INPUT_DIR = "scratch/keynode/project/real_datasets/run_1"
EDGE_FILE = os.path.join(INPUT_DIR, "edgelist.csv")

OUTPUT_DIR = "scratch/keynode/project/real_process/run_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

            infected_nodes = np.where(I)[0]
            for i in infected_nodes:
                neighbors = adj_list[i]
                if len(neighbors) == 0:
                    continue
                susceptible_neighbors = neighbors[S[neighbors]]
                if len(susceptible_neighbors) == 0:
                    continue
                transmit = rng.random(len(susceptible_neighbors)) < beta
                new_I[susceptible_neighbors[transmit]] = True

            recover = rng.random(I.sum()) < MU
            recovered_nodes = np.where(I)[0][recover]

            R[recovered_nodes] = True
            I[recovered_nodes] = False

            # 新感染节点加入 I
            new_I = new_I & S
            I |= new_I
            S[new_I] = False

        infected_ratio.append(R.sum() / N)

    return np.mean(infected_ratio)


# =========================================================
# 读取图
# =========================================================
def load_graph(edge_file):
    if not os.path.exists(edge_file):
        raise FileNotFoundError(f"找不到图文件: {edge_file}")

    edge_df = pd.read_csv(edge_file)

    if edge_df.shape[1] < 2:
        raise ValueError("edgelist.csv 至少应包含两列，分别表示边的两个端点。")

    # 只取前两列作为边
    src_col = edge_df.columns[0]
    dst_col = edge_df.columns[1]

    G = nx.Graph()
    G.add_edges_from(edge_df[[src_col, dst_col]].itertuples(index=False, name=None))

    # 去掉自环
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


# =========================================================
# 主程序
# =========================================================
def main():
    print("========== 加载图 ==========")
    G = load_graph(EDGE_FILE)

    nodes = sorted(G.nodes())
    N = len(nodes)

    print(f"输入图文件: {EDGE_FILE}")
    print(f"节点数: {N}")
    print(f"边数: {G.number_of_edges()}")

    if N == 0:
        raise ValueError("图为空，无法进行 SIR 仿真。")

    # 节点重新映射到连续索引
    node2idx = {n: i for i, n in enumerate(nodes)}
    idx2node = {i: n for n, i in node2idx.items()}

    # 邻接表（索引形式）
    adj_list = []
    for n in nodes:
        nbrs = [node2idx[nbr] for nbr in G.neighbors(n)]
        adj_list.append(np.array(nbrs, dtype=int))

    # =====================================================
    # 理论相变阈值 βc
    # βc = μ <k> / (<k^2> - <k>)
    # =====================================================
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    mean_k = degrees.mean()
    mean_k2 = (degrees ** 2).mean()

    denom = (mean_k2 - mean_k)
    if abs(denom) < 1e-12:
        raise ValueError("图的 <k^2> - <k> 过小，无法稳定计算 beta_c。通常说明图过稀或结构异常。")

    beta_c = MU * mean_k / denom
    beta_list = np.linspace(0.5 * beta_c, 1.8 * beta_c, 12)

    print(f"MU = {MU}")
    print(f"βc = {beta_c:.8f}")
    print("beta_list =", beta_list)

    # =====================================================
    # 传播响应矩阵
    # 每行一个节点，每列一个 beta
    # =====================================================
    spread_matrix = np.zeros((N, len(beta_list)), dtype=float)

    ctx = get_context("spawn")
    with ctx.Pool(cpu_count()) as pool:
        for b_idx, beta in enumerate(beta_list):
            print(f"\n===== β = {beta:.8f} =====")
            args_list = [(i, adj_list, beta, N) for i in range(N)]
            results = list(tqdm(pool.imap(run_sir_mc, args_list), total=N))
            spread_matrix[:, b_idx] = results

    # =====================================================
    # 平滑
    # =====================================================
    spread_matrix = gaussian_filter1d(spread_matrix, sigma=1, axis=1)

    # =====================================================
    # 临界斜率 + AUC -> influence soft label
    # =====================================================
    crit_idx = np.argmin(np.abs(beta_list - beta_c))
    crit_idx = np.clip(crit_idx, 1, len(beta_list) - 2)

    crit_slope = spread_matrix[:, crit_idx + 1] - spread_matrix[:, crit_idx - 1]
    auc = np.trapezoid(spread_matrix, beta_list, axis=1)

    influence = 0.75 * crit_slope + 0.25 * auc
    influence = (influence - influence.min()) / (influence.max() - influence.min() + 1e-8)

    # =====================================================
    # 保存节点标签
    # =====================================================
    sir_df = pd.DataFrame({
        "node": nodes,
        "soft_label": influence
    })

    for i in range(len(beta_list)):
        sir_df[f"spread_beta_{i}"] = spread_matrix[:, i]

    sir_df.to_csv(os.path.join(OUTPUT_DIR, "sir_node_labels.csv"), index=False, encoding="utf-8-sig")
    print("SIR 节点标签已保存")

    # =====================================================
    # 保存相变曲线
    # =====================================================
    avg_curve = spread_matrix.mean(axis=0)
    curve_df = pd.DataFrame({
        "beta": beta_list,
        "avg_spread": avg_curve
    })
    curve_df.to_csv(os.path.join(OUTPUT_DIR, "sir_phase_curve.csv"), index=False, encoding="utf-8-sig")
    print("相变曲线数据已保存")

    # =====================================================
    # 保存 beta 配置
    # =====================================================
    beta_df = pd.DataFrame({
        "beta_index": np.arange(len(beta_list)),
        "beta": beta_list
    })
    beta_df.to_csv(os.path.join(OUTPUT_DIR, "beta_list.csv"), index=False, encoding="utf-8-sig")

    # =====================================================
    # 绘图
    # =====================================================
    plt.figure(figsize=(6, 4))
    plt.plot(beta_list, avg_curve, marker='o')
    plt.axvline(beta_c, linestyle='--', label='beta_c')
    plt.xlabel("Infection Rate β")
    plt.ylabel("Average Infected Ratio")
    plt.title("SIR Phase Transition - run_1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sir_phase_curve.png"), dpi=300)
    plt.close()
    print("相变曲线图已保存")

    print("\n=========== 处理完成 ===========")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()