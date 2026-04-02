import torch
import numpy as np
import pandas as pd
import networkx as nx
import json
import os
from torch_geometric.datasets import Planetoid, Amazon, Coauthor

# ======================================================
# 路径配置（与你工程一致）
# ======================================================
ROOT = "scratch/project/public_datasets"
OUT_BASE = "scratch/project/public_process/extract"

datasets = [
    ("Planetoid", "Cora"),
    ("Planetoid", "CiteSeer"),
    ("Planetoid", "PubMed"),
    ("Amazon", "Computers"),
    ("Amazon", "Photo"),
    ("Coauthor", "CS"),
    ("Coauthor", "Physics"),
]

def load_dataset(dtype, name):
    if dtype == "Planetoid":
        return Planetoid(root=f"{ROOT}/Planetoid", name=name)
    elif dtype == "Amazon":
        return Amazon(root=f"{ROOT}/Amazon", name=name)
    elif dtype == "Coauthor":
        return Coauthor(root=f"{ROOT}/Coauthor", name=name)

def process_one_dataset(name, data):
    print(f"\n====== 处理 {name} ======")
    edge_index = data.edge_index
    G = nx.Graph()
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    print("节点数:", G.number_of_nodes())
    print("边数:", G.number_of_edges())

    # ================= 结构特征 =================
    print("计算结构特征中...")
    degree_centrality = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)
    k_shell = nx.core_number(G)

    nodes = sorted(G.nodes())

    struct_df = pd.DataFrame({
        "node": nodes,
        "degree_centrality": [degree_centrality[n] for n in nodes],
        "closeness": [closeness[n] for n in nodes],
        "betweenness": [betweenness[n] for n in nodes],
        "clustering_coeff": [clustering[n] for n in nodes],
        "k_shell": [k_shell[n] for n in nodes]
    })

    # ================= 输出目录 =================
    out_dir = f"{OUT_BASE}/{name}_struct"
    os.makedirs(out_dir, exist_ok=True)

    # 1️⃣ node_features.csv
    struct_df.to_csv(os.path.join(out_dir, "node_features.csv"), index=False)

    # 2️⃣ x_struct.npy
    x_struct = struct_df.drop(columns=["node"]).values.astype(np.float32)
    np.save(os.path.join(out_dir, "x_struct.npy"), x_struct)

    # 3️⃣ edge_index.npy
    np.save(os.path.join(out_dir, "edge_index.npy"), edge_index.numpy())

    # 4️⃣ node_index_map.json
    node_index_map = {int(n): int(n) for n in nodes}
    with open(os.path.join(out_dir, "node_index_map.json"), "w") as f:
        json.dump(node_index_map, f, indent=2)

    # 5️⃣ edgelist.csv
    edge_df = pd.DataFrame(edges, columns=["src", "dst"])
    edge_df.to_csv(os.path.join(out_dir, "edgelist.csv"), index=False)

    print(f"{name} 完成 → {out_dir}")

# ======================================================
# 🚀 主循环
# ======================================================
for dtype, name in datasets:
    dataset = load_dataset(dtype, name)
    data = dataset[0]
    process_one_dataset(name, data)

print("\n🎯 全部 7 个数据集结构特征提取完毕")
