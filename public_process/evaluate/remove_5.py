import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

BASE_DIR = "scratch/project/public_process/extract"
OUT_DIR = "scratch/project/public_process/evaluate/robust"

datasets = [
"Cora","CiteSeer","PubMed",
"Computers","Photo","CS","Physics"
]

seeds = [0,1,2,3,4]

REMOVE_STEPS = np.linspace(0,0.2,11)

random.seed(42)
np.random.seed(42)


# ================= 近似全局效率 =================

def approx_global_efficiency(G, sample_size=200):

    nodes = list(G.nodes())
    N = len(nodes)

    if N < 2:
        return 0

    sample = random.sample(nodes,min(sample_size,N))

    inv_sum = 0
    pair_count = 0

    for s in sample:

        lengths = nx.single_source_shortest_path_length(G,s)

        for t,d in lengths.items():

            if s != t and d > 0:

                inv_sum += 1.0/d
                pair_count += 1

    if pair_count == 0:
        return 0

    return inv_sum/pair_count


# ================= 基线方法 =================

def pagerank_score(G):

    pr = nx.pagerank(G)
    return np.array([pr[n] for n in G.nodes()])


def degree_score(G):

    deg = dict(G.degree())
    return np.array([deg[n] for n in G.nodes()])


def betweenness_score(G):

    bc = nx.betweenness_centrality(G)
    return np.array([bc[n] for n in G.nodes()])


# ================= 读取GNN分数 =================

def load_gnn_scores(data_dir,seed):

    path = os.path.join(
        data_dir,
        f"results_seed_{seed}",
        "gnn_node_scores.csv"
    )

    df = pd.read_csv(path)

    df = df.sort_values("node")

    return df["gnn_score"].values


# ================= 单次删除实验 =================

def robustness_step(G_full, scores, ratio):

    N0 = G_full.number_of_nodes()

    nodes = np.array(list(G_full.nodes()))

    rank = nodes[np.argsort(-scores)]

    k = int(ratio * N0)

    remove_nodes = rank[:k]

    G = G_full.copy()
    G.remove_nodes_from(remove_nodes)

    if G.number_of_nodes() == 0:
        return 0,0,np.nan

    comp = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(comp)

    lcc = len(comp)/N0
    eff = approx_global_efficiency(G)

    if len(comp) > 10:
        try:
            dia = nx.diameter(Gc)
        except:
            dia = np.nan
    else:
        dia = np.nan

    return lcc,eff,dia


# ================= 主实验 =================

records = []

for dataset in datasets:

    print("\n======",dataset,"======")

    data_dir = f"{BASE_DIR}/{dataset}_struct"

    edge_df = pd.read_csv(
        os.path.join(data_dir,"edgelist.csv")
    )

    G = nx.Graph()
    G.add_edges_from(edge_df.values)

    pr_scores = pagerank_score(G)
    deg_scores = degree_score(G)
    bc_scores = betweenness_score(G)

    for seed in seeds:

        print("seed",seed)

        gnn_scores = load_gnn_scores(data_dir,seed)

        methods = {
            "Ours": gnn_scores,
            "PageRank": pr_scores,
            "Degree": deg_scores,
            "Betweenness": bc_scores
        }

        for method,scores in methods.items():

            for ratio in REMOVE_STEPS:

                lcc,eff,dia = robustness_step(G,scores,ratio)

                records.append([
                    dataset,
                    seed,
                    method,
                    ratio,
                    lcc,
                    eff,
                    dia
                ])


# ================= 保存原始结果 =================

df = pd.DataFrame(
    records,
    columns=[
        "dataset",
        "seed",
        "method",
        "remove_ratio",
        "LCC",
        "Efficiency",
        "Diameter"
    ]
)

os.makedirs(OUT_DIR,exist_ok=True)

raw_path = os.path.join(
    OUT_DIR,
    "robustness_full_results.csv"
)

df.to_csv(raw_path,index=False)

print("raw results saved:",raw_path)


# ================= 统计 mean std =================

summary = df.groupby(
    ["dataset","method","remove_ratio"]
).agg(
    LCC_mean=("LCC","mean"),
    LCC_std=("LCC","std"),
    Eff_mean=("Efficiency","mean"),
    Eff_std=("Efficiency","std"),
    Dia_mean=("Diameter","mean"),
    Dia_std=("Diameter","std")
).reset_index()

summary_path = os.path.join(
    OUT_DIR,
    "robustness_summary.csv"
)

summary.to_csv(summary_path,index=False)

print("summary saved:",summary_path)


# ================= 画鲁棒性曲线 =================

for dataset in datasets:

    sub = summary[summary.dataset==dataset]

    # ===== LCC =====
    plt.figure()

    for method in sub.method.unique():

        d = sub[sub.method==method]

        plt.plot(
            d.remove_ratio,
            d.LCC_mean,
            marker="o",
            label=method
        )

    plt.xlabel("Removed Node Ratio")
    plt.ylabel("Largest Connected Component")
    plt.title(dataset)

    plt.legend()

    plt.savefig(
        os.path.join(OUT_DIR,f"{dataset}_LCC_curve.pdf"),
        bbox_inches="tight"
    )

    plt.close()


    # ===== Efficiency =====

    plt.figure()

    for method in sub.method.unique():

        d = sub[sub.method==method]

        plt.plot(
            d.remove_ratio,
            d.Eff_mean,
            marker="o",
            label=method
        )

    plt.xlabel("Removed Node Ratio")
    plt.ylabel("Global Efficiency")
    plt.title(dataset)

    plt.legend()

    plt.savefig(
        os.path.join(OUT_DIR,f"{dataset}_Efficiency_curve.pdf"),
        bbox_inches="tight"
    )

    plt.close()

print("robustness curves generated")