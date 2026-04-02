import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = "scratch/project/public_process/extract"
OUT_DIR = "scratch/project/public_process/evaluate/sir_results"

datasets = [
    "Cora",
    "CiteSeer","PubMed",
    "Computers","Photo","CS","Physics"
]

seeds = [0,1,2,3,4]

TOPK_LIST = [0.01,0.05,0.10]

BETA_SCALE = [
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.5,
    2.0
]
robust_records = []
MC = 100
SOURCE_SAMPLE = 100

random.seed(42)
np.random.seed(42)

# =========================
# SIR Simulation
# =========================

def run_sir_once(adj, source, beta):

    S = set(adj.keys())
    I = {source}
    R = set()

    S.remove(source)

    while I:

        new_I = set()

        for node in I:

            for nbr in adj[node]:

                if nbr in S and random.random() < beta:
                    new_I.add(nbr)

        R |= I
        S -= new_I
        I = new_I

    return len(R)
# =========================
# True SIR Influence
# =========================

def compute_true_influence(G, beta):

    nodes = list(G.nodes())
    adj = {n:list(G.neighbors(n)) for n in nodes}

    N = len(nodes)

    true_spread = []

    for node in tqdm(nodes,desc="True SIR"):

        sims = [
            run_sir_once(adj,node,beta)
            for _ in range(MC)
        ]

        true_spread.append(np.mean(sims)/N)

    return np.array(true_spread)

def compute_avg_spread(G, beta, sources):

    nodes = list(G.nodes())
    adj = {n:list(G.neighbors(n)) for n in nodes}

    N = len(nodes)

    valid_sources = [s for s in sources if s in adj]

    if len(valid_sources) == 0:
        return 0

    spreads = []

    for node in valid_sources:

        sims = [
            run_sir_once(adj,node,beta)
            for _ in range(MC)
        ]

        spreads.append(np.mean(sims)/N)

    return np.mean(spreads)

# =========================
# Baselines
# =========================

def pagerank_score(G):

    pr = nx.pagerank(G)

    nodes = list(G.nodes())

    return np.array([pr[n] for n in nodes])


def degree_score(G):
    deg = dict(G.degree())

    nodes = list(G.nodes())
    return np.array([deg[n] for n in nodes])


def betweenness_score(G):

    bc = nx.betweenness_centrality(
        G,
        k=200,
        seed=42
    )

    nodes = list(G.nodes())
    return np.array([bc[n] for n in nodes])

# =========================
# Blocking Experiment
# =========================

def run_blocking(G, scores, beta, ratio, sources, I_base):

    N = len(scores)

    k = int(ratio * N)

    nodes = np.arange(N)

    rank = nodes[np.argsort(-scores)]

    remove_nodes = rank[:k]

    G_removed = G.copy()
    G_removed.remove_nodes_from(remove_nodes)

    I_after = compute_avg_spread(G_removed,beta,sources)

    delta_I = (I_base - I_after) / (I_base + 1e-8)

    return delta_I

# =========================
# Adaptive Blocking
# =========================

def run_adaptive_blocking(G, scores, beta, ratio, sources, I_base):

    G_tmp = G.copy()

    N = len(scores)
    k = int(ratio * N)

    nodes = np.arange(N)
    rank = nodes[np.argsort(-scores)]

    removed = []

    for i in range(k):

        node = rank[i]

        if node in G_tmp:
            G_tmp.remove_node(node)
            removed.append(node)

    I_after = compute_avg_spread(G_tmp, beta, sources)

    delta_I = (I_base - I_after) / (I_base + 1e-8)

    return delta_I
# =========================
# Top-k Precision
# =========================

def compute_topk_precision(pred,true,k_ratio):

    N = len(pred)

    k = int(N * k_ratio)

    pred_rank = np.argsort(-pred)[:k]
    true_rank = np.argsort(-true)[:k]

    overlap = len(set(pred_rank) & set(true_rank))

    return overlap / k
# =========================
# Load GNN scores
# =========================

def load_gnn_scores(data_dir,seed):

    path = os.path.join(
        data_dir,
        f"results_seed_{seed}",
        "gnn_node_scores.csv"
    )

    df = pd.read_csv(path)

    df = df.sort_values("node")

    return df["gnn_score"].values

# =========================
# Main Experiment
# =========================

all_records = []

for dataset in datasets:

    print("\n========",dataset,"========")

    data_dir = f"{BASE_DIR}/{dataset}_struct"

    edge_df = pd.read_csv(
        os.path.join(data_dir,"edgelist.csv")
    )

    G = nx.Graph()
    G.add_edges_from(edge_df.values)
    G = nx.convert_node_labels_to_integers(G)

    degrees = np.array([d for _,d in G.degree()])
    mean_k = degrees.mean()
    mean_k2 = np.mean(degrees**2)

    beta_c = mean_k/(mean_k2-mean_k+1e-8)

    pr_scores = pagerank_score(G)
    deg_scores = degree_score(G)
    bc_scores = betweenness_score(G)

    for beta_scale in BETA_SCALE:

        beta = beta_scale * beta_c

        print("beta =",beta)

        for ratio in TOPK_LIST:

            print("remove ratio =",ratio)

            for seed in seeds:

                print("seed",seed)

                random.seed(seed)
                np.random.seed(seed)

                gnn_scores = load_gnn_scores(data_dir,seed)

                methods = {
                    "Ours": gnn_scores,
                    "PageRank": pr_scores,
                    "Degree": deg_scores,
                    "Betweenness": bc_scores
                }

                nodes = list(G.nodes())

                sources = random.sample(
                    nodes,
                    min(SOURCE_SAMPLE,len(nodes))
                )

                I_base = compute_avg_spread(
                    G,
                    beta,
                    sources
                )

                for method,scores in methods.items():

                    delta = run_blocking(
                        G,
                        scores,
                        beta,
                        ratio,
                        sources,
                        I_base
                    )
                    if ratio == 0.05:
                        robust_records.append([
                            dataset,
                            seed,
                            method,
                            beta_scale,
                            delta
                        ])

                    all_records.append([
                        dataset,
                        seed,
                        method,
                        beta_scale,
                        ratio,
                        delta
                    ])
                    # Adaptive blocking
                    delta_adp = run_adaptive_blocking(
                        G,
                        scores,
                        beta,
                        ratio,
                        sources,
                        I_base
                    )

                    all_records.append([
                        dataset,
                        seed,
                        method+"_Adaptive",
                        beta_scale,
                        ratio,
                        delta_adp
                    ])

# =========================
# Top-k Precision Experiment
# =========================
precision_records = []

for dataset in datasets:

    print("Precision test:",dataset)

    data_dir = f"{BASE_DIR}/{dataset}_struct"

    edge_df = pd.read_csv(
        os.path.join(data_dir,"edgelist.csv")
    )

    G = nx.Graph()
    G.add_edges_from(edge_df.values)
    G = nx.convert_node_labels_to_integers(G)

    degrees = np.array([d for _,d in G.degree()])
    mean_k = degrees.mean()
    mean_k2 = np.mean(degrees**2)

    beta_c = mean_k/(mean_k2-mean_k+1e-8)
    beta = 1.5 * beta_c

    true_spread = compute_true_influence(G,beta)

    pr_scores = pagerank_score(G)
    deg_scores = degree_score(G)
    bc_scores = betweenness_score(G)

    for seed in seeds:

        print("seed",seed)

        gnn_scores = load_gnn_scores(data_dir,seed)

        methods = {
            "Ours":gnn_scores,
            "PageRank":pr_scores,
            "Degree":deg_scores,
            "Betweenness":bc_scores
        }

        for k_ratio in [0.01,0.05,0.10]:

            for method,scores in methods.items():

                p = compute_topk_precision(
                    scores,
                    true_spread,
                    k_ratio
                )

                precision_records.append([
                    dataset,
                    seed,
                    method,
                    k_ratio,
                    p
                ])
# =========================
# Save Results
# =========================
robust_df = pd.DataFrame(
    robust_records,
    columns=[
        "dataset",
        "seed",
        "method",
        "beta_scale",
        "Delta_I"
    ]
)

robust_path = os.path.join(
    OUT_DIR,
    "sir_robustness_results.csv"
)

robust_df.to_csv(robust_path,index=False)

print("robustness saved:",robust_path)
precision_df = pd.DataFrame(
    precision_records,
    columns=[
        "dataset",
        "seed",
        "method",
        "k_ratio",
        "precision"
    ]
)

precision_path = os.path.join(
    OUT_DIR,
    "topk_precision.csv"
)

precision_df.to_csv(precision_path,index=False)

print("precision saved:",precision_path)
results_df = pd.DataFrame(
    all_records,
    columns=[
        "dataset",
        "seed",
        "method",
        "beta_scale",
        "ratio",
        "Delta_I"
    ]
)

os.makedirs(OUT_DIR,exist_ok=True)

results_path = os.path.join(
    OUT_DIR,
    "sir_blocking_full_results.csv"
)

results_df.to_csv(results_path,index=False)

print("results saved:",results_path)

# =========================
# Summary
# =========================

summary = results_df.groupby(
    ["dataset","method","beta_scale","ratio"]
)["Delta_I"].agg(["mean","std"]).reset_index()

summary_path = os.path.join(
    OUT_DIR,
    "sir_blocking_summary.csv"
)

summary.to_csv(summary_path,index=False)

print("summary saved:",summary_path)

# =========================
# Plot curves
# =========================

for dataset in datasets:

    df = summary[summary.dataset==dataset]

    plt.figure()

    for method in df.method.unique():

        sub = df[
            (df.method==method) &
            (df.beta_scale==1.5)
        ]

        x = sub["ratio"]
        y = sub["mean"]

        plt.plot(x,y,marker="o",label=method)

    plt.xlabel("Removed Node Ratio")
    plt.ylabel("Propagation Drop (ΔI)")
    plt.title(dataset)

    plt.legend()

    fig_path = os.path.join(
        OUT_DIR,
        f"{dataset}_blocking_curve.pdf"
    )

    plt.savefig(fig_path,bbox_inches="tight")

    plt.close()
# =========================
# Robustness Curve
# =========================

for dataset in datasets:

    df = robust_df[robust_df.dataset==dataset]

    plt.figure()

    for method in df.method.unique():
        sub = df[df.method == method]

        sub2 = sub.groupby("beta_scale")["Delta_I"].mean().reset_index()

        plt.plot(
            sub2["beta_scale"],
            sub2["Delta_I"],
            marker="o",
            label=method
        )

    plt.xlabel("β / βc")
    plt.ylabel("Propagation Drop (ΔI)")
    plt.title(dataset+" Robustness")

    plt.legend()

    fig_path = os.path.join(
        OUT_DIR,
        f"{dataset}_robustness_curve.pdf"
    )

    plt.savefig(fig_path,bbox_inches="tight")

    plt.close()
# =========================
# Spread Prediction Curve
# =========================

for dataset in datasets:

    print("Spread curve:",dataset)

    data_dir = f"{BASE_DIR}/{dataset}_struct"

    edge_df = pd.read_csv(
        os.path.join(data_dir,"edgelist.csv")
    )

    G = nx.Graph()
    G.add_edges_from(edge_df.values)
    G = nx.convert_node_labels_to_integers(G)

    degrees = np.array([d for _,d in G.degree()])
    mean_k = degrees.mean()
    mean_k2 = np.mean(degrees**2)

    beta_c = mean_k/(mean_k2-mean_k+1e-8)
    beta = 1.5 * beta_c

    true_spread = compute_true_influence(G,beta)

    gnn_scores = load_gnn_scores(data_dir,0)

    plt.figure()

    plt.scatter(
        gnn_scores,
        true_spread,
        s=8,
        alpha=0.6
    )

    plt.xlabel("Predicted Influence")
    plt.ylabel("True SIR Spread")
    plt.title(dataset+" Spread Prediction")

    fig_path = os.path.join(
        OUT_DIR,
        f"{dataset}_spread_prediction.pdf"
    )

    plt.savefig(fig_path,bbox_inches="tight")

    plt.close()

# =========================
# Precision Curve
# =========================

for dataset in datasets:

    df = precision_df[precision_df.dataset==dataset]

    plt.figure()

    for method in df.method.unique():

        sub = df[df.method==method]

        sub2 = sub.groupby("k_ratio")["precision"].mean().reset_index()

        plt.plot(
            sub2["k_ratio"],
            sub2["precision"],
            marker="o",
            label=method
        )

    plt.xlabel("Top-k Ratio")
    plt.ylabel("Precision@k")
    plt.title(dataset+" Top-k Precision")

    plt.legend()

    fig_path = os.path.join(
        OUT_DIR,
        f"{dataset}_precision_curve.pdf"
    )

    plt.savefig(fig_path,bbox_inches="tight")
    plt.close()
print("Blocking curves generated")