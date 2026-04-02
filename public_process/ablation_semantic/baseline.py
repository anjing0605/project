# baseline.py

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.nn import GCNConv, SAGEConv,GATConv
from scipy.stats import spearmanr, kendalltau


# ============================================================
# 1️⃣ 统一划分 mask（完全公平）
# ============================================================



# ============================================================
# 2️⃣ 评价指标（只在 test 上）
# ============================================================

def compute_metrics(pred, true):

    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    else:
        pred = np.asarray(pred)

    if torch.is_tensor(true):
        true = true.detach().cpu().numpy()
    else:
        true = np.asarray(true)

    pred = pred.reshape(-1)
    true = true.reshape(-1)

    sp = spearmanr(pred, true).correlation
    kd = kendalltau(pred, true).correlation

    def topk_overlap(k):
        p_top = set(np.argsort(-pred)[:k])
        t_top = set(np.argsort(-true)[:k])
        return len(p_top & t_top) / k

    return {
        "SP": float(sp),
        "KD": float(kd),
        "Top10": topk_overlap(10),
        "Top20": topk_overlap(20)
    }



# ============================================================
# 3️⃣ 传统图指标 Baselines（无需训练）
# ============================================================

def pyg_to_nx(data):
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index.T)
    return G


def baseline_degree(data):
    G = pyg_to_nx(data)
    scores = np.array([G.degree(i) for i in range(data.num_nodes)])
    return torch.tensor(scores, dtype=torch.float)


def baseline_pagerank(data):
    G = pyg_to_nx(data)
    pr = nx.pagerank(G)
    scores = np.array([pr[i] for i in range(data.num_nodes)])
    return torch.tensor(scores, dtype=torch.float)


def baseline_kshell(data):
    G = pyg_to_nx(data)
    core = nx.core_number(G)
    scores = np.array([core[i] for i in range(data.num_nodes)])
    return torch.tensor(scores, dtype=torch.float)


# ============================================================
# 4️⃣ 统一 GNN 训练框架
# ============================================================

class MLPBaseline(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class GCNBaseline(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.lin = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x).squeeze(1)


class SAGEBaseline(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, 128)
        self.conv2 = SAGEConv(128, 64)
        self.lin = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x).squeeze(1)

class GATBaseline(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.conv1 = GATConv(in_dim, 128, heads=4, concat=True)
        self.conv2 = GATConv(128*4, 64, heads=4, concat=True)

        self.lin = nn.Linear(64*4, 1)

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        return self.lin(x).squeeze(1)
# ============================================================
# 5️⃣ 统一 EarlyStop + Spearman 选优
# ============================================================

def train_model(model, data, mask, device,
                max_epoch=10000,
                patience=200):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_sp = -1
    best_state = None
    counter = 0

    x = data.x.to(device)
    edge = data.edge_index.to(device)
    y = data.y_sir.to(device)

    for epoch in range(max_epoch):

        model.train()
        optimizer.zero_grad()

        if isinstance(model, MLPBaseline):
            pred = model(x)
        else:
            pred = model(x, edge)

        loss = F.mse_loss(
            pred[mask["train"]],
            y[mask["train"]]
        )

        loss.backward()
        optimizer.step()

        # ===== 验证 =====
        model.eval()
        with torch.no_grad():
            if isinstance(model, MLPBaseline):
                val_pred = model(x)
            else:
                val_pred = model(x, edge)

            sp = spearmanr(
                val_pred[mask["val"]].cpu(),
                y[mask["val"]].cpu()
            ).correlation

        if sp > best_sp:
            best_sp = sp
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if isinstance(model, MLPBaseline):
            final_pred = model(x)
        else:
            final_pred = model(x, edge)

    return final_pred.cpu()


# ============================================================
# 6️⃣ 主入口（与你的调用完全兼容）
# ============================================================

def run_all_baselines(data, model_ours, device):

    mask = {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask
    }

    results = {}
    true = data.y_sir[mask["test"]]
    # ----- 传统方法 -----
    for name, func in {
        "Degree": baseline_degree,
        "PR": baseline_pagerank,
        "K-shell": baseline_kshell
    }.items():

        pred = func(data)[mask["test"]]
        results[name] = compute_metrics(pred, true)

    # ----- MLP -----
    pred = train_model(
        MLPBaseline(data.x.size(1)),
        data, mask, device
    )[mask["test"]]
    results["MLP"] = compute_metrics(pred, true)

    # ----- GCN -----
    pred = train_model(
        GCNBaseline(data.x.size(1)),
        data, mask, device
    )[mask["test"]]
    results["GCN"] = compute_metrics(pred, true)

    # ----- SAGE -----
    pred = train_model(
        SAGEBaseline(data.x.size(1)),
        data, mask, device
    )[mask["test"]]
    results["SAGE"] = compute_metrics(pred, true)
    # ----- GAT -----
    pred = train_model(
        GATBaseline(data.x.size(1)),
        data, mask, device
    )[mask["test"]]

    results["GAT"] = compute_metrics(pred, true)
    '''
    sage:
    # ----- Ours -----
    model_ours.eval()
    with torch.no_grad():
        pred = model_ours(data.to(device)).mean(dim=1).cpu()
        #pred = model_ours(data.to(device)).squeeze().cpu()

    pred = pred[mask["test"]]
    results["Ours"] = compute_metrics(pred, true)
    '''
    # ----- Ours -----
    model_ours.eval()
    data_gpu = data.to(device)

    with torch.no_grad():
        pred = model_ours(data_gpu)

    if pred.dim() == 2:
        pred = pred.mean(dim=1)
    pred = pred.cpu().numpy()
    true = data.y_sir.cpu().numpy()

    test_mask = mask["test"].cpu().numpy()

    pred = pred[test_mask]
    true = true[test_mask]

    results["Ours"] = compute_metrics(pred, true)
    return results


# ============================================================
# 7️⃣ 打印表格
# ============================================================

def print_results(results):

    print(f"{'Method':<10} {'SP':<8} {'KD':<8} {'Top10':<8} {'Top20':<8}")

    for method, m in results.items():
        print(
            f"{method:<10} "
            f"{m['SP']:.4f}   "
            f"{m['KD']:.4f}   "
            f"{m['Top10']:.4f}   "
            f"{m['Top20']:.4f}"
        )