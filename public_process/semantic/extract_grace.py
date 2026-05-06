import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Amazon, Planetoid, Coauthor
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj

############################################
# 全局参数
############################################

ROOT = "scratch/project/public_datasets"
RANDOM_STATE = 42

# 设定计算设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################
# 数据集列表
############################################

DATASETS = [
    ("Amazon", "Computers"),
    ("Amazon", "Photo"),
    ("Coauthor", "CS"),
    ("Coauthor", "Physics"),
    ("Planetoid", "Cora"),
    ("Planetoid", "CiteSeer"),
    ("Planetoid", "PubMed"),
]


############################################
# 加载数据
############################################

def load_dataset(dataset_type, dataset_name):
    if dataset_type == "Amazon":
        dataset = Amazon(
            root=os.path.join(ROOT, "Amazon"),
            name=dataset_name
        )

    elif dataset_type == "Planetoid":
        dataset = Planetoid(
            root=os.path.join(ROOT, "Planetoid"),
            name=dataset_name
        )

    elif dataset_type == "Coauthor":
        dataset = Coauthor(
            root=os.path.join(ROOT, "Coauthor"),
            name=dataset_name
        )

    else:
        raise ValueError("Unsupported dataset type")

    return dataset[0]


############################################
# 固定语义维度
############################################

def get_semantic_dimension(data):
    num_nodes = data.num_nodes

    if num_nodes < 5000:
        return 16
    elif num_nodes < 20000:
        return 16
    else:
        return 16


############################################
# 图对比学习 (GRACE) 模型定义
############################################

class GraceEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraceEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        return self.conv2(x, edge_index)


def drop_feature(x, drop_prob):
    """特征掩码增广"""
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x_aug = x.clone()
    x_aug[:, drop_mask] = 0
    return x_aug


def info_nce_loss(z1, z2, temperature=0.5):
    """计算归一化温度缩放交叉熵损失 (NT-Xent)"""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # 计算点积相似度矩阵 (N x N)
    sim_matrix = torch.exp(torch.matmul(z1, z2.T) / temperature)

    # 提取对角线作为正样本对
    pos_sim = torch.diag(sim_matrix)

    # 计算损失：-log(正样本 / 所有样本)
    loss = -torch.log(pos_sim / sim_matrix.sum(dim=1))
    return loss.mean()


############################################
# 提取语义特征 (替换核心)
############################################

def extract_semantic_features(data, epochs=200, lr=0.001):
    print("Original feature shape:", data.x.shape)

    # 数据转移至 GPU/CPU
    x = data.x.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)

    # 获取目标输出维度
    out_dim = get_semantic_dimension(data)
    hidden_dim = out_dim * 2
    print(f"Target semantic dimension: {out_dim}")

    # 初始化编码器与优化器
    print("Training GRACE Contrastive Encoder...")
    encoder = GraceEncoder(x.size(1), hidden_dim, out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)

    encoder.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 视图 1：轻度扰动
        edge_index1, _ = dropout_adj(edge_index, p=0.2)
        x1 = drop_feature(x, 0.3)
        z1 = encoder(x1, edge_index1)

        # 视图 2：重度扰动
        edge_index2, _ = dropout_adj(edge_index, p=0.4)
        x2 = drop_feature(x, 0.4)
        z2 = encoder(x2, edge_index2)

        # 计算对称对比损失
        loss = info_nce_loss(z1, z2) + info_nce_loss(z2, z1)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1:03d}/{epochs}], Contrastive Loss: {loss.item():.4f}')

    # 推理模式提取最终特征
    encoder.eval()
    with torch.no_grad():
        semantic_features = encoder(x, edge_index)

        # 为了稳定下游，执行特征级标准化
        semantic_features = F.normalize(semantic_features, p=2, dim=1)

    # 图对比学习无法提供严谨的特征方差解释率，故返回占位符 0.0 以保持接口兼容
    return semantic_features.cpu(), 0.0


############################################
# 语义桥接度
############################################

def compute_semantic_bridge_score(data, semantic):
    # 保持原逻辑不变，在 CPU 上计算
    semantic = semantic.cpu()
    row, col = data.edge_index.cpu()
    N = semantic.size(0)

    neighbor_sum = torch.zeros_like(semantic)
    neighbor_count = torch.zeros(N)

    neighbor_sum.index_add_(0, row, semantic[col])
    neighbor_count.index_add_(0, row, torch.ones_like(row, dtype=torch.float))

    neighbor_count = neighbor_count.clamp(min=1).unsqueeze(1)
    mean_neighbor = neighbor_sum / neighbor_count

    diff = semantic - mean_neighbor
    score = torch.norm(diff, dim=1)

    return score


############################################
# 主流程
############################################

def main():
    # 设定随机种子保证可复现性
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    for dataset_type, dataset_name in DATASETS:

        print("\n======================================")
        print(f"Processing {dataset_type} - {dataset_name} on {DEVICE}")

        # 确保保存目录存在
        save_dir = os.path.join(ROOT, dataset_type)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"semantic_{dataset_name}.pt")

        if os.path.exists(save_path):
            print("Semantic file already exists. Skipping.")
            continue

        data = load_dataset(dataset_type, dataset_name)

        semantic_features, explained_var = \
            extract_semantic_features(data, epochs=200)

        bridge_score = compute_semantic_bridge_score(
            data, semantic_features
        )

        torch.save({
            "semantic_features": semantic_features,
            "explained_variance": explained_var,
            "semantic_bridge_score": bridge_score
        }, save_path)

        print("Saved to:", save_path)
        print("Semantic feature shape:", semantic_features.shape)


if __name__ == "__main__":
    main()