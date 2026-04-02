import os
import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Amazon, Planetoid, Coauthor


############################################
# 全局参数
############################################

ROOT = "scratch/project/public_datasets"
RANDOM_STATE = 42


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
# 固定语义维度（核心修改）
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
# 提取语义特征
############################################

def extract_semantic_features(data):

    print("Original feature shape:", data.x.shape)

    # 转为 float32
    X = data.x.cpu().numpy().astype(np.float32)

    # 稀疏化
    X_sparse = csr_matrix(X)

    # TF-IDF
    print("Applying TF-IDF...")
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_sparse)

    # 固定维度
    dim = min(get_semantic_dimension(data), X_tfidf.shape[1] - 1)
    print("Using semantic dimension:", dim)

    # SVD
    print("Performing SVD...")
    svd = TruncatedSVD(n_components=dim, random_state=RANDOM_STATE)
    X_semantic = svd.fit_transform(X_tfidf)

    explained_var = svd.explained_variance_ratio_.sum()

    # 标准化
    scaler = StandardScaler()
    X_semantic = scaler.fit_transform(X_semantic)

    return torch.tensor(X_semantic, dtype=torch.float32), explained_var


############################################
# 语义桥接度
############################################

def compute_semantic_bridge_score(data, semantic):

    row, col = data.edge_index
    N = semantic.size(0)

    neighbor_sum = torch.zeros_like(semantic)
    neighbor_count = torch.zeros(N, device=semantic.device)

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

    for dataset_type, dataset_name in DATASETS:

        print("\n======================================")
        print(f"Processing {dataset_type} - {dataset_name}")

        save_path = os.path.join(
            ROOT,
            dataset_type,
            f"semantic_{dataset_name}.pt"
        )

        if os.path.exists(save_path):
            print("Semantic file already exists. Skipping.")
            continue

        data = load_dataset(dataset_type, dataset_name)

        semantic_features, explained_var = \
            extract_semantic_features(data)

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
        print("Explained variance:", round(explained_var, 4))


if __name__ == "__main__":
    main()