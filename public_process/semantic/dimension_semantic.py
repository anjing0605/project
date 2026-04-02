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
EXTRACT_ROOT = "scratch/project/public_process/extract"
RANDOM_STATE = 42

# 只给 tfidf_svd_bridge 做多语义维度对比
BRIDGE_SEMANTIC_DIMS = [16, 32, 64, 128, 256]


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

SEMANTIC_MODES = [
    "raw",
    "tfidf",
    "tfidf_svd",
    "tfidf_svd_bridge"
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


def get_extract_dir(dataset_name):

    mapping = {
        "Cora": "Cora_struct",
        "CiteSeer": "CiteSeer_struct",
        "PubMed": "PubMed_struct",
        "Computers": "Computers_struct",
        "Photo": "Photo_struct",
        "CS": "CS_struct",
        "Physics": "Physics_struct"
    }

    return os.path.join(EXTRACT_ROOT, mapping[dataset_name])


############################################
# 固定语义维度（保留原逻辑，给 tfidf_svd 用）
############################################

def get_semantic_dimension(data):

    num_nodes = data.num_nodes

    if num_nodes < 5000:
        return 64
    elif num_nodes < 20000:
        return 128
    else:
        return 256


############################################
# 获取 tfidf_svd_bridge 可用维度
############################################

def get_valid_bridge_dims(data, candidate_dims):
    X = data.x.cpu().numpy().astype(np.float32)
    X_sparse = csr_matrix(X)

    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_sparse)

    # TruncatedSVD 要求 n_components < n_features
    max_dim = X_tfidf.shape[1] - 1

    valid_dims = [d for d in candidate_dims if d <= max_dim]

    if len(valid_dims) == 0:
        raise ValueError(
            f"No valid bridge dims. max allowed dim = {max_dim}, "
            f"candidates = {candidate_dims}"
        )

    return valid_dims, X_tfidf


############################################
# 提取语义特征
############################################

def extract_semantic_features(data, mode, semantic_dim=None, X_tfidf_cache=None):

    print("Original feature shape:", data.x.shape)

    X = data.x.cpu().numpy().astype(np.float32)

    if mode == "raw":

        X_feat = X
        explained_var = None

    elif mode == "tfidf":

        X_sparse = csr_matrix(X)

        print("Applying TF-IDF...")
        tfidf = TfidfTransformer()
        X_feat = tfidf.fit_transform(X_sparse).toarray()

        explained_var = None

    elif mode in ["tfidf_svd", "tfidf_svd_bridge"]:

        if X_tfidf_cache is None:
            X_sparse = csr_matrix(X)

            print("Applying TF-IDF...")
            tfidf = TfidfTransformer()
            X_tfidf = tfidf.fit_transform(X_sparse)
        else:
            X_tfidf = X_tfidf_cache

        # tfidf_svd 仍然使用原来的固定维度逻辑
        if semantic_dim is None:
            semantic_dim = min(get_semantic_dimension(data), X_tfidf.shape[1] - 1)

        if semantic_dim >= X_tfidf.shape[1]:
            raise ValueError(
                f"semantic_dim={semantic_dim} must be smaller than "
                f"num_features={X_tfidf.shape[1]}"
            )

        print("Using semantic dimension:", semantic_dim)

        print("Performing SVD...")
        svd = TruncatedSVD(n_components=semantic_dim, random_state=RANDOM_STATE)
        X_feat = svd.fit_transform(X_tfidf)

        explained_var = svd.explained_variance_ratio_.sum()

        scaler = StandardScaler()
        X_feat = scaler.fit_transform(X_feat)

    else:
        raise ValueError("Unknown mode")

    return torch.tensor(X_feat, dtype=torch.float32), explained_var


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
# 保存路径
############################################

def build_save_path(save_dir, mode, semantic_dim=None):
    if mode == "tfidf_svd_bridge" and semantic_dim is not None:
        filename = f"semantic_{mode}_dim{semantic_dim}.pt"
    else:
        filename = f"semantic_{mode}.pt"

    return os.path.join(save_dir, filename)


############################################
# 主流程
############################################

def main():

    for dataset_type, dataset_name in DATASETS:

        print("\n======================================")
        print(f"Processing Dataset: {dataset_type} - {dataset_name}")

        # 读取数据（只读取一次）
        data = load_dataset(dataset_type, dataset_name)

        save_dir = get_extract_dir(dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # 只为 tfidf_svd_bridge 多维度实验预先准备
        valid_bridge_dims, X_tfidf_cache = get_valid_bridge_dims(
            data,
            BRIDGE_SEMANTIC_DIMS
        )
        print("Valid bridge dims:", valid_bridge_dims)

        for mode in SEMANTIC_MODES:

            print("\n--------------------------------------")
            print(f"Semantic Mode: {mode}")

            # raw / tfidf / tfidf_svd 保持单次运行
            if mode in ["raw", "tfidf", "tfidf_svd"]:

                save_path = build_save_path(save_dir, mode)

                if os.path.exists(save_path):
                    print("File already exists. Skipping.")
                    continue

                semantic_features, explained_var = extract_semantic_features(
                    data,
                    mode
                )

                if mode == "tfidf_svd_bridge":
                    bridge_score = compute_semantic_bridge_score(
                        data,
                        semantic_features
                    )
                else:
                    bridge_score = None

                if mode == "tfidf_svd":
                    semantic_dim_value = min(
                        get_semantic_dimension(data),
                        data.x.shape[1] - 1
                    )
                else:
                    semantic_dim_value = None

                torch.save({
                    "dataset_type": dataset_type,
                    "dataset_name": dataset_name,
                    "semantic_mode": mode,
                    "semantic_dim": semantic_dim_value,
                    "semantic_features": semantic_features,
                    "explained_variance": explained_var,
                    "semantic_bridge_score": bridge_score
                }, save_path)

                print("Saved to:", save_path)
                print("Semantic feature shape:", semantic_features.shape)

                if explained_var is not None:
                    print("Explained variance:", round(explained_var, 4))

            # 只有 tfidf_svd_bridge 跑多个维度
            elif mode == "tfidf_svd_bridge":

                for semantic_dim in valid_bridge_dims:

                    print(f"\n  >>> Bridge Semantic Dim: {semantic_dim}")

                    save_path = build_save_path(save_dir, mode, semantic_dim)

                    if os.path.exists(save_path):
                        print("  File already exists. Skipping.")
                        continue

                    semantic_features, explained_var = extract_semantic_features(
                        data,
                        mode,
                        semantic_dim=semantic_dim,
                        X_tfidf_cache=X_tfidf_cache
                    )

                    bridge_score = compute_semantic_bridge_score(
                        data,
                        semantic_features
                    )

                    torch.save({
                        "dataset_type": dataset_type,
                        "dataset_name": dataset_name,
                        "semantic_mode": mode,
                        "semantic_dim": semantic_dim,
                        "semantic_features": semantic_features,
                        "explained_variance": explained_var,
                        "semantic_bridge_score": bridge_score
                    }, save_path)

                    print("  Saved to:", save_path)
                    print("  Semantic feature shape:", semantic_features.shape)
                    print("  Explained variance:", round(explained_var, 4))

            else:
                raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()