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
# 固定语义维度（核心修改）
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
# 提取语义特征
############################################

def extract_semantic_features(data, mode):

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

        X_sparse = csr_matrix(X)

        print("Applying TF-IDF...")
        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X_sparse)

        dim = min(get_semantic_dimension(data), X_tfidf.shape[1] - 1)
        print("Using semantic dimension:", dim)

        print("Performing SVD...")
        svd = TruncatedSVD(n_components=dim, random_state=RANDOM_STATE)
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
# 主流程
############################################

def main():

    for dataset_type, dataset_name in DATASETS:

        print("\n======================================")
        print(f"Processing Dataset: {dataset_type} - {dataset_name}")

        # 读取数据（只读取一次）
        data = load_dataset(dataset_type, dataset_name)

        for mode in SEMANTIC_MODES:

            print("\n--------------------------------------")
            print(f"Semantic Mode: {mode}")

            save_dir = get_extract_dir(dataset_name)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"semantic_{mode}.pt"
            )


            if os.path.exists(save_path):
                print("File already exists. Skipping.")
                continue

            # 提取语义特征
            semantic_features, explained_var = \
                extract_semantic_features(data, mode)

            # 只有完整方法才计算 bridge
            if mode == "tfidf_svd_bridge":

                bridge_score = compute_semantic_bridge_score(
                    data,
                    semantic_features
                )

            else:
                bridge_score = None

            # 保存
            torch.save({
                "semantic_features": semantic_features,
                "explained_variance": explained_var,
                "semantic_bridge_score": bridge_score
            }, save_path)

            print("Saved to:", save_path)
            print("Semantic feature shape:", semantic_features.shape)

            if explained_var is not None:
                print("Explained variance:", round(explained_var, 4))

if __name__ == "__main__":
    main()