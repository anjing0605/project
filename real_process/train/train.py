import os
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv,SAGEConv
from scipy.stats import spearmanr,kendalltau
import torch_geometric
import json
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
from baseline import run_all_baselines, print_results
torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
import copy
config = {
    "w_g_reg":  2.5,   # 保证整体影响力稳定
    "w_g_rank": 3.5,

    "w_b_reg":  0.1,   # 学传播曲线形状
    "w_b_rank": 0.5,

    "w_mono": 0.05,
    "w_topk":4.0,
    "use_struct": True,
    "use_perf": False,
    "use_semantic":True,
    # ===== architecture control =====
    "use_graph": True,
    "num_layers": 2,
    "multi_beta": True,
}

def safe_torch_load(pth):
    """
    安全加载 PyTorch 对象，允许 PyG 的 DataEdgeAttr 反序列化
    """
    try:
        # PyG edge attr safe loading if available
        from torch_geometric.data.data import DataEdgeAttr
        with torch.serialization.safe_globals([DataEdgeAttr]):
            return torch.load(pth, weights_only=False)
    except Exception:
        return torch.load(pth, map_location='cpu')

def canon_node_entity(node_label: str) -> str:
    """
    将图节点标签规范化为 KG 实体标签格式。
    例如:
      9      -> node_9
      "9"    -> node_9
      "9.0"  -> node_9
      "node9" -> node_9
      "node_9" -> node_9
    """
    s = str(node_label).strip()

    if s.startswith("node_"):
        return s

    # 处理 "9" / "9.0"
    try:
        v = float(s)
        if v.is_integer():
            return f"node_{int(v)}"
    except Exception:
        pass

    # 处理 "node9"
    if s.startswith("node") and s[4:].isdigit():
        return f"node_{int(s[4:])}"

    return s

def candidate_entity_keys(node_value, node_idx, dataset_name=None):
    """
    针对一个图节点，生成一组可能对应的 KG 实体名候选。
    """
    cands = []

    raw = str(node_value).strip()
    cands.append(raw)

    ds = str(dataset_name).strip() if dataset_name is not None else ""
    ds_no_us = ds.replace("_", "")   # run_1 -> run1

    # 若是 0.0 / 1.0 这种，可额外尝试 0 / 1
    try:
        fv = float(raw)
        if fv.is_integer():
            iv = int(fv)

            cands.extend([
                str(iv),                    # "0"
                f"{fv:.1f}",                # "0.0"
                f"node_{iv}",               # "node_0"
                f"node{iv}",                # "node0"
                f"{ds}_node_{iv}" if ds else "",     # "run_1_node_0"
                f"{ds}_node{iv}" if ds else "",      # "run_1_node0"
                f"{ds_no_us}_node_{iv}" if ds_no_us else "",  # "run1_node_0"
                f"{ds_no_us}_node{iv}" if ds_no_us else "",   # "run1_node0"
                f"run1_node_{iv}",          # "run1_node_0"
                f"run1_node{iv}",           # "run1_node0"
                f"run_1_node_{iv}",         # "run_1_node_0"
                f"run_1_node{iv}",          # "run_1_node0"
            ])
    except Exception:
        pass

    # 再兜底尝试 node_idx 对应形式
    cands.extend([
        str(node_idx),
        f"{float(node_idx):.1f}",
        f"node_{node_idx}",
        f"node{node_idx}",
        f"{ds}_node_{node_idx}" if ds else "",
        f"{ds}_node{node_idx}" if ds else "",
        f"{ds_no_us}_node_{node_idx}" if ds_no_us else "",
        f"{ds_no_us}_node{node_idx}" if ds_no_us else "",
        f"run1_node_{node_idx}",
        f"run1_node{node_idx}",
        f"run_1_node_{node_idx}",
        f"run_1_node{node_idx}",
    ])

    # 去重并去掉空串
    out = []
    seen = set()
    for x in cands:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out
def load_run1_kg_semantic_aligned(root_dir, df, dataset_name="run_1"):
    """
    从 run_1 的 kg_embeddings.npz 中读取:
      - entity_embeddings
      - entity_mapping
    再结合当前 df 的 node/node_idx，将实体嵌入重排为按 node_idx 排列的语义特征矩阵。
    """
    sem_path = os.path.join(root_dir, "kg_embeddings.npz")
    if not os.path.exists(sem_path):
        raise RuntimeError("kg_embeddings.npz 不存在")

    sem_npz = np.load(sem_path, allow_pickle=True)

    # ---------- 1) 读取 entity_embeddings ----------
    if "entity_embeddings" not in sem_npz:
        raise RuntimeError(
            f"kg_embeddings.npz 缺少 entity_embeddings，现有键: {list(sem_npz.keys())}"
        )

    entity_embeddings = sem_npz["entity_embeddings"]

    # 若是 ComplEx / complex dtype，保留更多信息而不是直接丢弃虚部
    if np.iscomplexobj(entity_embeddings):
        # 方案A：拼接实部和虚部，信息更完整
        entity_embeddings = np.concatenate(
            [entity_embeddings.real, entity_embeddings.imag], axis=1
        )

    entity_embeddings = np.asarray(entity_embeddings, dtype=np.float32)

    # ---------- 2) 读取 entity_mapping ----------
    if "entity_mapping" not in sem_npz:
        raise RuntimeError(
            f"kg_embeddings.npz 缺少 entity_mapping，现有键: {list(sem_npz.keys())}"
        )

    raw_mapping = sem_npz["entity_mapping"]

    entity_to_id = {}
    try:
        for item in raw_mapping:
            if len(item) != 2:
                continue
            k, v = item[0], item[1]
            entity_to_id[str(k)] = int(v)
    except Exception as e:
        raise RuntimeError(f"entity_mapping 解析失败: {e}")

    if len(entity_to_id) == 0:
        raise RuntimeError("entity_mapping 解析后为空")

    # ---------- 3) 按 node_idx 重排 ----------
    num_nodes = len(df)
    emb_dim = entity_embeddings.shape[1]
    semantic = np.zeros((num_nodes, emb_dim), dtype=np.float32)

    matched = 0
    missed_nodes = []
    matched_examples = []

    print("\n===== KG Entity Mapping Debug =====")
    print("entity_embeddings.shape =", entity_embeddings.shape)
    print("entity_mapping sample =", list(entity_to_id.keys())[:30])
    print("df node sample =", df["node"].head(10).tolist())
    print("df node_idx sample =", df["node_idx"].head(10).tolist())
    print("===================================\n")

    for _, row in df.iterrows():
        node_idx = int(row["node_idx"])
        raw_node = row["node"]

        ent_idx = None
        used_key = None

        for cand in candidate_entity_keys(raw_node, node_idx, dataset_name=dataset_name):
            if cand in entity_to_id:
                ent_idx = entity_to_id[cand]
                used_key = cand
                break

        if ent_idx is None:
            missed_nodes.append(str(raw_node))
            continue

        if not (0 <= ent_idx < entity_embeddings.shape[0]):
            raise RuntimeError(
                f"实体索引越界: raw_node={raw_node}, used_key={used_key}, ent_idx={ent_idx}, "
                f"entity_embeddings.shape[0]={entity_embeddings.shape[0]}"
            )

        semantic[node_idx] = entity_embeddings[ent_idx]
        matched += 1

        if len(matched_examples) < 10:
            matched_examples.append((str(raw_node), node_idx, used_key, ent_idx))

    print("\n===== KG Semantic Align Debug =====")
    print("entity_embeddings.shape =", entity_embeddings.shape)
    print("num_nodes =", num_nodes)
    print("matched =", matched)
    print("missed =", len(missed_nodes))
    print("matched examples =", matched_examples)
    if len(missed_nodes) > 0:
        print("missed sample =", missed_nodes[:10])
    print("===================================\n")

    if matched == 0:
        raise RuntimeError(
            "KG 语义特征对齐失败：一个节点都没有匹配到 entity_mapping。"
        )

    return semantic
class SlimRunDataset(Dataset):
    """
    适配真实网络的单图 SIR Dataset
    保持 Dataset 接口，但内部只有 1 个图
    """

    def __init__(self,
                 root_dir,
                 dataset_name=None,
                 use_struct=True,
                 use_perf=False,
                 use_semantic=True,
                 seed=0
                 ):

        super().__init__(root=None, transform=None)

        if not (use_struct or use_perf or use_semantic):
            raise ValueError("至少启用一种特征")
        self.seed = seed

        self.use_struct = use_struct
        self.use_perf = use_perf
        self.root = root_dir
        self.dataset_name = dataset_name
        self.use_semantic = use_semantic

        feat_tag = f"s{int(self.use_struct)}_p{int(self.use_perf)}_k{int(self.use_semantic)}"
        self.graph_path = os.path.join(self.root, f"processed_sir/real_graph_{feat_tag}.pt")

        self._process_single()   # 不要做 exists 判断，单图处理成本很低
    
        self.data = safe_torch_load(self.graph_path)

    # ===================== 单图构建 =====================
    def _process_single(self):

        node_file = os.path.join(self.root, "node_features.csv")
        edge_file = os.path.join(self.root, "edgelist.csv")
        sir_file  = os.path.join(self.root, "sir_node_labels.csv")
        phase_file = os.path.join(self.root, "sir_phase_curve.csv")
        for f in [node_file, edge_file, sir_file, phase_file]:
            if not os.path.exists(f):
                raise RuntimeError(f"{f} 不存在，请检查数据集目录")

        # ===================== 读取节点特征 =====================
        df = pd.read_csv(node_file)

        print("\n===== Node Feature Debug =====")
        print("Loaded node file path:", node_file)
        print("node max =", df["node"].max())
        print("node min =", df["node"].min())
        print("node unique =", df["node"].nunique())
        print("len(df) =", len(df))
        print("==============================\n")

        # ===================== 节点编号对齐 =====================
        # 其他数据集：保留原来的 node -> node_index_map -> node_idx 逻辑
        # run_1：node_features.csv 中的 node 已经是 0,1,2,... 形式的节点编号，直接作为 node_idx

        if self.dataset_name == "run_1":
            # run_1 的 node 已经是节点编号，直接使用
            try:
                df["node_idx"] = df["node"].astype(float).astype(int)
            except Exception as e:
                raise RuntimeError(f"run_1 的 node 列无法直接转成整数编号: {e}")

        else:
            # 其他数据集沿用原逻辑
            df["node"] = df["node"].astype(str)

            with open(os.path.join(self.root, "node_index_map.json")) as f:
                node_index_map = json.load(f)

            df["node_idx"] = df["node"].map(node_index_map)

            if df["node_idx"].isna().any():
                unmapped = df.loc[df["node_idx"].isna(), "node"].tolist()
                raise RuntimeError(
                    f"存在未映射到 index 的 node，前20个未匹配值: {unmapped[:20]}"
                )

            df["node_idx"] = df["node_idx"].astype(int)

        # ===================== 读取 SIR 标签 =====================
        sir_df = pd.read_csv(sir_file)

        # sir 文件里的 node 本来就是 index
        sir_df["node_idx"] = sir_df["node"].astype(int)
        sir_df = sir_df.drop(columns=["node"])
        # node_features.csv 里若已有监督标签列，先删除，避免 merge 后变成 soft_label_x / soft_label_y
        for col in ["soft_label", "true_label", "run_split", "mask"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        # ===================== 正确 merge（index 对 index）=====================
        df = df.merge(sir_df, on="node_idx", how="left")

        # 验证是否成功匹配
        if "soft_label" not in df.columns:
            raise RuntimeError("缺少 soft_label 列，SIR 文件格式错误")

        if df["soft_label"].isna().all():
            raise RuntimeError("soft_label 全为 NaN，node 对齐失败")
        beta_cols = [c for c in df.columns if c.startswith("spread_beta_")]
        if len(beta_cols) == 0:
            raise RuntimeError("未找到 spread_beta_* 列，SIR 文件格式错误")

        # ===================== 节点顺序对齐 =====================
        df = df.sort_values("node_idx").reset_index(drop=True)
        # ===================== 重新编号（解决不连续问题） =====================
        old_ids = df["node_idx"].values.copy()
        new_id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids)}

        df["node_idx"] = df["node_idx"].map(new_id_map)        

        # ===================== 读取 β 相变曲线 =====================
        phase_df = pd.read_csv(phase_file)

        beta_list = phase_df["beta"].values.astype(np.float32)
        avg_spread = phase_df["avg_spread"].values.astype(np.float32)

        # ===================== SIR 多β标签 =====================
        beta_cols = [c for c in df.columns if c.startswith("spread_beta_")]
        y_sir_multi = df[beta_cols].values.astype(np.float32)
        self.num_beta = len(beta_cols)

        # ---- 1️⃣ 自动寻找相变点 βc（斜率最大）----
        d_spread = np.gradient(avg_spread, beta_list)
        beta_c = beta_list[np.argmax(d_spread)]

        # ---- 2️⃣ 相变区加权（增强临界传播差异）----
        sigma = 0.18 * beta_c
        weights = np.exp(-((beta_list - beta_c)**2) / (2 * sigma**2))
        weights = weights / weights.sum()


        # ---- 3️⃣ 再做标准化（保持监督尺度稳定）----
        y_sir_multi = (y_sir_multi - y_sir_multi.mean(0)) / (y_sir_multi.std(0) + 1e-6)
        y_sir_multi = y_sir_multi * weights   # ⭐ 先加权
        y_sir_multi = np.nan_to_num(y_sir_multi, nan=0.0)

        # ===================== 主监督标签 =====================
        y_sir = df["soft_label"].values.astype(np.float32)
        labeled_mask_sir = ~np.isnan(y_sir)
        # ===================== Train / Val / Test Split =====================
        num_nodes = len(df)
        all_idx = np.where(labeled_mask_sir)[0]

        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_idx)

        n = len(all_idx)
        n_train = int(n * 0.6)
        n_val   = int(n * 0.2)

        train_idx = all_idx[:n_train]
        val_idx   = all_idx[n_train:n_train+n_val]
        test_idx  = all_idx[n_train+n_val:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
 
        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        test_mask[test_idx]   = True

        # ===================== 排名监督 =====================
        rank_sir = np.full(len(df), -1, dtype=np.int64)
        valid = np.where(labeled_mask_sir)[0]
        if len(valid) > 1:
            order = np.argsort(-y_sir[valid])
            for r, i in enumerate(valid[order]):
                rank_sir[i] = r


        # ===================== 节点特征 =====================
        feat_list = []
        start = 0
        modal_slices = {}

        if self.use_perf:
            perf = np.load(os.path.join(self.root, "x_perf.npy"))
            feat_list.append(perf)
            modal_slices["perf"] = torch.tensor([start, start + perf.shape[1]])
            start += perf.shape[1]

        if self.use_struct:
            struct = np.load(os.path.join(self.root, "x_struct.npy"))
            if struct.shape[0] != len(df):
                raise RuntimeError(
                    f"struct 行数 {struct.shape[0]} 与节点数 {len(df)} 不一致"
            )
            feat_list.append(struct)
            modal_slices["struct"] = torch.tensor([start, start + struct.shape[1]])
            start += struct.shape[1]
        
        if self.use_semantic:
            if self.dataset_name is None:
                raise ValueError("dataset_name 必须指定")

            # run_1 使用 kg_embeddings.npz，并按 entity_mapping 重排到 node_idx 顺序
            if self.dataset_name == "run_1":
                semantic = load_run1_kg_semantic_aligned(self.root, df, dataset_name=self.dataset_name)

            else:
                # 公共数据集仍使用 semantic_{name}.pt
                sem_file = f"semantic_{self.dataset_name}.pt"
                sem_path = os.path.join(self.root, sem_file)

                if not os.path.exists(sem_path):
                    raise RuntimeError(f"{sem_file} 不存在")

                sem_data = safe_torch_load(sem_path)
                semantic = sem_data["semantic_features"].numpy()

                node_ids = df["node"].to_numpy(dtype=np.int64)
                semantic = semantic[node_ids]

            feat_list.append(semantic)
            modal_slices["semantic"] = torch.tensor([start, start + semantic.shape[1]])
            start += semantic.shape[1]
        x = torch.from_numpy(np.concatenate(feat_list, axis=1)).float()
        print("\n===== X Tensor Debug =====")
        print("x.shape =", x.shape)
        print("==========================\n")

        # ===================== 图结构 =====================
        edges_df = pd.read_csv(edge_file)
        # 使用新编号映射 edge
        edges_df["src"] = edges_df["src"].map(new_id_map)
        edges_df["dst"] = edges_df["dst"].map(new_id_map)

        # 删除不存在于 feature 中的节点
        edges_df = edges_df.dropna().copy()

        edges_df["src"] = edges_df["src"].astype(int)
        edges_df["dst"] = edges_df["dst"].astype(int)

        edge_index = torch.tensor(
            edges_df[["src", "dst"]].values.T,
            dtype=torch.long
        ).contiguous()

        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        print("\n===== Edge vs X Debug =====")
        print("x.shape[0] =", x.shape[0])
        print("edge_index max =", edge_index.max().item())
        print("edge_index min =", edge_index.min().item())
        print("===========================\n")

        assert edge_index.max().item() < x.shape[0], "❌ edge_index 超出 x 范围"
        # ===================== 调试：检查节点范围 =====================
        num_nodes = len(df)

        print("\n========== Graph Debug Info ==========")
        print("num_nodes (len(df)) =", num_nodes)
        print("max node id in feature =", df["node"].max())
        print("min node id in feature =", df["node"].min())
        print("max edge src =", edges_df["src"].max())
        print("min edge src =", edges_df["src"].min())
        print("max edge dst =", edges_df["dst"].max())
        print("min edge dst =", edges_df["dst"].min())
        print("edge_index max =", edge_index.max().item())
        print("edge_index min =", edge_index.min().item())
        print("======================================\n")

        assert edge_index.max().item() < num_nodes, "❌ edge_index 存在越界节点"
        assert edge_index.min().item() >= 0, "❌ edge_index 存在负编号"

        # ===================== PyG Data =====================
        data = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=len(df),

            y_sir=torch.tensor(y_sir, dtype=torch.float),
            y_sir_multi=torch.tensor(y_sir_multi, dtype=torch.float),

            mask_sir=torch.from_numpy(labeled_mask_sir).bool(),
            rank_sir=torch.tensor(rank_sir, dtype=torch.long),
            modal_slices=modal_slices,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_beta=self.num_beta,
        )
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)


        torch.save(data, self.graph_path)

    # Dataset 仍然保持接口，但长度=1
    def __len__(self):
        return 1

    def __getitem__(self, idx):
         return self.data

class MultiHeadModalFusion(nn.Module):
    def __init__(self, dim, num_heads=4, max_modal=3):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # 模态类型编码 (关键！！！)
        self.modal_embed = nn.Parameter(torch.randn(max_modal, dim))

        self.ln = nn.LayerNorm(dim)

    def forward(self, modal_list):
        x_list = [m for m in modal_list if m is not None]
        M = len(x_list)
        if M == 0:
            raise ValueError("No modal input")

        N, D = x_list[0].shape
        x = torch.stack(x_list, dim=1)  # (N, M, D)

        # ===== 加模态类型编码 =====
        x = x + self.modal_embed[:M]

        # ===== QKV =====
        qkv = self.qkv(x)  # (N, M, 3D)
        qkv = qkv.reshape(N, M, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (N, M, heads, head_dim)

        q = q.transpose(1, 2)  # (N, heads, M, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ===== Attention across MODALS =====
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (N, heads, M, head_dim)
        out = out.transpose(1, 2).reshape(N, M, D)

        # ===== 模态聚合 =====
        out = out.mean(dim=1)  # (N, D)

        # ===== 残差 + LN =====
        residual = x.mean(dim =1)
        out = self.proj(out)
        out = self.ln(out+residual)

        return out

def modal_norm(x: torch.Tensor):
    if x is None:
        return None
    if not torch.is_tensor(x):
        return None
    if x.numel() == 0:
        return x
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std

class SlimMultiGAT(nn.Module):
    """
    Configurable Multi-Modal Dual-Encoder GAT
    Modal switches:
        use_perf
        use_struct
        use_semantic
    Graph switches:
        use_graph
        num_layers
    """

    def __init__(self,
                 perf_dim=3,
                 struct_dim=5,
                 semantic_dim=0,
                 hidden_dim=128,
                 modal_dim=128,
                 fusion_heads=4,
                 out_dim=1,
                 config=None):

        super().__init__()
        self.config = config or {}

        # ========= 控制开关 =========
        self.use_graph = self.config.get("use_graph", True)
        self.num_layers = self.config.get("num_layers", 3)
        self.multi_beta = self.config.get("multi_beta", True)

        # ===== Modal Switches =====
        self.use_perf = self.config.get("use_perf", False)
        self.use_struct = self.config.get("use_struct", True)
        self.use_semantic = self.config.get("use_semantic", True)

        self.perf_dim = perf_dim
        self.struct_dim = struct_dim
        self.semantic_dim = semantic_dim
        self.modal_dim = modal_dim
        self.hidden_dim = hidden_dim
        self.dropout = self.config.get("dropout", 0.3)

        # ================= Modal Encoders =================
        def make_encoder(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, modal_dim)
            )

        if self.use_perf:
            self.perf_encoder = make_encoder(perf_dim)

        if self.use_struct:
            self.struct_encoder = make_encoder(struct_dim)

        if self.use_semantic and semantic_dim > 0:
            self.semantic_encoder = make_encoder(semantic_dim)

        # 统计有效模态数
        self.num_modal = 0
        if self.use_perf: self.num_modal += 1
        if self.use_struct: self.num_modal += 1
        if self.use_semantic and semantic_dim > 0:
            self.num_modal += 1

        # ================= Fusion (only if >1 modal) =================
        if self.num_modal > 1:
            self.fusion = MultiHeadModalFusion(
                dim=modal_dim,
                num_heads=fusion_heads
            )
        else:
            self.fusion = None
            self.single_modal_proj = nn.Identity()

        # =====================================================
        # ================= Dual GAT Backbone =================
        # =====================================================

        gat_heads = 4
        gat_hidden = hidden_dim // gat_heads
        # ---- Perf Branch ----
        if self.use_perf and perf_dim > 0:
            self.perf_gats = nn.ModuleList([
                GATConv(
                    modal_dim if i == 0 else hidden_dim,
                    gat_hidden,
                    heads=gat_heads,
                    concat=True,
                    dropout=self.dropout
                )
                for i in range(self.num_layers)
            ])
        # ---- Structure Branch ----
        if self.use_struct:
            self.struct_gats = nn.ModuleList([
                GATConv(
                    modal_dim if i == 0 else hidden_dim,
                    gat_hidden,
                    heads=gat_heads,
                    concat=True,
                    dropout=self.dropout
                )
                for i in range(self.num_layers)
            ])

        # ---- Semantic Branch ----
        if self.use_semantic and semantic_dim > 0:
            self.semantic_gats = nn.ModuleList([
                GATConv(
                    modal_dim if i == 0 else hidden_dim,
                    gat_hidden,
                    heads=gat_heads,
                    concat=True,
                    dropout=self.dropout
                )
                for i in range(self.num_layers)
            ])

        # ---- Branch Fusion ----
        self.branch_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        # ---- MLP fallback ----
        self.mlp = nn.Sequential(
            nn.Linear(modal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # ================= Output Head =================
        if self.multi_beta:
            self.num_beta = self.config.get("num_beta")
        else:
            self.num_beta = 1

        self.layernorm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_beta)
        )

        self._init_weights()

    # ================= Init =================
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ================= Modal Split =================
    def _split_modal(self, x, data):
        perf = struct = semantic = None

        if self.use_perf and "perf" in data.modal_slices:
            s, e = data.modal_slices["perf"].tolist()
            perf = x[:, s:e]

        if self.use_struct and "struct" in data.modal_slices:
            s, e = data.modal_slices["struct"].tolist()
            struct = x[:, s:e]

        if self.use_semantic and "semantic" in data.modal_slices:
            s, e = data.modal_slices["semantic"].tolist()
            semantic = x[:, s:e]

        return perf, struct, semantic

    # ================= Forward =================
    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        perf, struct, semantic = self._split_modal(x, data)

        # =====================================================
        # 1️⃣ Modal Encoding（只编码一次）
        # =====================================================

        perf_emb = None
        struct_emb = None
        semantic_emb = None

        if self.use_perf and perf is not None:
            perf_emb = self.perf_encoder(perf)

        if self.use_struct and struct is not None:
            struct_emb = self.struct_encoder(struct)

        if self.use_semantic and semantic is not None:
            semantic_emb = self.semantic_encoder(semantic)

        # 收集可用模态
        modal_list = [m for m in [perf_emb, struct_emb, semantic_emb] if m is not None]

        if len(modal_list) == 0:
            raise ValueError("No valid modal input")

        # =====================================================
        # 2️⃣ Early Fusion
        # =====================================================

        if self.fusion is not None and len(modal_list) > 1:
            x = self.fusion(modal_list)
        else:
            x = modal_list[0]

        x = F.dropout(x, p=self.dropout, training=self.training)

        # =====================================================
        # 3️⃣ Graph Branch
        # =====================================================

        if (not self.use_graph) or self.num_layers == 0:
            h = self.mlp(x)
            self.last_modal_attn = None

        else:

            perf_out = None
            struct_out = None
            semantic_out = None

            # ---- Perf Branch ----
            if perf_emb is not None and hasattr(self, "perf_gats"):
                p = perf_emb
                for gat in self.perf_gats:
                    p_new = F.elu(gat(p, edge_index))
                    p_new = F.dropout(p_new, p=self.dropout, training=self.training)

                    if p_new.shape == p.shape:
                        p = p + p_new
                    else:
                        p = p_new
                perf_out = p

            # ---- Structure Branch ----
            if struct_emb is not None and hasattr(self, "struct_gats"):
                s = struct_emb
                for gat in self.struct_gats:
                    s_new = F.elu(gat(s, edge_index))
                    s_new = F.dropout(s_new, p=self.dropout, training=self.training)

                    if s_new.shape == s.shape:
                        s = s + s_new
                    else:
                        s = s_new
                struct_out = s

            # ---- Semantic Branch ----
            if semantic_emb is not None and hasattr(self, "semantic_gats"):
                m = semantic_emb
                for gat in self.semantic_gats:
                    m_new = F.elu(gat(m, edge_index))
                    m_new = F.dropout(m_new, p=self.dropout, training=self.training)

                    if m_new.shape == m.shape:
                        m = m + m_new
                    else:
                        m = m_new
                semantic_out = m

            # ---- Branch-level Fusion ----
            branch_list = [b for b in [perf_out, struct_out, semantic_out] if b is not None]

            if len(branch_list) >= 2:
                stacked = torch.stack(branch_list, dim=1)   # [N, M, H]
                fused, attn_weights = self.branch_fusion(stacked, stacked, stacked)
                graph_h = fused.mean(dim=1)
                self.last_modal_attn = attn_weights.detach()
            elif len(branch_list) == 1:
                graph_h = branch_list[0]
                self.last_modal_attn = None
            else:
                graph_h = x
                self.last_modal_attn = None

            # 保留 early fusion 信息，避免图分支完全覆盖原始模态交互
            h = graph_h + x

        # =====================================================
        # 4️⃣ Output
        # =====================================================

        h = self.layernorm(h)
        self.last_hidden = h.detach()

        #out = self.classifier(h).squeeze(-1)
        out = self.classifier(h)

        return out

def sir_huber_loss(pred, target, mask, delta=1.0):
    mask = mask.unsqueeze(1).expand_as(target)
    diff = pred - target
    diff = diff[mask]

    abs_diff = diff.abs()
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))
    lin  = abs_diff - quad
    return (0.5 * quad**2 + delta * lin).mean()

def stable_pairwise_rank_loss(pred, target, mask, margin=0.05, max_pairs=8000):
    idx = mask.nonzero(as_tuple=True)[0]
    if len(idx) < 2:
        return pred.sum() * 0.0

    p = pred[idx]
    t = target[idx]

    N = len(p)
    i = torch.randint(0, N, (max_pairs,), device=p.device)
    j = torch.randint(0, N, (max_pairs,), device=p.device)

    diff_true = t[i] - t[j]
    sign = torch.sign(diff_true)

    valid = sign != 0
    if valid.sum() == 0:
        return pred.sum() * 0.0

    scale = (p.std() + 1e-6)
    diff_pred = (p[i] - p[j]) / scale


    # ⭐ 核心：margin ranking
    gap = (t[i] - t[j]).abs()
    adaptive_margin = margin + 0.5 * gap  # ← 核心

    loss = F.relu(adaptive_margin - sign * diff_pred)
    return loss.mean()
def topk_rank_boost(pred, target, mask, k_ratio=0.2):
    pred = pred[mask]
    target = target[mask]

    N = target.size(0)
    k = max(2, int(N * k_ratio))

    _, top_idx = torch.topk(target, k)

    return stable_pairwise_rank_loss(
        pred[top_idx],
        target[top_idx],
        torch.ones_like(top_idx, dtype=torch.bool)
    )
def tail_amplify_weight(target, alpha=2.0):
    """
    让高影响节点产生更大梯度
    """
    norm = (target - target.min()) / (target.max() - target.min() + 1e-6)
    return 1.0 + alpha * norm


def stable_monotonic_loss(pred_sir, mask, margin=0.0):
    diff = pred_sir[:, 1:] - pred_sir[:, :-1]
    valid = mask.unsqueeze(1).expand_as(diff)
    return F.relu(margin - diff)[valid].mean()

def compute_sir_only_loss(pred_sir, data, config):

    mask = data.train_mask
    y_multi = data.y_sir_multi
    y_mean  = data.y_sir.unsqueeze(1)

    pred_mean = pred_sir.mean(dim=1, keepdim=True)

    weight = tail_amplify_weight(data.y_sir).unsqueeze(1)

    # ===== 全局回归 =====
    loss_g_reg = sir_huber_loss(pred_mean * weight, y_mean * weight, mask)

    # ===== 全局排序 =====
    loss_g_rank = stable_pairwise_rank_loss(
        pred_mean.squeeze(1),
        data.y_sir,
        mask
    )

    # ===== TopK =====
    loss_topk = topk_rank_boost(
        pred_mean.squeeze(1),
        data.y_sir,
        mask,
        k_ratio=0.2
    )

    # ===== β相关 =====
    if config.get("multi_beta", True):

        B = pred_sir.size(1)

        loss_b_reg = sir_huber_loss(pred_sir, y_multi, mask) * 0.3

        loss_b_rank = 0
        for b in range(B):
            loss_b_rank += stable_pairwise_rank_loss(
                pred_sir[:, b],
                y_multi[:, b],
                mask
            )
        loss_b_rank /= B

        loss_mono = stable_monotonic_loss(pred_sir, mask)

    else:
        loss_b_reg = torch.tensor(0., device=pred_sir.device)
        loss_b_rank = torch.tensor(0., device=pred_sir.device)
        loss_mono = torch.tensor(0., device=pred_sir.device)

    # ===== 总损失 =====
    loss = (
        config["w_g_reg"]  * loss_g_reg  +
        config["w_g_rank"] * loss_g_rank +
        config["w_topk"]   * loss_topk   +
        config["w_b_reg"]  * loss_b_reg  +
        config["w_b_rank"] * loss_b_rank +
        config["w_mono"]   * loss_mono
    )

    return loss, {
        "g_reg": loss_g_reg.item(),
        "g_rank": loss_g_rank.item(),
        "topk": loss_topk.item(),
        "b_reg": loss_b_reg.item(),
        "b_rank": loss_b_rank.item(),
        "mono": loss_mono.item()
    }
def train_one_epoch(model, loader, optimizer, device, config):
    
    model.train()
    total_loss = 0
    stats_sum = defaultdict(float)

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()

        pred_sir = model(data)   # ← 不再解包

        loss, stats = compute_sir_only_loss(pred_sir, data, config)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in stats.items():
            stats_sum[k] += v

    n = len(loader)
    for k in stats_sum:
        stats_sum[k] /= n

    return total_loss / n, stats_sum

def rank_metrics(pred, target):
    pred = pred.numpy()
    target = target.numpy()

    sp = spearmanr(pred, target).correlation
    kd = kendalltau(pred, target).correlation

    return sp, kd

@torch.no_grad()
def evaluate(model, loader, device, split="val"):
    model.eval()
    all_pred, all_true = [], []

    for data in loader:
        data = data.to(device)
        pred_sir = model(data)

        # ⭐ 与训练目标对齐
        if split == "val":
            mask = data.val_mask
        elif split == "test":
            mask = data.test_mask
        else:
            mask = data.mask_sir

        pred = pred_sir.mean(dim=1)[mask].cpu()
        true = data.y_sir[mask].cpu()

        all_pred.append(pred)
        all_true.append(true)

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)

    sp, kd = rank_metrics(pred, true)

    return {"spearman": sp, "kendall": kd}
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
import networkx as nx
import os

def generate_all_plots(
        model,
        data,
        loader,
        history,
        out_dir,
        edge_file=None
):
    """
    自动生成完整可视化结果
    """

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    model.eval()

    # ======================================================
    # 1️⃣ 获取预测
    # ======================================================
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)
            pred_sir = model(batch)
            pred_mean = pred_sir.mean(dim=1).cpu().numpy()
            true_mean = batch.y_sir.cpu().numpy()
            true_multi = batch.y_sir_multi.cpu().numpy()
            pred_multi = pred_sir.cpu().numpy()

    hist_df = pd.DataFrame(history)

    # ======================================================
    # 2️⃣ Loss 曲线
    # ======================================================
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["loss"])
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(fig_dir, "loss_curve.png"))
    plt.close()

    # ======================================================
    # 3️⃣ 排序指标曲线
    # ======================================================
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["spearman"], label="Spearman")
    plt.plot(hist_df["epoch"], hist_df["kendall"], label="Kendall")
    plt.legend()
    plt.title("Ranking Metrics")
    plt.savefig(os.path.join(fig_dir, "ranking_curve.png"))
    plt.close()

    # ======================================================
    # 4️⃣ 预测散点图
    # ======================================================
    plt.figure()
    plt.scatter(true_mean, pred_mean, alpha=0.4)
    plt.xlabel("True Influence")
    plt.ylabel("Predicted Influence")
    plt.title("Prediction Scatter")
    plt.savefig(os.path.join(fig_dir, "scatter.png"))
    plt.close()

    # ======================================================
    # 5️⃣ Top-K Overlap 曲线
    # ======================================================
    def topk_overlap(pred, true, k):
        p_top = set(np.argsort(-pred)[:k])
        t_top = set(np.argsort(-true)[:k])
        return len(p_top & t_top) / k

    ks = [10, 20, 50, 100]
    overlaps = [topk_overlap(pred_mean, true_mean, k) for k in ks]

    plt.figure()
    plt.plot(ks, overlaps)
    plt.xlabel("K")
    plt.ylabel("Overlap Ratio")
    plt.title("Top-K Overlap")
    plt.savefig(os.path.join(fig_dir, "topk_overlap.png"))
    plt.close()

    # ======================================================
    # 6️⃣ 平均传播曲线对比
    # ======================================================
    plt.figure()
    plt.plot(true_multi.mean(0), label="True")
    plt.plot(pred_multi.mean(0), label="Pred")
    plt.legend()
    plt.title("Average Spread Curve")
    plt.savefig(os.path.join(fig_dir, "avg_spread_curve.png"))
    plt.close()

    # ======================================================
    # 7️⃣ 单节点传播曲线示例
    # ======================================================
    sample_nodes = np.random.choice(len(pred_mean), 3, replace=False)

    for node_id in sample_nodes:
        plt.figure()
        plt.plot(true_multi[node_id], label="True")
        plt.plot(pred_multi[node_id], label="Pred")
        plt.legend()
        plt.title(f"Node {node_id} Spread Curve")
        plt.savefig(os.path.join(fig_dir, f"node_{node_id}_curve.png"))
        plt.close()
    # ======================================================
    # 8️⃣ 传播曲线拟合图（论文常用）
    # ======================================================

    plt.figure()

    # 随机选5个节点
    sample_nodes = np.random.choice(len(pred_mean), 5, replace=False)

    for idx in sample_nodes:

        plt.plot(true_multi[idx], alpha=0.6)
        plt.plot(pred_multi[idx], '--', alpha=0.6)

    plt.xlabel("Beta Index")
    plt.ylabel("Spread")
    plt.title("Propagation Curve Fitting")

    plt.savefig(os.path.join(fig_dir, "beta_curve_fit.png"))
    plt.close()

    # ======================================================
    # 8️⃣ 不同 β 下的 Spearman
    # ======================================================
    B = pred_multi.shape[1]
    beta_sps = []

    for b in range(B):
        sp = spearmanr(pred_multi[:, b], true_multi[:, b]).correlation
        beta_sps.append(sp)

    plt.figure()
    plt.plot(range(B), beta_sps)
    plt.xlabel("Beta Index")
    plt.ylabel("Spearman")
    plt.title("Spearman Across Beta")
    plt.savefig(os.path.join(fig_dir, "beta_spearman.png"))
    plt.close()

    # ======================================================
    # 9️⃣ 嵌入空间 t-SNE（结构 / 语义 / 双模态对比）
    # ======================================================
    with torch.no_grad():

        batch = next(iter(loader))
        device = next(model.parameters()).device
        batch = batch.to(device)

        # ==================================================
        # 1️⃣ 双模态
        # ==================================================
        model.use_struct = True
        model.use_semantic = True
        model.use_perf = False

        _ = model(batch)
        emb_dual = model.last_hidden.cpu().numpy()
        attn_weights = model.last_modal_attn


        # ==================================================
        # 2️⃣ 结构-only
        # ==================================================
        model.use_struct = True
        model.use_semantic = False
        model.use_perf = False

        _ = model(batch)
        emb_struct = model.last_hidden.cpu().numpy()


        # ==================================================
        # 3️⃣ 语义-only
        # ==================================================
        model.use_struct = False
        model.use_semantic = True
        model.use_perf = False

        _ = model(batch)
        emb_sem = model.last_hidden.cpu().numpy()


    # ======================================================
    # t-SNE 可视化函数（内联）
    # ======================================================

    # -------- Struct Only --------
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb_struct)

    plt.figure()
    plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=true_mean,
        cmap="viridis",
        s=15
    )
    plt.colorbar()
    plt.title("t-SNE (Structure Only)")
    plt.savefig(os.path.join(fig_dir, "tsne_struct.png"))
    plt.close()


    # -------- Semantic Only --------
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb_sem)

    plt.figure()
    plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=true_mean,
        cmap="viridis",
        s=15
    )
    plt.colorbar()
    plt.title("t-SNE (Semantic Only)")
    plt.savefig(os.path.join(fig_dir, "tsne_semantic.png"))
    plt.close()


    # -------- Dual Modal --------
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb_dual)

    plt.figure()
    plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=true_mean,
        cmap="viridis",
        s=15
    )
    plt.colorbar()
    plt.title("t-SNE (Struct + Semantic)")
    plt.savefig(os.path.join(fig_dir, "tsne_dual.png"))
    plt.close()

    '''
    # ======================================================
    # 🔟 模态 Attention 可视化
    # ======================================================
    if attn_weights is not None:
        # attn_weights shape: [N, num_heads, num_modal, num_modal]
        attn_np = attn_weights.cpu().numpy()

        # 平均 batch + heads
        attn_mean = attn_np.mean(axis=0).mean(axis=0)  # shape -> [num_modal, num_modal]

        # 取对角线并保证是 1D
        attn_diag = np.diag(attn_mean).flatten()

        plt.figure()
        plt.bar(range(len(attn_diag)), attn_diag)
        plt.xticks(range(len(attn_diag)), ["Struct", "Semantic"][:len(attn_diag)])
        plt.title("Average Modal Attention Weight")
        plt.savefig(os.path.join(fig_dir, "modal_attention.png"))
        plt.close()
        '''
    # ======================================================
    # 🔟 Degree 对比（基于当前图）
    # ======================================================
    edge_index = batch.edge_index.cpu().numpy()

    num_nodes = pred_mean.shape[0]
    degrees = np.zeros(num_nodes)

    # 统计入度（无向图相当于度）
    for src, dst in edge_index.T:
        degrees[src] += 1
        degrees[dst] += 1

    plt.figure()
    plt.scatter(degrees, pred_mean, alpha=0.4)
    plt.xlabel("Degree")
    plt.ylabel("GNN Score")
    plt.title("Degree vs GNN Score")
    plt.savefig(os.path.join(fig_dir, "degree_vs_score.png"))
    plt.close()

    print(f"所有图已生成至: {fig_dir}")
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE = "scratch/keynode/project/public_process/extract"
    datasets = [
        #"Cora", "CiteSeer", "PubMed",
        #"Computers", "Photo",
        #"CS", "Physics",
        "run_1"
    ]

    # ================= Early Stopping 超参数 =================
    max_epoch = 30000
    patience = 200
    seeds = [0, 1, 2, 3, 4]

    for name in datasets:

        print(f"\n================ {name} ================\n")

        data_root = f"{BASE}/{name}_struct"
        seed_results = []

        # ================= 多随机种子 =================
        for seed in seeds:

            print(f"\n----- Seed {seed} -----\n")

            torch.manual_seed(seed)
            np.random.seed(seed)
            use_struct = True
            use_semantic = True
            use_perf = (name == "run_1")   # 只有 run_1 开启 perf

            dataset = SlimRunDataset(
                root_dir=data_root,
                dataset_name=name,
                use_struct=use_struct,
                use_perf=use_perf,
                use_semantic=use_semantic,
                seed=seed
            )

            data = dataset[0]
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            semantic_dim = 0
            if use_semantic:
                if name == "run_1":
                    sem_path = os.path.join(data_root, "kg_embeddings.npz")
                    if not os.path.exists(sem_path):
                        raise RuntimeError("kg_embeddings.npz 不存在")

                    sem_npz = np.load(sem_path, allow_pickle=True)

                    if "entity_embeddings" not in sem_npz:
                        raise RuntimeError(
                            f"kg_embeddings.npz 中未找到 entity_embeddings，现有键: {list(sem_npz.keys())}"
                        )

                    entity_embeddings = sem_npz["entity_embeddings"]

                    # ComplEx 等模型可能产生复数嵌入；与前面 load_run1_kg_semantic_aligned 保持一致
                    if np.iscomplexobj(entity_embeddings):
                        entity_embeddings = np.concatenate(
                            [entity_embeddings.real, entity_embeddings.imag],
                            axis=1
                        )

                    entity_embeddings = np.asarray(entity_embeddings, dtype=np.float32)
                    semantic_dim = entity_embeddings.shape[1]

                else:
                    sem_file = f"semantic_{name}.pt"
                    sem_path = os.path.join(data_root, sem_file)

                    if not os.path.exists(sem_path):
                        raise RuntimeError(f"{sem_file} 不存在")

                    sem_data = safe_torch_load(sem_path)
                    semantic_dim = sem_data["semantic_features"].shape[1]

            model = SlimMultiGAT(
                perf_dim=3,
                struct_dim=5,
                semantic_dim=semantic_dim,
                hidden_dim=128,
                modal_dim=128,
                fusion_heads=4,
                config={
                    **config,
                    "use_struct": use_struct,
                    "use_perf": use_perf,
                    "use_semantic": use_semantic,
                    "num_beta": dataset.data.num_beta
                }
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=40
            )

            # ================= Early Stopping 初始化 =================
            best_metric = -1.0
            best_state = None
            patience_counter = 0
            history = []

            print("Training...")

            for epoch in range(1, max_epoch + 1):

                loss, stats = train_one_epoch(
                    model, loader, optimizer, device, config
                )

                metrics = evaluate(model, loader, device, split="val")

                record = {
                    "epoch": epoch,
                    "loss": loss,
                    **stats,
                    **metrics
                }
                history.append(record)

                current_sp = metrics["spearman"]
                scheduler.step(current_sp)

                # ===== Early Stopping =====
                if current_sp > best_metric:
                    best_metric = current_sp
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 20 == 0:
                    print(
                        f"Epoch {epoch:04d} | "
                        f"Loss {loss:.4f} | "
                        f"SP {metrics['spearman']:.4f} | "
                        f"KD {metrics['kendall']:.4f}"
                    )

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # ================= 恢复最佳模型 =================
            model.load_state_dict(best_state)

            test_metrics = evaluate(model, loader, device, split="test")
            print("Final Test:", test_metrics)

            seed_results.append(test_metrics["spearman"])

            # ================= 每个 seed 单独保存 =================
            out_dir = os.path.join(
                data_root,
                f"results_seed_{seed}"
            )
            os.makedirs(out_dir, exist_ok=True)

            torch.save(
                model.state_dict(),
                os.path.join(out_dir, "sir_gnn.pt")
            )

            hist_df = pd.DataFrame(history)
            hist_df.to_csv(
                os.path.join(out_dir, "train_log.csv"),
                index=False
            )
            #print("Train label:", y_sir[:10])
            '''

            # ===== 导出预测分数 =====
            @torch.no_grad()
            def export_scores():
                model.eval()
                for batch in loader:
                    batch = batch.to(device)
                    pred = model(batch).cpu().numpy().reshape(-1)
                    df = pd.DataFrame({
                        "node": np.arange(len(pred)),
                        "gnn_score": pred
                    })
                    df.to_csv(
                        os.path.join(out_dir, "gnn_node_scores.csv"),
                        index=False
                    )
            '''
            @torch.no_grad()
            def export_scores():
                model.eval()
                for batch in loader:
                    batch = batch.to(device)
                    pred = model(batch).cpu().numpy()
                    print("pred raw shape =", pred.shape)

                    # ===== 去掉 batch 维度 =====
                    pred = np.squeeze(pred)

                    print("pred squeezed shape =", pred.shape)

                    if pred.ndim == 2:
                        score = pred.mean(axis=1)
                    else:
                        score = pred
                    score = np.asarray(score, dtype=float).reshape(-1)
                    node_ids = np.arange(score.shape[0], dtype=int)

                    print("score shape =", score.shape)

                    assert score.ndim == 1
                    assert node_ids.ndim == 1


                    # 先创建 DataFrame，再赋值列（避免 pandas 推断问题）
                    df = pd.DataFrame(index=np.arange(len(score)))
                    df["node"] = node_ids
                    df["gnn_score"] = score

                    out_path = os.path.join(out_dir, "gnn_node_scores.csv")
                    df.to_csv(out_path, index=False)

                    print("saved:", out_path)
            

            export_scores()

            generate_all_plots(
                model=model,
                data=data,
                loader=loader,
                history=history,
                out_dir=out_dir,
                edge_file=os.path.join(data_root, "edgelist.csv")
            )

            best_model = SlimMultiGAT(
                perf_dim=3,
                struct_dim=5,
                semantic_dim=semantic_dim,
                hidden_dim=128,
                modal_dim=128,
                fusion_heads=4,
                config={
                **config,
                "use_struct": use_struct,
                "use_perf": use_perf,
                "use_semantic": use_semantic,
                "num_beta": dataset.data.num_beta
                }
            ).to(device)

            best_model.load_state_dict(
                torch.load(os.path.join(out_dir, "sir_gnn.pt"),
                map_location=device)
            )

            results = run_all_baselines(
                data=data,
                model_ours=best_model,
                device=device
            )

            print_results(results)

            pd.DataFrame(results).T.to_csv(
                os.path.join(out_dir, "baseline_results.csv")
            )

        # ================= 多种子统计 =================
        mean_sp = np.mean(seed_results)
        std_sp = np.std(seed_results)

        print(f"\n{name} Final Spearman: {mean_sp:.4f} ± {std_sp:.4f}\n")

        pd.DataFrame({
            "seed": seeds,
            "spearman": seed_results
        }).to_csv(
            os.path.join(data_root, "multi_seed_summary.csv"),
            index=False
        )

    print("✅ 全部数据集训练 + 多种子统计完成")