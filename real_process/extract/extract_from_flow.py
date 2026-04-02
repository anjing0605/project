#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_from_flow.py

改进说明：
- 将性能特征与结构特征分别做 StandardScaler + PCA(n_components=1) 来学习各自内部权重
- 使用 alpha 融合两者的 PC1 得到每个节点的原始质量 Q，然后 per-run min-max 归一化到 [0,1] 得到 soft_label
- 保存 x_perf.npy, x_struct.npy, soft_label.npy, node_features.csv, node_index_map.json
- 保存 PCA 模型、scaler 与权重以便复现和展示
- 可选 stratified mask（半监督）
"""
'''
python3 scratch/project/scripts/extract_from_flow.py \
  --output_dir scratch/project/global_training_data \
  --num_runs 1 \
  --apply_mask True \
  --mask_frac 0.5
# 注意：这里不加 --reveal_test

'''
import os
import re
import glob
import time
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import xml.etree.ElementTree as ET
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA

import joblib
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---------------- Config ----------------
DEFAULT_CONFIG = {
    "time_window": 10.0,
    "decay_factor": 0.85,
    "semi_ratio": 0.1,
    "training_data_dir": "scratch/keynode/project/real_datasets",
    "output_dir": "scratch/keynode/project/real_datasets",
    "run_id": None,
    "num_runs": None,
    "alpha": 0.5,
    "apply_mask": False,
    "mask_frac": 0.5,
    "pca_on_runs": None,
    "seed": 123
}

parser = argparse.ArgumentParser(description='网络节点特征提取与 分离 PCA 权重学习 + soft_label 生成')
parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG["output_dir"], help='训练数据目录（包含 run_*）')
parser.add_argument('--run_id', type=int, default=None, help='只处理一个 run_X 的 run id')
parser.add_argument('--num_runs', type=int, default=None, help='处理前 N 个 run（按 run_id 排序）')
parser.add_argument('--alpha', type=float, default=DEFAULT_CONFIG["alpha"], help='性能/结构融合系数 alpha')
parser.add_argument('--apply_mask', action='store_true', help='是否在生成 soft_label 后应用 stratified mask（半监督隐藏）')
parser.add_argument('--mask_frac', type=float, default=DEFAULT_CONFIG["mask_frac"], help='mask 比例（隐藏比例），例如 0.5 表示隐藏 50% 的标签')
parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG["seed"], help='随机种子')
parser.add_argument('--train_runs', type=str, default="",help='可选：指定训练用 runs，例如 "1,2,3" 或 "1-10"；留空表示自动扫描/全量')
parser.add_argument('--reveal_test', action='store_true',help='是否揭示 test 标签用于最终评估（默认关闭，防止泄露）')

args = parser.parse_args()

np.random.seed(args.seed)

# ---------------- 辅助函数 ----------------
def safe_get(element, attr, dtype=float, default=0):
    try:
        value = element.attrib.get(attr, default)
        if isinstance(value, str) and dtype in (int, float):
            value = re.sub(r'[^0-9eE\.\+\-]', '', value)
        if dtype == int:
            return int(value) if value != '' else default
        elif dtype == float:
            return float(value) if value != '' else default
        return value
    except Exception:
        return default

def time_weighted_average(values, times, decay=0.85):
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    weights = []
    t_last = times[-1]
    for t in times:
        dt = max(t_last - t, 0.0)
        weights.append(decay ** dt)
    weights = np.array(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    values = np.array(values, dtype=float)
    return float((weights * values).sum())

def generate_topology_visualization(G, save_path):
    plt.figure(figsize=(8, 6))
    try:
        pos = nx.spring_layout(G, seed=42)
    except Exception:
        pos = None
    nx.draw(G, pos=pos, node_size=50, with_labels=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def stratified_mask(labels, mask_frac=0.5, n_bins=10, seed=123):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    mask = np.ones(n, dtype=bool)

    if n == 0:
        return mask

    bins = np.linspace(labels.min(), labels.max(), n_bins + 1)
    inds = np.digitize(labels, bins) - 1
    inds = np.clip(inds, 0, n_bins - 1)

    for b in range(n_bins):
        idx = np.where(inds == b)[0]
        if len(idx) == 0:
            continue
        k = int(np.floor(len(idx) * mask_frac))
        if k <= 0:
            continue
        hide = rng.choice(idx, size=k, replace=False)
        mask[hide] = False
    return mask

# ------------------------- 关键修改：process_single_run -------------------------
def process_single_run(run_id):
    """
    处理一个 run 的输入文件，输出 node_features.csv, x_perf.npy, x_struct.npy, edgelist.csv, node_index_map.json
    返回：运行目录路径或 None（失败）
    """
    print(f"\n--- 处理 run_{run_id} ---")
    run_dir = os.path.join(args.output_dir, f"run_{run_id}")
    if not os.path.isdir(run_dir):
        print(f"run_dir 不存在: {run_dir}")
        return None

    flow_file = os.path.join(run_dir, "flow_results.xml")
    mapping_file = os.path.join(run_dir, "node_ip_mapping.json")
    nodes_json_file = os.path.join(run_dir, "nodes.json")

    # 必需文件检查
    for f in [flow_file, mapping_file, nodes_json_file]:
        if not os.path.exists(f):
            print(f"缺少文件: {f}; 跳过 run {run_id}")
            return None

    # ---------------------- 读取 nodes.json ----------------------
    try:
        with open(nodes_json_file) as f:
            nodes_data = json.load(f)
        node_info_dict = {n.get('nodeId', i): n for i, n in enumerate(nodes_data)}
    except Exception as e:
        print(f"读取 nodes.json 错误: {e}")
        node_info_dict = {}

    # ---------------------- 读取 mapping ----------------------
    try:
        with open(mapping_file) as f:
            raw_mapping = json.load(f)

        mapping = {}  # ip → nodeId (int)
        for ip, name in raw_mapping.items():
            if isinstance(name, list):
                name = name[0]
            try:
                node_id = int(str(name).replace("node", "").replace("n", ""))
            except Exception:
                node_id = int(name)
            mapping[ip] = node_id

    except Exception as e:
        print(f"读取 mapping 错误: {e}")
        mapping = {}

    # ---------------------- 解析 FlowMonitor XML ----------------------
    try:
        tree = ET.parse(flow_file)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] parse XML failed in run {run_id}: {e}")
        return None

    def _clean_ip(ip):
        if ip is None:
            return None
        s = str(ip)
        if ':' in s and '.' in s:
            s = s.split(':')[0]
        if '/' in s:
            s = s.split('/')[0]
        return s

    def _parse_int(elem, name):
        v = elem.get(name)
        if v is None:
            child = elem.find(name)
            if child is not None and child.text:
                v = child.text
            else:
                return 0
        try:
            return int(float(str(v)))
        except Exception:
            return 0

    def _parse_time_s(elem, name):
        v = elem.get(name)
        if v is None:
            child = elem.find(name)
            if child is not None and child.text:
                v = child.text
            else:
                return 0.0

        s = str(v).strip().lower()
        try:
            if s.startswith("+"):
                s = s[1:].strip()
            if s.endswith("ns"):
                return float(s[:-2].strip()) * 1e-9
            if s.endswith("us"):
                return float(s[:-2].strip()) * 1e-6
            if s.endswith("ms"):
                return float(s[:-2].strip()) * 1e-3
            if s.endswith("s"):
                return float(s[:-1].strip())
            return float(s)
        except Exception:
            return 0.0

    # --- flowId → srcIP, dstIP 映射 ----
    flowid_to_ips = {}

    for cls in root.findall('.//Ipv4FlowClassifier'):
        for f in cls.findall('Flow'):
            fid = f.get('flowId')
            src = f.get('sourceAddress')
            dst = f.get('destinationAddress')
            if fid:
                flowid_to_ips[str(fid)] = (_clean_ip(src), _clean_ip(dst))

    for cls in root.findall('.//FlowClassifier'):
        for f in cls.findall('flow'):
            fid = f.get('flowId') or f.get('id')
            src = f.get('sourceAddress') or f.get('source')
            dst = f.get('destinationAddress') or f.get('destination')
            if fid:
                flowid_to_ips[str(fid)] = (_clean_ip(src), _clean_ip(dst))

    flows_stats = root.findall('.//FlowStats/Flow')
    if not flows_stats:
        flows_stats = root.findall('.//Flow')

    # 端到端质量归因到 src；dst 只记录负载
    per_node = defaultdict(lambda: {
        'tx': 0, 'rx': 0, 'lost': 0,
        'rx_bytes': 0,
        'delay_sum_s': 0.0,
        'dur_sum_s': 0.0,
        'flow_cnt': 0,
        'fail_flow_cnt': 0,
        'in_rx': 0,
        'in_bytes': 0
    })

    sim_start_s = None
    sim_end_s = None

    for f in flows_stats:
        fid = f.get('flowId') or f.get('id')
        fid = str(fid) if fid else None

        tx = _parse_int(f, 'txPackets')
        rx = _parse_int(f, 'rxPackets')
        lost = _parse_int(f, 'lostPackets')
        rxBytes = _parse_int(f, 'rxBytes')

        delaySum_s = _parse_time_s(f, 'delaySum')
        timeFirstTx_s = _parse_time_s(f, 'timeFirstTxPacket')
        timeLastRx_s = _parse_time_s(f, 'timeLastRxPacket')

        if timeFirstTx_s > 0:
            sim_start_s = timeFirstTx_s if sim_start_s is None else min(sim_start_s, timeFirstTx_s)
        if timeLastRx_s > 0:
            sim_end_s = timeLastRx_s if sim_end_s is None else max(sim_end_s, timeLastRx_s)

        if lost == 0 and tx > 0:
            lost = max(tx - rx, 0)

        dur_s = 0.0
        if timeFirstTx_s > 0 and timeLastRx_s > 0 and timeLastRx_s > timeFirstTx_s:
            dur_s = timeLastRx_s - timeFirstTx_s

        src_ip, dst_ip = None, None
        if fid and fid in flowid_to_ips:
            src_ip, dst_ip = flowid_to_ips[fid]

        if not src_ip:
            src_ip = _clean_ip(f.get('sourceAddress') or f.get('source'))
        if not dst_ip:
            dst_ip = _clean_ip(f.get('destinationAddress') or f.get('destination'))

        src_node = mapping.get(src_ip, None) if src_ip else None
        dst_node = mapping.get(dst_ip, None) if dst_ip else None

        if src_node is not None:
            h = per_node[src_node]
            h['tx'] += tx
            h['rx'] += rx
            h['lost'] += lost
            h['rx_bytes'] += rxBytes
            if rx > 0:
                h['delay_sum_s'] += delaySum_s
            if dur_s > 0:
                h['dur_sum_s'] += dur_s
            h['flow_cnt'] += 1
            if tx > 0 and rx == 0:
                h['fail_flow_cnt'] += 1

        if dst_node is not None:
            h2 = per_node[dst_node]
            h2['in_rx'] += rx
            h2['in_bytes'] += rxBytes

    if sim_start_s is None or sim_end_s is None or sim_end_s <= sim_start_s:
        sim_time_s = 1.0
    else:
        sim_time_s = sim_end_s - sim_start_s

    all_nodes = set()
    for n in list(mapping.values()) + list(per_node.keys()):
        try:
            all_nodes.add(int(n))
        except Exception:
            continue

    node_records = []
    for node in sorted(all_nodes):
        h = per_node.get(node, {})
        tx = int(h.get('tx', 0))
        rx = int(h.get('rx', 0))
        lost = int(h.get('lost', 0))
        rx_bytes = int(h.get('rx_bytes', 0))
        in_rx = int(h.get('in_rx', 0))
        in_bytes = int(h.get('in_bytes', 0))

        delay_avg_s = float(h.get('delay_sum_s', 0.0)) / max(rx, 1)
        loss_raw = lost / max(tx, 1)

        dur_sum_s = float(h.get('dur_sum_s', 0.0))
        denom_s = dur_sum_s if dur_sum_s > 1e-9 else sim_time_s
        throughput_mbps = (rx_bytes * 8.0) / max(denom_s, 1e-9) / 1e6

        flow_cnt = int(h.get('flow_cnt', 0))
        fail_flow_cnt = int(h.get('fail_flow_cnt', 0))
        fail_ratio = (fail_flow_cnt / max(flow_cnt, 1)) if flow_cnt > 0 else 0.0

        node_records.append({
            'node': int(node),
            'throughput_raw': throughput_mbps,
            'delay_raw': delay_avg_s,
            'loss_raw': loss_raw,
            'txPackets': tx,
            'rxPackets': rx,
            'lostPackets': lost,
            'rx_bytes': rx_bytes,
            'flow_cnt': flow_cnt,
            'fail_ratio': fail_ratio,
            'in_rxPackets': in_rx,
            'in_bytes': in_bytes,
            'uptime_ratio': 1.0 - fail_ratio
        })

    df_nodes = pd.DataFrame(node_records)

    # ------------------ 结构特征：links.json ------------------
    links_json_path = os.path.join(run_dir, "links.json")
    edges = []

    if os.path.exists(links_json_path):
        try:
            with open(links_json_path, "r") as f:
                links = json.load(f)
            if isinstance(links, dict) and "links" in links:
                links = links["links"]

            for e in links:
                try:
                    u = int(e.get("src", e.get("source")))
                    v = int(e.get("dst", e.get("target")))
                    if u == v:
                        continue
                    edges.append((u, v))
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] 读取 links.json 失败: {links_json_path}, err={e}")
    else:
        print(f"[WARN] links.json 不存在: {links_json_path}，结构特征将退化为0")

    df_edges = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
    df_edges.to_csv(os.path.join(run_dir, "edgelist.csv"), index=False)

    if df_edges.empty:
        degree, closeness, betweenness, clustering, core = {}, {}, {}, {}, {}
    else:
        G = nx.from_pandas_edgelist(df_edges, "source", "target")
        degree = nx.degree_centrality(G)
        closeness = nx.closeness_centrality(G)
        try:
            betweenness = nx.betweenness_centrality(G)
        except Exception:
            betweenness = {n: 0.0 for n in G.nodes()}
        clustering = nx.clustering(G)
        try:
            core = nx.core_number(G)
        except Exception:
            core = {n: 0 for n in G.nodes()}

    df_nodes['degree_centrality'] = df_nodes['node'].map(degree).fillna(0)
    df_nodes['closeness'] = df_nodes['node'].map(closeness).fillna(0)
    df_nodes['betweenness'] = df_nodes['node'].map(betweenness).fillna(0)
    df_nodes['clustering_coeff'] = df_nodes['node'].map(clustering).fillna(0)
    df_nodes['k_shell'] = df_nodes['node'].map(core).fillna(0)

    # ------------------ 性能特征转换 ------------------
    df_nodes['throughput'] = np.sqrt(df_nodes['throughput_raw'] + 1e-6)
    df_nodes['delay'] = 1.0 / (1.0 + df_nodes['delay_raw'])
    df_nodes['loss'] = 1.0 - np.clip(df_nodes['loss_raw'], 0, 1)

    # ------------------ ✅ 自检输出 ------------------
    try:
        print("\n[SELF-CHECK] 典型节点样例（run_{}）".format(run_id))

        pure_recv = df_nodes[(df_nodes['txPackets'] == 0) & (df_nodes['in_rxPackets'] > 0)].copy()
        pure_recv = pure_recv.sort_values('in_bytes', ascending=False).head(5)
        if len(pure_recv) > 0:
            print("\n  - 纯接收(承载负载)节点 Top5: tx=0 & in_rx>0（按 in_bytes 排序）")
            print(pure_recv[['node','txPackets','rxPackets','in_rxPackets','in_bytes']].to_string(index=False))
        else:
            print("\n  - 纯接收(承载负载)节点: 未发现 tx=0 且 in_rx>0 的节点")

        fail_send = df_nodes[(df_nodes['txPackets'] > 0) & (df_nodes['rxPackets'] == 0)].copy()
        fail_send = fail_send.sort_values('txPackets', ascending=False).head(5)
        if len(fail_send) > 0:
            print("\n  - 失败发送节点 Top5: tx>0 & rx=0（按 txPackets 排序）")
            print(fail_send[['node','txPackets','rxPackets','lostPackets','fail_ratio']].to_string(index=False))
        else:
            print("\n  - 失败发送节点: 未发现 tx>0 & rx=0 的节点（通常是好现象）")

        high_loss = df_nodes[df_nodes['txPackets'] > 0].copy().sort_values('loss_raw', ascending=False).head(5)
        if len(high_loss) > 0:
            print("\n  - 高丢包节点 Top5（按 loss_raw 排序，tx>0）")
            print(high_loss[['node','txPackets','rxPackets','lostPackets','loss_raw']].to_string(index=False))

        high_delay = df_nodes[df_nodes['rxPackets'] > 0].copy().sort_values('delay_raw', ascending=False).head(5)
        if len(high_delay) > 0:
            print("\n  - 高时延节点 Top5（按 delay_raw 排序，rx>0）")
            print(high_delay[['node','rxPackets','delay_raw','throughput_raw','loss_raw']].to_string(index=False))

        high_thr = df_nodes.copy().sort_values('throughput_raw', ascending=False).head(5)
        if len(high_thr) > 0:
            print("\n  - 高吞吐节点 Top5（按 throughput_raw 排序）")
            print(high_thr[['node','throughput_raw','txPackets','rxPackets','loss_raw']].to_string(index=False))
    except Exception as e:
        print(f"[SELF-CHECK] 输出失败: {e}")

    # ------------------ 保存节点特征 ------------------
    df_nodes.to_csv(os.path.join(run_dir, "node_features.csv"), index=False)

    perf_cols = ['delay', 'throughput', 'loss']
    struct_cols = ['degree_centrality', 'betweenness', 'closeness', 'k_shell', 'clustering_coeff']

    x_perf = df_nodes[perf_cols].values.astype(float)
    x_struct = df_nodes[struct_cols].values.astype(float)

    np.save(os.path.join(run_dir, "x_perf.npy"), x_perf)
    np.save(os.path.join(run_dir, "x_struct.npy"), x_struct)

    os.makedirs(os.path.join(run_dir, "scalers"), exist_ok=True)

    node_index_map = {int(row['node']): i for i, row in df_nodes.iterrows()}
    with open(os.path.join(run_dir, "node_index_map.json"), 'w') as f:
        json.dump(node_index_map, f, indent=2, ensure_ascii=False)

    print(f"Run {run_id} feature saved: x_perf shape={x_perf.shape}, x_struct shape={x_struct.shape}")
    return run_dir

def _parse_tier(position: str):
    """
    position 形如 "tier=access" / "tier=agg" / "tier=core"
    """
    if not position:
        return None
    m = re.search(r"tier\s*=\s*([a-zA-Z0-9_-]+)", str(position))
    return m.group(1) if m else None

def _parse_bw_to_gbps(s: str):
    """
    "1Gbps" / "800Mbps" / "1000Kbps" -> 以 Gbps 返回 float
    解析失败返回 None
    """
    if not s:
        return None
    t = str(s).strip().lower()
    m = re.match(r"([0-9.]+)\s*([a-z]+)", t)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit.startswith("g"):
        return val
    if unit.startswith("m"):
        return val / 1000.0
    if unit.startswith("k"):
        return val / 1_000_000.0
    return None

def _bucket_num(x, bins, labels):
    """
    把连续数值分桶成离散标签，避免 KG 里 literal 太碎。
    bins: e.g. [0, 2, 5, 10, 20, 50, 1e9]
    labels: e.g. ["<=2", "2-5", ...]
    """
    if x is None:
        return None
    for i in range(len(bins)-1):
        if bins[i] <= x < bins[i+1]:
            return labels[i]
    return labels[-1]

def generate_kg_triples(run_dir: str,
                        nodes_json_name="nodes.json",
                        links_json_name="links.json",
                        out_name="triples.tsv",
                        include_links=True,
                        include_bw_stats=True,
                        add_device_tier_prior=True,
                        typical_min_count=3,
                        typical_min_ratio=0.6):
    """
    读取 nodes.json / links.json，输出 triples.tsv
    三元组格式：head \\t relation \\t tail

    ✅ 改动点：
    1) deviceType -> typicalTier：加入频率阈值过滤（避免噪声）
    2) include_links 默认开启：把边属性(bw/delay/type/loss)离散分桶注入为 typed relation（不引入link实体，稳）
       关系形式：linkTo__type_p2p__bw_20-50G__d_10-20__loss_0
    """

    nodes_path = os.path.join(run_dir, nodes_json_name)
    links_path = os.path.join(run_dir, links_json_name)
    out_path = os.path.join(run_dir, out_name)

    if not os.path.exists(nodes_path):
        print(f"[KG] nodes.json not found: {nodes_path}")
        return None

    with open(nodes_path, "r") as f:
        nodes = json.load(f)

    triples = []

    # ========== helpers ==========
    def _canon_text(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s == "" or s == "none" or s == "null" or s == "unknown":
            return None
        return s

    # 统一 entity 命名，避免和纯数字冲突
    def N(i):    return f"node_{int(i)}"
    def AS(i):   return f"as_{int(i)}"
    def DEV(s):  return f"device_{_canon_text(s) or 'unknown'}"
    def TIER(s): return f"tier_{_canon_text(s) or 'unknown'}"
    def ROLE(s): return f"role_{_canon_text(s) or 'unknown'}"
    def LIT(s):  return f"lit_{_canon_text(s) or 'unknown'}"  # literal wrapper（可选）

    def _parse_bw_to_gbps(s: str):
        if s is None:
            return None
        t = str(s).strip().lower()
        m = re.match(r"([0-9.]+)\s*([a-z]+)", t)
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2)
        if unit.startswith("g"):
            return val
        if unit.startswith("m"):
            return val / 1000.0
        if unit.startswith("k"):
            return val / 1_000_000.0
        return None

    def _parse_delay_to_ms(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        try:
            if s.endswith("ms"):
                return float(s[:-2].strip())
            if s.endswith("us"):
                return float(s[:-2].strip()) / 1000.0
            if s.endswith("ns"):
                return float(s[:-2].strip()) / 1e6
            if s.endswith("s"):
                return float(s[:-1].strip()) * 1000.0
            # assume already ms
            return float(s)
        except Exception:
            return None

    def _bucket_num(x, bins, labels):
        if x is None:
            return None
        for i in range(len(bins) - 1):
            if bins[i] <= x < bins[i + 1]:
                return labels[i]
        return labels[-1]

    # ========== tier prior (optional) ==========
    tier_order = ["access", "agg", "core"]
    for a, b in zip(tier_order, tier_order[1:]):
        triples.append((TIER(a), "below", TIER(b)))
        triples.append((TIER(b), "above", TIER(a)))

    # ========== buckets ==========
    port_bins   = [0, 2, 4, 8, 16, 10**9]
    port_labels = ["<=2", "2-4", "4-8", "8-16", ">=16"]

    bw_bins   = [0, 1, 2, 5, 10, 20, 50, 10**9]  # Gbps
    bw_labels = ["<1G", "1-2G", "2-5G", "5-10G", "10-20G", "20-50G", ">=50G"]

    delay_bins   = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 10**9]  # ms
    delay_labels = ["<0.2", "0.2-0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50", ">=50"]

    loss_bins   = [0, 1e-12, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10**9]  # rate
    loss_labels = ["0", "1e-12~1e-4", "1e-4~1e-3", "1e-3~1e-2", "1e-2~1e-1", "1e-1~1", ">=1"]

    # ========== 1) node semantic triples ==========
    # 先统计 device->tier 频率（用于 typicalTier 阈值过滤）
    dev_tier_cnt = defaultdict(lambda: defaultdict(int))
    dev_cnt = defaultdict(int)

    # 先收集每个 node 的 tier/dev/role/as
    for n in nodes:
        node_id = n.get("nodeId")
        if node_id is None:
            continue
        dev  = _canon_text(n.get("deviceType", None))
        tier = _canon_text(_parse_tier(n.get("position", "")) or None)
        if dev is not None and tier is not None:
            dev_cnt[dev] += 1
            dev_tier_cnt[dev][tier] += 1

    for n in nodes:
        node_id = n.get("nodeId")
        if node_id is None:
            continue

        role = n.get("role", "unknown")
        as_id = n.get("asId", -1)
        dev = n.get("deviceType", "unknown")
        port = n.get("portCount", None)
        tier = _parse_tier(n.get("position", "")) or "unknown"

        h = N(node_id)

        triples.append((h, "hasRole", ROLE(role)))
        # asId 可能是 -1 或 None
        try:
            as_int = int(as_id)
        except Exception:
            as_int = -1
        if as_int >= 0:
            triples.append((h, "inAS", AS(as_int)))
            triples.append((AS(as_int), "hasMember", h))

        triples.append((h, "hasDeviceType", DEV(dev)))
        triples.append((h, "inTier", TIER(tier)))

        # port bucket（减少 literal 稀疏）
        if port is not None:
            try:
                port_i = int(port)
                triples.append((h, "hasPortCount", f"port_{port_i}"))
                pb = _bucket_num(port_i, port_bins, port_labels)
                if pb:
                    triples.append((h, "portBucket", f"portBucket_{pb}"))
            except Exception:
                pass

        # ifaceBandwidths bucket stats
        if include_bw_stats:
            bws = n.get("ifaceBandwidths", []) or []
            bw_vals = [_parse_bw_to_gbps(x) for x in bws]
            bw_vals = [x for x in bw_vals if x is not None]
            if len(bw_vals) > 0:
                bw_min = min(bw_vals)
                bw_max = max(bw_vals)
                bw_avg = sum(bw_vals) / len(bw_vals)
                triples.append((h, "bwMinBucket", f"bw_{_bucket_num(bw_min, bw_bins, bw_labels)}"))
                triples.append((h, "bwMaxBucket", f"bw_{_bucket_num(bw_max, bw_bins, bw_labels)}"))
                triples.append((h, "bwAvgBucket", f"bw_{_bucket_num(bw_avg, bw_bins, bw_labels)}"))

    # ========== 1.1) deviceType -> typicalTier 先验（带阈值） ==========
    if add_device_tier_prior:
        for dev, tier_map in dev_tier_cnt.items():
            total = dev_cnt.get(dev, 0)
            if total <= 0:
                continue
            tier_best, c_best = max(tier_map.items(), key=lambda kv: kv[1])
            if c_best >= int(typical_min_count) and (c_best / max(1, total)) >= float(typical_min_ratio):
                triples.append((DEV(dev), "typicalTier", TIER(tier_best)))

    # ========== 2) links typed relations ==========
    if include_links and os.path.exists(links_path):
        try:
            with open(links_path, "r") as f:
                links = json.load(f)
            # 你的 links.json 结构是 list[dict]，不需要 "links" 包装；这里兼容一下
            if isinstance(links, dict) and "links" in links:
                links = links["links"]

            seen = set()
            for e in links:
                u = e.get("src", e.get("source"))
                v = e.get("dst", e.get("target"))
                if u is None or v is None:
                    continue
                try:
                    u = int(u)
                    v = int(v)
                except Exception:
                    continue
                if u == v:
                    continue

                # 去重（无向）
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in seen:
                    continue
                seen.add((a, b))

                typ = _canon_text(e.get("type", None)) or "unknown"

                bw_g = _parse_bw_to_gbps(e.get("bw", e.get("bandwidth", None)))
                d_ms = _parse_delay_to_ms(e.get("delay", e.get("latency", None)))

                loss = e.get("lossRate", e.get("loss", None))
                try:
                    loss_f = float(loss) if loss is not None else None
                except Exception:
                    loss_f = None

                bw_bucket = _bucket_num(bw_g, bw_bins, bw_labels) if bw_g is not None else "unknown"
                d_bucket  = _bucket_num(d_ms, delay_bins, delay_labels) if d_ms is not None else "unknown"
                l_bucket  = _bucket_num(loss_f, loss_bins, loss_labels) if loss_f is not None else "unknown"

                # ✅ typed relation：信息量更强，实体数不爆炸
                rel = f"linkTo__type_{typ}__bw_{bw_bucket}__d_{d_bucket}__loss_{l_bucket}"

                triples.append((N(a), rel, N(b)))
                triples.append((N(b), rel, N(a)))

                # 可选：把边属性也作为“节点的局部属性”写进去（更容易学习）
                triples.append((N(a), "hasAdjBwBucket", f"bw_{bw_bucket}"))
                triples.append((N(b), "hasAdjBwBucket", f"bw_{bw_bucket}"))
                triples.append((N(a), "hasAdjDelayBucket", f"d_{d_bucket}"))
                triples.append((N(b), "hasAdjDelayBucket", f"d_{d_bucket}"))

        except Exception as ex:
            print(f"[KG] failed to read links.json: {ex}")

    # ========== dedup + write ==========
    triples = list(dict.fromkeys(triples))  # 保序去重
    with open(out_path, "w") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

    print(f"KG triples saved: {out_path} (num triples={len(triples)})")
    return out_path

# ---------------- PCA 学习与 soft_label 生成（分离 perf/struct） ----------------
def learn_pca_weights_separate(perf_arrays, struct_arrays):
    """
    perf_arrays: list of arrays [n_i x d_p]
    struct_arrays: list of arrays [n_i x d_s]
    返回：scaler_perf, scaler_struct, w_perf (len d_p), w_struct (len d_s), pca_perf, pca_struct
    """
    Xp = np.vstack(perf_arrays)
    Xs = np.vstack(struct_arrays)
    # 使用 StandardScaler 来让 PCA 对方差敏感
    scaler_perf = StandardScaler().fit(Xp)
    scaler_struct = StandardScaler().fit(Xs)
    Xp_std = scaler_perf.transform(Xp)
    Xs_std = scaler_struct.transform(Xs)

    pca_perf = PCA(n_components=1, random_state=42).fit(Xp_std)
    pca_struct = PCA(n_components=1, random_state=42).fit(Xs_std)

    load_perf = pca_perf.components_[0]
    load_struct = pca_struct.components_[0]

    w_perf = np.abs(load_perf)
    w_struct = np.abs(load_struct)
    # 归一化为和为1（可解释）
    w_perf = w_perf / (w_perf.sum() + 1e-12)
    w_struct = w_struct / (w_struct.sum() + 1e-12)

    return scaler_perf, scaler_struct, w_perf, w_struct, pca_perf, pca_struct

def compute_soft_label_for_run(x_perf, x_struct,
                               scaler_perf, scaler_struct,
                               pca_perf, pca_struct,
                               alpha=0.5):
    x_perf_std = scaler_perf.transform(x_perf)
    x_struct_std = scaler_struct.transform(x_struct)

    pc1_perf   = pca_perf.transform(x_perf_std).ravel()      # ✅ 真·PC1 score
    pc1_struct = pca_struct.transform(x_struct_std).ravel()  # ✅ 真·PC1 score

    Q = alpha * pc1_perf + (1.0 - alpha) * pc1_struct
    soft = (Q - Q.min()) / (Q.max() - Q.min() + 1e-9)
    return soft.astype(np.float32), pc1_perf, pc1_struct

# ------------------ 用这个 main-block 替换你脚本中对应的 __main__ 的 run-level 处理部分 ------------------
if __name__ == "__main__":
    # ===============================
    # 1. 获取 run 列表
    # ===============================
    if args.run_id is not None and args.run_id >= 0:
        run_ids = [args.run_id]
    else:
        run_dirs = glob.glob(os.path.join(args.output_dir, "run_*"))
        run_ids = sorted([int(os.path.basename(d).split("_")[-1]) for d in run_dirs if os.path.isdir(d)])
        if args.num_runs is not None and args.num_runs > 0:
            run_ids = run_ids[:args.num_runs]

    if not run_ids:
        raise RuntimeError("未发现任何 run_* 目录，请确认 output_dir 是否正确")

    print(f"发现 {len(run_ids)} 个 runs: {run_ids}")

    # ===============================
    # 2. 特征提取
    # ===============================
    processed_runs = []
    for rid in run_ids:
        rdir = process_single_run(rid)
        if rdir is not None:
            processed_runs.append(rid)

    processed_runs = sorted(processed_runs)

    if not processed_runs:
        raise RuntimeError("没有成功处理的 runs，退出")

    # ===============================
    # 3. run-level 按顺序划分（70/15/15 + 小样本保护）
    #    说明：
    #      - 你的目标是跨 run 泛化：用 train runs 训练，用 val runs 早停/调参，用 test runs 最终评估
    #      - test 默认不揭示标签（--reveal_test 才揭示）
    # ===============================
    N = len(processed_runs)

    if N == 1:
        train_ids, val_ids, test_ids = processed_runs, [], []
    elif N == 2:
        train_ids, val_ids, test_ids = [processed_runs[0]], [], [processed_runs[1]]
    elif N == 3:
        train_ids, val_ids, test_ids = [processed_runs[0]], [processed_runs[1]], [processed_runs[2]]
    elif N == 4:
        train_ids, val_ids, test_ids = processed_runs[:2], [processed_runs[2]], [processed_runs[3]]
    else:
        n_train = max(1, int(round(0.70 * N)))
        n_val   = max(1, int(round(0.15 * N)))
        n_test  = N - n_train - n_val

        # 保证 test >= 1
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_val = max(1, n_val - 1)

        train_ids = processed_runs[:n_train]
        val_ids   = processed_runs[n_train:n_train + n_val]
        test_ids  = processed_runs[n_train + n_val:]

    # PCA 只用 train runs（避免 val/test 泄露）
    pca_runs = train_ids[:]

    print(f"Split (ordered) -> train: {train_ids}")
    print(f"Split (ordered) -> val  : {val_ids}")
    print(f"Split (ordered) -> test : {test_ids}")

    # ===============================
    # 4. 保存 split 元信息
    # ===============================
    pca_dir = os.path.join(args.output_dir, "pca_models")
    os.makedirs(pca_dir, exist_ok=True)

    split_meta = {
        "processed_runs": processed_runs,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "pca_runs": pca_runs,
        "alpha": args.alpha,
        "apply_mask": bool(getattr(args, "apply_mask", False)),
        "mask_frac": float(getattr(args, "mask_frac", 0.0)),
        "reveal_test": bool(getattr(args, "reveal_test", False)),
    }

    with open(os.path.join(pca_dir, "split_meta.json"), "w") as f:
        json.dump(split_meta, f, indent=2)

    # ===============================
    # 5. PCA 权重学习 (仅在 train runs)
    # ===============================
    perf_list, struct_list = [], []
    for r in pca_runs:
        rdir = os.path.join(args.output_dir, f"run_{r}")
        os.makedirs(rdir,exist_ok=True)
        perf_list.append(np.load(os.path.join(rdir, "x_perf.npy")))
        struct_list.append(np.load(os.path.join(rdir, "x_struct.npy")))

    scaler_perf_global, scaler_struct_global, w_perf, w_struct, pca_perf, pca_struct = learn_pca_weights_separate(
        perf_list, struct_list
    )

    # ===============================
    # 6. 每个 run 生成 label + mask + run_split
    #
    # mask 的统一语义（非常重要）：
    #   - mask == 1：该节点标签对“训练 loss”可见（只在 train run 上使用）
    #   - val run：mask 全 1（便于评估/早停），但训练脚本必须确保 loss 只在 train run 上计算
    #   - test run：默认 mask 全 0 且 true_label=-1（训练期间完全不可见），--reveal_test 才打开
    # ===============================
    for r in processed_runs:
        rdir = os.path.join(args.output_dir, f"run_{r}")
        os.makedirs(rdir,exist_ok=True)
        x_perf = np.load(os.path.join(rdir, "x_perf.npy"))
        x_struct = np.load(os.path.join(rdir, "x_struct.npy"))
        node_file = os.path.join(rdir, "node_features.csv")

        soft, _, _ = compute_soft_label_for_run(
            x_perf, x_struct,
            scaler_perf_global, scaler_struct_global,
            pca_perf, pca_struct,
            alpha=args.alpha
        )

        df_nodes = pd.read_csv(node_file)
        df_nodes['soft_label'] = soft

        # ---------- run_split 字段 ----------
        if r in train_ids:
            run_split = 'train'
        elif r in val_ids:
            run_split = 'val'
        else:
            run_split = 'test'
        df_nodes['run_split'] = run_split

        # ---------- mask / true_label ----------
        # true_label 默认不可见（-1），严格避免 test 泄露
        df_nodes['true_label'] = -1.0

        if run_split == "train":
            if getattr(args, "apply_mask", False):
                frac = float(args.mask_frac)  # frac 表示“隐藏比例”
                soft_vals = df_nodes['soft_label'].values

                # 你需要保证 stratified_mask 支持 mask_frac + seed：
                #   mask=1 表示可见标签
                mask = stratified_mask(soft_vals, mask_frac=frac, seed=getattr(args, "seed", 42)).astype(int)

                df_nodes['mask'] = mask
                df_nodes.loc[mask == 1, 'true_label'] = df_nodes.loc[mask == 1, 'soft_label']
            else:
                df_nodes['mask'] = 1
                df_nodes['true_label'] = df_nodes['soft_label']

        elif run_split == "val":
            # val：标签全可见用于早停/调参（但训练脚本不能把 val 加入 loss）
            df_nodes['mask'] = 1
            df_nodes['true_label'] = df_nodes['soft_label']

        else:  # test
            if getattr(args, "reveal_test", False):
                # 最终评估阶段：显式打开 test 标签
                df_nodes['mask'] = 1
                df_nodes['true_label'] = df_nodes['soft_label']
            else:
                # 训练期间：test 完全不可见
                df_nodes['mask'] = 0
                df_nodes['true_label'] = -1.0

        # ===============================
        # 保存 labels_and_masks.csv & 更新 node_features.csv
        # ===============================
        out_cols = ['node', 'soft_label', 'true_label', 'mask', 'run_split']
        df_nodes[out_cols].to_csv(os.path.join(rdir, "labels_and_masks.csv"), index=False)
        df_nodes.to_csv(node_file, index=False)

        print(f"✅ run_{r} => mask ratio = {df_nodes['mask'].mean():.3f}, split={run_split}")

        # 在 PCA soft_label 生成之后，再生成 KG triples
        generate_kg_triples(rdir)

    print("\n✅ 全流程完成：")
    print(" - run-level 顺序切分 train/val/test（70/15/15，小样本保护）")
    print(" - train run 支持 stratified 半监督 mask")
    print(" - val run 标签全可见（仅评估/早停，不参与训练 loss）")
    print(" - test run 默认不可见（--reveal_test 才揭示最终评估）")
    print(" - 输出 labels_and_masks.csv 可直接用于 GraphSAGE")
