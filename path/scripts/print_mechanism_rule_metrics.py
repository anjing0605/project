'''
输出：

mechanism_summary_wide.csv
mechanism_summary_long.csv
marginal_damage_long.csv
mechanism_metrics.json

它统计你要的三类证据：

overlap 机制
路径类型机制
per-k marginal damage 机
'''
#python path/scripts/print_mechanism_rule_metrics.py --input path/outputs/metrics/cora_rule_metrics.json
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


# =========================
# basic io
# =========================

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =========================
# path helpers
# =========================

def normalize_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def path_to_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    if nodes is None or len(nodes) < 2:
        return []
    return [normalize_edge(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


def internal_nodes(nodes: List[int]) -> List[int]:
    if nodes is None or len(nodes) <= 2:
        return []
    return nodes[1:-1]


def jaccard_overlap(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def safe_std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    mu = safe_mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


# =========================
# payload parsing
# =========================

def parse_k_values_arg(k_values_str: Optional[str]) -> Optional[List[int]]:
    if not k_values_str:
        return None
    vals = []
    for x in k_values_str.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals if vals else None


def get_k_values(payload: Dict[str, Any], cli_k_values: Optional[List[int]] = None) -> List[int]:
    """
    Priority:
      1) payload["k_list"]
      2) payload["comparison"]["k_list"]
      3) cli_k_values
      4) fallback to [1, 3, 5, 10]
    """
    if isinstance(payload.get("k_list"), list) and payload["k_list"]:
        return [int(x) for x in payload["k_list"]]

    comparison = payload.get("comparison", {})
    if isinstance(comparison, dict) and isinstance(comparison.get("k_list"), list) and comparison["k_list"]:
        return [int(x) for x in comparison["k_list"]]

    if cli_k_values:
        return [int(x) for x in cli_k_values]

    return [1, 3, 5, 10]


def get_method_to_paths(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Priority:
      1) payload["top_paths"][method]
      2) payload["comparison"]["methods"][method]["top_paths"]
      3) payload["comparison"]["methods"][method]["selected_paths"]
    """
    method_to_paths: Dict[str, List[Dict[str, Any]]] = {}

    top_paths = payload.get("top_paths")
    if isinstance(top_paths, dict) and top_paths:
        for method, recs in top_paths.items():
            if isinstance(recs, list):
                method_to_paths[method] = recs

    if not method_to_paths:
        methods = payload.get("comparison", {}).get("methods", {})
        if isinstance(methods, dict):
            for method, block in methods.items():
                if not isinstance(block, dict):
                    continue
                recs = block.get("top_paths")
                if recs is None:
                    recs = block.get("selected_paths")
                if isinstance(recs, list):
                    method_to_paths[method] = recs

    if not method_to_paths:
        raise ValueError(
            "Cannot find top path records. "
            "Expected one of: payload['top_paths'], "
            "payload['comparison']['methods'][method]['top_paths'], "
            "or payload['comparison']['methods'][method]['selected_paths']."
        )

    return method_to_paths


def get_method_to_damage_curves(payload: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    """
    Read damage curves from payload["comparison"]["methods"][method].

    Expected keys like:
      - delta_E_curve
      - delta_LCC_curve
      - delta_ASP_curve
      - fragility_score_curve
    """
    out: Dict[str, Dict[str, List[float]]] = {}
    methods = payload.get("comparison", {}).get("methods", {})
    if not isinstance(methods, dict):
        return out

    for method, block in methods.items():
        if not isinstance(block, dict):
            continue

        out[method] = {
            "delta_E_curve": list(block.get("delta_E_curve", []) or []),
            "delta_LCC_curve": list(block.get("delta_LCC_curve", []) or []),
            "delta_ASP_curve": list(block.get("delta_ASP_curve", []) or []),
            "fragility_score_curve": list(block.get("fragility_score_curve", []) or []),
        }

    return out


# =========================
# overlap analysis
# =========================

def compute_overlap_metrics(path_records: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    recs = list(path_records[:top_k])

    edge_sets: List[set] = []
    node_sets: List[set] = []

    for r in recs:
        nodes = list(r.get("nodes", []) or [])
        edges = set(path_to_edges(nodes))
        ints = set(internal_nodes(nodes))
        edge_sets.append(edges)
        node_sets.append(ints)

    pair_edge_overlaps: List[float] = []
    pair_node_overlaps: List[float] = []

    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            pair_edge_overlaps.append(jaccard_overlap(edge_sets[i], edge_sets[j]))
            pair_node_overlaps.append(jaccard_overlap(node_sets[i], node_sets[j]))

    unique_edges = set()
    unique_internal_nodes = set()
    for es in edge_sets:
        unique_edges |= es
    for ns in node_sets:
        unique_internal_nodes |= ns

    incremental_max_edge_overlap: List[float] = []
    incremental_mean_edge_overlap: List[float] = []
    incremental_max_node_overlap: List[float] = []
    incremental_mean_node_overlap: List[float] = []

    for i in range(len(recs)):
        if i == 0:
            incremental_max_edge_overlap.append(0.0)
            incremental_mean_edge_overlap.append(0.0)
            incremental_max_node_overlap.append(0.0)
            incremental_mean_node_overlap.append(0.0)
            continue

        edge_ovs = [jaccard_overlap(edge_sets[i], edge_sets[j]) for j in range(i)]
        node_ovs = [jaccard_overlap(node_sets[i], node_sets[j]) for j in range(i)]

        incremental_max_edge_overlap.append(max(edge_ovs) if edge_ovs else 0.0)
        incremental_mean_edge_overlap.append(safe_mean(edge_ovs))
        incremental_max_node_overlap.append(max(node_ovs) if node_ovs else 0.0)
        incremental_mean_node_overlap.append(safe_mean(node_ovs))

    return {
        "num_paths": len(recs),
        "avg_pairwise_edge_overlap": safe_mean(pair_edge_overlaps),
        "std_pairwise_edge_overlap": safe_std(pair_edge_overlaps),
        "max_pairwise_edge_overlap": max(pair_edge_overlaps) if pair_edge_overlaps else 0.0,
        "avg_pairwise_node_overlap": safe_mean(pair_node_overlaps),
        "std_pairwise_node_overlap": safe_std(pair_node_overlaps),
        "max_pairwise_node_overlap": max(pair_node_overlaps) if pair_node_overlaps else 0.0,
        "unique_edge_coverage": len(unique_edges),
        "unique_internal_node_coverage": len(unique_internal_nodes),
        "incremental_max_edge_overlap": incremental_max_edge_overlap,
        "incremental_mean_edge_overlap": incremental_mean_edge_overlap,
        "incremental_max_node_overlap": incremental_max_node_overlap,
        "incremental_mean_node_overlap": incremental_mean_node_overlap,
        "avg_unique_edges_per_path": len(unique_edges) / max(len(recs), 1),
        "avg_unique_internal_nodes_per_path": len(unique_internal_nodes) / max(len(recs), 1),
        "total_path_edges": sum(len(es) for es in edge_sets),
        "total_internal_nodes_with_repetition": sum(len(ns) for ns in node_sets),
        "edge_coverage_ratio": len(unique_edges) / max(sum(len(es) for es in edge_sets), 1),
        "internal_node_coverage_ratio": len(unique_internal_nodes) / max(sum(len(ns) for ns in node_sets), 1),
    }


# =========================
# path type analysis
# =========================

def get_feature(r: Dict[str, Any], key: str, default: float = 0.0) -> float:
    features = r.get("features", {}) or {}
    if key in features:
        try:
            return float(features[key])
        except Exception:
            return default
    return default


def compute_path_type_metrics(path_records: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    recs = list(path_records[:top_k])

    lengths = []
    cross_comm_ratios = []
    fragility_scores = []
    internal_node_importances = []

    for r in recs:
        lengths.append(float(r.get("length", get_feature(r, "path_length", 0.0))))
        cross_comm_ratios.append(get_feature(r, "cross_comm_ratio", 0.0))
        fragility_scores.append(get_feature(r, "fragility_score", 0.0))
        internal_node_importances.append(get_feature(r, "internal_node_importance", 0.0))

    return {
        "num_paths": len(recs),
        "avg_path_length": safe_mean(lengths),
        "std_path_length": safe_std(lengths),
        "avg_cross_comm_ratio": safe_mean(cross_comm_ratios),
        "std_cross_comm_ratio": safe_std(cross_comm_ratios),
        "avg_fragility_score": safe_mean(fragility_scores),
        "std_fragility_score": safe_std(fragility_scores),
        "avg_internal_node_importance": safe_mean(internal_node_importances),
        "std_internal_node_importance": safe_std(internal_node_importances),
    }


# =========================
# marginal damage analysis
# =========================

def diff_curve(curve: List[float]) -> List[float]:
    if not curve:
        return []
    out = [curve[0]]
    for i in range(1, len(curve)):
        out.append(curve[i] - curve[i - 1])
    return out


def compute_marginal_damage_metrics(
    damage_curves: Dict[str, List[float]],
    k_values: List[int],
) -> Dict[str, Any]:
    delta_E_curve = list(damage_curves.get("delta_E_curve", []) or [])
    delta_LCC_curve = list(damage_curves.get("delta_LCC_curve", []) or [])
    delta_ASP_curve = list(damage_curves.get("delta_ASP_curve", []) or [])
    fragility_score_curve = list(damage_curves.get("fragility_score_curve", []) or [])

    lengths = [
        len(delta_E_curve),
        len(delta_LCC_curve),
        len(delta_ASP_curve),
        len(fragility_score_curve),
    ]
    max_len = max(lengths) if lengths else 0

    if len(k_values) < max_len:
        raise ValueError(
            f"k_list length ({len(k_values)}) is shorter than available damage curve length ({max_len}). "
            f"k_list = {k_values}"
        )

    used_k_values = k_values[:max_len]

    return {
        "k_values": used_k_values,
        "delta_E_curve": delta_E_curve,
        "delta_LCC_curve": delta_LCC_curve,
        "delta_ASP_curve": delta_ASP_curve,
        "fragility_score_curve": fragility_score_curve,
        "marginal_delta_E": diff_curve(delta_E_curve),
        "marginal_delta_LCC": diff_curve(delta_LCC_curve),
        "marginal_delta_ASP": diff_curve(delta_ASP_curve),
        "marginal_fragility_score": diff_curve(fragility_score_curve),
    }


# =========================
# table builders
# =========================

def build_overlap_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for method, res in results.items():
        ov = res["overlap"]
        rows.append({
            "method": method,
            "num_paths": ov["num_paths"],
            "avg_pairwise_edge_overlap": ov["avg_pairwise_edge_overlap"],
            "std_pairwise_edge_overlap": ov["std_pairwise_edge_overlap"],
            "max_pairwise_edge_overlap": ov["max_pairwise_edge_overlap"],
            "avg_pairwise_node_overlap": ov["avg_pairwise_node_overlap"],
            "std_pairwise_node_overlap": ov["std_pairwise_node_overlap"],
            "max_pairwise_node_overlap": ov["max_pairwise_node_overlap"],
            "unique_edge_coverage": ov["unique_edge_coverage"],
            "unique_internal_node_coverage": ov["unique_internal_node_coverage"],

            "avg_unique_edges_per_path": ov["avg_unique_edges_per_path"],
            "avg_unique_internal_nodes_per_path": ov["avg_unique_internal_nodes_per_path"],
            "total_path_edges": ov["total_path_edges"],
            "total_internal_nodes_with_repetition": ov["total_internal_nodes_with_repetition"],
            "edge_coverage_ratio": ov["edge_coverage_ratio"],
            "internal_node_coverage_ratio": ov["internal_node_coverage_ratio"],
        })
    return pd.DataFrame(rows)


def build_path_type_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for method, res in results.items():
        pt = res["path_type"]
        rows.append({
            "method": method,
            "num_paths": pt["num_paths"],
            "avg_path_length": pt["avg_path_length"],
            "std_path_length": pt["std_path_length"],
            "avg_cross_comm_ratio": pt["avg_cross_comm_ratio"],
            "std_cross_comm_ratio": pt["std_cross_comm_ratio"],
            "avg_fragility_score": pt["avg_fragility_score"],
            "std_fragility_score": pt["std_fragility_score"],
            "avg_internal_node_importance": pt["avg_internal_node_importance"],
            "std_internal_node_importance": pt["std_internal_node_importance"],
        })
    return pd.DataFrame(rows)


def build_marginal_damage_long_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for method, res in results.items():
        mg = res["marginal_damage"]
        k_values = mg.get("k_values", [])

        n = max(
            len(mg.get("delta_E_curve", [])),
            len(mg.get("delta_LCC_curve", [])),
            len(mg.get("delta_ASP_curve", [])),
            len(mg.get("marginal_delta_E", [])),
            len(mg.get("marginal_delta_LCC", [])),
            len(mg.get("marginal_delta_ASP", [])),
        )

        for i in range(n):
            rows.append({
                "method": method,
                "k": k_values[i] if i < len(k_values) else None,
                "delta_E": mg["delta_E_curve"][i] if i < len(mg["delta_E_curve"]) else None,
                "delta_LCC": mg["delta_LCC_curve"][i] if i < len(mg["delta_LCC_curve"]) else None,
                "delta_ASP": mg["delta_ASP_curve"][i] if i < len(mg["delta_ASP_curve"]) else None,
                "marginal_delta_E": mg["marginal_delta_E"][i] if i < len(mg["marginal_delta_E"]) else None,
                "marginal_delta_LCC": mg["marginal_delta_LCC"][i] if i < len(mg["marginal_delta_LCC"]) else None,
                "marginal_delta_ASP": mg["marginal_delta_ASP"][i] if i < len(mg["marginal_delta_ASP"]) else None,
            })
    return pd.DataFrame(rows)


# =========================
# pretty print
# =========================

def print_overlap_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No overlap metrics.")
        return
    cols = [
        "method",
        "avg_pairwise_edge_overlap",
        "avg_pairwise_node_overlap",
        "unique_edge_coverage",
        "unique_internal_node_coverage",
    ]
    print("=" * 90)
    print("Mechanism Summary: Overlap / Coverage")
    print("=" * 90)
    print(df[cols].to_string(index=False))


def print_path_type_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No path-type metrics.")
        return
    cols = [
        "method",
        "avg_path_length",
        "avg_cross_comm_ratio",
        "avg_fragility_score",
        "avg_internal_node_importance",
    ]
    print("\n" + "=" * 90)
    print("Mechanism Summary: Path Type")
    print("=" * 90)
    print(df[cols].to_string(index=False))


def print_marginal_damage_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No marginal-damage metrics.")
        return

    cols = [
        "method",
        "k",
        "delta_E",
        "delta_LCC",
        "delta_ASP",
        "marginal_delta_E",
        "marginal_delta_LCC",
        "marginal_delta_ASP",
    ]
    print("\n" + "=" * 90)
    print("Mechanism Summary: Marginal Damage (Long Table)")
    print("=" * 90)
    print(df[cols].to_string(index=False))


# =========================
# main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print mechanism metrics for critical-path comparison results."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to metrics json, e.g. path/outputs/metrics/cora_rule_metrics.json"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="How many top paths per method to analyze (default: 10)"
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default=None,
        help="Optional fallback k values, e.g. '1,3,5,10'. "
             "If payload contains k_list, payload k_list takes precedence."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="path/outputs/metrics_tables",
        help="Output directory for mechanism tables."
    )
    args = parser.parse_args()

    payload = load_json(args.input)
    cli_k_values = parse_k_values_arg(args.k_values)
    k_values = get_k_values(payload, cli_k_values)

    method_to_paths = get_method_to_paths(payload)
    method_to_damage = get_method_to_damage_curves(payload)

    input_path = Path(args.input)
    if args.output_dir is None:
        output_dir = input_path.parent.parent / "mechanism_tables"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for method, recs in method_to_paths.items():
        results[method] = {
            "overlap": compute_overlap_metrics(recs, top_k=args.topk),
            "path_type": compute_path_type_metrics(recs, top_k=args.topk),
            "marginal_damage": compute_marginal_damage_metrics(
                method_to_damage.get(method, {}),
                k_values=k_values,
            ),
        }

    overlap_df = build_overlap_table(results).sort_values(
        by=["avg_pairwise_edge_overlap", "avg_pairwise_node_overlap"],
        ascending=[True, True]
    )
    path_type_df = build_path_type_table(results).sort_values(
        by=["avg_fragility_score"],
        ascending=[False]
    )
    marginal_damage_df = build_marginal_damage_long_table(results)

    overlap_csv = output_dir / "mechanism_overlap_metrics.csv"
    path_type_csv = output_dir / "mechanism_path_type_metrics.csv"
    marginal_damage_csv = output_dir / "mechanism_marginal_damage_metrics.csv"
    mechanism_json = output_dir / "mechanism_metrics.json"

    save_csv(overlap_csv, overlap_df)
    save_csv(path_type_csv, path_type_df)
    save_csv(marginal_damage_csv, marginal_damage_df)
    save_json(mechanism_json, results)

    print(f"[INFO] using k_list = {k_values}")
    print_overlap_summary(overlap_df)
    print_path_type_summary(path_type_df)
    print_marginal_damage_summary(marginal_damage_df)

    print()
    print(f"Saved overlap table to: {overlap_csv}")
    print(f"Saved path-type table to: {path_type_csv}")
    print(f"Saved marginal-damage table to: {marginal_damage_csv}")
    print(f"Saved mechanism json to: {mechanism_json}")


if __name__ == "__main__":
    main()