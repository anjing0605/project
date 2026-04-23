from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


DEFAULT_K_LIST = [1, 3, 5, 10]
#完整 comparison 里各方法的 ΔE@1/3/5/10、ΔLCC@1/3/5/10、ΔASP@1/3/5/10
'''
输出文件位于：    wide_path = path/outputs/metrics_tables / "comparison_topk_metrics_wide.csv"
    long_path = path/outputs/metrics_tables / "comparison_topk_metrics_long.csv"
'''
# python path/scripts/print_comparison_rule_metrics.py --input path/outputs/metrics/cora_rule_metrics.json
def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_comparison_root(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    支持两种输入：
    1) 完整 payload:
       {
         "shared_base_metrics": ...,
         "candidate_coverage": ...,
         "comparison": {
             "shared_base_metrics": ...,
             "methods": {...}
         }
       }

    2) comparison 本体:
       {
         "shared_base_metrics": ...,
         "methods": {...}
       }
    """
    if "comparison" in payload and isinstance(payload["comparison"], dict):
        comp = payload["comparison"]
    else:
        comp = payload

    if "methods" not in comp or not isinstance(comp["methods"], dict):
        raise ValueError(
            "Invalid comparison json: cannot find comparison['methods'] "
            "or payload['methods']."
        )
    return comp


def try_get_k_list(comp: Dict[str, Any], fallback: List[int]) -> List[int]:
    """
    优先从 comparison 内提取 k_list；
    如果没有，就用 fallback=[1,3,5,10]。
    """
    if "k_list" in comp and isinstance(comp["k_list"], list):
        return [int(x) for x in comp["k_list"]]

    methods = comp.get("methods", {})
    for _, m in methods.items():
        if isinstance(m, dict) and "k_list" in m and isinstance(m["k_list"], list):
            return [int(x) for x in m["k_list"]]

    return list(fallback)


def get_curve(method_payload: Dict[str, Any], key: str) -> List[float]:
    """
    兼容几种可能的字段名：
    - delta_E_curve / delta_LCC_curve / delta_ASP_curve
    - curves: {delta_E_curve: ...}
    """
    if key in method_payload and isinstance(method_payload[key], list):
        return [float(x) for x in method_payload[key]]

    curves = method_payload.get("curves", None)
    if isinstance(curves, dict) and key in curves and isinstance(curves[key], list):
        return [float(x) for x in curves[key]]

    raise KeyError(f"Cannot find curve '{key}' in method payload.")


def curve_to_k_map(curve: List[float], k_list: List[int]) -> Dict[int, float]:
    if len(curve) != len(k_list):
        raise ValueError(
            f"Curve length mismatch: len(curve)={len(curve)} vs len(k_list)={len(k_list)}"
        )
    return {int(k): float(v) for k, v in zip(k_list, curve)}


def build_wide_table(comp: Dict[str, Any], k_list: List[int]) -> pd.DataFrame:
    """
    输出宽表：
    method | ΔE@1 | ΔE@3 | ... | ΔLCC@1 | ... | ΔASP@10
    """
    rows: List[Dict[str, Any]] = []

    for method_name, method_payload in comp["methods"].items():
        row: Dict[str, Any] = {"method": method_name}

        delta_E_curve = get_curve(method_payload, "delta_E_curve")
        delta_LCC_curve = get_curve(method_payload, "delta_LCC_curve")
        delta_ASP_curve = get_curve(method_payload, "delta_ASP_curve")

        e_map = curve_to_k_map(delta_E_curve, k_list)
        lcc_map = curve_to_k_map(delta_LCC_curve, k_list)
        asp_map = curve_to_k_map(delta_ASP_curve, k_list)

        for k in k_list:
            row[f"ΔE@{k}"] = e_map[k]
        for k in k_list:
            row[f"ΔLCC@{k}"] = lcc_map[k]
        for k in k_list:
            row[f"ΔASP@{k}"] = asp_map[k]

        rows.append(row)

    df = pd.DataFrame(rows)

    # 按常见方法顺序排一下；不存在的自动忽略
    preferred_order = ["rule", "shortest", "betweenness", "node_score", "random", "rank", "rl"]
    existing = [m for m in preferred_order if m in df["method"].tolist()]
    others = [m for m in df["method"].tolist() if m not in existing]
    final_order = existing + sorted(others)

    df["__order__"] = df["method"].apply(lambda x: final_order.index(x))
    df = df.sort_values("__order__").drop(columns="__order__").reset_index(drop=True)
    return df


def build_long_table(comp: Dict[str, Any], k_list: List[int]) -> pd.DataFrame:
    """
    输出长表：
    method | k | delta_E | delta_LCC | delta_ASP
    """
    records: List[Dict[str, Any]] = []

    for method_name, method_payload in comp["methods"].items():
        delta_E_curve = get_curve(method_payload, "delta_E_curve")
        delta_LCC_curve = get_curve(method_payload, "delta_LCC_curve")
        delta_ASP_curve = get_curve(method_payload, "delta_ASP_curve")

        if not (len(delta_E_curve) == len(delta_LCC_curve) == len(delta_ASP_curve) == len(k_list)):
            raise ValueError(f"Curve length mismatch in method '{method_name}'.")

        for k, de, dlcc, dasp in zip(k_list, delta_E_curve, delta_LCC_curve, delta_ASP_curve):
            records.append(
                {
                    "method": method_name,
                    "k": int(k),
                    "delta_E": float(de),
                    "delta_LCC": float(dlcc),
                    "delta_ASP": float(dasp),
                }
            )

    df = pd.DataFrame(records)
    return df.sort_values(["method", "k"]).reset_index(drop=True)


def print_section(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to metrics.json or comparison json")
    parser.add_argument("--outdir", type=str, default="path/outputs/metrics_tables", help="Directory to save csv tables")
    parser.add_argument("--k_list", type=int, nargs="*", default=None, help="Override k list, e.g. --k_list 1 3 5 10")
    args = parser.parse_args()

    payload = load_json(args.input)
    comp = resolve_comparison_root(payload)

    k_list = args.k_list if args.k_list is not None and len(args.k_list) > 0 else try_get_k_list(comp, DEFAULT_K_LIST)

    wide_df = build_wide_table(comp, k_list)
    long_df = build_long_table(comp, k_list)

    print_section(wide_df, "Comparison Summary: ΔE / ΔLCC / ΔASP @ k")
    print_section(long_df, "Comparison Long Table")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wide_path = outdir / "comparison_topk_metrics_wide.csv"
    long_path = outdir / "comparison_topk_metrics_long.csv"

    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved wide table to: {wide_path}")
    print(f"Saved long table to: {long_path}")


if __name__ == "__main__":
    main()