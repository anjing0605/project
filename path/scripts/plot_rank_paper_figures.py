from __future__ import annotations

from pathlib import Path
import json
from itertools import combinations
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

from path.src.analysis.metrics_report import MetricsReporter
from path.src.analysis.plotting import Plotter


PATH_ROOT = Path(__file__).resolve().parents[1]

RANK_METHODS = [
    "rank_pure",
    "rank_set_pred_score_selector",
    "rank_set_submodular_greedy",
]
METHOD_NAME_MAP = {
    "rank_pure": "PureRank",
    "rank_set_pred_score_selector": "RankSet-Pred",
    "rank_set_submodular_greedy": "RankSet-Submod",
}

RANK_METHOD_LABELS = [
    METHOD_NAME_MAP[m] for m in RANK_METHODS
]

DATASET_NAME_MAP = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "pubmed": "Pubmed",
    "computers": "Computers",
    "photo": "Photo",
    "cs": "CS",
    "physics": "Physics",
}

def _load_or_build_method_table(metrics_dir: Path) -> pd.DataFrame:
    table_path = metrics_dir / "all_method_comparison.csv"

    if table_path.exists():
        return pd.read_csv(table_path)

    df = MetricsReporter.build_method_comparison_table(metrics_dir)
    MetricsReporter.save_table(df, table_path)
    return df


def _load_or_build_path_quality_table(paths_dir: Path, metrics_dir: Path) -> pd.DataFrame:
    table_path = metrics_dir / "all_path_quality.csv"

    if table_path.exists():
        return pd.read_csv(table_path)

    df = MetricsReporter.build_path_quality_table(paths_dir=paths_dir, top_n=10)
    MetricsReporter.save_table(df, table_path)
    return df


def _plot_bar(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    if df.empty:
        print(f"[WARN] Empty dataframe, skip {out_path.name}")
        return

    df = df.copy()

    if x == "method_label":
        df[x] = pd.Categorical(
            df[x],
            categories=RANK_METHOD_LABELS,
            ordered=True,
        )
        df = df.sort_values(by=x)
    else:
        df = df.sort_values(by=y, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(df[x].astype(str), df[y])
    ax.set_xlabel("Rank method")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    if not out_path.exists():
        raise RuntimeError(f"Figure was not saved: {out_path}")


def _normalize_dataset_name(name: str) -> str:
    """
    兼容 all_path_quality.csv 中 rank path 的 dataset 可能被写成：
    cora_rank_pure_rank_paths
    cora_rank_rank_set_pred_score_selector_paths
    cora_rank_rank_set_submodular_greedy_paths

    统一转成：
    Cora
    """
    name = str(name)

    if "_rank_" in name:
        name = name.split("_rank_")[0]

    mapping = {
        "cora": "Cora",
        "citeseer": "Citeseer",
        "pubmed": "Pubmed",
        "computers": "Computers",
        "photo": "Photo",
        "cs": "CS",
        "physics": "Physics",
    }

    return mapping.get(name.lower(), name)

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_rank_method_from_filename(filename: str) -> str:
    if "pure_rank" in filename:
        return "rank_pure"
    if "pred_score_selector" in filename:
        return "rank_set_pred_score_selector"
    if "submodular_greedy" in filename:
        return "rank_set_submodular_greedy"
    return "rank_unknown"


def _parse_dataset_from_rank_filename(filename: str) -> str:
    """
    例如：
    cora_rank_pure_rank_paths.json -> Cora
    cora_rank_rank_set_pred_score_selector_paths.json -> Cora
    """
    raw = filename.split("_rank_")[0]
    return DATASET_NAME_MAP.get(raw.lower(), raw)


def _path_edges(nodes: list[int]) -> set[tuple[int, int]]:
    """
    Cora / Citeseer / Pubmed 通常按无向图处理，
    所以边统一排序，避免 (u,v) 和 (v,u) 被当成两条边。
    """
    edges = set()
    for u, v in zip(nodes[:-1], nodes[1:]):
        edges.add(tuple(sorted((int(u), int(v)))))
    return edges


def _internal_nodes(nodes: list[int]) -> set[int]:
    """
    只统计内部节点，不统计 source 和 target。
    因为 source / target 来自任务对，直接纳入 overlap 会污染结果。
    """
    if len(nodes) <= 2:
        return set()
    return set(map(int, nodes[1:-1]))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _mean_pairwise_overlap(items: list[set]) -> float:
    if len(items) < 2:
        return 0.0

    vals = []
    for a, b in combinations(items, 2):
        vals.append(_jaccard(a, b))

    if not vals:
        return 0.0

    return float(sum(vals) / len(vals))


def build_rank_overlap_coverage_table(paths_dir: Path, top_n: int = 10) -> pd.DataFrame:
    """
    读取 rank 三种模式的 selected paths，
    计算 Top-n 路径集合的：
    1. 平均 pairwise edge overlap
    2. 平均 pairwise internal-node overlap
    3. unique edge coverage
    4. unique internal-node coverage
    """
    rows = []

    for path in paths_dir.glob("*rank*paths.json"):
        name = path.name

        if "_rule_" in name or "_rl_eval_" in name:
            continue

        method = _parse_rank_method_from_filename(name)
        if method not in RANK_METHODS:
            continue

        dataset = _parse_dataset_from_rank_filename(name)

        records = _load_json(path)
        if not isinstance(records, list):
            continue

        records = records[:top_n]

        edge_sets = []
        internal_node_sets = []

        for r in records:
            nodes = r.get("nodes", [])
            if not nodes:
                continue

            edge_sets.append(_path_edges(nodes))
            internal_node_sets.append(_internal_nodes(nodes))

        unique_edges = set()
        for s in edge_sets:
            unique_edges |= s

        unique_internal_nodes = set()
        for s in internal_node_sets:
            unique_internal_nodes |= s

        total_edges = sum(len(s) for s in edge_sets)
        total_internal_nodes = sum(len(s) for s in internal_node_sets)

        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_NAME_MAP.get(method, method),
                "top_n": top_n,
                "avg_edge_overlap": _mean_pairwise_overlap(edge_sets),
                "avg_internal_node_overlap": _mean_pairwise_overlap(internal_node_sets),
                "unique_edge_coverage": len(unique_edges),
                "unique_internal_node_coverage": len(unique_internal_nodes),
                "total_edges": total_edges,
                "total_internal_nodes": total_internal_nodes,
                "edge_coverage_ratio": len(unique_edges) / total_edges if total_edges > 0 else 0.0,
                "internal_node_coverage_ratio": (
                    len(unique_internal_nodes) / total_internal_nodes
                    if total_internal_nodes > 0
                    else 0.0
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "method_label",
                "top_n",
                "avg_edge_overlap",
                "avg_internal_node_overlap",
                "unique_edge_coverage",
                "unique_internal_node_coverage",
                "total_edges",
                "total_internal_nodes",
                "edge_coverage_ratio",
                "internal_node_coverage_ratio",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["dataset", "method"]
    ).reset_index(drop=True)


def _plot_rank_grouped_bar(
    df: pd.DataFrame,
    dataset: str,
    metrics: list[str],
    save_path: Path,
    title: str,
    ylabel: str,
) -> None:
    sub = df[df["dataset"] == dataset].copy()

    if sub.empty:
        print(f"[WARN] No rows for dataset={dataset}, skip {save_path.name}")
        return

    sub = sub.set_index("method_label")

    missing = [m for m in metrics if m not in sub.columns]
    if missing:
        print(f"[WARN] Missing metrics {missing}, skip {save_path.name}")
        return

    plot_df = sub[metrics]

    ax = plot_df.plot(kind="bar", figsize=(8, 4.8))

    ax.set_xlabel("Rank method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220)
    plt.close()
def main() -> None:
    metrics_dir = PATH_ROOT / "outputs" / "metrics"
    paths_dir = PATH_ROOT / "outputs" / "paths"
    fig_dir = PATH_ROOT / "outputs" / "figures" / "rank"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. 画 rank 三模式的 Top-k damage curve
    method_df = _load_or_build_method_table(metrics_dir)

    if method_df.empty:
        raise RuntimeError("No metrics rows found. Run rank experiments first.")

    rank_df = method_df[method_df["method"].isin(RANK_METHODS)].copy()

    if rank_df.empty:
        raise RuntimeError(
            "No rank rows found in all_method_comparison.csv. "
            "Run run_rank first, then rebuild method comparison table."
        )

    damage_metrics = [
        "delta_E",
        "delta_LCC",
        "delta_ASP",
        "fragility_score",
        "num_removed_nodes",
    ]

    plot_df = rank_df.copy()
    plot_df["method"] = plot_df["method"].map(METHOD_NAME_MAP).fillna(plot_df["method"])

    datasets = sorted(plot_df["dataset"].dropna().unique())

    for dataset in datasets:
        for metric in damage_metrics:
            if metric not in plot_df.columns:
                continue

            out_path = fig_dir / f"{dataset}_rank_{metric}_curve.png"

            Plotter.plot_metric_curve(
                df=plot_df,
                dataset=dataset,
                metric=metric,
                methods=RANK_METHOD_LABELS,
                save_path=out_path,
                title=f"{dataset} - Rank methods: {metric} vs Top-k",
            )

            print(f"[SAVE] {out_path}")

    # 2. 画 rank 三模式的路径质量柱状图
    quality_df = _load_or_build_path_quality_table(paths_dir, metrics_dir)

    if quality_df.empty:
        print("[WARN] all_path_quality.csv is empty. Skip path quality figures.")
    else:
        quality_df = quality_df[quality_df["method"].isin(RANK_METHODS)].copy()

        if quality_df.empty:
            print("[WARN] No rank rows found in all_path_quality.csv. Skip path quality figures.")
        else:
            quality_df["dataset_norm"] = quality_df["dataset"].apply(_normalize_dataset_name)

            quality_metrics = [
                "path_length",
                "cross_comm_ratio",
                "avg_node_importance",
                "avg_edge_bc",
                "fragility_score",
            ]

            for dataset in sorted(quality_df["dataset_norm"].dropna().unique()):
                sub = quality_df[quality_df["dataset_norm"] == dataset].copy()

                for metric in quality_metrics:
                    if metric not in sub.columns:
                        continue

                    bar_df = (
                        sub.groupby("method", as_index=False)[metric]
                        .mean()
                        .dropna()
                    )

                    bar_df["method_label"] = bar_df["method"].map(METHOD_NAME_MAP).fillna(bar_df["method"])

                    out_path = fig_dir / f"{dataset}_rank_{metric}_bar.png"
                    _plot_bar(
                        df=bar_df,
                        x="method_label",
                        y=metric,
                        out_path=out_path,
                        title=f"{dataset} - Rank methods: average {metric}",
                    )

                    if out_path.exists():
                        print(f"[SAVE] {out_path}")
                    else:
                        print(f"[WARN] Figure not found after plotting: {out_path}")


    # =========================
    # Rank overlap / coverage figures
    # =========================
    paths_dir = PATH_ROOT / "outputs" / "paths"

    rank_mech_df = build_rank_overlap_coverage_table(
        paths_dir=paths_dir,
        top_n=10,
    )

    if rank_mech_df.empty:
        print("[WARN] No rank path records found. Skip overlap / coverage figures.")
        return

    mech_csv = metrics_dir / "rank_overlap_coverage.csv"
    rank_mech_df.to_csv(mech_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {mech_csv}")

    for dataset in sorted(rank_mech_df["dataset"].dropna().unique()):
        overlap_path = fig_dir / f"{dataset}_rank_overlap_bar.png"
        coverage_path = fig_dir / f"{dataset}_rank_coverage_bar.png"

        _plot_rank_grouped_bar(
            df=rank_mech_df,
            dataset=dataset,
            metrics=[
                "avg_edge_overlap",
                "avg_internal_node_overlap",
            ],
            save_path=overlap_path,
            title=f"{dataset} - Rank path overlap",
            ylabel="Average pairwise overlap",
        )
        print(f"[SAVE] {overlap_path}")

        _plot_rank_grouped_bar(
            df=rank_mech_df,
            dataset=dataset,
            metrics=[
                "unique_edge_coverage",
                "unique_internal_node_coverage",
            ],
            save_path=coverage_path,
            title=f"{dataset} - Rank path coverage",
            ylabel="Unique coverage count",
        )
        print(f"[SAVE] {coverage_path}")

    print(f"[DONE] rank figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()