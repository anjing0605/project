from __future__ import annotations

from pathlib import Path

import pandas as pd

from path.src.analysis.plotting import Plotter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_ROOT = Path(__file__).resolve().parents[1]


def plot_bar(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        raise RuntimeError(f"Empty dataframe for {title}")

    df = df.sort_values(by=y, ascending=False)

    plt.figure(figsize=(8, 4.5))
    plt.bar(df[x], df[y])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    metrics_dir = PATH_ROOT / "outputs" / "metrics"
    tables_dir = PATH_ROOT / "outputs" / "metrics_tables"
    figures_dir = PATH_ROOT / "outputs" / "figures" /"rule"
    figures_dir.mkdir(parents=True, exist_ok=True)

    overlap_csv = tables_dir / "mechanism_overlap_metrics.csv"
    path_type_csv = tables_dir / "mechanism_path_type_metrics.csv"
    budget_csv = metrics_dir / "fixed_node_budget_comparison.csv"

    if overlap_csv.exists():
        overlap_df = pd.read_csv(overlap_csv)

        plot_bar(
            df=overlap_df,
            x="method",
            y="avg_pairwise_node_overlap",
            out_path=figures_dir / "rule_avg_pairwise_node_overlap.png",
            title="Average pairwise internal-node overlap",
        )

        plot_bar(
            df=overlap_df,
            x="method",
            y="unique_internal_node_coverage",
            out_path=figures_dir / "rule_unique_internal_node_coverage.png",
            title="Unique internal-node coverage",
        )

        plot_bar(
            df=overlap_df,
            x="method",
            y="unique_edge_coverage",
            out_path=figures_dir / "rule_unique_edge_coverage.png",
            title="Unique edge coverage",
        )

    if path_type_csv.exists():
        path_type_df = pd.read_csv(path_type_csv)

        plot_bar(
            df=path_type_df,
            x="method",
            y="avg_path_length",
            out_path=figures_dir / "rule_avg_path_length.png",
            title="Average path length",
        )

        plot_bar(
            df=path_type_df,
            x="method",
            y="avg_cross_comm_ratio",
            out_path=figures_dir / "rule_avg_cross_comm_ratio.png",
            title="Average cross-community ratio",
        )

    if budget_csv.exists():
        budget_df = pd.read_csv(budget_csv)

        for metric in ["fragility_score", "delta_E", "delta_LCC", "delta_ASP"]:
            if metric not in budget_df.columns:
                continue

            for dataset in sorted(budget_df["dataset"].dropna().unique()):
                Plotter.plot_budget_curve(
                    df=budget_df,
                    dataset=dataset,
                    metric=metric,
                    save_path=figures_dir / f"{dataset}_fixed_budget_{metric}.png",
                    title=f"{dataset} - fixed node budget {metric}",
                )

    print(f"[DONE] figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()