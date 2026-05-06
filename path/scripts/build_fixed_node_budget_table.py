from __future__ import annotations

from pathlib import Path

from path.src.analysis.metrics_report import MetricsReporter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metrics_dir = PATH_ROOT / "outputs" / "metrics"
    out_csv = metrics_dir / "fixed_node_budget_comparison.csv"

    df = MetricsReporter.build_fixed_node_budget_table(metrics_dir)
    if df.empty:
        raise RuntimeError(
            f"No fixed-node-budget metrics found under: {metrics_dir}. "
            "Run path.scripts.run_rule first and make sure "
            "fixed_node_budget_comparison is saved in rule metrics json."
        )

    MetricsReporter.save_table(df, out_csv)

    print(f"[DONE] fixed-node-budget table saved to: {out_csv}")
    print(df.head(30))


if __name__ == "__main__":
    main()