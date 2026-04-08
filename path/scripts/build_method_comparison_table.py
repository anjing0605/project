from __future__ import annotations

from pathlib import Path

from path.src.analysis.metrics_report import MetricsReporter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_ROOT = Path(__file__).resolve().parents[1]
'''python -m path.scripts.build_method_comparison_table
python -m path.scripts.build_path_quality_table
python -m path.scripts.plot_rl_training_curves
python -m path.scripts.run_ablation_experiments
python -m path.scripts.build_ablation_table'''

def main() -> None:
    metrics_dir = PATH_ROOT / "outputs" / "metrics"
    out_csv = metrics_dir / "all_method_comparison.csv"

    df = MetricsReporter.build_method_comparison_table(metrics_dir)
    if df.empty:
        raise RuntimeError(f"No comparison metrics found under: {metrics_dir}")

    MetricsReporter.save_table(df, out_csv)

    print(f"[DONE] method comparison table saved to: {out_csv}")
    print(df.head(20))


if __name__ == "__main__":
    main()