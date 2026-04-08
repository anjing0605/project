from __future__ import annotations

from pathlib import Path

from path.src.analysis.ablation import AblationManager
from path.src.analysis.metrics_report import MetricsReporter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metrics_dir = PATH_ROOT / "outputs" / "metrics"
    out_csv = metrics_dir / "ablation_summary.csv"

    df = AblationManager.build_ablation_summary(metrics_dir)
    if df.empty:
        raise RuntimeError(f"No ablation metrics found under: {metrics_dir}")

    MetricsReporter.save_table(df, out_csv)

    print(f"[DONE] ablation table saved to: {out_csv}")
    print(df.head(20))


if __name__ == "__main__":
    main()