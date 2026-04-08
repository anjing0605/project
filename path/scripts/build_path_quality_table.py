from __future__ import annotations

from pathlib import Path

from path.src.analysis.metrics_report import MetricsReporter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths_dir = PATH_ROOT / "outputs" / "paths"
    out_csv = PATH_ROOT / "outputs" / "metrics" / "all_path_quality.csv"

    df = MetricsReporter.build_path_quality_table(paths_dir=paths_dir, top_n=10)
    if df.empty:
        raise RuntimeError(f"No path json found under: {paths_dir}")

    MetricsReporter.save_table(df, out_csv)

    print(f"[DONE] path quality table saved to: {out_csv}")
    print(df.head(20))


if __name__ == "__main__":
    main()