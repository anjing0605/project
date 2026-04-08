from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"D:\project\keynode\project")
EXTRACT_ROOT = PROJECT_ROOT / "public_process" / "extract"

DATASETS = [
    "CiteSeer_struct",
    "Computers_struct",
    "Cora_struct",
    "CS_struct",
    "Photo_struct",
    "Physics_struct",
    "PubMed_struct",
]

SEED_DIRS = [f"results_seed_{i}" for i in range(5)]
SCORE_FILENAME = "gnn_node_scores.csv"
OUTPUT_FILENAME = "gnn_node_scores_mean.csv"


def load_one_score_file(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"node", "gnn_score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{csv_path} 缺少必须列，当前列为: {list(df.columns)}，需要列: {required_cols}"
        )

    df = df[["node", "gnn_score"]].copy()
    df["node"] = df["node"].astype(int)
    df["gnn_score"] = df["gnn_score"].astype(float)

    if df["node"].duplicated().any():
        dup = df.loc[df["node"].duplicated(), "node"].tolist()
        raise ValueError(f"{csv_path} 中 node 重复，例如: {dup[:10]}")

    return df.sort_values("node").reset_index(drop=True)


def merge_dataset_scores(dataset_name: str) -> None:
    dataset_root = EXTRACT_ROOT / dataset_name
    result_root = dataset_root / "result"

    seed_dfs = []
    for seed_dir in SEED_DIRS:
        csv_path = result_root / seed_dir / SCORE_FILENAME
        df = load_one_score_file(csv_path)
        df = df.rename(columns={"gnn_score": f"gnn_score_{seed_dir}"})
        seed_dfs.append(df)

    merged = seed_dfs[0]
    for df in seed_dfs[1:]:
        merged = merged.merge(df, on="node", how="inner")

    expected_rows = len(seed_dfs[0])
    if len(merged) != expected_rows:
        raise ValueError(
            f"{dataset_name} 合并后行数异常: 原始 {expected_rows}, 合并后 {len(merged)}。"
            f"说明不同 seed 的 node 集不一致。"
        )

    score_cols = [c for c in merged.columns if c.startswith("gnn_score_results_seed_")]
    merged["gnn_score"] = merged[score_cols].mean(axis=1)

    out_df = merged[["node", "gnn_score"]].copy()
    out_df = out_df.sort_values("node").reset_index(drop=True)

    output_path = dataset_root / OUTPUT_FILENAME
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] {dataset_name} -> {output_path}")


def main() -> None:
    for dataset_name in DATASETS:
        try:
            merge_dataset_scores(dataset_name)
        except Exception as e:
            print(f"[ERROR] {dataset_name}: {e}")


if __name__ == "__main__":
    main()