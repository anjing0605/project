# paper_results_pipeline_advanced.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


# =========================================================
# 全局绘图风格（论文打印版）
# =========================================================

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",

    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    "axes.linewidth": 1.0,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,

    "savefig.dpi": 300,
    "figure.dpi": 120
})


# =========================================================
# 配置
# =========================================================

BASE = "scratch/project/public_process/extract"

DATASETS = [
    "Cora", "CiteSeer", "PubMed",
    "Computers", "Photo",
    "CS", "Physics"
]

SEEDS = [0, 1, 2, 3, 4]

RESULT_FILE = "baseline_results.csv"

OUTPUT_DIR = "scratch/project/public_process/train/baseline_results"

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR = os.path.join(OUTPUT_DIR, "tables")
STAT_DIR = os.path.join(OUTPUT_DIR, "stats")

for d in [OUTPUT_DIR, FIG_DIR, TAB_DIR, STAT_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================================================
# 工具函数
# =========================================================

def save_fig(name):
    plt.savefig(
        os.path.join(FIG_DIR, f"{name}.pdf"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(FIG_DIR, f"{name}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def highlight_best_second(df, metric):
    values = df[metric].values
    order = np.argsort(values)[::-1]
    best = order[0]
    second = order[1]
    return best, second


def reorder_methods(existing_methods):
    preferred = [
        "Degree",
        "K-shell",
        "PR",
        "MLP",
        "GCN",
        "GAT",
        "SAGE",
        "Ours"
    ]
    preferred = [m for m in preferred if m in existing_methods]
    remaining = [m for m in existing_methods if m not in preferred]
    return preferred + remaining


# =========================================================
# 读取多seed结果
# =========================================================

dataset_results = []

for dataset in DATASETS:
    records = []

    for seed in SEEDS:
        path = os.path.join(
            BASE,
            f"{dataset}_struct",
            "result",
            f"results_seed_{seed}",
            RESULT_FILE
        )

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, index_col=0)
        df["Method"] = df.index
        df["Seed"] = seed

        records.append(df.reset_index(drop=True))

    if len(records) == 0:
        continue

    seed_df = pd.concat(records, ignore_index=True)

    mean_df = seed_df.groupby("Method")[["SP", "KD", "Top10", "Top20"]].mean()
    std_df = seed_df.groupby("Method")[["SP", "KD", "Top10", "Top20"]].std()

    summary = mean_df.copy()
    summary["SP_std"] = std_df["SP"]
    summary["KD_std"] = std_df["KD"]
    summary["Top10_std"] = std_df["Top10"]
    summary["Top20_std"] = std_df["Top20"]

    summary["Dataset"] = dataset
    summary["Method"] = summary.index

    dataset_results.append(summary.reset_index(drop=True))

summary_df = pd.concat(dataset_results, ignore_index=True)


# =========================================================
# overall统计
# =========================================================

overall_mean = summary_df.groupby("Method")[["SP", "KD", "Top10", "Top20"]].mean()
overall_std = summary_df.groupby("Method")[["SP", "KD", "Top10", "Top20"]].std()

overall = overall_mean.copy()
overall["SP_std"] = overall_std["SP"]
overall["KD_std"] = overall_std["KD"]
overall["Top10_std"] = overall_std["Top10"]
overall["Top20_std"] = overall_std["Top20"]

overall = overall.sort_values("SP", ascending=False)


# =========================================================
# Excel表
# =========================================================

overall.to_csv(
    os.path.join(TAB_DIR, "overall_results.csv")
)

summary_df.to_csv(
    os.path.join(TAB_DIR, "dataset_results.csv"),
    index=False
)


# =========================================================
# Latex表
# =========================================================

best_sp, second_sp = highlight_best_second(overall, "SP")
best_kd, second_kd = highlight_best_second(overall, "KD")

lines = []
lines.append("\\begin{tabular}{lcc}")
lines.append("\\toprule")
lines.append("Method & Spearman & Kendall\\\\")
lines.append("\\midrule")

for i, (m, row) in enumerate(overall.iterrows()):
    sp = f"{row['SP']:.4f}±{row['SP_std']:.4f}"
    kd = f"{row['KD']:.4f}±{row['KD_std']:.4f}"

    if i == best_sp:
        sp = "\\textbf{" + sp + "}"
    elif i == second_sp:
        sp = "\\underline{" + sp + "}"

    if i == best_kd:
        kd = "\\textbf{" + kd + "}"
    elif i == second_kd:
        kd = "\\underline{" + kd + "}"

    lines.append(f"{m} & {sp} & {kd}\\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")

with open(os.path.join(TAB_DIR, "latex_table.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# =========================================================
# 显著性检验
# =========================================================

best_method = overall.index[0]

records = []

for m in overall.index[1:]:
    a = summary_df[summary_df["Method"] == best_method]["SP"]
    b = summary_df[summary_df["Method"] == m]["SP"]

    stat, p = ttest_rel(a, b)

    records.append({
        "compare": f"{best_method} vs {m}",
        "p_value": p
    })

pd.DataFrame(records).to_csv(
    os.path.join(STAT_DIR, "significance_tests.csv"),
    index=False
)


# =========================================================
# 绘图前准备
# =========================================================

methods = list(overall.index)
methods_ordered = reorder_methods(methods)

x = np.arange(len(methods))
width = 0.35


# =========================================================
# 图1 平均SP
# =========================================================

plt.figure(figsize=(8.2, 4.8))

vals = overall["SP"]
errs = overall["SP_std"]

bars = plt.bar(methods, vals, yerr=errs, capsize=4)

bars[best_sp].set_color("tab:red")
bars[second_sp].set_color("tab:orange")

plt.xticks(rotation=35, ha="right")
plt.ylabel("Spearman")
plt.xlabel("Method")
plt.title("Average Performance")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
save_fig("fig1_avg_bar")


# =========================================================
# 图2 grouped
# =========================================================

plt.figure(figsize=(8.8, 4.8))

plt.bar(x - width / 2, overall["SP"], width, label="SP")
plt.bar(x + width / 2, overall["KD"], width, label="KD")

plt.xticks(x, methods, rotation=35, ha="right")
plt.ylabel("Score")
plt.xlabel("Method")
plt.title("Overall SP and KD Comparison")
plt.legend(frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
save_fig("fig2_grouped_metrics")


# =========================================================
# 图3 dataset comparison
# 同时输出柱状图 + 折线图
# =========================================================

pivot = summary_df.pivot(index="Dataset", columns="Method", values="SP")
pivot = pivot.reindex(DATASETS)

method_order = reorder_methods(list(pivot.columns))
pivot = pivot[method_order]

# 图3-1 柱状图
ax = pivot.plot(
    kind="bar",
    figsize=(11, 5.0),
    width=0.82
)

ax.set_ylabel("Spearman")
ax.set_xlabel("Dataset")
ax.set_title("Dataset-wise Spearman Comparison")

ax.legend(
    title="Method",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
    frameon=True
)

ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
save_fig("fig3_dataset_bar")

# 图3-2 折线图
plt.figure(figsize=(11, 5.0))

markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', '+']

for i, method in enumerate(method_order):
    plt.plot(
        pivot.index,
        pivot[method],
        marker=markers[i % len(markers)],
        linewidth=2.2,
        markersize=7,
        label=method
    )

plt.ylabel("Spearman")
plt.xlabel("Dataset")
plt.title("Dataset-wise Spearman Comparison (Line Plot)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.legend(
    title="Method",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
    frameon=True
)

plt.tight_layout()
save_fig("fig3_dataset_line")


# =========================================================
# 图4 boxplot
# =========================================================

methods_box = reorder_methods(list(summary_df["Method"].unique()))

data = [
    summary_df[summary_df["Method"] == m]["SP"].values
    for m in methods_box
]

plt.figure(figsize=(9.2, 4.8))

plt.boxplot(
    data,
    labels=methods_box,
    patch_artist=False,
    showmeans=True
)

plt.xticks(rotation=35, ha="right")
plt.ylabel("Spearman")
plt.xlabel("Method")
plt.title("Distribution of Spearman Across Datasets")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
save_fig("fig4_boxplot_sp")


# =========================================================
# 图5 topk
# =========================================================

plt.figure(figsize=(8.8, 4.8))

plt.bar(x - width / 2, overall["Top10"], width, label="Top10")
plt.bar(x + width / 2, overall["Top20"], width, label="Top20")

plt.xticks(x, methods, rotation=35, ha="right")
plt.ylabel("Score")
plt.xlabel("Method")
plt.title("Top-k Performance Comparison")
plt.legend(frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
save_fig("fig5_topk_bar")


# =========================================================
# 图6 radar
# =========================================================

radar_labels = ["SP", "KD", "Top10", "Top20"]
angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(7.0, 7.0))
ax = plt.subplot(111, polar=True)

top_methods = list(overall.index[:4])

for m in top_methods:
    values = overall.loc[m, radar_labels].astype(float).tolist()
    values += values[:1]
    ax.plot(angles, values, label=m, linewidth=2.2)
    ax.fill(angles, values, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels)
ax.set_title("Radar Comparison of Top Methods", pad=20)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(
    bbox_to_anchor=(1.12, 1.08),
    loc="upper left",
    frameon=True
)

plt.tight_layout()
save_fig("fig6_radar")


# =========================================================
# 图7 ablation
# =========================================================

if "L0_MLP" in overall.index:
    ablation_methods = [m for m in overall.index if m.startswith("L")]
    ablation_vals = overall.loc[ablation_methods]["SP"]

    plt.figure(figsize=(8.2, 4.8))

    plt.plot(
        ablation_methods,
        ablation_vals,
        marker="o",
        linewidth=2.2,
        markersize=7
    )

    plt.ylabel("Spearman")
    plt.xlabel("Variant")
    plt.title("Ablation Study")
    plt.grid(True, linestyle="--", alpha=0.5)

    for i, v in enumerate(ablation_vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_fig("fig7_ablation")


print("All results generated.")