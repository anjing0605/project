import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
plt.style.use("seaborn-v0_8-whitegrid")

BASE = "scratch/keynode/project/public_process/extract"

DATASETS = [
    "Cora",
    "CiteSeer",
    "PubMed",
    "Computers",
    "Photo",
    "CS",
    "Physics"
]

OUT = "scratch/keynode/project/public_process/train/ablation_results"
os.makedirs(OUT, exist_ok=True)


def save_fig(name):
    plt.savefig(os.path.join(OUT, name + ".pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close()


def load_results(filename):
    rows = []
    for dataset in DATASETS:
        path = os.path.join(BASE, f"{dataset}_struct", filename)
        df = pd.read_csv(path, index_col=0)
        df["dataset"] = dataset
        df["method"] = df.index
        rows.append(df.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)

def filter_progressive(df):

    keep = [
        "L0_MLP",
        "L1_1Layer_GNN",
        "L2_2Layer_GNN",
        "L5_Full"
    ]

    df = df[df["method"].isin(keep)].copy()

    # rename
    df["method"] = df["method"].replace({
        "L0_MLP": "L0",
        "L1_1Layer_GNN": "L1",
        "L2_2Layer_GNN": "L2",
        "L5_Full": "L3_Full"
    })

    return df
def aggregate(df):
    mean = df.groupby("method")[["SP_mean","KD_mean"]].mean()
    std = df.groupby("method")[["SP_mean","KD_mean"]].std()

    mean["SP_std"] = std["SP_mean"]
    mean["KD_std"] = std["KD_mean"]

    return mean


def highlight_best_second(series):
    order = series.sort_values(ascending=False)
    best = order.index[0]
    second = order.index[1]

    formatted = {}
    for idx,val in series.items():
        text = f"{val:.4f}"
        if idx == best:
            text = f"\\textbf{{{text}}}"
        elif idx == second:
            text = f"\\underline{{{text}}}"
        formatted[idx] = text
    return formatted


def export_latex(df, filename):
    sp_fmt = highlight_best_second(df["SP_mean"])
    kd_fmt = highlight_best_second(df["KD_mean"])

    lines = []
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Method & Spearman & Kendall\\\\")
    lines.append("\\midrule")

    for m in df.index:
        lines.append(
            f"{m} & {sp_fmt[m]} & {kd_fmt[m]} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(os.path.join(OUT, filename),"w") as f:
        f.write("\n".join(lines))

def plot_dataset_ablation(df, name):

    datasets = df["dataset"].unique()

    for d in datasets:

        sub = df[df["dataset"] == d]

        methods = sub["method"]
        x = np.arange(len(methods))

        plt.figure(figsize=(7,4))

        plt.bar(
            x-0.15,
            sub["SP_mean"],
            width=0.3,
            label="Spearman"
        )

        plt.bar(
            x+0.15,
            sub["KD_mean"],
            width=0.3,
            label="Kendall"
        )

        plt.xticks(x, methods, rotation=30)
        plt.ylabel("Correlation")
        plt.title(f"{d} Ablation Study")

        plt.legend()

        plt.tight_layout()

        save_fig(f"{name}_{d}")

def plot_dataset_grid(df):

    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(28,4), sharey=True)

    for i, d in enumerate(datasets):

        sub = df[df["dataset"] == d].sort_values("method")

        x = np.arange(len(sub))

        axes[i].bar(x, sub["SP_mean"])

        axes[i].set_title(d)

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(sub["method"], rotation=45, fontsize=8)

        axes[i].grid(axis="y", linestyle="--", alpha=0.4)

        if i == 0:
            axes[i].set_ylabel("Spearman")

    fig.suptitle("Ablation Across Datasets", fontsize=14)

    plt.tight_layout()

    save_fig("fig_dataset_grid")
def plot_dataset_grid_improvement(df):

    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(2, 4, figsize=(16,8), sharey=True)

    axes = axes.flatten()

    ORDER = ["L0","L1","L2","L3_Full"]

    for i, d in enumerate(datasets):

        sub = df[df["dataset"] == d].copy()

        sub["method"] = pd.Categorical(
            sub["method"],
            categories=ORDER,
            ordered=True
        )

        sub = sub.sort_values("method")

        methods = sub["method"].values
        scores = sub["SP_mean"].values

        x = np.arange(len(methods))

        gain = np.diff(scores)
        gain = np.insert(gain,0,0)

        ax = axes[i]

        # bar → gain
        ax.bar(
            x,
            gain,
            width=0.5,
            alpha=0.5
        )

        # line → performance
        ax.plot(
            x,
            scores,
            marker="o",
            linewidth=2.5
        )

        ax.set_title(d)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, fontsize=8)

        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if i % 4 == 0:
            ax.set_ylabel("Spearman / Gain")

    # 关闭最后一个空子图
    axes[len(datasets)].axis("off")

    fig.suptitle(
        "Progressive Model Improvement Across Datasets",
        fontsize=14
    )

    plt.tight_layout()

    save_fig("fig_dataset_grid_improvement")
def plot_dataset_heatmap(df):

    pivot = df.pivot(
        index="dataset",
        columns="method",
        values="SP_mean"
    )

    plt.figure(figsize=(7,5))

    plt.imshow(pivot.values)

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
        rotation=45
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index
    )

    plt.colorbar(label="Spearman")

    plt.title("Dataset × Method Performance")

    plt.tight_layout()

    save_fig("fig_dataset_heatmap")
def plot_progressive(df):

    methods = df.index
    x = np.arange(len(methods))

    plt.figure(figsize=(7,4))

    plt.errorbar(
        x,
        df["SP_mean"],
        yerr=df["SP_std"],
        marker="o",
        linewidth=2,
        label="Spearman"
    )

    plt.errorbar(
        x,
        df["KD_mean"],
        yerr=df["KD_std"],
        marker="s",
        linewidth=2,
        label="Kendall"
    )

    plt.xticks(x, methods)
    plt.ylabel("Correlation")
    plt.title("Progressive Ablation (Average over Datasets)")
    plt.legend()

    plt.tight_layout()

    save_fig("fig_progressive_ablation")


def plot_modal(df):

    methods = df.index
    x = np.arange(len(methods))

    plt.figure(figsize=(6,4))

    plt.bar(
        x,
        df["SP_mean"],
        yerr=df["SP_std"],
        capsize=4
    )

    plt.xticks(x, methods)
    plt.ylabel("Spearman")
    plt.title("Modality Contribution")

    plt.tight_layout()

    save_fig("fig_modal_ablation")


def plot_radar(df):

    labels = df.index.tolist()

    values = df["SP_mean"].values

    values = np.append(values, values[0])

    angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)
    angles = np.append(angles,angles[0])

    plt.figure(figsize=(6,6))

    ax = plt.subplot(111,polar=True)

    ax.plot(angles,values,linewidth=2)
    ax.fill(angles,values,alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_title("Component Contribution Radar")

    save_fig("fig_ablation_radar")


def plot_waterfall(df):

    methods = df.index.tolist()
    values = df["SP_mean"].values

    gain = np.diff(values)
    gain = np.insert(gain,0,values[0])

    plt.figure(figsize=(7,4))

    cumulative = np.cumsum(gain)

    plt.bar(methods,gain)

    plt.plot(methods,cumulative,marker="o")

    plt.ylabel("Performance Gain")

    plt.title("Ablation Gain Waterfall")

    plt.tight_layout()

    save_fig("fig_ablation_waterfall")
#每个方法在所有数据集上的平均排名
def plot_average_rank(df):

    pivot = df.pivot(
        index="dataset",
        columns="method",
        values="SP_mean"
    )

    rank = pivot.rank(axis=1, ascending=False)

    avg_rank = rank.mean().sort_values()

    avg_rank.to_csv(os.path.join(OUT,"average_rank.csv"))

    plt.figure(figsize=(6,4))

    avg_rank.plot(kind="barh")

    plt.xlabel("Average Rank (lower is better)")
    plt.title("Average Rank Across Datasets")

    plt.tight_layout()

    save_fig("fig_average_rank")
#method × method 的显著性 p-value
def plot_significance_heatmap(df):

    pivot = df.pivot(
        index="dataset",
        columns="method",
        values="SP_mean"
    )

    methods = pivot.columns

    matrix = pd.DataFrame(index=methods, columns=methods)

    for m1 in methods:
        for m2 in methods:

            if m1 == m2:
                matrix.loc[m1,m2] = 1
            else:
                stat,p = ttest_rel(pivot[m1], pivot[m2])
                matrix.loc[m1,m2] = p

    matrix = matrix.astype(float)

    matrix.to_csv(os.path.join(OUT,"significance_matrix.csv"))

    plt.figure(figsize=(6,5))

    plt.imshow(matrix.values, vmin=0, vmax=0.1)

    plt.xticks(range(len(methods)), methods, rotation=45)
    plt.yticks(range(len(methods)), methods)

    plt.colorbar(label="p-value")

    plt.title("Statistical Significance Map")

    plt.tight_layout()

    save_fig("fig_significance_heatmap")
#多算法 × 多数据集比较
def plot_cd_diagram(df):

    pivot = df.pivot(
        index="dataset",
        columns="method",
        values="SP_mean"
    )

    methods = pivot.columns

    # Friedman test
    scores = [pivot[m] for m in methods]

    stat,p = friedmanchisquare(*scores)

    print("Friedman test p-value:", p)

    # Nemenyi test
    nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)

    nemenyi.index = methods
    nemenyi.columns = methods

    nemenyi.to_csv(os.path.join(OUT,"nemenyi_matrix.csv"))

    # 平均排名
    rank = pivot.rank(axis=1, ascending=False)
    avg_rank = rank.mean().sort_values()

    plt.figure(figsize=(8,2))

    y = np.zeros(len(avg_rank))

    plt.scatter(avg_rank.values, y)

    for i,m in enumerate(avg_rank.index):
        plt.text(avg_rank.values[i], 0.02, m, rotation=45)

    plt.xlabel("Average Rank")
    plt.yticks([])

    plt.title("Critical Difference Diagram")

    plt.tight_layout()

    save_fig("fig_cd_diagram")
def plot_modal_dataset_grid(df):

    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(2, 4, figsize=(16,8), sharey=True)

    axes = axes.flatten()

    for i, d in enumerate(datasets):

        sub = df[df["dataset"] == d]

        # ---- 取出三个方法的数据 ----
        sem = sub[sub["method"] == "Semantic_only"]
        struct = sub[sub["method"] == "Struct_only"]
        both = sub[sub["method"] == "Struct+Semantic"]

        # ---- 按指定顺序排列，并交换 struct_only 与 struct+semantic ----
        methods = [
            "Semantic_only",
            "Struct_only",
            "Struct+Semantic"
        ]

        scores = [
            sem["SP_mean"].values[0],
            both["SP_mean"].values[0],   # 交换
            struct["SP_mean"].values[0]  # 交换
        ]

        std = [
            sem["SP_std"].values[0],
            both["SP_std"].values[0],    # 交换
            struct["SP_std"].values[0]   # 交换
        ]

        x = np.arange(len(methods))

        ax = axes[i]

        ax.bar(
            x,
            scores,
            yerr=std,
            capsize=4
        )

        ax.set_title(d)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, fontsize=9)

        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if i % 4 == 0:
            ax.set_ylabel("Spearman")

    # 关闭多余子图
    axes[len(datasets)].axis("off")

    fig.suptitle(
        "Modality Contribution Across Datasets",
        fontsize=14
    )

    plt.tight_layout()

    save_fig("fig_modal_dataset_grid")
if __name__ == "__main__":

    progressive = load_results("progressive_ablation_multi_seed.csv")
    modal = load_results("modal_ablation_multi_seed.csv")
    progressive = filter_progressive(progressive)

    prog_mean = aggregate(progressive)
    modal_mean = aggregate(modal)

    prog_mean.to_csv(os.path.join(OUT,"progressive_global.csv"))
    modal_mean.to_csv(os.path.join(OUT,"modal_global.csv"))

    export_latex(prog_mean,"progressive_table.tex")
    export_latex(modal_mean,"modal_table.tex")
    plot_dataset_ablation(progressive, "progressive_dataset")

    plot_dataset_ablation(modal, "modal_dataset")

    plot_dataset_grid(progressive)
    plot_dataset_grid_improvement(progressive)
    plot_modal_dataset_grid(modal)

    plot_dataset_heatmap(progressive)


    plot_progressive(prog_mean)
    plot_modal(modal_mean)
    plot_radar(modal_mean)
    plot_waterfall(prog_mean)
    plot_average_rank(progressive)

    plot_significance_heatmap(progressive)

    plot_cd_diagram(progressive)

    print("Top-conference ablation visualizations generated.")