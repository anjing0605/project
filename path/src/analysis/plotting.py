from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
'''方法对比折线图：k vs delta_E / delta_LCC / delta_ASP
消融柱状图
RL 训练曲线图
'''

class Plotter:
    """
    Plot utility for:
    1) method comparison curves
    2) RL training curves
    3) ablation bar charts
    """

    @staticmethod
    def _ensure_parent(path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def plot_metric_curve(
        df: pd.DataFrame,
        dataset: str,
        metric: str,
        save_path: str | Path,
        methods: Optional[Iterable[str]] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        df columns required:
            dataset, method, k, metric
        """
        if metric not in df.columns:
            raise ValueError(f"metric '{metric}' not found in dataframe columns.")

        sub = df[df["dataset"] == dataset].copy()
        if methods is not None:
            methods = list(methods)
            sub = sub[sub["method"].isin(methods)]

        if sub.empty:
            raise ValueError(f"No rows found for dataset={dataset}, metric={metric}")

        plt.figure(figsize=(7, 4.5))

        for method in sorted(sub["method"].unique()):
            one = sub[sub["method"] == method].sort_values(by="k")
            plt.plot(one["k"], one[metric], marker="o", label=method)

        plt.xlabel("Top-k removed paths")
        plt.ylabel(metric)
        plt.title(title or f"{dataset} - {metric}")
        plt.legend()
        plt.tight_layout()

        save_path = Plotter._ensure_parent(save_path)
        plt.savefig(save_path, dpi=220)
        plt.close()

    @staticmethod
    def plot_training_curve(
        y,
        ylabel: str,
        save_path: str | Path,
        xlabel: str = "Epoch",
        title: Optional[str] = None,
    ) -> None:
        plt.figure(figsize=(7, 4.5))
        x = list(range(1, len(y) + 1))
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title or ylabel)
        plt.tight_layout()

        save_path = Plotter._ensure_parent(save_path)
        plt.savefig(save_path, dpi=220)
        plt.close()

    @staticmethod
    def plot_rl_training_curves_from_log(
        log_obj: dict,
        tag: str,
        save_dir: str | Path,
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if "avg_reward" in log_obj:
            Plotter.plot_training_curve(
                y=log_obj["avg_reward"],
                ylabel="Average Reward",
                save_path=save_dir / f"{tag}_avg_reward.png",
                title=f"{tag} - avg_reward",
            )

        if "arrival_rate" in log_obj:
            Plotter.plot_training_curve(
                y=log_obj["arrival_rate"],
                ylabel="Arrival Rate",
                save_path=save_dir / f"{tag}_arrival_rate.png",
                title=f"{tag} - arrival_rate",
            )

        if "avg_steps" in log_obj:
            Plotter.plot_training_curve(
                y=log_obj["avg_steps"],
                ylabel="Average Steps",
                save_path=save_dir / f"{tag}_avg_steps.png",
                title=f"{tag} - avg_steps",
            )

    @staticmethod
    def plot_ablation_bar(
        df: pd.DataFrame,
        dataset: str,
        metric: str,
        k_value: int,
        save_path: str | Path,
        title: Optional[str] = None,
    ) -> None:
        """
        df columns required:
            dataset, ablation, k, metric
        """
        if metric not in df.columns:
            raise ValueError(f"metric '{metric}' not found in dataframe columns.")

        sub = df[(df["dataset"] == dataset) & (df["k"] == k_value)].copy()
        if sub.empty:
            raise ValueError(f"No rows found for dataset={dataset}, k={k_value}")

        sub = sub.sort_values(by=metric, ascending=False)

        plt.figure(figsize=(9, 4.8))
        plt.bar(sub["ablation"], sub[metric])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric)
        plt.title(title or f"{dataset} - {metric} @ k={k_value}")
        plt.tight_layout()

        save_path = Plotter._ensure_parent(save_path)
        plt.savefig(save_path, dpi=220)
        plt.close()