from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from path.src.data.preprocess import GraphPreprocessor
from path.src.core.keynode import KeyNodeSelector
from path.src.core.task_sampler import TaskPairBuilder
from path.src.baselines.rule_based import RuleBasedCriticalPath
from path.src.core.evaluator import MethodEvaluator
from path.src.core.fragility import FragilityEvaluator

try:
    from path.src.utils.seed import set_seed
except ImportError:
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
'''
python -m path.scripts.search_rule_weights `
  --dataset_name Cora `
  --root D:\project\keynode\project\public_datasets\Planetoid `
  --importance_path D:\project\keynode\project\public_process\extract\Cora_struct\gnn_node_scores_mean.csv `
  --node_features_path D:\project\keynode\project\public_process\extract\Cora_struct\node_features.csv `
  --old_id_col node `
  --strict_importance_alignment `
  --verbose `
  --topk 30 `
  --path_k 3 `
  --max_hops 8 `
  --delta 2 `
  --top_q 10 `
  --num_trials 60 `
  --seed 42 `
  --outdir path/outputs/weight_search
'''

# -----------------------------
# 基础工具
# -----------------------------

def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def bucket_shortest_len(x: int) -> str:
    if x <= 2:
        return "len2"
    if x == 3:
        return "len3"
    return "len4p"


def stratified_split_tasks(
    tasks: List[Any],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    按 (shortest_len bucket, same_community) 做弱分层切分。
    """
    rng = random.Random(seed)
    groups: Dict[Tuple[str, bool], List[Any]] = {}

    for t in tasks:
        key = (bucket_shortest_len(int(t.shortest_len)), bool(t.same_community))
        groups.setdefault(key, []).append(t)

    train_tasks, val_tasks, test_tasks = [], [], []

    for _, group in groups.items():
        rng.shuffle(group)
        n = len(group)

        n_val = max(1, int(round(n * val_ratio))) if n >= 5 else max(0, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio))) if n >= 5 else max(0, int(round(n * test_ratio)))

        # 防止超界
        if n_val + n_test > n:
            overflow = n_val + n_test - n
            if n_test >= overflow:
                n_test -= overflow
            else:
                overflow -= n_test
                n_test = 0
                n_val = max(0, n_val - overflow)

        val_part = group[:n_val]
        test_part = group[n_val:n_val + n_test]
        train_part = group[n_val + n_test:]

        train_tasks.extend(train_part)
        val_tasks.extend(val_part)
        test_tasks.extend(test_part)

    rng.shuffle(train_tasks)
    rng.shuffle(val_tasks)
    rng.shuffle(test_tasks)
    return train_tasks, val_tasks, test_tasks


def summarize_task_split(tasks: List[Any]) -> Dict[str, Any]:
    out = {
        "num_tasks": len(tasks),
        "len2": 0,
        "len3": 0,
        "len4p": 0,
        "same_community_true": 0,
        "same_community_false": 0,
    }
    for t in tasks:
        out[bucket_shortest_len(int(t.shortest_len))] += 1
        if bool(t.same_community):
            out["same_community_true"] += 1
        else:
            out["same_community_false"] += 1
    return out


# -----------------------------
# 权重采样
# -----------------------------

WEIGHT_KEYS = [
    "avg_node_importance",
    "avg_edge_bc",
    "cross_comm_ratio",
    "fragility_score",
    "path_length",
]


def default_weights() -> Dict[str, float]:
    """
    用你当前更接近 paper_default 的风格：
    fragility 仍为主项，长度为惩罚项对应的正权重。
    注意：score_path 内部应当对 path_length 做减项。
    """
    return {
        "avg_node_importance": 0.12,
        "avg_edge_bc": 0.10,
        "cross_comm_ratio": 0.03,
        "fragility_score": 0.65,
        "path_length": 0.10,
    }


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in w.values())
    if s <= 0:
        raise ValueError("Weight sum must be positive.")
    return {k: float(v) / s for k, v in w.items()}


def sample_weight_vector(rng: np.random.Generator) -> Dict[str, float]:
    """
    随机搜索，但限制在“仍符合方法定义”的区域：
    - fragility 必须偏高
    - cross_comm 不宜过大
    - path_length 不宜过大
    """
    alpha = np.array([2.0, 2.0, 0.7, 6.0, 1.5], dtype=float)

    for _ in range(5000):
        x = rng.dirichlet(alpha)
        w = {k: float(v) for k, v in zip(WEIGHT_KEYS, x)}

        if w["fragility_score"] < 0.35:
            continue
        if w["cross_comm_ratio"] > 0.20:
            continue
        if not (0.05 <= w["path_length"] <= 0.25):
            continue
        if w["avg_node_importance"] < 0.05:
            continue
        if w["avg_edge_bc"] < 0.05:
            continue

        return normalize_weights(w)

    raise RuntimeError("Failed to sample a valid weight vector under current constraints.")


# -----------------------------
# 单次评估
# -----------------------------

def run_rule_once(
    bundle: Any,
    tasks: List[Any],
    weights: Dict[str, float],
    path_k: int,
    max_hops: int,
    delta: int,
    top_q: int,
    k_list: List[int],
    shared_base_metrics: Dict[str, float],
    lambda_E: float,
    lambda_LCC: float,
    lambda_ASP: float,
) -> Dict[str, Any]:
    evaluator = MethodEvaluator(
        lambda_E=lambda_E,
        lambda_LCC=lambda_LCC,
        lambda_ASP=lambda_ASP,
    )

    # 兼容你本地 rule_based.py 可能还没统一签名的情况
    try:
        selected_paths = RuleBasedCriticalPath.run(
            bundle=bundle,
            tasks=tasks,
            path_k=path_k,
            max_hops=max_hops,
            delta=delta,
            weights=weights,
            top_q=top_q,
            shared_base_metrics=shared_base_metrics,
        )
    except TypeError:
        selected_paths = RuleBasedCriticalPath.run(
            bundle=bundle,
            tasks=tasks,
            path_k=path_k,
            max_hops=max_hops,
            delta=delta,
            weights=weights,
            top_q=top_q,
        )

    metrics = evaluator.evaluate_topk_damage(
        G=bundle.nx_graph,
        selected_paths=selected_paths,
        k_list=k_list,
    )

    return {
        "weights": dict(weights),
        "selected_paths": selected_paths,
        "metrics": metrics,
    }


# -----------------------------
# 搜索目标
# -----------------------------

def flatten_metric_dims(
    trial_metrics_list: List[Dict[str, Any]],
    target_ks: List[int],
    metric_names: List[str],
    k_list: List[int],
) -> pd.DataFrame:
    """
    把每个 trial 的 metrics 展平为：
    delta_E@3, delta_E@5, ...
    """
    rows = []
    for i, item in enumerate(trial_metrics_list):
        metrics = item["metrics"]
        row = {"trial_id": i}

        for metric_name in metric_names:
            curve_key = f"{metric_name}_curve"
            curve = metrics[curve_key]
            if len(curve) != len(k_list):
                raise ValueError(
                    f"{curve_key} length mismatch: len(curve)={len(curve)} vs len(k_list)={len(k_list)}"
                )

            for k, v in zip(k_list, curve):
                if k in target_ks or k == 1:
                    row[f"{metric_name}@{k}"] = float(v)

        rows.append(row)

    return pd.DataFrame(rows)


def minmax_normalize_column(x: pd.Series) -> pd.Series:
    lo = float(x.min())
    hi = float(x.max())
    if math.isclose(lo, hi):
        return pd.Series([0.5] * len(x), index=x.index)
    return (x - lo) / (hi - lo)


def attach_search_score(
    trials_df: pd.DataFrame,
    metric_weights: Dict[str, float],
    set_ks: List[int],
    top1_bonus: float = 0.10,
) -> pd.DataFrame:
    """
    主目标优化集合质量（k=3/5/10），
    仅用很小的 top1 bonus 防止单路径过弱。
    """
    df = trials_df.copy()

    # 先对每个 metric@k 做 min-max
    metric_cols = []
    for metric_name in ["delta_E", "delta_LCC", "delta_ASP"]:
        for k in [1] + list(set_ks):
            col = f"{metric_name}@{k}"
            if col in df.columns:
                norm_col = f"norm_{col}"
                df[norm_col] = minmax_normalize_column(df[col])
                metric_cols.append(norm_col)

    # set-level objective
    set_scores = []
    for _, row in df.iterrows():
        s = 0.0
        for k in set_ks:
            s += metric_weights["delta_E"] * float(row[f"norm_delta_E@{k}"])
            s += metric_weights["delta_LCC"] * float(row[f"norm_delta_LCC@{k}"])
            s += metric_weights["delta_ASP"] * float(row[f"norm_delta_ASP@{k}"])
        s /= max(1, len(set_ks))
        set_scores.append(s)

    df["set_score"] = set_scores

    # top1 bonus
    top1_scores = []
    for _, row in df.iterrows():
        s1 = (
            metric_weights["delta_E"] * float(row["norm_delta_E@1"])
            + metric_weights["delta_LCC"] * float(row["norm_delta_LCC@1"])
            + metric_weights["delta_ASP"] * float(row["norm_delta_ASP@1"])
        )
        top1_scores.append(s1)

    df["top1_score"] = top1_scores
    df["search_score"] = (1.0 - top1_bonus) * df["set_score"] + top1_bonus * df["top1_score"]
    df = df.sort_values("search_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# -----------------------------
# 主流程
# -----------------------------

def build_bundle(args: argparse.Namespace) -> Any:
    kwargs = dict(
        name=args.dataset_name,
        root=args.root,
        importance_path=args.importance_path,
        community_mode=args.community_mode,
    )

    # 兼容你本地 preprocess 可能扩展过的参数
    if args.node_features_path is not None:
        kwargs["node_features_path"] = args.node_features_path
        kwargs["old_id_col_in_node_features"] = args.old_id_col
        kwargs["strict_importance_alignment"] = bool(args.strict_importance_alignment)
        kwargs["importance_fill_value"] = float(args.importance_fill_value)
        kwargs["verbose"] = bool(args.verbose)

    return GraphPreprocessor.build_graph_bundle(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Cora")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--importance_path", type=str, required=True)

    # 兼容你本地 GraphPreprocessor 扩展版
    parser.add_argument("--node_features_path", type=str, default=None)
    parser.add_argument("--old_id_col", type=str, default="node")
    parser.add_argument("--strict_importance_alignment", action="store_true")
    parser.add_argument("--importance_fill_value", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--community_mode", type=str, default="louvain")

    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--min_shortest_len", type=int, default=2)

    parser.add_argument("--path_k", type=int, default=3)
    parser.add_argument("--max_hops", type=int, default=8)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--top_q", type=int, default=10)

    parser.add_argument("--k_list", type=int, nargs="*", default=[1, 3, 5, 10])

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    parser.add_argument("--num_trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)

    # evaluator / objective
    parser.add_argument("--lambda_E", type=float, default=0.4)
    parser.add_argument("--lambda_LCC", type=float, default=0.4)
    parser.add_argument("--lambda_ASP", type=float, default=0.2)

    parser.add_argument("--obj_delta_E", type=float, default=1.0 / 3.0)
    parser.add_argument("--obj_delta_LCC", type=float, default=1.0 / 3.0)
    parser.add_argument("--obj_delta_ASP", type=float, default=1.0 / 3.0)

    parser.add_argument("--top1_bonus", type=float, default=0.10)
    parser.add_argument("--set_ks", type=int, nargs="*", default=[3, 5, 10])

    parser.add_argument("--outdir", type=str, default="path/outputs/weight_search")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Building graph bundle...")
    bundle = build_bundle(args)

    print("Selecting key nodes and building task pairs...")
    key_nodes = KeyNodeSelector.select_topk_nodes(bundle.importance, args.topk)
    tasks = TaskPairBuilder.build_task_pairs(
        G=bundle.nx_graph,
        key_nodes=key_nodes,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=args.min_shortest_len,
    )

    train_tasks, val_tasks, test_tasks = stratified_split_tasks(
        tasks=tasks,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Total tasks = {len(tasks)}")
    print("Train split =", summarize_task_split(train_tasks))
    print("Val split   =", summarize_task_split(val_tasks))
    print("Test split  =", summarize_task_split(test_tasks))

    # 固定 shared_base_metrics，确保所有 trial 公平
    fragility_evaluator = FragilityEvaluator()
    shared_base_metrics = fragility_evaluator.compute_base_metrics(bundle.nx_graph)

    trial_records: List[Dict[str, Any]] = []

    # trial 0 固定跑 default
    candidate_weights = [default_weights()]
    for _ in range(max(0, args.num_trials - 1)):
        candidate_weights.append(sample_weight_vector(rng))

    print("=" * 100)
    print(f"Running random search on val split: {len(candidate_weights)} trials")

    for trial_id, weights in enumerate(candidate_weights):
        result = run_rule_once(
            bundle=bundle,
            tasks=val_tasks,
            weights=weights,
            path_k=args.path_k,
            max_hops=args.max_hops,
            delta=args.delta,
            top_q=args.top_q,
            k_list=args.k_list,
            shared_base_metrics=shared_base_metrics,
            lambda_E=args.lambda_E,
            lambda_LCC=args.lambda_LCC,
            lambda_ASP=args.lambda_ASP,
        )

        rec = {
            "trial_id": trial_id,
            "weights": dict(weights),
            "metrics": result["metrics"],
            "num_selected_paths": len(result["selected_paths"]),
        }
        trial_records.append(rec)

        m = result["metrics"]
        e_curve = m["delta_E_curve"]
        lcc_curve = m["delta_LCC_curve"]
        asp_curve = m["delta_ASP_curve"]

        msg = (
            f"[Trial {trial_id:03d}] "
            f"weights={weights} | "
            f"ΔE@10={e_curve[-1]:.6f}, "
            f"ΔLCC@10={lcc_curve[-1]:.6f}, "
            f"ΔASP@10={asp_curve[-1]:.6f}"
        )
        print(msg)

    # 两阶段：先评估全部 trial，再归一化打分
    metrics_df = flatten_metric_dims(
        trial_metrics_list=trial_records,
        target_ks=args.set_ks,
        metric_names=["delta_E", "delta_LCC", "delta_ASP"],
        k_list=args.k_list,
    )

    weights_df = pd.DataFrame(
        [
            {
                "trial_id": x["trial_id"],
                **x["weights"],
                "num_selected_paths": x["num_selected_paths"],
            }
            for x in trial_records
        ]
    )

    trials_df = weights_df.merge(metrics_df, on="trial_id", how="left")

    obj_weights = {
        "delta_E": float(args.obj_delta_E),
        "delta_LCC": float(args.obj_delta_LCC),
        "delta_ASP": float(args.obj_delta_ASP),
    }
    s = sum(obj_weights.values())
    obj_weights = {k: v / s for k, v in obj_weights.items()}

    scored_df = attach_search_score(
        trials_df=trials_df,
        metric_weights=obj_weights,
        set_ks=args.set_ks,
        top1_bonus=args.top1_bonus,
    )

    best_trial = scored_df.iloc[0]
    best_trial_id = int(best_trial["trial_id"])
    best_weights = {
        "avg_node_importance": float(best_trial["avg_node_importance"]),
        "avg_edge_bc": float(best_trial["avg_edge_bc"]),
        "cross_comm_ratio": float(best_trial["cross_comm_ratio"]),
        "fragility_score": float(best_trial["fragility_score"]),
        "path_length": float(best_trial["path_length"]),
    }

    print("=" * 100)
    print("Top-10 trials on validation split:")
    show_cols = [
        "rank", "trial_id",
        "avg_node_importance", "avg_edge_bc", "cross_comm_ratio", "fragility_score", "path_length",
        "delta_E@1", "delta_E@3", "delta_E@5", "delta_E@10",
        "delta_LCC@1", "delta_LCC@3", "delta_LCC@5", "delta_LCC@10",
        "delta_ASP@1", "delta_ASP@3", "delta_ASP@5", "delta_ASP@10",
        "set_score", "top1_score", "search_score",
    ]
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 240,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        print(scored_df[show_cols].head(10).to_string(index=False))

    print("=" * 100)
    print("Best weights selected on validation split:")
    print(best_weights)

    # 在 test 上只跑一次 best
    best_test_result = run_rule_once(
        bundle=bundle,
        tasks=test_tasks,
        weights=best_weights,
        path_k=args.path_k,
        max_hops=args.max_hops,
        delta=args.delta,
        top_q=args.top_q,
        k_list=args.k_list,
        shared_base_metrics=shared_base_metrics,
        lambda_E=args.lambda_E,
        lambda_LCC=args.lambda_LCC,
        lambda_ASP=args.lambda_ASP,
    )

    # 同时跑 default，方便直接比较
    default_test_result = run_rule_once(
        bundle=bundle,
        tasks=test_tasks,
        weights=default_weights(),
        path_k=args.path_k,
        max_hops=args.max_hops,
        delta=args.delta,
        top_q=args.top_q,
        k_list=args.k_list,
        shared_base_metrics=shared_base_metrics,
        lambda_E=args.lambda_E,
        lambda_LCC=args.lambda_LCC,
        lambda_ASP=args.lambda_ASP,
    )

    print("=" * 100)
    print("Test metrics: default vs best")
    comp_rows = []
    for idx, k in enumerate(args.k_list):
        comp_rows.append({
            "k": k,
            "default_delta_E": float(default_test_result["metrics"]["delta_E_curve"][idx]),
            "best_delta_E": float(best_test_result["metrics"]["delta_E_curve"][idx]),
            "default_delta_LCC": float(default_test_result["metrics"]["delta_LCC_curve"][idx]),
            "best_delta_LCC": float(best_test_result["metrics"]["delta_LCC_curve"][idx]),
            "default_delta_ASP": float(default_test_result["metrics"]["delta_ASP_curve"][idx]),
            "best_delta_ASP": float(best_test_result["metrics"]["delta_ASP_curve"][idx]),
        })
    comp_df = pd.DataFrame(comp_rows)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        print(comp_df.to_string(index=False))

    # 保存
    scored_csv_path = outdir / f"{args.dataset_name.lower()}_rule_weight_search_trials.csv"
    scored_df.to_csv(scored_csv_path, index=False, encoding="utf-8-sig")

    result_json = {
        "dataset_name": args.dataset_name,
        "search_config": {
            "topk": args.topk,
            "min_shortest_len": args.min_shortest_len,
            "path_k": args.path_k,
            "max_hops": args.max_hops,
            "delta": args.delta,
            "top_q": args.top_q,
            "k_list": args.k_list,
            "set_ks": args.set_ks,
            "num_trials": args.num_trials,
            "seed": args.seed,
            "objective_metric_weights": obj_weights,
            "top1_bonus": args.top1_bonus,
        },
        "split_summary": {
            "train": summarize_task_split(train_tasks),
            "val": summarize_task_split(val_tasks),
            "test": summarize_task_split(test_tasks),
        },
        "shared_base_metrics": shared_base_metrics,
        "default_weights": default_weights(),
        "best_trial_id": best_trial_id,
        "best_weights": best_weights,
        "best_val_row": {k: to_jsonable(v) for k, v in best_trial.to_dict().items()},
        "test_metrics_default": default_test_result["metrics"],
        "test_metrics_best": best_test_result["metrics"],
    }

    result_json_path = outdir / f"{args.dataset_name.lower()}_rule_weight_search_best.json"
    save_json(result_json_path, result_json)

    print("=" * 100)
    print(f"Saved trials csv to: {scored_csv_path}")
    print(f"Saved best result json to: {result_json_path}")


if __name__ == "__main__":
    main()