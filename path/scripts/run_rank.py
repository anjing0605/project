from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required for scripts/run_rank.py") from exc

# 你的实际路径是:
# D:\project\keynode\project\path\scripts\run_rank.py
# 因此:
# parents[0] = ...\path\scripts
# parents[1] = ...\path
# parents[2] = ...\project   <- 需要加到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PATH_ROOT = Path(__file__).resolve().parents[1]      # D:/project/keynode/project/path

from path.src.core.fragility import FragilityEvaluator
from path.src.core.evaluator import MethodEvaluator
from path.src.core.keynode import KeyNodeSelector
from path.src.core.task_sampler import TaskPairBuilder
from path.src.data.preprocess import GraphPreprocessor
from path.src.ranking.dataset import PathRankingDatasetBuilder
from path.src.ranking.xgb_ranker import XGBPathRanker
from path.src.utils.tb_logger import TBLogger
from path.src.core.set_scorer import SubmodularPathSelector
from path.src.core.pred_score_selector import PredScoreSelector
"""
cd D:\project\keynode\project
跑纯 rank
python -m path.scripts.run_rank --config path/configs/cora_rank_pure.yaml --debug
跑 rank + pred score set selection
python -m path.scripts.run_rank --config path/configs/cora_rank_set_pred.yaml --debug
跑 rank + submodular greedy保留的“更强集合选择”版本：
python -m path.scripts.run_rank --config path/configs/cora_rank_set_submod.yaml --debug
python -m path.scripts.build_method_comparison_table
 D:\project\keynode\project\path\outputs\metrics\all_method_comparison.csv
python -m path.scripts.build_path_quality_table
D:\project\keynode\project\path\outputs\metrics\all_path_quality.csv
python -m path.scripts.plot_rank_paper_figures
 D:\project\keynode\project\path\outputs\figures\rank
"""

def resolve_output_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(PATH_ROOT / p)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _serialize_pathrecords(records: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        features = getattr(r, "features", None) or {}
        fragility = getattr(r, "fragility", None) or {}
        metadata = getattr(r, "metadata", None) or {}

        out.append(
            {
                "source": int(r.source),
                "target": int(r.target),
                "nodes": [int(n) for n in r.nodes],
                "score": None if r.score is None else float(r.score),
                "method": r.method,
                "features": {str(k): float(v) for k, v in features.items()},
                "fragility": {str(k): float(v) for k, v in fragility.items()},
                "metadata": dict(metadata),
            }
        )
    return out


def _select_key_nodes_from_cfg(importance, key_cfg: Dict[str, Any]) -> List[int]:
    mode = key_cfg.get("mode", "topk")

    if mode == "topk":
        k = int(key_cfg.get("k", 30))
        return KeyNodeSelector.select_topk_nodes(importance, k)

    elif mode in ("top_ratio", "ratio"):
        ratio = float(key_cfg.get("ratio", 0.05))
        return KeyNodeSelector.select_top_ratio_nodes(importance, ratio)

    else:
        raise ValueError(
            f"Unsupported keynode mode: {mode}. "
            f"Expected one of ['topk', 'top_ratio', 'ratio']"
        )


def _build_tasks_from_cfg(bundle, key_nodes: List[int], key_cfg: Dict[str, Any]):
    """
    Build full key-node task pool, then sample it according to task_sampling_mode.
    This makes ranking stage consistent with rule/RL task sampling.
    """
    all_tasks = TaskPairBuilder.build_task_pairs(
        G=bundle.nx_graph,
        key_nodes=key_nodes,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=int(key_cfg.get("min_shortest_len", 2)),
    )

    max_pairs = key_cfg.get("max_pairs", None)
    if max_pairs is None or int(max_pairs) <= 0:
        return all_tasks

    mode = str(key_cfg.get("task_sampling_mode", "paper_default")).strip().lower()
    random_seed = int(key_cfg.get("random_seed", 42))
    random_ratio = float(key_cfg.get("random_task_ratio", 0.1))

    tasks = TaskPairBuilder.sample_task_pairs(
        task_pairs=all_tasks,
        num_samples=int(max_pairs),
        mode=mode,
        random_seed=random_seed,
        G=bundle.nx_graph,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=int(key_cfg.get("min_shortest_len", 2)),
        random_ratio=random_ratio,
    )

    return tasks


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))

def _dbg(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[DEBUG] {msg}")


def _dbg_kv(enabled: bool, title: str, kv: Dict[str, Any]) -> None:
    if not enabled:
        return
    print(f"[DEBUG] {title}")
    for k, v in kv.items():
        print(f"    - {k}: {v}")


def _dbg_df(enabled: bool, name: str, df, max_rows: int = 3) -> None:
    if not enabled:
        return
    print(f"[DEBUG] dataframe<{name}> shape = {getattr(df, 'shape', None)}")
    cols = list(df.columns) if hasattr(df, "columns") else []
    print(f"[DEBUG] dataframe<{name}> columns = {cols}")
    if hasattr(df, "head"):
        try:
            print(f"[DEBUG] dataframe<{name}> head({max_rows}) =")
            print(df.head(max_rows).to_string())
        except Exception as exc:
            print(f"[DEBUG] dataframe<{name}> head print failed: {exc}")


def _dbg_list(enabled: bool, name: str, values: List[Any], max_items: int = 5) -> None:
    if not enabled:
        return
    print(f"[DEBUG] list<{name}> size = {len(values)}")
    preview = values[: min(max_items, len(values))]
    print(f"[DEBUG] list<{name}> preview = {preview}")


def _dbg_path_records(enabled: bool, name: str, records: List[Any], max_items: int = 3) -> None:
    if not enabled:
        return
    print(f"[DEBUG] path_records<{name}> size = {len(records)}")
    for i, r in enumerate(records[: min(max_items, len(records))]):
        print(f"[DEBUG] path_records<{name}>[{i}]")
        print(f"    source={getattr(r, 'source', None)}, target={getattr(r, 'target', None)}")
        print(f"    nodes={getattr(r, 'nodes', None)}")
        print(f"    score={getattr(r, 'score', None)}")
        print(f"    method={getattr(r, 'method', None)}")
        print(f"    features={getattr(r, 'features', None)}")
        print(f"    fragility={getattr(r, 'fragility', None)}")

def _round_robin_truncate_by_task(records: List[Any], top_q: int) -> List[Any]:
    """
    Truncate selected records without random sampling.

    Purpose:
    - avoid over-selecting from a few high-score tasks;
    - preserve within-task selection utility ordering;
    - keep deterministic behavior.
    """
    if top_q <= 0:
        return []
    if len(records) <= top_q:
        return records

    from collections import defaultdict

    groups = defaultdict(list)
    for r in records:
        groups[(int(r.source), int(r.target))].append(r)

    for key in groups:
        groups[key].sort(
            key=lambda r: float(
                (getattr(r, "metadata", None) or {}).get(
                    "selection_utility",
                    getattr(r, "score", 0.0) or 0.0,
                )
            ),
            reverse=True,
        )

    ordered_keys = sorted(
        groups.keys(),
        key=lambda key: float(
            (groups[key][0].metadata or {}).get(
                "selection_utility",
                groups[key][0].score or 0.0,
            )
        ),
        reverse=True,
    )

    out = []
    cursor = 0

    while len(out) < top_q:
        progressed = False

        for key in ordered_keys:
            bucket = groups[key]
            if cursor < len(bucket):
                out.append(bucket[cursor])
                progressed = True
                if len(out) >= top_q:
                    break

        if not progressed:
            break

        cursor += 1

    return out

def _stage_tic(enabled: bool, name: str) -> float:
    if enabled:
        print(f"\n[DEBUG][STAGE-BEGIN] {name}")
    return time.perf_counter()


def _stage_toc(enabled: bool, name: str, t0: float) -> None:
    if enabled:
        dt = time.perf_counter() - t0
        print(f"[DEBUG][STAGE-END] {name} | elapsed = {dt:.4f}s")
def _log_dataset_info(
    tb: TBLogger,
    bundle,
    key_nodes: List[int],
    tasks: List[Any],
    df,
    train_df,
    test_df,
    feature_cols: List[str],
    backend: str,
) -> None:
    tb.add_scalar("rank/data/num_nodes", int(bundle.num_nodes), 0)
    tb.add_scalar("rank/data/num_edges", int(bundle.nx_graph.number_of_edges()), 0)
    tb.add_scalar("rank/data/num_key_nodes", int(len(key_nodes)), 0)
    tb.add_scalar("rank/data/num_tasks", int(len(tasks)), 0)

    tb.add_scalar("rank/data/num_samples", int(len(df)), 0)
    tb.add_scalar("rank/data/num_train_samples", int(len(train_df)), 0)
    tb.add_scalar("rank/data/num_test_samples", int(len(test_df)), 0)

    tb.add_scalar("rank/data/importance_mean", float(bundle.importance.mean()), 0)
    tb.add_scalar("rank/data/importance_std", float(bundle.importance.std()), 0)

    if tasks:
        shortest_lens = [float(t.shortest_len) for t in tasks]
        pair_scores = [float(t.pair_score) for t in tasks]
        cross_comm = [0.0 if t.same_community else 1.0 for t in tasks]

        tb.add_scalar("rank/tasks/avg_shortest_len", _safe_mean(shortest_lens), 0)
        tb.add_scalar("rank/tasks/avg_pair_score", _safe_mean(pair_scores), 0)
        tb.add_scalar("rank/tasks/cross_community_ratio", _safe_mean(cross_comm), 0)

    tb.add_text("rank/info/backend", str(backend), 0)
    tb.add_text("rank/info/feature_cols", "\n".join(feature_cols), 0)


def _log_selected_path_stats(
    tb: TBLogger,
    selected: List[Any],
) -> None:
    tb.add_scalar("rank/paths/num_selected_paths", len(selected), 0)

    if not selected:
        return

    lengths = [float(len(p.nodes)) for p in selected]
    tb.add_scalar("rank/paths/avg_path_length", _safe_mean(lengths), 0)
    tb.add_scalar("rank/paths/max_path_length", max(lengths), 0)
    tb.add_scalar("rank/paths/min_path_length", min(lengths), 0)
    tb.add_histogram("rank/paths/path_length_hist", lengths, 0)

    scores = [float(p.score) for p in selected if getattr(p, "score", None) is not None]
    if scores:
        tb.add_scalar("rank/paths/avg_path_score", _safe_mean(scores), 0)
        tb.add_scalar("rank/paths/max_path_score", max(scores), 0)
        tb.add_scalar("rank/paths/min_path_score", min(scores), 0)
        tb.add_histogram("rank/paths/path_score_hist", scores, 0)

    delta_Es = []
    delta_LCCs = []
    delta_ASPs = []
    fragility_scores = []

    for p in selected:
        frag = getattr(p, "fragility", None) or {}
        if "delta_E" in frag:
            delta_Es.append(float(frag["delta_E"]))
        if "delta_LCC" in frag:
            delta_LCCs.append(float(frag["delta_LCC"]))
        if "delta_ASP" in frag:
            delta_ASPs.append(float(frag["delta_ASP"]))
        if "fragility_score" in frag:
            fragility_scores.append(float(frag["fragility_score"]))

    if delta_Es:
        tb.add_scalar("rank/paths/avg_delta_E", _safe_mean(delta_Es), 0)
    if delta_LCCs:
        tb.add_scalar("rank/paths/avg_delta_LCC", _safe_mean(delta_LCCs), 0)
    if delta_ASPs:
        tb.add_scalar("rank/paths/avg_delta_ASP", _safe_mean(delta_ASPs), 0)
    if fragility_scores:
        tb.add_scalar("rank/paths/avg_fragility_score", _safe_mean(fragility_scores), 0)


def _log_comparison_to_tb(
    tb: TBLogger,
    comparison: Dict[str, Any],
    k_list: List[int],
) -> None:
    """
    comparison 典型形式:
    {
        "xgb_rank": {
            "delta_E_curve": [...],
            "delta_LCC_curve": [...],
            "delta_ASP_curve": [...]
        }
    }
    """
    for method_name, metrics in comparison.items():
        delta_E_curve = metrics.get("delta_E_curve", [])
        delta_LCC_curve = metrics.get("delta_LCC_curve", [])
        delta_ASP_curve = metrics.get("delta_ASP_curve", [])

        for k, v in zip(k_list, delta_E_curve):
            tb.add_scalar(f"{method_name}/damage/delta_E_at_k", v, int(k))

        for k, v in zip(k_list, delta_LCC_curve):
            tb.add_scalar(f"{method_name}/damage/delta_LCC_at_k", v, int(k))

        for k, v in zip(k_list, delta_ASP_curve):
            tb.add_scalar(f"{method_name}/damage/delta_ASP_at_k", v, int(k))

        if delta_E_curve:
            tb.add_scalar(f"{method_name}/damage/top_last_delta_E", float(delta_E_curve[-1]), 0)
        if delta_LCC_curve:
            tb.add_scalar(f"{method_name}/damage/top_last_delta_LCC", float(delta_LCC_curve[-1]), 0)
        if delta_ASP_curve:
            tb.add_scalar(f"{method_name}/damage/top_last_delta_ASP", float(delta_ASP_curve[-1]), 0)


def main() -> None:
    total_t0 = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Run supervised ranking baseline for critical paths."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging."
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)

    dataset_cfg = cfg.get("dataset", {})
    key_cfg = cfg.get("keynode", {})
    path_cfg = cfg.get("paths", {})
    frag_cfg = cfg.get("fragility", {})
    rank_cfg = cfg.get("ranking", {})
    out_cfg = cfg.get("output", {})
    tb_cfg = cfg.get("tensorboard", {})
    debug_cfg = cfg.get("debug", {})
    debug = bool(args.debug or debug_cfg.get("enabled", False))

    _dbg_kv(debug, "loaded config sections", {
        "config_path": args.config,
        "dataset_name": dataset_cfg.get("name"),
        "dataset_root": dataset_cfg.get("root"),
        "importance_path": dataset_cfg.get("importance_path"),
        "node_features_path": dataset_cfg.get("node_features_path"),
        "community_mode": dataset_cfg.get("community_mode", "louvain"),
        "key_mode": key_cfg.get("mode", "topk"),
        "k_shortest": path_cfg.get("k_shortest", 3),
        "max_hops": path_cfg.get("max_hops", 8),
        "delta": path_cfg.get("delta", 2),
        "train_ratio": rank_cfg.get("train_ratio", 0.8),
        "top_per_task": rank_cfg.get("top_per_task", 1),
        "top_q": path_cfg.get("top_q", 10),
    })

    tb = TBLogger(
        log_root=resolve_output_path(tb_cfg.get("log_root", "outputs/tb")),
        experiment_name="rank",
        run_name=tb_cfg.get("run_name", f"{dataset_cfg['name']}_rank"),
        enabled=bool(tb_cfg.get("enabled", True)),
    )
    tb.add_config(cfg)

    # -------------------------
    # Stage 1: build graph bundle
    # -------------------------
    t0 = _stage_tic(debug, "build_graph_bundle")
    bundle = GraphPreprocessor.build_graph_bundle(
        name=dataset_cfg["name"],
        root=dataset_cfg["root"],
        importance_path=dataset_cfg["importance_path"],
        community_mode=dataset_cfg.get("community_mode", "louvain"),
        node_features_path=dataset_cfg.get("node_features_path"),
        old_id_col_in_node_features=dataset_cfg.get("old_id_col_in_node_features", "node"),
        strict_importance_alignment=dataset_cfg.get("strict_importance_alignment", True),
        importance_fill_value=dataset_cfg.get("importance_fill_value", 0.0),
        verbose=dataset_cfg.get("verbose", True),
    )
    _stage_toc(debug, "build_graph_bundle", t0)

    _dbg_kv(debug, "bundle summary", {
        "bundle.name": bundle.name,
        "bundle.num_nodes": bundle.num_nodes,
        "bundle.num_edges": bundle.nx_graph.number_of_edges(),
        "importance_alignment": bundle.metadata.get("importance_alignment"),
        "community_mode": bundle.metadata.get("community_mode"),
        "importance_mean": float(bundle.importance.mean()) if hasattr(bundle.importance, "mean") else None,
        "importance_std": float(bundle.importance.std()) if hasattr(bundle.importance, "std") else None,
    })

    # -------------------------
    # Stage 2: select key nodes
    # -------------------------
    t0 = _stage_tic(debug, "select_key_nodes")
    key_nodes = _select_key_nodes_from_cfg(bundle.importance, key_cfg)
    _stage_toc(debug, "select_key_nodes", t0)

    if not key_nodes:
        raise RuntimeError("No key nodes were selected.")

    _dbg_list(debug, "key_nodes", key_nodes, max_items=10)

    # -------------------------
    # Stage 3: build tasks
    # -------------------------
    t0 = _stage_tic(debug, "build_tasks")
    tasks = _build_tasks_from_cfg(bundle, key_nodes, key_cfg)
    _stage_toc(debug, "build_tasks", t0)

    if not tasks:
        raise RuntimeError(
            "No valid task pairs were constructed. "
            "Check key-node selection, graph connectivity, and shortest-path constraints."
        )

    _dbg_list(debug, "tasks", tasks, max_items=5)

    fragility_weights = {
        "lambda_E": float(frag_cfg.get("lambda_E", 0.4)),
        "lambda_LCC": float(frag_cfg.get("lambda_LCC", 0.4)),
        "lambda_ASP": float(frag_cfg.get("lambda_ASP", 0.2)),
        "lambda_red": float(rank_cfg.get("lambda_red", 0.2)),
    }
    _dbg_kv(debug, "fragility weights", fragility_weights)

    # -------------------------
    # Stage 4: build path samples
    # -------------------------
    t0 = _stage_tic(debug, "build_path_samples")
    df = PathRankingDatasetBuilder.build_path_samples(
        bundle=bundle,
        tasks=tasks,
        path_k=int(path_cfg.get("k_shortest", 12)),
        max_hops=int(path_cfg.get("max_hops", 10)),
        delta=int(path_cfg.get("delta", 3)),

        # ===== NEW: enlarge candidate pool =====
        raw_k_multiplier=int(path_cfg.get("raw_k_multiplier", 5)),
        raw_k_min_extra=int(path_cfg.get("raw_k_min_extra", 20)),
        final_k=int(path_cfg.get("final_k", path_cfg.get("k_shortest", 12))),
        max_internal_overlap=float(path_cfg.get("max_internal_overlap", 0.80)),
        fallback_relax_overlap=float(path_cfg.get("fallback_relax_overlap", 0.95)),
        fallback_extra_hops=int(path_cfg.get("fallback_extra_hops", 2)),

        fragility_weights=fragility_weights,
        fragility_mode=rank_cfg.get("fragility_mode", "hybrid"),
        cache_path=out_cfg.get("fragility_cache_json", "outputs/cache/rank_fragility_cache.json"),
        exact_every_n_tasks=int(rank_cfg.get("exact_every_n_tasks", 20)),
        exact_top_ranks=int(rank_cfg.get("exact_top_ranks", 1)),
        exact_max_path_len=int(rank_cfg.get("exact_max_path_len", 4)),
        progress_every=int(rank_cfg.get("progress_every", 10)),
        label_mode=str(rank_cfg.get("label_mode", "marginal")),
        debug=debug,
    )
    _stage_toc(debug, "build_path_samples", t0)

    if df.empty:
        raise RuntimeError("No candidate path samples were generated.")

    _dbg_df(debug, "all_samples", df, max_rows=5)

    feature_cols = rank_cfg.get("feature_cols") or PathRankingDatasetBuilder.feature_columns(df)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Configured feature columns are missing from ranking dataframe: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    _dbg_list(debug, "feature_cols", feature_cols, max_items=50)

    # -------------------------
    # Stage 5: split train/test
    # -------------------------
    t0 = _stage_tic(debug, "split_by_task")
    train_df, test_df = PathRankingDatasetBuilder.split_by_task(
        df,
        train_ratio=float(rank_cfg.get("train_ratio", 0.8)),
    )
    _stage_toc(debug, "split_by_task", t0)

    if train_df.empty:
        raise RuntimeError("Training split is empty. Increase task count or train_ratio.")

    if test_df.empty:
        if debug:
            test_df = train_df.copy()
            _dbg(debug, "test_df was empty, fallback to train_df.copy() in debug mode")
        else:
            raise RuntimeError(
                "Test split is empty in non-debug mode. "
                "Increase task count or adjust train_ratio."
            )

    _dbg_df(debug, "train_df", train_df, max_rows=3)
    _dbg_df(debug, "test_df", test_df, max_rows=3)

    # -------------------------
    # Stage 6: fit ranker
    # -------------------------
    t0 = _stage_tic(debug, "ranker_fit")
    ranker = XGBPathRanker(
        feature_cols=feature_cols,
        params=rank_cfg.get("xgb_params", None),
    )
    _dbg_kv(debug, "ranker init", {
        "backend_before_fit": getattr(ranker, "backend", None),
        "xgb_params": rank_cfg.get("xgb_params", None),
    })

    # 【修改点】：传入 val_df 以便在 TensorBoard 和控制台监控 NDCG 指标
    ranker.fit(train_df, val_df=test_df)
    _stage_toc(debug, "ranker_fit", t0)

    _dbg_kv(debug, "ranker fitted", {
        "backend_after_fit": getattr(ranker, "backend", None),
        "num_train_rows": len(train_df),
        "num_features": len(feature_cols),
    })

    # -------------------------
    # Stage 7: score dataframe
    # -------------------------
    t0 = _stage_tic(debug, "score_dataframe")
    scored_test_df = ranker.score_dataframe(test_df)
    scored_test_df["backend"] = ranker.backend
    _stage_toc(debug, "score_dataframe", t0)

    _dbg_df(debug, "scored_test_df", scored_test_df, max_rows=5)

    # -------------------------
    # Stage 8: dataframe -> pathrecords
    # -------------------------
    t0 = _stage_tic(debug, "dataframe_to_pathrecords")

    # 【修改点】：放宽提取限制。不要只取 top_per_task=1。
    # 我们提取每个任务的 Top-15（或全量），交给下游的 MMR 去优中选优。
    extract_top_k = int(rank_cfg.get("top_per_task", 15))
    if extract_top_k <= 1:
        extract_top_k = 15  # 强制保底，否则 MMR 无效

    path_records = XGBPathRanker.dataframe_to_pathrecords(
        scored_test_df,
        top_per_task=extract_top_k,
    )
    _stage_toc(debug, "dataframe_to_pathrecords", t0)
    '''
    # -------------------------
    # Stage 9: deduplicate
    # -------------------------
 
    t0 = _stage_tic(debug, "greedy_deduplicate")
    selected = PathDeduplicator.greedy_deduplicate(
        path_records,
        overlap_threshold=float(path_cfg.get("overlap_threshold", 0.6)),
        top_q=int(path_cfg.get("top_q", 10)),
    )
    _stage_toc(debug, "greedy_deduplicate", t0)

    _dbg_path_records(debug, "selected", selected, max_items=5)
    '''

    # -------------------------
    # Stage 9: select paths
    # -------------------------
    t0 = _stage_tic(debug, "select_paths")
    shared_base_metrics = FragilityEvaluator(
        lambda_E=fragility_weights["lambda_E"],
        lambda_LCC=fragility_weights["lambda_LCC"],
        lambda_ASP=fragility_weights["lambda_ASP"],
    ).compute_base_metrics(bundle.nx_graph)

    rank_mode = str(rank_cfg.get("mode", "pure_rank")).strip().lower()
    global_top_q = int(rank_cfg.get("global_top_q", path_cfg.get("top_q", 10)))

    _dbg_kv(debug, "rank mode", {
        "mode": rank_mode,
        "top_per_task": int(rank_cfg.get("top_per_task", 5)),
        "global_top_q": global_top_q,
    })

    if rank_mode == "pure_rank":
        # 版本 1：纯 rank
        # 只按 pred_score 全局排序，不做子模选择，不做冗余惩罚
        pure_global_sort = bool(rank_cfg.get("pure_global_sort", False))

        if pure_global_sort:
            ranked_candidates = sorted(
                path_records,
                key=lambda r: float(r.score) if getattr(r, "score", None) is not None else 0.0,
                reverse=True,
            )
            selected = ranked_candidates[:global_top_q]

        else:
            from collections import defaultdict

            candidates_by_task = defaultdict(list)
            for cand in path_records:
                candidates_by_task[(cand.source, cand.target)].append(cand)

            selected = []
            for task_pair, cands in candidates_by_task.items():
                cands = sorted(
                    cands,
                    key=lambda r: float(r.score) if getattr(r, "score", None) is not None else 0.0,
                    reverse=True,
                )
                selected.extend(cands)

            selected = _round_robin_truncate_by_task(selected, global_top_q)
        for i, cand in enumerate(selected):
            if getattr(cand, "metadata", None) is None:
                cand.metadata = {}
            cand.metadata.update({
                "selector": "pure_rank",
                "selection_step": i + 1,
            })


    elif rank_mode == "rank_set":

        selector_name = str(rank_cfg.get("selector", "pred_score_selector")).strip().lower()

        if selector_name == "pred_score_selector":

            # 1. 实例化 MMR Selector

            selector = PredScoreSelector(

                lambda_red=float(rank_cfg.get("lambda_red", 0.2)),

                edge_overlap_threshold=float(path_cfg.get("overlap_threshold", 0.8)),

                max_shared_internal_nodes=int(rank_cfg.get("max_shared_internal_nodes", 5)),

               top_q=int(rank_cfg.get("per_task_set_top_q", 3)),

            )

            # 2. 【核心修改】：XGBRanker 分数不可跨任务比较，必须按 Task 分组！

            from collections import defaultdict

            candidates_by_task = defaultdict(list)

            for cand in path_records:
                candidates_by_task[(cand.source, cand.target)].append(cand)

            selected = []

            for task_pair, cands in candidates_by_task.items():
                # 3. 对每个任务内部的候选路径进行 MMR 多样性去重

                task_selected = selector.select(cands)

                selected.extend(task_selected)

            if len(selected) > global_top_q:
                selected = _round_robin_truncate_by_task(selected, global_top_q)


        elif selector_name == "submodular_greedy":

            # SubmodularPathSelector 依赖的是计算真实的连通性下降值 (True Marginal Gain)

            # 因为是真实物理指标，所以【可以】跨 Task 比较！这里保持全局队列逻辑不变

            ranked_candidates = sorted(

                path_records,

                key=lambda r: float(r.score) if getattr(r, "score", None) is not None else 0.0,

                reverse=True,

            )

            selector = SubmodularPathSelector(

                lambda_E=fragility_weights["lambda_E"],

                lambda_LCC=fragility_weights["lambda_LCC"],

                lambda_ASP=fragility_weights["lambda_ASP"],

                lambda_red=float(rank_cfg.get("lambda_red", 0.2)),

                edge_overlap_threshold=float(path_cfg.get("overlap_threshold", 0.8)),

                max_shared_internal_nodes=int(rank_cfg.get("max_shared_internal_nodes", 5)),

                min_marginal_gain=float(rank_cfg.get("min_marginal_gain", 1e-6)),

                allow_negative_gain=float(rank_cfg.get("allow_negative_gain", -0.005)),

                use_lazy=True,

            )

            selected = selector.select(

                G=bundle.nx_graph,

                candidates=ranked_candidates,

                top_q=global_top_q,

                shared_base_metrics=shared_base_metrics,

            )

        else:

            raise ValueError(f"Unsupported selector: {selector_name}")

    else:
        raise ValueError(f"Unsupported ranking mode: {rank_mode}")

    _stage_toc(debug, "select_paths", t0)
    _dbg_path_records(debug, "selected", selected, max_items=10)

    # -------------------------
    # Stage 10: evaluate
    # -------------------------
    t0 = _stage_tic(debug, "evaluate_and_compare")
    evaluator = MethodEvaluator(
        lambda_E=fragility_weights["lambda_E"],
        lambda_LCC=fragility_weights["lambda_LCC"],
        lambda_ASP=fragility_weights["lambda_ASP"],
    )

    requested_k_list = list(out_cfg.get("k_list", [1, 3, 5, 10]))
    effective_k_list = [int(k) for k in requested_k_list if int(k) <= len(selected)]

    comparison = evaluator.compare_methods(
        {"xgb_rank": selected},
        bundle.nx_graph,
        effective_k_list,
        mode=rank_cfg.get("eval_mode", "hybrid"),
        early_stop=bool(rank_cfg.get("eval_early_stop", True)),
        tol=float(rank_cfg.get("eval_tol", 1e-4)),
        debug=bool(rank_cfg.get("eval_debug", False)),
        shared_base_metrics=shared_base_metrics,
    )

    summary = evaluator.summarize_top_paths(
        selected,
        top_n=int(out_cfg.get("top_n_summary", 10)),
    )
    _stage_toc(debug, "evaluate_and_compare", t0)

    _dbg_kv(debug, "comparison keys", {
        "methods": list(comparison.keys()),
        "requested_k_list": requested_k_list,
        "effective_k_list": effective_k_list,
    })
    _dbg_kv(debug, "summary info", {
        "summary_len": len(summary) if hasattr(summary, "__len__") else None
    })

    _log_dataset_info(
        tb=tb,
        bundle=bundle,
        key_nodes=key_nodes,
        tasks=tasks,
        df=df,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        backend=ranker.backend,
    )
    _log_selected_path_stats(tb, selected)
    _log_comparison_to_tb(tb, comparison, effective_k_list)

    tb.add_text(
        "rank/info/xgb_params",
        json.dumps(rank_cfg.get("xgb_params", {}), indent=2, ensure_ascii=False),
        0,
    )

    # -------------------------
    # Stage 11: resolve outputs
    # -------------------------
    metrics_out = resolve_output_path(
        out_cfg.get("metrics_json", "outputs/metrics/rank_metrics.json")
    )
    paths_out = resolve_output_path(
        out_cfg.get("paths_json", "outputs/paths/rank_paths.json")
    )
    dataset_out = resolve_output_path(
        out_cfg.get("dataset_csv", "outputs/metrics/rank_dataset.csv")
    )
    scored_test_out = resolve_output_path(
        out_cfg.get("scored_test_csv", "outputs/metrics/rank_scored_test.csv")
    )

    _dbg_kv(debug, "output paths", {
        "metrics_out": metrics_out,
        "paths_out": paths_out,
        "dataset_out": dataset_out,
        "scored_test_out": scored_test_out,
    })

    _ensure_parent(metrics_out)
    _ensure_parent(paths_out)
    _ensure_parent(dataset_out)
    _ensure_parent(scored_test_out)

    # ===== 新增：统一读取 mode / selector =====
    rank_mode = str(rank_cfg.get("mode", "pure_rank")).strip().lower()
    selector_name = str(rank_cfg.get("selector", "none")).strip().lower()

    # pure_rank 时，不应该显示 selector
    effective_selector = selector_name if rank_mode == "rank_set" else "none"

    # ===== 新增：兼容 shared_base_metrics 可能为空 =====
    shared_base_metrics_payload = None
    if "shared_base_metrics" in locals() and shared_base_metrics is not None:
        shared_base_metrics_payload = {
            "global_efficiency": float(shared_base_metrics["global_efficiency"]),
            "lcc_ratio": float(shared_base_metrics["lcc_ratio"]),
            "avg_shortest_path_lcc": float(shared_base_metrics["avg_shortest_path_lcc"]),
        }

    # ===== 新增：最终选中路径数 =====
    num_selected_paths = int(len(selected)) if "selected" in locals() and selected is not None else 0


    metrics_payload = {
        "dataset": {
            "name": bundle.name,
            "num_nodes": int(bundle.num_nodes),
            "num_edges": int(bundle.nx_graph.number_of_edges()),
            "community_mode": bundle.metadata.get("community_mode"),
            "importance_alignment": bundle.metadata.get("importance_alignment"),
        },

        "key_nodes": {
            "count": int(len(key_nodes)),
            "preview": [int(x) for x in key_nodes[: min(20, len(key_nodes))]],
        },

        "tasks": {
            "count": int(len(tasks)),
            "preview": [
                {
                    "source": int(t.source),
                    "target": int(t.target),
                    "shortest_len": int(t.shortest_len),
                    "same_community": bool(t.same_community),
                    "pair_score": float(t.pair_score),
                }
                for t in tasks[: min(20, len(tasks))]
            ],
        },

        "ranking": {
            # ===== 新增：实验模式信息 =====
            "mode": rank_mode,  # pure_rank | rank_set
            "selector": effective_selector,  # none | pred_score_selector | submodular_greedy
            "label_mode": str(rank_cfg.get("label_mode", "marginal")),

            # ===== 原有信息 =====
            "backend": ranker.backend,
            "feature_cols": list(feature_cols),
            "num_samples": int(len(df)),
            "num_train_samples": int(len(train_df)),
            "num_test_samples": int(len(test_df)),
            "top_per_task": int(rank_cfg.get("top_per_task", 1)),
            "global_top_q": int(rank_cfg.get("global_top_q", path_cfg.get("top_q", 10))),

            # ===== selector / set selection 参数 =====
            "lambda_red": float(rank_cfg.get("lambda_red", 0.2)),
            "allow_negative_gain": float(rank_cfg.get("allow_negative_gain", -0.005)),
            "max_shared_internal_nodes": int(rank_cfg.get("max_shared_internal_nodes", 3)),
            "min_marginal_gain": float(rank_cfg.get("min_marginal_gain", 1e-6)),
            "alpha_pred": float(rank_cfg.get("alpha_pred", 1.0)),
            "alpha_gain": float(rank_cfg.get("alpha_gain", 0.0)),
            "normalize_pred_score": bool(rank_cfg.get("normalize_pred_score", False)),
            "normalize_marginal_gain": bool(rank_cfg.get("normalize_marginal_gain", False)),

            # ===== 候选路径构造参数（新增，方便复现实验） =====
            "path_k": int(path_cfg.get("k_shortest", 12)),
            "final_k": int(path_cfg.get("final_k", path_cfg.get("k_shortest", 12))),
            "raw_k_multiplier": int(path_cfg.get("raw_k_multiplier", 5)),
            "raw_k_min_extra": int(path_cfg.get("raw_k_min_extra", 20)),
            "max_hops": int(path_cfg.get("max_hops", 10)),
            "delta": int(path_cfg.get("delta", 3)),
            "max_internal_overlap": float(path_cfg.get("max_internal_overlap", 0.80)),
            "edge_overlap_threshold": float(path_cfg.get("overlap_threshold", 0.80)),

            # ===== fragility dataset/eval 配置（保留） =====
            "fragility_mode": str(rank_cfg.get("fragility_mode", "hybrid")),
            "exact_every_n_tasks": int(rank_cfg.get("exact_every_n_tasks", 10)),
            "exact_top_ranks": int(rank_cfg.get("exact_top_ranks", 3)),
            "exact_max_path_len": int(rank_cfg.get("exact_max_path_len", 5)),
            "progress_every": int(rank_cfg.get("progress_every", 20)),
            "eval_mode": str(rank_cfg.get("eval_mode", "hybrid")),
            "eval_early_stop": bool(rank_cfg.get("eval_early_stop", True)),
            "eval_tol": float(rank_cfg.get("eval_tol", 1e-4)),
            "eval_debug": bool(rank_cfg.get("eval_debug", False)),

            # ===== damage curve 相关（新增） =====
            "requested_k_list": requested_k_list,
            "effective_k_list": effective_k_list,
            "num_selected_paths": num_selected_paths,

            # ===== shared base metrics（兼容 None） =====
            "shared_base_metrics": shared_base_metrics_payload,
        },

        "comparison": comparison,
        "top_summary": summary,
    }
    # -------------------------
    # Stage 12: save outputs
    # -------------------------
    t0 = _stage_tic(debug, "save_outputs")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    with open(paths_out, "w", encoding="utf-8") as f:
        json.dump(_serialize_pathrecords(selected), f, indent=2, ensure_ascii=False)

    df.to_csv(dataset_out, index=False)
    scored_test_df.to_csv(scored_test_out, index=False)

    tb.save_json("rank_metrics.json", metrics_payload)
    tb.save_json("rank_selected_paths.json", {"paths": _serialize_pathrecords(selected)})
    _stage_toc(debug, "save_outputs", t0)

    total_dt = time.perf_counter() - total_t0
    tb.add_scalar("rank/summary/num_key_nodes", len(key_nodes), 0)
    tb.add_scalar("rank/summary/num_tasks", len(tasks), 0)
    tb.add_scalar("rank/summary/num_selected_paths", len(selected), 0)
    tb.add_scalar("rank/summary/total_elapsed_sec", total_dt, 0)

    tb.flush()
    tb.close()

    print(f"[DONE] ranking baseline finished: {bundle.name}")
    print(f"[INFO] importance_alignment = {bundle.metadata.get('importance_alignment')}")
    print(f"[INFO] backend = {ranker.backend}")
    print(f"[INFO] #key_nodes = {len(key_nodes)}")
    print(f"[INFO] #tasks = {len(tasks)}, #samples = {len(df)}")
    print(f"[INFO] #selected_paths = {len(selected)}")
    print(f"[INFO] tensorboard_log_dir -> {tb.log_dir}")
    print(f"[INFO] total_elapsed_sec -> {total_dt:.4f}")
    print(f"[SAVE] metrics -> {metrics_out}")
    print(f"[SAVE] paths   -> {paths_out}")
    print(f"[SAVE] dataset -> {dataset_out}")
    print(f"[SAVE] scored  -> {scored_test_out}")
if __name__ == "__main__":
    main()