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
    raise ImportError("PyYAML is required for scripts/run_rule.py") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PATH_ROOT = Path(__file__).resolve().parents[1]  # D:/project/keynode/project/path

from path.src.baselines.betweenness_path import BetweennessPathBaseline
from path.src.baselines.node_score_path import NodeScorePathBaseline
from path.src.baselines.random_path import RandomPathBaseline
from path.src.baselines.rule_based import RuleBasedCriticalPath
from path.src.baselines.shortest_path import ShortestPathBaseline
from path.src.core.evaluator import MethodEvaluator
from path.src.core.types import TaskPair
from path.src.data.preprocess import GraphPreprocessor
from path.src.utils.tb_logger import TBLogger
from path.src.core.fragility import FragilityEvaluator
from path.src.core.keynode import KeyNodeSelector
from path.src.core.task_sampler import TaskPairBuilder
'''
cd D:\project\keynode\project
python -m path.scripts.run_rule --config path/configs/cora_rule.yaml
tensorboard --logdir D:\project\keynode\project\path\outputs\tb --port 6006
'''
'''
python -m path.scripts.run_rule --config path/configs/cora_rule.yaml
打印 rule 对比表
python -m path.scripts.print_comparison_rule_metrics --input path/outputs/metrics/cora_rule_metrics.json
生成 overlap / coverage / path type / marginal damage 表
python -m path.scripts.print_mechanism_rule_metrics --input path/outputs/metrics/cora_rule_metrics.json --topk 10 --output_dir path/outputs/metrics_tables
生成普通 top-k 方法对比 CSV
python -m path.scripts.build_method_comparison_table
生成路径质量表
python -m path.scripts.build_path_quality_table
生成 fixed-node-budget CSV
python -m path.scripts.build_fixed_node_budget_table
生成论文机制图
python -m path.scripts.plot_rule_mechanism_figures
'''
def resolve_output_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(PATH_ROOT / p)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def stage_print(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def timed_call(stage_name: str, fn, *args, **kwargs):
    stage_print(f"[STAGE-START] {stage_name}")
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
    except Exception as e:
        dt = time.perf_counter() - t0
        stage_print(f"[STAGE-ERROR] {stage_name} failed after {dt:.2f}s: {repr(e)}")
        raise
    dt = time.perf_counter() - t0
    stage_print(f"[STAGE-DONE] {stage_name} finished in {dt:.2f}s")
    return out


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def save_json(path_str: str, obj: Dict[str, Any]) -> None:
    ensure_parent(path_str)
    with open(path_str, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def log_dataset_and_task_info(
    tb: TBLogger,
    bundle,
    key_nodes: List[int],
    tasks: List[TaskPair],
) -> None:
    tb.add_scalar("rule/data/num_nodes", int(bundle.num_nodes), 0)
    tb.add_scalar("rule/data/num_edges", int(bundle.nx_graph.number_of_edges()), 0)
    tb.add_scalar("rule/data/num_key_nodes", int(len(key_nodes)), 0)
    tb.add_scalar("rule/data/num_tasks", int(len(tasks)), 0)

    tb.add_scalar("rule/data/importance_mean", float(bundle.importance.mean()), 0)
    tb.add_scalar("rule/data/importance_std", float(bundle.importance.std()), 0)

    if tasks:
        shortest_lens = [float(t.shortest_len) for t in tasks]
        pair_scores = [float(t.pair_score) for t in tasks]
        cross_comm = [0.0 if t.same_community else 1.0 for t in tasks]

        tb.add_scalar("rule/tasks/avg_shortest_len", _safe_mean(shortest_lens), 0)
        tb.add_scalar("rule/tasks/avg_pair_score", _safe_mean(pair_scores), 0)
        tb.add_scalar("rule/tasks/cross_community_ratio", _safe_mean(cross_comm), 0)


def log_path_set_stats(
    tb: TBLogger,
    method_name: str,
    path_records: List[Any],
) -> None:
    tb.add_scalar(f"{method_name}/paths/num_selected_paths", len(path_records), 0)

    if not path_records:
        return

    lengths = [float(len(p.nodes)) for p in path_records]
    tb.add_scalar(f"{method_name}/paths/avg_path_length", _safe_mean(lengths), 0)
    tb.add_scalar(f"{method_name}/paths/max_path_length", max(lengths), 0)
    tb.add_scalar(f"{method_name}/paths/min_path_length", min(lengths), 0)
    tb.add_histogram(f"{method_name}/paths/path_length_hist", lengths, 0)

    scores = [float(p.score) for p in path_records if getattr(p, "score", None) is not None]
    if scores:
        tb.add_scalar(f"{method_name}/paths/avg_path_score", _safe_mean(scores), 0)
        tb.add_scalar(f"{method_name}/paths/max_path_score", max(scores), 0)
        tb.add_scalar(f"{method_name}/paths/min_path_score", min(scores), 0)
        tb.add_histogram(f"{method_name}/paths/path_score_hist", scores, 0)

    fragility_scores = []
    delta_Es = []
    delta_LCCs = []
    delta_ASPs = []

    for p in path_records:
        frag = getattr(p, "fragility", None) or {}
        if "fragility_score" in frag:
            fragility_scores.append(float(frag["fragility_score"]))
        if "delta_E" in frag:
            delta_Es.append(float(frag["delta_E"]))
        if "delta_LCC" in frag:
            delta_LCCs.append(float(frag["delta_LCC"]))
        if "delta_ASP" in frag:
            delta_ASPs.append(float(frag["delta_ASP"]))

    if fragility_scores:
        tb.add_scalar(f"{method_name}/paths/avg_fragility_score", _safe_mean(fragility_scores), 0)
    if delta_Es:
        tb.add_scalar(f"{method_name}/paths/avg_delta_E", _safe_mean(delta_Es), 0)
    if delta_LCCs:
        tb.add_scalar(f"{method_name}/paths/avg_delta_LCC", _safe_mean(delta_LCCs), 0)
    if delta_ASPs:
        tb.add_scalar(f"{method_name}/paths/avg_delta_ASP", _safe_mean(delta_ASPs), 0)


def log_comparison_to_tb(
    tb: TBLogger,
    comparison: Dict[str, Any],
    k_list: List[int],
) -> None:
    if "methods" in comparison:
        method_dict = comparison["methods"]
    else:
        method_dict = comparison

    for method_name, metrics in method_dict.items():
        real_k_list = metrics.get("k_list", k_list)

        delta_E_curve = metrics.get("delta_E_curve", [])
        delta_LCC_curve = metrics.get("delta_LCC_curve", [])
        delta_ASP_curve = metrics.get("delta_ASP_curve", [])

        for k, v in zip(real_k_list, delta_E_curve):
            tb.add_scalar(f"{method_name}/damage/delta_E_at_k", v, int(k))

        for k, v in zip(real_k_list, delta_LCC_curve):
            tb.add_scalar(f"{method_name}/damage/delta_LCC_at_k", v, int(k))

        for k, v in zip(real_k_list, delta_ASP_curve):
            tb.add_scalar(f"{method_name}/damage/delta_ASP_at_k", v, int(k))

        if delta_E_curve:
            tb.add_scalar(f"{method_name}/damage/top_last_delta_E", float(delta_E_curve[-1]), 0)
        if delta_LCC_curve:
            tb.add_scalar(f"{method_name}/damage/top_last_delta_LCC", float(delta_LCC_curve[-1]), 0)
        if delta_ASP_curve:
            tb.add_scalar(f"{method_name}/damage/top_last_delta_ASP", float(delta_ASP_curve[-1]), 0)

def log_fixed_budget_to_tb(tb, comparison):
    method_dict = comparison.get("methods", comparison)

    for method_name, metrics in method_dict.items():
        budgets = metrics.get("node_budget_list", [])
        frag_curve = metrics.get("fragility_score_curve", [])
        delta_E_curve = metrics.get("delta_E_curve", [])
        delta_LCC_curve = metrics.get("delta_LCC_curve", [])
        delta_ASP_curve = metrics.get("delta_ASP_curve", [])

        for b, v in zip(budgets, frag_curve):
            tb.add_scalar(f"{method_name}/fixed_budget/fragility_at_budget", v, int(b))

        for b, v in zip(budgets, delta_E_curve):
            tb.add_scalar(f"{method_name}/fixed_budget/delta_E_at_budget", v, int(b))

        for b, v in zip(budgets, delta_LCC_curve):
            tb.add_scalar(f"{method_name}/fixed_budget/delta_LCC_at_budget", v, int(b))

        for b, v in zip(budgets, delta_ASP_curve):
            tb.add_scalar(f"{method_name}/fixed_budget/delta_ASP_at_budget", v, int(b))

def check_shared_base_metrics_consistency(
    shared_base_metrics: Dict[str, float],
    comparison: Dict[str, Any],
) -> None:
    methods = comparison.get("methods", {})
    if not methods:
        raise ValueError("comparison['methods'] is empty.")

    for method_name, metrics in methods.items():
        method_base = metrics.get("base_metrics")
        if method_base is None:
            raise ValueError(f"{method_name} missing base_metrics in comparison result.")

        for k, v in shared_base_metrics.items():
            mv = float(method_base[k])
            sv = float(v)
            if abs(mv - sv) > 1e-12:
                raise ValueError(
                    f"shared_base_metrics mismatch in method={method_name}, key={k}: "
                    f"method={mv}, shared={sv}"
                )

def compute_base_metrics_for_mode(
    evaluator: FragilityEvaluator,
    G,
    mode: str,
) -> Dict[str, float]:
    mode = str(mode).lower().strip()

    if mode == "exact":
        return {
            "global_efficiency": evaluator.global_efficiency_exact(G),
            "lcc_ratio": evaluator.lcc_ratio(G, G.number_of_nodes()),
            "avg_shortest_path_lcc": evaluator.avg_shortest_path_of_lcc_exact(G),
        }

    if mode == "approx":
        return evaluator.compute_base_metrics(G)

    if mode == "hybrid":
        # hybrid 的 top-k 前几项会用 exact，base 也建议 exact，避免混合基准。
        return {
            "global_efficiency": evaluator.global_efficiency_exact(G),
            "lcc_ratio": evaluator.lcc_ratio(G, G.number_of_nodes()),
            "avg_shortest_path_lcc": evaluator.avg_shortest_path_of_lcc_exact(G),
        }

    raise ValueError(f"Unsupported eval mode: {mode}")
def main() -> None:
    total_t0 = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Run rule-based critical-path identification and baselines."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    stage_print(f"[INFO] run_rule.py started")
    stage_print(f"[INFO] config path = {args.config}")

    cfg = load_config(args.config)

    dataset_cfg = cfg["dataset"]
    keynode_cfg = cfg["keynode"]
    paths_cfg = cfg["paths"]
    fragility_cfg = cfg.get(
        "fragility",
        {"lambda_E": 0.4, "lambda_LCC": 0.4, "lambda_ASP": 0.2}
    )
    baselines_cfg = cfg.get("baselines", {})
    output_cfg = cfg.get("output", {})
    tb_cfg = cfg.get("tensorboard", {})

    tb = TBLogger(
        log_root=resolve_output_path(tb_cfg.get("log_root", "outputs/tb")),
        experiment_name="rule",
        run_name=tb_cfg.get("run_name", f"{dataset_cfg['name']}_rule"),
        enabled=bool(tb_cfg.get("enabled", True)),
    )
    tb.add_config(cfg)

    stage_print(
        "[CONFIG] "
        f"dataset={dataset_cfg['name']}, "
        f"root={dataset_cfg['root']}, "
        f"importance_path={dataset_cfg['importance_path']}"
    )
    stage_print(
        "[CONFIG] "
        f"keynode_mode={keynode_cfg.get('mode', 'topk')}, "
        f"k={keynode_cfg.get('k')}, "
        f"ratio={keynode_cfg.get('ratio')}, "
        f"min_shortest_len={keynode_cfg.get('min_shortest_len', 2)}, "
        f"max_pairs={keynode_cfg.get('max_pairs')}"
    )
    stage_print(
        "[CONFIG] "
        f"k_shortest={paths_cfg.get('k_shortest', 3)}, "
        f"max_hops={paths_cfg.get('max_hops', 8)}, "
        f"delta={paths_cfg.get('delta', 2)}, "
        f"top_q={paths_cfg.get('top_q', 10)}, "
        f"overlap_threshold={paths_cfg.get('overlap_threshold', 0.6)}, "
        f"top_m_for_fragility={paths_cfg.get('top_m_for_fragility', 3)}, "
        f"fragility_gate={cfg.get('scorer', {}).get('fragility_gate', 0.50)}, "
        f"gate_penalty={cfg.get('scorer', {}).get('gate_penalty', 0.08)}"
    )
    stage_print(
        "[CONFIG] "
        f"random_num_trials={baselines_cfg.get('random_num_trials', 30)}, "
        f"random_num_samples={baselines_cfg.get('random_num_samples', 5)}, "
        f"random_seed={baselines_cfg.get('random_seed', 42)}, "
        f"k_candidates={baselines_cfg.get('k_candidates', 5)}, "
        f"use_internal_node_importance={baselines_cfg.get('use_internal_node_importance', False)}"
    )
    stage_print(
        "[CONFIG] "
        f"fragility={fragility_cfg}, "
        f"k_list={output_cfg.get('k_list', [1, 3, 5, 10])}, "
        f"top_n_summary={output_cfg.get('top_n_summary', 10)}"
    )

    bundle = timed_call(
        "build_graph_bundle",
        GraphPreprocessor.build_graph_bundle,
        name=dataset_cfg["name"],
        root=dataset_cfg["root"],
        importance_path=dataset_cfg["importance_path"],
        community_mode=dataset_cfg.get("community_mode", "louvain"),
        node_features_path=dataset_cfg["node_features_path"],
        old_id_col_in_node_features=dataset_cfg.get("old_id_col_in_node_features", "node"),
        strict_importance_alignment=dataset_cfg.get("strict_importance_alignment", True),
        importance_fill_value=dataset_cfg.get("importance_fill_value", 0.0),
        verbose=dataset_cfg.get("verbose", True),
    )

    stage_print(
        "[BUNDLE] "
        f"name={bundle.name}, "
        f"num_nodes={bundle.num_nodes}, "
        f"num_edges={bundle.nx_graph.number_of_edges()}, "
        f"community_mode={bundle.metadata.get('community_mode')}, "
        f"importance_alignment={bundle.metadata.get('importance_alignment')}"
    )

    keynode_mode = keynode_cfg.get("mode", "topk")

    if keynode_mode == "topk":
        key_nodes = timed_call(
            "KeyNodeSelector.select_topk_nodes",
            KeyNodeSelector.select_topk_nodes,
            importance=bundle.importance,
            k=keynode_cfg.get("k"),
        )
    elif keynode_mode == "top_ratio":
        key_nodes = timed_call(
            "KeyNodeSelector.select_top_ratio_nodes",
            KeyNodeSelector.select_top_ratio_nodes,
            importance=bundle.importance,
            ratio=keynode_cfg.get("ratio"),
        )
    elif keynode_mode == "stratified":
        node_buckets = timed_call(
            "KeyNodeSelector.select_stratified_nodes",
            KeyNodeSelector.select_stratified_nodes,
            importance=bundle.importance,
            total_k=keynode_cfg.get("k", 60),
            high_ratio=keynode_cfg.get("high_ratio", 0.4),
            mid_ratio=keynode_cfg.get("mid_ratio", 0.4),
            low_ratio=keynode_cfg.get("low_ratio", 0.2),
        )
        key_nodes = node_buckets["all"]
    else:
        raise ValueError(f"Unsupported keynode mode: {keynode_mode}")

    stage_print(
        f"[KEYNODES] count={len(key_nodes)}, "
        f"preview={key_nodes[:min(20, len(key_nodes))]}"
    )

    all_task_pairs = timed_call(
        "TaskPairBuilder.build_task_pairs",
        TaskPairBuilder.build_task_pairs,
        G=bundle.nx_graph,
        key_nodes=key_nodes,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=keynode_cfg.get("min_shortest_len", 2),
    )

    task_sampling_mode = str(
        keynode_cfg.get("task_sampling_mode", "hybrid")
    ).strip().lower()

    sample_kwargs = dict(
        task_pairs=all_task_pairs,
        num_samples=int(keynode_cfg.get("max_pairs", 200)),
        mode=task_sampling_mode,
        random_seed=int(keynode_cfg.get("random_seed", 42)),
        G=bundle.nx_graph,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=int(keynode_cfg.get("min_shortest_len", 2)),
        random_ratio=float(keynode_cfg.get("random_task_ratio", 0.10)),
    )

    tasks = timed_call(
        "TaskPairBuilder.sample_task_pairs",
        TaskPairBuilder.sample_task_pairs,
        **sample_kwargs,
    )
    if not tasks:
        raise RuntimeError(
            "No valid task pairs were constructed. "
            "Check key-node selection, graph connectivity, and shortest-path constraints."
        )
    fragility_evaluator = FragilityEvaluator(**fragility_cfg)

    selection_base_metrics = timed_call(
        "compute selection_base_metrics",
        fragility_evaluator.compute_base_metrics,
        bundle.nx_graph,
    )

    eval_mode = output_cfg.get("eval_mode", "exact")

    eval_base_metrics = timed_call(
        "compute eval_base_metrics",
        compute_base_metrics_for_mode,
        fragility_evaluator,
        bundle.nx_graph,
        eval_mode,
    )

    stage_print(f"[SELECTION-BASE-METRICS] {selection_base_metrics}")
    stage_print(f"[EVAL-BASE-METRICS] {eval_base_metrics}")

    tb.add_scalar(
        "rule/selection_base/global_efficiency",
        float(selection_base_metrics["global_efficiency"]),
        0,
    )
    tb.add_scalar(
        "rule/selection_base/lcc_ratio",
        float(selection_base_metrics["lcc_ratio"]),
        0,
    )
    tb.add_scalar(
        "rule/selection_base/avg_shortest_path_lcc",
        float(selection_base_metrics["avg_shortest_path_lcc"]),
        0,
    )

    tb.add_scalar(
        "rule/eval_base/global_efficiency",
        float(eval_base_metrics["global_efficiency"]),
        0,
    )
    tb.add_scalar(
        "rule/eval_base/lcc_ratio",
        float(eval_base_metrics["lcc_ratio"]),
        0,
    )
    tb.add_scalar(
        "rule/eval_base/avg_shortest_path_lcc",
        float(eval_base_metrics["avg_shortest_path_lcc"]),
        0,
    )

    log_dataset_and_task_info(tb, bundle, key_nodes, tasks)

    rule_candidate_stats: Dict[str, Any] = {}
    stage_print(f"[DEBUG] effective top_m_for_fragility = {paths_cfg.get('top_m_for_fragility', 3)}")
    baseline_top_m_for_fragility = int(
        baselines_cfg.get(
            "top_m_for_fragility",
            paths_cfg.get("top_m_for_fragility", 1),
        )
    )

    rule_paths = timed_call(
        "RuleBasedCriticalPath.run",
        RuleBasedCriticalPath.run,
        bundle=bundle,
        tasks=tasks,
        path_k=paths_cfg.get("k_shortest", 3),
        max_hops=paths_cfg.get("max_hops", 8),
        delta=paths_cfg.get("delta", 2),
        weights=cfg.get("rule_weights"),
        top_q=paths_cfg.get("top_q", 10),
        overlap_threshold=paths_cfg.get("overlap_threshold", 0.6),
        fragility_weights=fragility_cfg,
        top_m_for_fragility=paths_cfg.get("top_m_for_fragility", 3),
        fragility_gate=cfg.get("scorer", {}).get("fragility_gate", 0.50),
        gate_penalty=cfg.get("scorer", {}).get("gate_penalty", 0.08),
        shared_base_metrics=selection_base_metrics,
        candidate_stats=rule_candidate_stats,
        raw_k_multiplier=int(paths_cfg.get("raw_k_multiplier", 3)),
        raw_k_min_extra=int(paths_cfg.get("raw_k_min_extra", 10)),
        final_k=int(paths_cfg.get("final_k", paths_cfg.get("k_shortest", 3))),
        max_internal_overlap=float(paths_cfg.get("max_internal_overlap", 0.60)),
        fallback_relax_overlap=float(paths_cfg.get("fallback_relax_overlap", 0.95)),
        fallback_extra_hops=int(paths_cfg.get("fallback_extra_hops", 2)),
    )
    stage_print(f"[RESULT] rule_paths count = {len(rule_paths)}")

    shortest_paths = timed_call(
        "ShortestPathBaseline.run",
        ShortestPathBaseline.run,
        bundle=bundle,
        tasks=tasks,
        fragility_weights=fragility_cfg,
        shared_base_metrics=selection_base_metrics,
    )
    stage_print(f"[RESULT] shortest_paths count = {len(shortest_paths)}")

    random_paths = timed_call(
        "RandomPathBaseline.run",
        RandomPathBaseline.run,
        bundle=bundle,
        tasks=tasks,
        max_hops=paths_cfg.get("max_hops", 8),
        num_trials=baselines_cfg.get("random_num_trials", 30),
        num_samples=baselines_cfg.get("random_num_samples", 5),
        seed=baselines_cfg.get("random_seed", 42),
        fragility_weights=fragility_cfg,
        shared_base_metrics=selection_base_metrics,
        top_m_for_fragility=baseline_top_m_for_fragility,
    )
    stage_print(f"[RESULT] random_paths count = {len(random_paths)}")

    betweenness_paths = timed_call(
        "BetweennessPathBaseline.run",
        BetweennessPathBaseline.run,
        bundle=bundle,
        tasks=tasks,
        k_candidates=baselines_cfg.get("k_candidates", 5),
        max_hops=paths_cfg.get("max_hops", 8),
        delta=paths_cfg.get("delta", 2),
        fragility_weights=fragility_cfg,
        shared_base_metrics=selection_base_metrics,
        top_m_for_fragility=baseline_top_m_for_fragility,
    )
    stage_print(f"[RESULT] betweenness_paths count = {len(betweenness_paths)}")

    node_score_paths = timed_call(
        "NodeScorePathBaseline.run",
        NodeScorePathBaseline.run,
        bundle=bundle,
        tasks=tasks,
        k_candidates=baselines_cfg.get("k_candidates", 5),
        max_hops=paths_cfg.get("max_hops", 8),
        delta=paths_cfg.get("delta", 2),
        use_internal_only=baselines_cfg.get("use_internal_node_importance", False),
        fragility_weights=fragility_cfg,
        shared_base_metrics=selection_base_metrics,
        top_m_for_fragility=baseline_top_m_for_fragility,
    )
    stage_print(f"[RESULT] node_score_paths count = {len(node_score_paths)}")

    # 各方法路径集统计
    log_path_set_stats(tb, "rule", rule_paths)
    log_path_set_stats(tb, "shortest", shortest_paths)
    log_path_set_stats(tb, "random", random_paths)
    log_path_set_stats(tb, "betweenness", betweenness_paths)
    log_path_set_stats(tb, "node_score", node_score_paths)

    evaluator = MethodEvaluator(**fragility_cfg)
    k_list = output_cfg.get("k_list", [1, 3, 5, 10])

    comparison = timed_call(
        "MethodEvaluator.compare_methods",
        evaluator.compare_methods,
        result_dict={
            "rule": rule_paths,
            "shortest": shortest_paths,
            "random": random_paths,
            "betweenness": betweenness_paths,
            "node_score": node_score_paths,
        },
        G=bundle.nx_graph,
        k_list=k_list,
        mode=eval_mode,
        early_stop=False,
        tol=1e-4,
        shared_base_metrics=eval_base_metrics,
    )
    node_budget_list = output_cfg.get("node_budget_list", [5, 10, 20, 30, 50])

    fixed_node_budget_comparison = timed_call(
        "MethodEvaluator.compare_methods_fixed_node_budget",
        evaluator.compare_methods_fixed_node_budget,
        result_dict={
            "rule": rule_paths,
            "shortest": shortest_paths,
            "random": random_paths,
            "betweenness": betweenness_paths,
            "node_score": node_score_paths,
        },
        G=bundle.nx_graph,
        node_budget_list=node_budget_list,
        mode=eval_mode,
        shared_base_metrics=eval_base_metrics,
    )

    comparison.setdefault(
        "shared_base_metrics",
        {k: float(v) for k, v in eval_base_metrics.items()},
    )

    check_shared_base_metrics_consistency(
        shared_base_metrics=eval_base_metrics,
        comparison=comparison,
    )

    log_comparison_to_tb(tb, comparison, k_list)
    log_fixed_budget_to_tb(tb, fixed_node_budget_comparison)

    top_n = output_cfg.get("top_n_summary", 10)


    payload = {
        "dataset": {
            "name": bundle.name,
            "num_nodes": int(bundle.num_nodes),
            "num_edges": int(bundle.nx_graph.number_of_edges()),
            "community_mode": bundle.metadata.get("community_mode"),
            "importance_alignment": bundle.metadata.get("importance_alignment"),
        },
        "selection_base_metrics": {
            k: float(v) for k, v in selection_base_metrics.items()
        },
        "eval_base_metrics": {
            k: float(v) for k, v in eval_base_metrics.items()
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
        "fixed_node_budget_comparison": fixed_node_budget_comparison,
        "candidate_coverage": {
            "rule": rule_candidate_stats
        },
        "comparison": comparison,
        "top_paths": {
            "rule": evaluator.summarize_top_paths(rule_paths, top_n=top_n),
            "shortest": evaluator.summarize_top_paths(shortest_paths, top_n=top_n),
            "random": evaluator.summarize_top_paths(random_paths, top_n=top_n),
            "betweenness": evaluator.summarize_top_paths(betweenness_paths, top_n=top_n),
            "node_score": evaluator.summarize_top_paths(node_score_paths, top_n=top_n),
        },
    }

    metrics_out = resolve_output_path(output_cfg.get("metrics_json", "outputs/metrics/rule_metrics.json"))
    paths_out = resolve_output_path(output_cfg.get("paths_json", "outputs/paths/rule_paths.json"))
    candidate_out = resolve_output_path(
        output_cfg.get("candidate_json", "outputs/metrics/candidate_coverage.json")
    )

    stage_print(f"[OUTPUT] candidate_out = {candidate_out}")

    timed_call("save candidate json", save_json, candidate_out, rule_candidate_stats)
    tb.save_json("candidate_coverage.json", rule_candidate_stats)

    stage_print(f"[OUTPUT] metrics_out = {metrics_out}")
    stage_print(f"[OUTPUT] paths_out   = {paths_out}")

    timed_call("save metrics json", save_json, metrics_out, payload)
    timed_call("save paths json", save_json, paths_out, payload["top_paths"])

    # 同步保存到 TensorBoard run 目录
    tb.save_json("rule_metrics.json", payload)
    tb.save_json("rule_top_paths.json", payload["top_paths"])

    total_dt = time.perf_counter() - total_t0

    tb.add_scalar("rule/summary/num_key_nodes", len(key_nodes), 0)
    tb.add_scalar("rule/summary/num_tasks", len(tasks), 0)
    tb.add_scalar("rule/summary/num_rule_paths", len(rule_paths), 0)
    tb.add_scalar("rule/summary/total_elapsed_sec", total_dt, 0)

    tb.flush()
    tb.close()

    stage_print(f"[INFO] Dataset: {bundle.name}")
    stage_print(f"[INFO] #Key nodes: {len(key_nodes)}")
    stage_print(f"[INFO] #Tasks: {len(tasks)}")
    stage_print(f"[INFO] #Rule paths: {len(rule_paths)}")
    stage_print(f"[INFO] TensorBoard log dir: {tb.log_dir}")
    stage_print(f"[INFO] Metrics saved to: {metrics_out}")
    stage_print(f"[INFO] Top paths saved to: {paths_out}")
    stage_print(f"[INFO] TOTAL ELAPSED = {total_dt:.2f}s")


if __name__ == "__main__":
    main()