from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

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
'''
cd D:\project\keynode\project
python -m path.scripts.run_rule --config path/configs/cora_rule.yaml
tensorboard --logdir D:\project\keynode\project\path\outputs\tb --port 6006
'''
'''
数据与任务信息
rule/data/num_nodes
rule/data/num_edges
rule/data/num_key_nodes
rule/data/num_tasks
rule/data/importance_mean
rule/data/importance_std
rule/tasks/avg_shortest_len
rule/tasks/avg_pair_score
rule/tasks/cross_community_ratio
各方法路径集合统计

对 rule / shortest / random / betweenness / node_score 都会记录：

*/paths/num_selected_paths
*/paths/avg_path_length
*/paths/max_path_length
*/paths/min_path_length
*/paths/avg_path_score（若有）
*/paths/avg_fragility_score
*/paths/avg_delta_E
*/paths/avg_delta_LCC
*/paths/avg_delta_ASP
top-k damage 曲线

对每个方法都会记录：

*/damage/delta_E_at_k
*/damage/delta_LCC_at_k
*/damage/delta_ASP_at_k
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


def select_key_nodes(
    importance,
    mode: str,
    k: int | None = None,
    ratio: float | None = None
) -> List[int]:
    num_nodes = len(importance)

    if mode == "topk":
        if k is None or k <= 1:
            raise ValueError("When keynode.mode='topk', keynode.k must be > 1.")
        num_select = int(k)

    elif mode == "top_ratio":
        if ratio is None or not (0.0 < float(ratio) <= 1.0):
            raise ValueError("When keynode.mode='top_ratio', keynode.ratio must be in (0,1].")
        num_select = max(2, int(round(num_nodes * float(ratio))))

    else:
        raise ValueError(f"Unsupported keynode mode: {mode}")

    ranked = sorted(range(num_nodes), key=lambda i: float(importance[i]), reverse=True)
    selected = ranked[:num_select]
    return selected


def build_task_pairs(
    G: nx.Graph,
    key_nodes: List[int],
    community,
    importance,
    min_shortest_len: int = 2,
    max_pairs: int | None = None,
) -> List[TaskPair]:
    tasks: List[TaskPair] = []
    skipped_no_path = 0
    skipped_too_short = 0
    total_pairs_considered = 0

    t0 = time.perf_counter()

    for i, source in enumerate(key_nodes):
        for target in key_nodes[i + 1:]:
            total_pairs_considered += 1

            try:
                shortest_len = int(nx.shortest_path_length(G, source=source, target=target))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                skipped_no_path += 1
                continue

            if shortest_len < int(min_shortest_len):
                skipped_too_short += 1
                continue

            same_community = bool(community[source] == community[target])
            pair_score = float(importance[source] + importance[target])

            tasks.append(
                TaskPair(
                    source=int(source),
                    target=int(target),
                    shortest_len=shortest_len,
                    same_community=same_community,
                    pair_score=pair_score,
                )
            )

    tasks.sort(
        key=lambda t: (
            int(t.same_community),   # False(0) 优先于 True(1)
            -t.pair_score,
            -t.shortest_len,
            t.source,
            t.target,
        )
    )

    if max_pairs is not None and max_pairs > 0:
        tasks = tasks[: int(max_pairs)]

    dt = time.perf_counter() - t0
    stage_print(
        "[TASKS] "
        f"considered={total_pairs_considered}, "
        f"valid={len(tasks)}, "
        f"skipped_no_path={skipped_no_path}, "
        f"skipped_too_short={skipped_too_short}, "
        f"elapsed={dt:.2f}s"
    )

    if tasks:
        preview = tasks[: min(5, len(tasks))]
        for idx, t in enumerate(preview):
            stage_print(
                f"[TASKS-PREVIEW-{idx}] "
                f"source={t.source}, target={t.target}, "
                f"shortest_len={t.shortest_len}, "
                f"same_community={t.same_community}, "
                f"pair_score={t.pair_score:.6f}"
            )

    return tasks


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
    """
    comparison 典型形式:
    {
        "rule": {
            "delta_E_curve": [...],
            "delta_LCC_curve": [...],
            "delta_ASP_curve": [...]
        },
        ...
    }
    """
    if "methods" in comparison:
        method_dict = comparison["methods"]
    else:
        method_dict = comparison
    for method_name, metrics in method_dict.items():
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
        f"top_m_for_fragility={paths_cfg.get('top_m_for_fragility', 1)}, "
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

    key_nodes = timed_call(
        "select_key_nodes",
        select_key_nodes,
        importance=bundle.importance,
        mode=keynode_cfg.get("mode", "topk"),
        k=keynode_cfg.get("k"),
        ratio=keynode_cfg.get("ratio"),
    )

    stage_print(
        f"[KEYNODES] count={len(key_nodes)}, "
        f"preview={key_nodes[:min(20, len(key_nodes))]}"
    )

    tasks = timed_call(
        "build_task_pairs",
        build_task_pairs,
        G=bundle.nx_graph,
        key_nodes=key_nodes,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=keynode_cfg.get("min_shortest_len", 2),
        max_pairs=keynode_cfg.get("max_pairs"),
    )

    if not tasks:
        raise RuntimeError(
            "No valid task pairs were constructed. "
            "Check key-node selection, graph connectivity, and shortest-path constraints."
        )
    shared_base_metrics = timed_call(
        "compute shared_base_metrics",
        FragilityEvaluator(**fragility_cfg).compute_base_metrics,
        bundle.nx_graph,
    )

    stage_print(f"[SHARED-BASE-METRICS] {shared_base_metrics}")
    tb.add_scalar("rule/shared_base/global_efficiency", float(shared_base_metrics["global_efficiency"]), 0)
    tb.add_scalar("rule/shared_base/lcc_ratio", float(shared_base_metrics["lcc_ratio"]), 0)
    tb.add_scalar("rule/shared_base/avg_shortest_path_lcc", float(shared_base_metrics["avg_shortest_path_lcc"]), 0)

    log_dataset_and_task_info(tb, bundle, key_nodes, tasks)

    rule_candidate_stats: Dict[str, Any] = {}

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
        top_m_for_fragility=paths_cfg.get("top_m_for_fragility", 1),
        fragility_gate=cfg.get("scorer", {}).get("fragility_gate", 0.50),
        gate_penalty=cfg.get("scorer", {}).get("gate_penalty", 0.08),
        shared_base_metrics=shared_base_metrics,
        candidate_stats=rule_candidate_stats,
    )
    stage_print(f"[RESULT] rule_paths count = {len(rule_paths)}")

    shortest_paths = timed_call(
        "ShortestPathBaseline.run",
        ShortestPathBaseline.run,
        bundle=bundle,
        tasks=tasks,
        fragility_weights=fragility_cfg,
        shared_base_metrics=shared_base_metrics,
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
        shared_base_metrics=shared_base_metrics,
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
        shared_base_metrics=shared_base_metrics,
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
        shared_base_metrics=shared_base_metrics,
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
        shared_base_metrics=shared_base_metrics,
    )

    log_comparison_to_tb(tb, comparison, k_list)

    top_n = output_cfg.get("top_n_summary", 10)

    payload = {
        "dataset": {
            "name": bundle.name,
            "num_nodes": int(bundle.num_nodes),
            "num_edges": int(bundle.nx_graph.number_of_edges()),
            "community_mode": bundle.metadata.get("community_mode"),
            "importance_alignment": bundle.metadata.get("importance_alignment"),
        },
        "shared_base_metrics": {
            k: float(v) for k, v in shared_base_metrics.items()
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