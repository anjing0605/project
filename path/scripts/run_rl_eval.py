from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required for scripts/run_rl_eval.py") from exc

# 项目根目录:
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PATH_ROOT = Path(__file__).resolve().parents[1]

from path.src.baselines.betweenness_path import BetweennessPathBaseline
from path.src.baselines.node_score_path import NodeScorePathBaseline
from path.src.baselines.random_path import RandomPathBaseline
from path.src.baselines.rule_based import RuleBasedCriticalPath
from path.src.baselines.shortest_path import ShortestPathBaseline
from path.src.core.evaluator import MethodEvaluator
from path.src.core.keynode import KeyNodeSelector
from path.src.core.task_sampler import TaskPairBuilder
from path.src.data.preprocess import GraphPreprocessor
from path.src.models.gnn_encoder import FrozenGNNEncoder
from path.src.rl.env import CriticalPathEnv
from path.src.rl.inference import RLPathInferencer
from path.src.rl.ppo_agent import PPOAgent
from path.src.rl.state_encoder import StateEncoder
from path.src.utils.tb_logger import TBLogger


def resolve_output_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(PATH_ROOT / p)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def save_json(path_str: str, obj: Dict[str, Any]) -> None:
    ensure_parent(path_str)
    with open(path_str, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def serialize_paths(records) -> List[Dict[str, Any]]:
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
                "success": bool(getattr(r, "success", True)),
                "method": r.method,
                "features": {str(k): float(v) for k, v in features.items()},
                "fragility": {str(k): float(v) for k, v in fragility.items()},
                "metadata": dict(metadata),
            }
        )
    return out


def select_key_nodes_from_cfg(importance, keynode_cfg: Dict[str, Any]) -> List[int]:
    mode = keynode_cfg.get("mode", "topk")

    if mode == "topk":
        k = int(keynode_cfg.get("k", 30))
        return KeyNodeSelector.select_topk_nodes(importance, k)

    if mode in ("top_ratio", "ratio"):
        ratio = float(keynode_cfg.get("ratio", 0.05))
        return KeyNodeSelector.select_top_ratio_nodes(importance, ratio)

    raise ValueError(
        f"Unsupported keynode mode: {mode}. "
        f"Expected one of ['topk', 'top_ratio', 'ratio']"
    )


def build_tasks_from_cfg(bundle, key_nodes: List[int], keynode_cfg: Dict[str, Any]):
    tasks = TaskPairBuilder.build_task_pairs(
        G=bundle.nx_graph,
        key_nodes=key_nodes,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=int(keynode_cfg.get("min_shortest_len", 2)),
    )

    max_pairs = keynode_cfg.get("max_pairs", None)
    if max_pairs is not None and int(max_pairs) > 0:
        tasks = tasks[: int(max_pairs)]

    return tasks


def load_agent_checkpoint(agent: PPOAgent, ckpt_path: str) -> None:
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"RL checkpoint not found: {ckpt_path}")
    agent.load(str(ckpt))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _log_dataset_info(
    tb: TBLogger,
    bundle,
    key_nodes: List[int],
    tasks: List[Any],
    sampled: List[Any],
    rl_paths: List[Any],
    emb_path: str,
    rl_ckpt: str,
    num_success_sampled: int,
    success_rate: float,
    selection_rate: float,
) -> None:
    tb.add_scalar("rl_eval/data/num_nodes", int(bundle.num_nodes), 0)
    tb.add_scalar("rl_eval/data/num_edges", int(bundle.nx_graph.number_of_edges()), 0)
    tb.add_scalar("rl_eval/data/num_key_nodes", int(len(key_nodes)), 0)
    tb.add_scalar("rl_eval/data/num_tasks", int(len(tasks)), 0)
    tb.add_scalar("rl_eval/data/num_sampled_paths", int(len(sampled)), 0)
    tb.add_scalar("rl_eval/data/num_success_sampled_paths", int(num_success_sampled), 0)
    tb.add_scalar("rl_eval/data/num_selected_rl_paths", int(len(rl_paths)), 0)
    tb.add_scalar("rl_eval/data/success_rate", float(success_rate), 0)
    tb.add_scalar("rl_eval/data/selection_rate", float(selection_rate), 0)
    tb.add_scalar("rl_eval/data/importance_mean", float(bundle.importance.mean()), 0)
    tb.add_scalar("rl_eval/data/importance_std", float(bundle.importance.std()), 0)

    if tasks:
        shortest_lens = [float(t.shortest_len) for t in tasks]
        pair_scores = [float(t.pair_score) for t in tasks]
        cross_comm = [0.0 if t.same_community else 1.0 for t in tasks]

        tb.add_scalar("rl_eval/tasks/avg_shortest_len", _safe_mean(shortest_lens), 0)
        tb.add_scalar("rl_eval/tasks/avg_pair_score", _safe_mean(pair_scores), 0)
        tb.add_scalar("rl_eval/tasks/cross_community_ratio", _safe_mean(cross_comm), 0)

    tb.add_text("rl_eval/info/embedding_ckpt", str(emb_path), 0)
    tb.add_text("rl_eval/info/rl_ckpt", str(rl_ckpt), 0)


def _log_path_set_stats(
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

    delta_Es: List[float] = []
    delta_LCCs: List[float] = []
    delta_ASPs: List[float] = []
    single_scores: List[float] = []
    terminal_rewards: List[float] = []
    marginal_gains: List[float] = []
    compressed_gains: List[float] = []

    for p in path_records:
        frag = getattr(p, "fragility", None) or {}
        if "delta_E" in frag:
            delta_Es.append(float(frag["delta_E"]))
        if "delta_LCC" in frag:
            delta_LCCs.append(float(frag["delta_LCC"]))
        if "delta_ASP" in frag:
            delta_ASPs.append(float(frag["delta_ASP"]))
        if "single_path_score" in frag:
            single_scores.append(float(frag["single_path_score"]))
        if "terminal_reward" in frag:
            terminal_rewards.append(float(frag["terminal_reward"]))
        if "marginal_gain" in frag:
            marginal_gains.append(float(frag["marginal_gain"]))
        if "compressed_marginal_gain" in frag:
            compressed_gains.append(float(frag["compressed_marginal_gain"]))

    if delta_Es:
        tb.add_scalar(f"{method_name}/paths/avg_delta_E", _safe_mean(delta_Es), 0)
    if delta_LCCs:
        tb.add_scalar(f"{method_name}/paths/avg_delta_LCC", _safe_mean(delta_LCCs), 0)
    if delta_ASPs:
        tb.add_scalar(f"{method_name}/paths/avg_delta_ASP", _safe_mean(delta_ASPs), 0)
    if single_scores:
        tb.add_scalar(f"{method_name}/paths/avg_single_path_score", _safe_mean(single_scores), 0)
    if terminal_rewards:
        tb.add_scalar(f"{method_name}/paths/avg_terminal_reward", _safe_mean(terminal_rewards), 0)
    if marginal_gains:
        tb.add_scalar(f"{method_name}/paths/avg_marginal_gain", _safe_mean(marginal_gains), 0)
    if compressed_gains:
        tb.add_scalar(
            f"{method_name}/paths/avg_compressed_marginal_gain",
            _safe_mean(compressed_gains),
            0,
        )


def _log_comparison_to_tb(
    tb: TBLogger,
    comparison: Dict[str, Any],
    k_list: List[int],
) -> None:
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
        description="Evaluate trained PPO agent and compare with baselines."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    dataset_cfg = cfg["dataset"]
    keynode_cfg = cfg["keynode"]
    paths_cfg = cfg["paths"]
    fragility_cfg = cfg.get(
        "fragility",
        {"lambda_E": 0.55, "lambda_LCC": 0.0, "lambda_ASP": 0.45},
    )
    rl_cfg = cfg["rl"]
    baselines_cfg = cfg.get("baselines", {})
    output_cfg = cfg.get("output", {})
    tb_cfg = cfg.get("tensorboard", {})

    tb = TBLogger(
        log_root=resolve_output_path(tb_cfg.get("log_root", "outputs/tb")),
        experiment_name="rl_eval",
        run_name=tb_cfg.get("run_name", f"{dataset_cfg['name']}_rl_eval"),
        enabled=bool(tb_cfg.get("enabled", True)),
    )
    tb.add_config(cfg)

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

    key_nodes = select_key_nodes_from_cfg(bundle.importance, keynode_cfg)
    if not key_nodes:
        raise RuntimeError("No key nodes were selected.")

    tasks = build_tasks_from_cfg(bundle, key_nodes, keynode_cfg)
    if not tasks:
        raise RuntimeError(
            "No evaluation tasks constructed. "
            "Check key-node selection, graph connectivity, and shortest-path constraints."
        )

    emb_path = resolve_output_path(
        output_cfg.get("embedding_ckpt", "outputs/checkpoints/node_embeddings.pt")
    )
    embeddings = FrozenGNNEncoder.fit_or_load(
        bundle=bundle,
        ckpt_path=emb_path,
        hidden_dim=int(rl_cfg.get("gnn_hidden_dim", 128)),
        out_dim=int(rl_cfg.get("embedding_dim", 64)),
        epochs=int(rl_cfg.get("gnn_epochs", 120)),
        lr=float(rl_cfg.get("gnn_lr", 1e-2)),
    )

    env = CriticalPathEnv(
        bundle=bundle,
        node_embeddings=embeddings,
        max_hops=int(rl_cfg.get("max_hops", 8)),
        reward_mode=rl_cfg.get("eval_reward_mode", "fragility_topk_align"),
        reward_kwargs=rl_cfg.get("reward", {}),
    )

    dummy_state = env.reset(tasks[0])
    state_dim = int(StateEncoder.encode_state(dummy_state).numel())
    action_feats, _ = StateEncoder.encode_actions(
        state=dummy_state,
        node_embeddings=embeddings,
        importance=bundle.importance,
        edge_bc=bundle.edge_bc,
        graph=env.G,
        community=bundle.community,
    )
    action_dim = int(action_feats.shape[1])

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=int(rl_cfg.get("hidden_dim", 128)),
        policy_lr=float(rl_cfg.get("policy_lr", 1e-4)),
        value_lr=float(rl_cfg.get("value_lr", 1e-4)),
        gamma=float(rl_cfg.get("gamma", 0.99)),
        gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
        clip_eps=float(rl_cfg.get("clip_eps", 0.2)),
        entropy_coef=float(rl_cfg.get("entropy_coef", 0.01)),
        value_coef=float(rl_cfg.get("value_coef", 0.5)),
        ppo_epochs=int(rl_cfg.get("ppo_epochs", 4)),
        device=str(rl_cfg.get("device", "cpu")),
        action_temperature=float(rl_cfg.get("action_temperature", 1.0)),
    )

    rl_ckpt = resolve_output_path(
        output_cfg.get("rl_ckpt", "outputs/checkpoints/ppo_agent.pt")
    )
    load_agent_checkpoint(agent, rl_ckpt)

    sampled = RLPathInferencer.sample_paths(
        agent=agent,
        env=env,
        tasks=tasks,
        num_samples_per_task=int(rl_cfg.get("num_samples_per_task", 10)),
        keep_failed=True,
        deterministic=bool(rl_cfg.get("deterministic_eval", True)),
    )

    num_sampled_total = int(len(sampled))
    num_success_sampled = int(sum(1 for p in sampled if bool(getattr(p, "success", False))))
    success_rate = float(num_success_sampled / max(num_sampled_total, 1))

    rl_paths = RLPathInferencer.rescore_and_select(
        bundle=bundle,
        path_records=sampled,
        top_q=int(paths_cfg.get("top_q", 10)),
        overlap_threshold=float(paths_cfg.get("overlap_threshold", 0.6)),
        fragility_weights=fragility_cfg,
        require_success=True,
    )

    selection_rate = float(len(rl_paths) / max(num_success_sampled, 1))

    rule_paths = RuleBasedCriticalPath.run(
        bundle=bundle,
        tasks=tasks,
        path_k=int(paths_cfg.get("k_shortest", 3)),
        max_hops=int(paths_cfg.get("max_hops", 8)),
        delta=int(paths_cfg.get("delta", 2)),
        weights=cfg.get("rule_weights"),
        top_q=int(paths_cfg.get("top_q", 10)),
        overlap_threshold=float(paths_cfg.get("overlap_threshold", 0.6)),
        fragility_weights=fragility_cfg,
    )

    shortest_paths = ShortestPathBaseline.run(
        bundle=bundle,
        tasks=tasks,
        fragility_weights=fragility_cfg,
    )

    random_paths = RandomPathBaseline.run(
        bundle=bundle,
        tasks=tasks,
        num_samples=int(baselines_cfg.get("random_num_samples", 5)),
        max_hops=int(paths_cfg.get("max_hops", 8)),
        num_trials=int(baselines_cfg.get("random_num_trials", 30)),
        seed=int(baselines_cfg.get("random_seed", 42)),
        fragility_weights=fragility_cfg,
    )

    betweenness_paths = BetweennessPathBaseline.run(
        bundle=bundle,
        tasks=tasks,
        k_candidates=int(baselines_cfg.get("k_candidates", 5)),
        max_hops=int(paths_cfg.get("max_hops", 8)),
        delta=int(paths_cfg.get("delta", 2)),
        fragility_weights=fragility_cfg,
    )

    node_score_paths = NodeScorePathBaseline.run(
        bundle=bundle,
        tasks=tasks,
        k_candidates=int(baselines_cfg.get("k_candidates", 5)),
        max_hops=int(paths_cfg.get("max_hops", 8)),
        delta=int(paths_cfg.get("delta", 2)),
        use_internal_only=bool(baselines_cfg.get("use_internal_node_importance", False)),
        fragility_weights=fragility_cfg,
    )

    evaluator = MethodEvaluator(
        lambda_E=float(fragility_cfg.get("lambda_E", 0.55)),
        lambda_LCC=float(fragility_cfg.get("lambda_LCC", 0.0)),
        lambda_ASP=float(fragility_cfg.get("lambda_ASP", 0.45)),
    )

    k_list = output_cfg.get("k_list", [1, 3, 5, 10])
    comparison = evaluator.compare_methods(
        result_dict={
            "rl": rl_paths,
            "rule_based": rule_paths,
            "shortest": shortest_paths,
            "random": random_paths,
            "betweenness": betweenness_paths,
            "node_score": node_score_paths,
        },
        G=bundle.nx_graph,
        k_list=k_list,
    )

    _log_dataset_info(
        tb=tb,
        bundle=bundle,
        key_nodes=key_nodes,
        tasks=tasks,
        sampled=sampled,
        rl_paths=rl_paths,
        emb_path=emb_path,
        rl_ckpt=rl_ckpt,
        num_success_sampled=num_success_sampled,
        success_rate=success_rate,
        selection_rate=selection_rate,
    )

    _log_path_set_stats(tb, "rl", rl_paths)
    _log_path_set_stats(tb, "rule_based", rule_paths)
    _log_path_set_stats(tb, "shortest", shortest_paths)
    _log_path_set_stats(tb, "random", random_paths)
    _log_path_set_stats(tb, "betweenness", betweenness_paths)
    _log_path_set_stats(tb, "node_score", node_score_paths)

    _log_comparison_to_tb(tb, comparison, k_list)

    top_n = int(output_cfg.get("top_n_summary", 10))
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
        "rl_eval": {
            "num_sampled_paths": num_sampled_total,
            "num_success_sampled_paths": num_success_sampled,
            "num_selected_paths": int(len(rl_paths)),
            "success_rate": float(success_rate),
            "selection_rate": float(selection_rate),
            "embedding_ckpt": emb_path,
            "rl_ckpt": rl_ckpt,
            "eval_reward_mode": str(rl_cfg.get("eval_reward_mode", "fragility_topk_align")),
            "deterministic_eval": bool(rl_cfg.get("deterministic_eval", True)),
        },
        "comparison": comparison,
        "top_paths": {
            "rl": evaluator.summarize_top_paths(rl_paths, top_n=top_n),
            "rule_based": evaluator.summarize_top_paths(rule_paths, top_n=top_n),
            "shortest": evaluator.summarize_top_paths(shortest_paths, top_n=top_n),
            "random": evaluator.summarize_top_paths(random_paths, top_n=top_n),
            "betweenness": evaluator.summarize_top_paths(betweenness_paths, top_n=top_n),
            "node_score": evaluator.summarize_top_paths(node_score_paths, top_n=top_n),
        },
    }

    metrics_json = resolve_output_path(
        output_cfg.get("rl_eval_metrics_json", "outputs/metrics/rl_eval_metrics.json")
    )
    paths_json = resolve_output_path(
        output_cfg.get("rl_eval_paths_json", "outputs/paths/rl_eval_paths.json")
    )

    save_json(metrics_json, metrics_payload)
    save_json(
        paths_json,
        {
            "rl": serialize_paths(rl_paths),
            "rule_based": serialize_paths(rule_paths),
            "shortest": serialize_paths(shortest_paths),
            "random": serialize_paths(random_paths),
            "betweenness": serialize_paths(betweenness_paths),
            "node_score": serialize_paths(node_score_paths),
        },
    )

    tb.save_json("rl_eval_metrics.json", metrics_payload)
    tb.save_json(
        "rl_eval_paths.json",
        {
            "rl": serialize_paths(rl_paths),
            "rule_based": serialize_paths(rule_paths),
            "shortest": serialize_paths(shortest_paths),
            "random": serialize_paths(random_paths),
            "betweenness": serialize_paths(betweenness_paths),
            "node_score": serialize_paths(node_score_paths),
        },
    )

    total_dt = time.perf_counter() - total_t0
    tb.add_scalar("rl_eval/summary/num_key_nodes", len(key_nodes), 0)
    tb.add_scalar("rl_eval/summary/num_tasks", len(tasks), 0)
    tb.add_scalar("rl_eval/summary/num_sampled_rl_paths", num_sampled_total, 0)
    tb.add_scalar("rl_eval/summary/num_success_sampled_rl_paths", num_success_sampled, 0)
    tb.add_scalar("rl_eval/summary/num_selected_rl_paths", len(rl_paths), 0)
    tb.add_scalar("rl_eval/summary/success_rate", success_rate, 0)
    tb.add_scalar("rl_eval/summary/selection_rate", selection_rate, 0)
    tb.add_scalar("rl_eval/summary/total_elapsed_sec", total_dt, 0)

    tb.flush()
    tb.close()

    print(f"[DONE] RL eval finished: {bundle.name}")
    print(f"[INFO] importance_alignment = {bundle.metadata.get('importance_alignment')}")
    print(f"[INFO] #key_nodes = {len(key_nodes)}")
    print(f"[INFO] #tasks = {len(tasks)}")
    print(f"[INFO] #sampled_rl_paths = {num_sampled_total}")
    print(f"[INFO] #success_sampled_rl_paths = {num_success_sampled}")
    print(f"[INFO] success_rate = {success_rate:.6f}")
    print(f"[INFO] #selected_rl_paths = {len(rl_paths)}")
    print(f"[INFO] selection_rate = {selection_rate:.6f}")
    print(f"[INFO] eval_reward_mode = {rl_cfg.get('eval_reward_mode', 'fragility_topk_align')}")
    print(f"[INFO] deterministic_eval = {rl_cfg.get('deterministic_eval', True)}")
    print(f"[INFO] tensorboard_log_dir -> {tb.log_dir}")
    print(f"[SAVE] metrics -> {metrics_json}")
    print(f"[SAVE] paths   -> {paths_json}")


if __name__ == "__main__":
    main()