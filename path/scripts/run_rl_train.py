from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from path.src.data.preprocess import GraphPreprocessor
from path.src.models.gnn_encoder import FrozenGNNEncoder
from path.src.core.keynode import KeyNodeSelector
from path.src.core.task_sampler import TaskPairBuilder
from path.src.core.types import TaskPair
from path.src.rl.env import CriticalPathEnv
from path.src.rl.ppo_agent import PPOAgent
from path.src.rl.inference import RLPathInferencer
from path.src.baselines.rule_based import RuleBasedCriticalPath
from path.src.core.evaluator import MethodEvaluator
from path.src.utils.seed import set_seed
from path.src.utils.io import save_json
from path.src.rl.state_encoder import StateEncoder

# =========================
# helpers
# =========================

def bucket_tasks_by_shortest_len(tasks: List[TaskPair]) -> Dict[str, List[TaskPair]]:
    out = {"len2": [], "len3": [], "len4p": []}
    for t in tasks:
        if int(t.shortest_len) == 2:
            out["len2"].append(t)
        elif int(t.shortest_len) == 3:
            out["len3"].append(t)
        else:
            out["len4p"].append(t)
    return out


def sample_from_buckets(
    buckets: Dict[str, List[TaskPair]],
    ratios: Dict[str, float],
    total_size: int,
    seed: int,
) -> List[TaskPair]:
    rng = np.random.default_rng(seed)
    selected = []

    for name, ratio in ratios.items():
        pool = buckets.get(name, [])
        if len(pool) == 0:
            continue
        k = max(1, int(round(total_size * ratio)))
        idx = rng.choice(len(pool), size=k, replace=(k > len(pool)))
        selected.extend([pool[i] for i in idx])

    rng.shuffle(selected)
    return selected[:total_size]


def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    alpha = float(step) / float(total_steps - 1)
    return (1.0 - alpha) * start + alpha * end


def ensure_dir(path: str | Path) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
#按 bucket 做 train/val 分层切分
def split_bucket_train_val(
    buckets: Dict[str, List[TaskPair]],
    val_ratio: float,
    seed: int,
) -> tuple[Dict[str, List[TaskPair]], Dict[str, List[TaskPair]]]:
    rng = np.random.default_rng(seed)
    train_buckets = {}
    val_buckets = {}

    for name, pool in buckets.items():
        pool = list(pool)
        if len(pool) == 0:
            train_buckets[name] = []
            val_buckets[name] = []
            continue

        idx = np.arange(len(pool))
        rng.shuffle(idx)

        val_size = max(1, int(round(len(pool) * val_ratio))) if len(pool) >= 5 else max(0, int(len(pool) * val_ratio))
        val_idx = set(idx[:val_size].tolist())

        train_pool = [pool[i] for i in range(len(pool)) if i not in val_idx]
        val_pool = [pool[i] for i in range(len(pool)) if i in val_idx]

        # 防止某个桶 train 为空
        if len(train_pool) == 0 and len(val_pool) > 0:
            train_pool.append(val_pool.pop())

        train_buckets[name] = train_pool
        val_buckets[name] = val_pool

    return train_buckets, val_buckets
#把多个 bucket 合并成一个任务列表
def flatten_bucket_dict(bucket_dict: Dict[str, List[TaskPair]], names: List[str]) -> List[TaskPair]:
    out = []
    for n in names:
        out.extend(bucket_dict.get(n, []))
    return out
#这个函数只评估“到达率”，不做 PPO 更新。
def evaluate_arrival_rate(
    agent,
    env,
    tasks: List[TaskPair],
    max_tasks: int | None = None,
    deterministic: bool = True,
) -> float:
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    if len(tasks) == 0:
        return 0.0

    success = 0
    total = 0

    for task in tasks:
        ep_result, _ = agent.rollout_episode(
            env,
            task,
            deterministic=deterministic,
        )
        success += int(ep_result.reached_target)
        total += 1

    return float(success) / float(total) if total > 0 else 0.0
# =========================
# config
# =========================

def build_default_config():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path("path/outputs")
    return {
        "dataset_name": "Cora",
        "dataset_root": r"D:\project\keynode\project\public_datasets\Planetoid",
        "importance_path": r"D:\project\keynode\project\public_process\extract\Cora_struct\gnn_node_scores_mean.csv",
        "community_mode": "louvain",
        "topk_key_nodes": 30,
        "min_shortest_len": 2,
        "num_task_pairs": 200,
        "embedding_ckpt": str(out_root / "checkpoints" / "cora_node_embeddings.pt"),
        "rl_ckpt_stage1": str(out_root / "checkpoints" / "cora_ppo_agent_stage1.pt"),
        "rl_ckpt_stage2": str(out_root / "checkpoints" / "cora_ppo_agent_stage2.pt"),
        "tensorboard_log_dir": str(out_root / "tb" / "rl" / f"cora_rl_stagewise_{timestamp}"),
        "metrics_out": str(out_root / "metrics" / f"cora_rl_stagewise_{timestamp}.json"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "max_hops": 8,
        "hidden_dim": 128,
        "state_dim": 133,
        "action_dim": 197,
        "policy_lr": 1e-4,
        "value_lr": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "ppo_epochs": 4,
        "num_samples_per_epoch": 100,

        # ===== Stage 1 gate config =====
        "stage1_max_epochs": 120,
        "stage1_eval_every": 5,
        "stage1_gate_threshold": 0.20,    # 达到这个才允许进 Stage 2
        "stage1_target_threshold": 0.30,  # 理想目标
        "stage1_val_ratio": 0.2,
        "stage1_gate_patience": 3,

        # 性能门控 curriculum
        "stage1_len2_pass_threshold": 0.40,
        "stage1_len23_mix_threshold": 0.30,
        "stage1_len3_pass_threshold": 0.20,

        # Stage 2
        "stage2_epochs": 30,
    }

# =========================
# main
# =========================

def main():
    cfg = build_default_config()
    set_seed(cfg["seed"])

    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=cfg["tensorboard_log_dir"])
    else:
        writer = None

    ensure_dir(Path(cfg["rl_ckpt_stage1"]).parent)
    ensure_dir(Path(cfg["metrics_out"]).parent)

    # ---------- load graph bundle ----------
    bundle = GraphPreprocessor.build_graph_bundle(
        name=cfg["dataset_name"],
        root=cfg["dataset_root"],
        importance_path=cfg["importance_path"],
        community_mode=cfg["community_mode"],
    )

    key_nodes = KeyNodeSelector.select_topk_nodes(bundle.importance, cfg["topk_key_nodes"])

    tasks = TaskPairBuilder.build_task_pairs(
        G=bundle.nx_graph,
        key_nodes=key_nodes,
        community=bundle.community,
        importance=bundle.importance,
        min_shortest_len=cfg["min_shortest_len"],
    )

    if cfg["num_task_pairs"] < len(tasks):
        tasks = tasks[: cfg["num_task_pairs"]]

    buckets = bucket_tasks_by_shortest_len(tasks)
    train_buckets, val_buckets = split_bucket_train_val(
        buckets=buckets,
        val_ratio=cfg["stage1_val_ratio"],
        seed=cfg["seed"],
    )

    print("=" * 80)
    print("[INFO] bundle ready")
    print("dataset =", bundle.name)
    print("num_nodes =", bundle.num_nodes)
    print("num_edges =", bundle.nx_graph.number_of_edges())
    print("num_tasks =", len(tasks))
    print("bucket sizes =", {k: len(v) for k, v in buckets.items()})
    print("train bucket sizes =", {k: len(v) for k, v in train_buckets.items()})
    print("val bucket sizes =", {k: len(v) for k, v in val_buckets.items()})
    print("=" * 80)

    # ---------- node embeddings ----------
    node_embeddings = FrozenGNNEncoder.fit_or_load(
        bundle=bundle,
        ckpt_path=cfg["embedding_ckpt"],
        hidden_dim=128,
        out_dim=64,
        epochs=10,
        lr=1e-2,
    )
    # ---------- probe state/action dims ----------
    probe_env = CriticalPathEnv(
        bundle=bundle,
        node_embeddings=node_embeddings,
        max_hops=cfg["max_hops"],
        reward_mode="reachability",
        reward_kwargs={
            "distance_weight": 1.0,
            "importance_weight": 0.0,
            "edge_bc_weight": 0.0,
            "cross_comm_weight": 0.0,
            "step_cost": 0.03,
            "repeat_penalty": 0.60,
            "reach_bonus": 4.0,
            "fail_penalty": 1.0,
        },
    )

    probe_state = probe_env.reset(tasks[0])

    from path.src.rl.state_encoder import StateEncoder

    probe_state_vec = StateEncoder.encode_state(probe_state)
    probe_action_feats, _ = StateEncoder.encode_actions(
        state=probe_state,
        node_embeddings=probe_env.node_embeddings,
        importance=probe_env.importance,
        edge_bc=probe_env.edge_bc,
        graph=probe_env.G,
        community=probe_env.community,
    )

    cfg["state_dim"] = int(probe_state_vec.shape[0])
    cfg["action_dim"] = int(probe_action_feats.shape[1])

    print("[INFO] detected state_dim =", cfg["state_dim"])
    print("[INFO] detected action_dim =", cfg["action_dim"])

    # ---------- init PPO ----------
    agent = PPOAgent(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        hidden_dim=cfg["hidden_dim"],
        policy_lr=cfg["policy_lr"],
        value_lr=cfg["value_lr"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_eps=cfg["clip_eps"],
        entropy_coef=0.03,
        value_coef=cfg["value_coef"],
        ppo_epochs=cfg["ppo_epochs"],
        device=cfg["device"],
        action_temperature=1.5,
    )

    train_log = {
        "stage1": {
            "avg_reward": [],
            "arrival_rate": [],
            "avg_steps": [],
            "success_avg_steps": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        },
        "stage2": {
            "avg_reward": [],
            "arrival_rate": [],
            "avg_steps": [],
            "success_avg_steps": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "success_single_path_score_mean": [],
            "success_terminal_reward_mean": [],
            "success_marginal_gain_mean": [],
            "success_compressed_marginal_gain_mean": [],
            "success_avg_edge_bc": [],
            "success_overlap_mean": [],
        },
    }

    # ==========================================================
    # Stage 1: reachability (performance-gated)
    # ==========================================================
    env_stage1 = CriticalPathEnv(
        bundle=bundle,
        node_embeddings=node_embeddings,
        max_hops=cfg["max_hops"],
        reward_mode="reachability",
        reward_kwargs={
            "distance_weight": 1.0,
            "importance_weight": 0.0,
            "edge_bc_weight": 0.0,
            "cross_comm_weight": 0.0,
            "step_cost": 0.03,
            "repeat_penalty": 0.60,
            "reach_bonus": 4.0,
            "fail_penalty": 1.0,
            "single_frag_weight": 0.0,
            "marginal_gain_weight": 0.0,
            "overlap_penalty_weight": 0.0,
        },
    )

    best_stage1_arrival = -1.0
    best_stage1_val_arrival = -1.0
    best_stage1_gate_score = -1.0
    stage1_passed = False
    stage1_curriculum_stage = "len2_only"
    stage1_gate_patience_count = 0

    # 记录验证集指标
    train_log["stage1"]["val_arrival_len2"] = []
    train_log["stage1"]["val_arrival_len3"] = []
    train_log["stage1"]["val_arrival_len23"] = []
    train_log["stage1"]["val_arrival_all"] = []
    train_log["stage1"]["curriculum_stage"] = []

    for epoch in range(cfg["stage1_max_epochs"]):
        # ----------------------------------
        # performance-gated curriculum
        # ----------------------------------
        epoch_stage = stage1_curriculum_stage
        if epoch_stage == "len2_only":
            ratios = {"len2": 1.0}
        elif epoch_stage == "len2_len3":
            ratios = {"len2": 0.6, "len3": 0.4}
        else:
            ratios = {"len2": 0.3, "len3": 0.3, "len4p": 0.4}

        epoch_tasks = sample_from_buckets(
            buckets=train_buckets,
            ratios=ratios,
            total_size=cfg["num_samples_per_epoch"],
            seed=cfg["seed"] + epoch,
        )

        entropy_coef = linear_schedule(0.03, 0.015, epoch, cfg["stage1_max_epochs"])
        action_temperature = linear_schedule(1.5, 1.1, epoch, cfg["stage1_max_epochs"])
        agent.set_entropy_coef(entropy_coef)
        agent.set_action_temperature(action_temperature)

        stats, _ = agent.train_epoch(env_stage1, epoch_tasks)

        for k in [
            "avg_reward",
            "arrival_rate",
            "avg_steps",
            "success_avg_steps",
            "policy_loss",
            "value_loss",
            "entropy",
        ]:
            train_log["stage1"][k].append(float(stats.get(k, 0.0)))

        train_log["stage1"]["curriculum_stage"].append(epoch_stage)

        if writer is not None:
            for k, v in stats.items():
                writer.add_scalar(f"stage1/train/{k}", float(v), epoch)
            writer.add_scalar("stage1/train/entropy_coef", entropy_coef, epoch)
            writer.add_scalar("stage1/train/action_temperature", action_temperature, epoch)

        if stats["arrival_rate"] > best_stage1_arrival:
            best_stage1_arrival = stats["arrival_rate"]

        # ----------------------------------
        # validation gate
        # ----------------------------------
        val_len2 = 0.0
        val_len3 = 0.0
        val_len23 = 0.0
        val_all = 0.0

        if (epoch + 1) % cfg["stage1_eval_every"] == 0:
            val_len2 = evaluate_arrival_rate(
                agent, env_stage1, val_buckets["len2"]
            )
            val_len3 = evaluate_arrival_rate(
                agent, env_stage1, val_buckets["len3"]
            )
            val_len23 = evaluate_arrival_rate(
                agent, env_stage1, flatten_bucket_dict(val_buckets, ["len2", "len3"])
            )
            val_all = evaluate_arrival_rate(
                agent, env_stage1, flatten_bucket_dict(val_buckets, ["len2", "len3", "len4p"])
            )

            train_log["stage1"]["val_arrival_len2"].append(float(val_len2))
            train_log["stage1"]["val_arrival_len3"].append(float(val_len3))
            train_log["stage1"]["val_arrival_len23"].append(float(val_len23))
            train_log["stage1"]["val_arrival_all"].append(float(val_all))

            if writer is not None:
                writer.add_scalar("stage1/val/arrival_len2", val_len2, epoch)
                writer.add_scalar("stage1/val/arrival_len3", val_len3, epoch)
                writer.add_scalar("stage1/val/arrival_len23", val_len23, epoch)
                writer.add_scalar("stage1/val/arrival_all", val_all, epoch)

            # track best validation arrival for monitoring only
            if val_all > best_stage1_val_arrival:
                best_stage1_val_arrival = val_all

            # ----------------------------------
            # curriculum transitions
            # NOTE: transition first, so later gate/ckpt use updated stage
            # ----------------------------------
            curriculum_upgraded_this_eval = False

            if stage1_curriculum_stage == "len2_only":
                if val_len2 >= cfg["stage1_len2_pass_threshold"]:
                    stage1_curriculum_stage = "len2_len3"
                    curriculum_upgraded_this_eval = True
                    print(
                        f"[Stage1] curriculum upgrade: len2_only -> len2_len3 "
                        f"(val_len2={val_len2:.4f})"
                    )

            elif stage1_curriculum_stage == "len2_len3":
                if (
                    val_len23 >= cfg["stage1_len23_mix_threshold"]
                    and val_len3 >= cfg["stage1_len3_pass_threshold"]
                ):
                    stage1_curriculum_stage = "all_buckets"
                    curriculum_upgraded_this_eval = True
                    print(
                        f"[Stage1] curriculum upgrade: len2_len3 -> all_buckets "
                        f"(val_len23={val_len23:.4f}, val_len3={val_len3:.4f})"
                    )

            # ----------------------------------
            # save gate-aware stage1 ckpt for Stage2 initialization
            # only save after curriculum has reached all_buckets
            # and do not start counting/saving on the same eval that just upgraded
            # ----------------------------------
            gate_ready = (
                (not curriculum_upgraded_this_eval)
                and stage1_curriculum_stage == "all_buckets"
                and val_len2 >= cfg["stage1_len2_pass_threshold"]
                and val_len3 >= cfg["stage1_len3_pass_threshold"]
                and val_len23 >= cfg["stage1_len23_mix_threshold"]
            )

            if gate_ready:
                gate_score = (
                    0.25 * val_len2
                    + 0.25 * val_len3
                    + 0.25 * val_len23
                    + 0.25 * val_all
                )
                if gate_score > best_stage1_gate_score:
                    best_stage1_gate_score = gate_score
                    agent.save(cfg["rl_ckpt_stage1"])

            # ----------------------------------
            # stage1 gate: require curriculum completed + stable multi-bucket pass
            # ----------------------------------
            gate_ok = (
                gate_ready
                and val_all >= cfg["stage1_gate_threshold"]
            )

            if gate_ok:
                stage1_gate_patience_count += 1
            else:
                stage1_gate_patience_count = 0

            if stage1_gate_patience_count >= cfg["stage1_gate_patience"]:
                stage1_passed = True
                print(
                    f"[Stage1] gate passed at epoch {epoch + 1}: "
                    f"stage={stage1_curriculum_stage}, "
                    f"patience={stage1_gate_patience_count}, "
                    f"val_len2={val_len2:.4f}, "
                    f"val_len3={val_len3:.4f}, "
                    f"val_len23={val_len23:.4f}, "
                    f"val_all={val_all:.4f}"
                )
                break

        print(
            f"[Stage1][Epoch {epoch+1:03d}/{cfg['stage1_max_epochs']}] "
            f"stage={epoch_stage} "
            f"reward={stats['avg_reward']:.4f} "
            f"train_arr={stats['arrival_rate']:.4f} "
            f"avg_steps={stats['avg_steps']:.4f} "
            f"succ_steps={stats['success_avg_steps']:.4f} "
            f"entropy={stats['entropy']:.4f} "
            f"val_len2={val_len2:.4f} "
            f"val_len3={val_len3:.4f} "
            f"val_len23={val_len23:.4f} "
            f"val_all={val_all:.4f}"
        )
    print(f"[Stage1] best train arrival_rate = {best_stage1_arrival:.4f}")
    print(f"[Stage1] best val arrival_rate   = {best_stage1_val_arrival:.4f}")

    # reload best stage1 checkpoint if exists
    if Path(cfg["rl_ckpt_stage1"]).exists():
        agent.load(cfg["rl_ckpt_stage1"])
    # ==========================================================
    # Gate before Stage 2
    # ==========================================================
    if not stage1_passed:
        print("=" * 80)
        print("[STOP] Stage 1 did not pass gate. Skip Stage 2.")
        print(f"best_stage1_train_arrival = {best_stage1_arrival:.4f}")
        print(f"best_stage1_val_arrival   = {best_stage1_val_arrival:.4f}")
        print("=" * 80)

        out = {
            "dataset": {
                "name": bundle.name,
                "num_nodes": bundle.num_nodes,
                "num_edges": bundle.nx_graph.number_of_edges(),
            },
            "tasks": {
                "count": len(tasks),
                "bucket_sizes": {k: len(v) for k, v in buckets.items()},
                "train_bucket_sizes": {k: len(v) for k, v in train_buckets.items()},
                "val_bucket_sizes": {k: len(v) for k, v in val_buckets.items()},
            },
            "stage1_passed": False,
            "stage1_best_arrival_rate": best_stage1_arrival,
            "stage1_best_val_arrival_rate": best_stage1_val_arrival,
            "train_log": train_log,
            "checkpoints": {
                "stage1": cfg["rl_ckpt_stage1"],
                "stage2": None,
            },
        }

        save_json(cfg["metrics_out"], out)

        if writer is not None:
            writer.close()
        return
    # ==========================================================
    # Stage 2: fragility_topk_align
    # ==========================================================
    env_stage2 = CriticalPathEnv(
        bundle=bundle,
        node_embeddings=node_embeddings,
        max_hops=cfg["max_hops"],
        reward_mode="fragility_topk_align",
        reward_kwargs={
            "distance_weight": 0.15,
            "importance_weight": 0.05,
            "edge_bc_weight": 0.12,
            "cross_comm_weight": 0.08,
            "step_cost": 0.02,
            "repeat_penalty": 0.40,
            "reach_bonus": 0.50,
            "fail_penalty": 1.0,
            "single_frag_weight": 0.50,
            "marginal_gain_weight": 3.00,
            "overlap_penalty_weight": 1.00,

            "lambda_E": 0.55,
            "lambda_ASP": 0.45,
            "lambda_LCC": 0.0,

            "top_k": 10,
            "overlap_threshold": 0.6,

            "efficiency_num_pairs": 2000,
            "asp_num_sources": 64,
            "random_seed": cfg["seed"],
            "use_bridge_shortcut": False,
        },
    )

    best_stage2_score = -1.0

    # 可选：单独记录 union / marginal 相关指标
    if "success_marginal_gain_mean" not in train_log["stage2"]:
        train_log["stage2"]["success_marginal_gain_mean"] = []
    if "success_overlap_mean" not in train_log["stage2"]:
        train_log["stage2"]["success_overlap_mean"] = []

    for epoch in range(cfg["stage2_epochs"]):
        env_stage2.reset_topk_context()
        # Stage 2 继续偏向较长/较复杂路径
        if epoch < 10:
            ratios = {"len3": 0.4, "len4p": 0.6}
        else:
            ratios = {"len2": 0.2, "len3": 0.3, "len4p": 0.5}

        epoch_tasks = sample_from_buckets(
            buckets=buckets,
            ratios=ratios,
            total_size=cfg["num_samples_per_epoch"],
            seed=cfg["seed"] + 1000 + epoch,
        )

        # Stage 2 仍然保留一定探索，但后期收敛
        entropy_coef = linear_schedule(0.02, 0.005, epoch, cfg["stage2_epochs"])
        action_temperature = linear_schedule(1.15, 0.95, epoch, cfg["stage2_epochs"])
        agent.set_entropy_coef(entropy_coef)
        agent.set_action_temperature(action_temperature)

        stats, episode_results = agent.train_epoch(env_stage2, epoch_tasks)

        # 从 episode_results 里额外统计 marginal_gain / overlap

        for k in train_log["stage2"].keys():
            train_log["stage2"][k].append(float(stats.get(k, 0.0)))

        if writer is not None:
            for k, v in stats.items():
                writer.add_scalar(f"stage2/{k}", float(v), epoch)
            writer.add_scalar("stage2/entropy_coef", entropy_coef, epoch)
            writer.add_scalar("stage2/action_temperature", action_temperature, epoch)

        # Stage 2 的 early-stopping / best-ckpt 指标：
        # 更强调 set-level marginal gain 与 fragility，而不是单纯 arrival
        stage2_score = (
                20.0 * float(stats.get("success_compressed_marginal_gain_mean", 0.0))
                + 8.0 * float(stats.get("success_single_path_score_mean", 0.0))
                + 0.5 * float(stats.get("arrival_rate", 0.0))
                - 1.0 * float(stats.get("success_overlap_mean", 0.0))
        )

        if stage2_score > best_stage2_score:
            best_stage2_score = stage2_score
            agent.save(cfg["rl_ckpt_stage2"])

        print(
            f"[Stage2][Epoch {epoch + 1:03d}/{cfg['stage2_epochs']}] "
            f"reward={stats['avg_reward']:.4f} "
            f"arrival={stats['arrival_rate']:.4f} "
            f"avg_steps={stats['avg_steps']:.4f} "
            f"single={stats.get('success_single_path_score_mean', 0.0):.6f} "
            f"terminal={stats.get('success_terminal_reward_mean', 0.0):.6f} "
            f"marginal={stats.get('success_marginal_gain_mean', 0.0):.6f} "
            f"cmg={stats.get('success_compressed_marginal_gain_mean', 0.0):.6f} "
            f"overlap={stats.get('success_overlap_mean', 0.0):.6f} "
            f"succ_bc={stats.get('success_avg_edge_bc', 0.0):.6f} "
            f"entropy={stats['entropy']:.4f}"
        )

    print(f"[Stage2] best score = {best_stage2_score:.6f}")
    agent.load(cfg["rl_ckpt_stage2"])
    # ==========================================================
    # post-eval: RL vs Rule-based
    # ==========================================================
    rl_paths = RLPathInferencer.sample_paths(
        agent=agent,
        env=env_stage2,
        tasks=tasks,
        num_samples_per_task=10,
    )

    # 注意：这里要求你的 RLPathInferencer 已经有这个方法
    rl_selected = RLPathInferencer.rescore_and_select(
        bundle=bundle,
        path_records=rl_paths,
        top_q=10,
        overlap_threshold=0.6,
    )

    rule_weights = {
        "avg_node_importance": 0.20,
        "avg_edge_bc": 0.15,
        "cross_comm_ratio": 0.10,
        "fragility_score": 0.45,
        "path_length": 0.10,
    }
    rule_paths = RuleBasedCriticalPath.run(
        bundle=bundle,
        tasks=tasks,
        path_k=3,
        max_hops=cfg["max_hops"],
        delta=2,
        weights=rule_weights,
        top_q=10,
    )

    evaluator = MethodEvaluator(lambda_E=0.55, lambda_LCC=0.0, lambda_ASP=0.45)
    comparison = evaluator.compare_methods(
        result_dict={
            "rule_based": rule_paths,
            "rl_stage2": rl_selected,
        },
        G=bundle.nx_graph,
        k_list=[1, 3, 5, 10],
    )

    out = {
        "dataset": {
            "name": bundle.name,
            "num_nodes": bundle.num_nodes,
            "num_edges": bundle.nx_graph.number_of_edges(),
        },
        "tasks": {
            "count": len(tasks),
            "bucket_sizes": {k: len(v) for k, v in buckets.items()},
        },
        "stage1_best_arrival_rate": best_stage1_arrival,
        "stage2_best_score": best_stage2_score,
        "train_log": train_log,
        "comparison": comparison,
        "checkpoints": {
            "stage1": cfg["rl_ckpt_stage1"],
            "stage2": cfg["rl_ckpt_stage2"],
        },
    }

    save_json(cfg["metrics_out"], out)

    print("=" * 80)
    print("[DONE] Stage-wise RL training finished.")
    print("Stage1 ckpt =", cfg["rl_ckpt_stage1"])
    print("Stage2 ckpt =", cfg["rl_ckpt_stage2"])
    print("Metrics out =", cfg["metrics_out"])
    print("=" * 80)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()