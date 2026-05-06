from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import nn
from torch.optim import Adam

from path.src.core.types import EpisodeResult, PathRecord, TaskPair
from path.src.rl.env import CriticalPathEnv
from path.src.rl.state_encoder import StateEncoder
from path.src.rl.policy_net import MaskedPolicyNet
from path.src.rl.value_net import ValueNet


@dataclass
class Transition:
    state_vec: torch.Tensor
    action_feats: torch.Tensor
    action_idx: int
    logprob: float
    value: float
    reward: float
    done: bool


class PPOAgent:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 128,
            policy_lr: float = 1e-4,
            value_lr: float = 1e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_eps: float = 0.2,
            entropy_coef: float = 0.02,
            value_coef: float = 0.5,
            ppo_epochs: int = 4,
            device: str = "cpu",
            action_temperature: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.device = torch.device(device)
        self.action_temperature = float(action_temperature)

        self.policy = MaskedPolicyNet(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.value_net = ValueNet(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=value_lr)

    def set_entropy_coef(self, value: float) -> None:
        self.entropy_coef = float(value)

    def set_action_temperature(self, value: float) -> None:
        self.action_temperature = float(value)

    def save(self, ckpt_path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value_net": self.value_net.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_eps": self.clip_eps,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "ppo_epochs": self.ppo_epochs,
                "action_temperature": self.action_temperature,
            },
            ckpt_path,
        )

    def load(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_net.load_state_dict(ckpt["value_net"])
        self.entropy_coef = float(ckpt.get("entropy_coef", self.entropy_coef))
        self.action_temperature = float(
            ckpt.get("action_temperature", self.action_temperature)
        )

    def _build_action_distribution(
        self,
        action_feats: torch.Tensor,
    ) -> Categorical:
        logits = self.policy(action_feats)
        logits = logits / self.action_temperature
        return Categorical(logits=logits)

    def select_action(
            self,
            state: Dict[str, Any],
            env: CriticalPathEnv,
            deterministic: bool = False,
    ) -> Tuple[int, int, float, float, torch.Tensor, torch.Tensor]:
        state_vec = StateEncoder.encode_state(state).to(self.device)
        action_feats, action_ids = StateEncoder.encode_actions(
            state=state,
            node_embeddings=env.node_embeddings,
            importance=env.importance,
            edge_bc=env.edge_bc,
            graph=env.G,
            community=env.community,
        )
        action_feats = action_feats.to(self.device)

        dist = self._build_action_distribution(action_feats)

        if deterministic:
            action_idx = torch.argmax(dist.logits)
        else:
            action_idx = dist.sample()

        logprob = dist.log_prob(action_idx)
        value = self.value_net(state_vec)

        chosen_idx = int(action_idx.item())
        chosen_action_node_id = int(action_ids[chosen_idx])

        return (
            chosen_action_node_id,
            chosen_idx,
            float(logprob.item()),
            float(value.item()),
            state_vec.detach(),
            action_feats.detach(),
        )

    def rollout_episode(
            self,
            env: CriticalPathEnv,
            task: TaskPair,
            deterministic: bool = False,
    ) -> Tuple[EpisodeResult, List[Transition]]:
        state = env.reset(task)
        done = False

        transitions: List[Transition] = []
        total_reward = 0.0

        last_info: Dict[str, Any] = {}
        while not done:
            if len(state.get("valid_actions", [])) == 0:
                env.done = True
                done = True
                last_info = {
                    "reached_target": 0,
                    "fragility": {
                        "terminal_reward": -1.0,
                        "fail_reason": "no_valid_actions",
                    },
                }
                break

            (
                chosen_action_node_id,
                chosen_action_index,
                logprob,
                value,
                state_vec,
                action_feats,
            ) = self.select_action(state, env, deterministic=deterministic)

            next_state, reward, done, info = env.step(chosen_action_node_id)

            transitions.append(
                Transition(
                    state_vec=state_vec,
                    action_feats=action_feats,
                    action_idx=chosen_action_index,
                    logprob=logprob,
                    value=value,
                    reward=float(reward),
                    done=bool(done),
                )
            )

            total_reward += float(reward)
            state = next_state
            last_info = info

        path_record = PathRecord(
            nodes=list(env.path),
            edges=list(zip(env.path[:-1], env.path[1:])),
            source=int(task.source),
            target=int(task.target),
            success=bool(last_info.get("reached_target", 0) == 1),
            method="rl",
            score=None,
            features=None,
            fragility=last_info.get("fragility", None),
        )

        ep_result = EpisodeResult(
            task=task,
            path=path_record,
            total_reward=float(total_reward),
            steps=int(len(env.path) - 1),
            reached_target=bool(last_info.get("reached_target", 0) == 1),
        )
        return ep_result, transitions

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)

        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def collect_rollouts(
        self,
        env: CriticalPathEnv,
        tasks: List[TaskPair],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], List[EpisodeResult]]:
        all_state_vecs = []
        all_action_feats = []
        all_action_indices = []
        all_old_logprobs = []
        all_rewards = []
        all_values = []
        all_dones = []

        episode_results: List[EpisodeResult] = []

        for task in tasks:
            ep_result, transitions = self.rollout_episode(env, task)
            episode_results.append(ep_result)

            rewards = [tr.reward for tr in transitions]
            values = [tr.value for tr in transitions]
            dones = [tr.done for tr in transitions]
            advantages, returns = self._compute_gae(rewards, values, dones)

            for i, tr in enumerate(transitions):
                all_state_vecs.append(tr.state_vec)
                all_action_feats.append(tr.action_feats)
                all_action_indices.append(tr.action_idx)
                all_old_logprobs.append(tr.logprob)
                all_rewards.append(returns[i])
                all_values.append(advantages[i])
                all_dones.append(tr.done)

        batch = {
            "state_vecs": all_state_vecs,
            "action_feats": all_action_feats,
            "action_indices": torch.tensor(all_action_indices, dtype=torch.long, device=self.device),
            "old_logprobs": torch.tensor(all_old_logprobs, dtype=torch.float32, device=self.device),
            "returns": torch.tensor(all_rewards, dtype=torch.float32, device=self.device),
            "advantages": torch.tensor(all_values, dtype=torch.float32, device=self.device),
        }

        stats = self._summarize_episode_results(episode_results)
        return batch, stats, episode_results

    def _evaluate_actions(
        self,
        state_vecs: List[torch.Tensor],
        action_feats_list: List[torch.Tensor],
        action_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_logprobs = []
        entropies = []
        values = []

        for i in range(len(state_vecs)):
            state_vec = state_vecs[i].to(self.device)
            action_feats = action_feats_list[i].to(self.device)
            action_idx = action_indices[i]

            dist = self._build_action_distribution(action_feats)
            new_logprobs.append(dist.log_prob(action_idx))
            entropies.append(dist.entropy())
            values.append(self.value_net(state_vec).squeeze(-1))

        new_logprobs = torch.stack(new_logprobs)
        entropies = torch.stack(entropies)
        values = torch.stack(values)

        return new_logprobs, entropies, values

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        old_logprobs = batch["old_logprobs"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        action_indices = batch["action_indices"]
        state_vecs = batch["state_vecs"]
        action_feats_list = batch["action_feats"]

        policy_losses = []
        value_losses = []
        entropy_vals = []

        for _ in range(self.ppo_epochs):
            new_logprobs, entropies, values = self._evaluate_actions(
                state_vecs=state_vecs,
                action_feats_list=action_feats_list,
                action_indices=action_indices,
            )

            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = entropies.mean()

            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.policy_optimizer.step()
            self.value_optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropy_vals.append(float(entropy.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_vals)),
        }

    def _summarize_episode_results(
            self,
            episode_results: List[EpisodeResult],
    ) -> Dict[str, float]:
        rewards = [ep.total_reward for ep in episode_results]
        steps = [ep.steps for ep in episode_results]
        success_flags = [1.0 if ep.reached_target else 0.0 for ep in episode_results]
        success_steps = [ep.steps for ep in episode_results if ep.reached_target]

        success_single_path_scores = []
        success_terminal_rewards = []
        success_marginal_gains = []
        success_compressed_marginal_gains = []
        success_avg_bc = []
        success_overlaps = []
        success_node_overlaps = []
        success_new_internal_nodes = []
        success_negative_marginal_flags = []
        success_selection_scores = []
        for ep in episode_results:
            if ep.reached_target and ep.path.fragility is not None:
                frag = ep.path.fragility

                # Stage B / Stage C unified fields
                success_single_path_scores.append(
                    float(
                        frag.get(
                            "single_path_score",
                            frag.get("raw_fragility_score", 0.0),  # backward compatibility
                        )
                    )
                )
                success_terminal_rewards.append(
                    float(
                        frag.get(
                            "terminal_reward",
                            frag.get("fragility_score", 0.0),  # backward compatibility
                        )
                    )
                )
                success_marginal_gains.append(
                    float(frag.get("marginal_gain", 0.0))
                )
                success_compressed_marginal_gains.append(
                    float(frag.get("compressed_marginal_gain", 0.0))
                )
                success_avg_bc.append(
                    float(frag.get("avg_edge_bc", 0.0))
                )
                success_overlaps.append(
                    float(frag.get("max_overlap", 0.0))
                )
                success_node_overlaps.append(
                    float(frag.get("max_node_overlap", 0.0))
                )
                success_new_internal_nodes.append(
                    float(frag.get("new_internal_nodes", 0.0))
                )
                success_negative_marginal_flags.append(
                    1.0 if float(frag.get("marginal_gain", 0.0)) < 0.0 else 0.0
                )
                success_selection_scores.append(
                    float(frag.get("selection_score", 0.0))
                )

        return {
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "arrival_rate": float(np.mean(success_flags)) if success_flags else 0.0,
            "avg_steps": float(np.mean(steps)) if steps else 0.0,
            "success_avg_steps": float(np.mean(success_steps)) if success_steps else 0.0,

            # unified stats
            "success_single_path_score_mean": float(np.mean(success_single_path_scores))
            if success_single_path_scores else 0.0,

            "success_terminal_reward_mean": float(np.mean(success_terminal_rewards))
            if success_terminal_rewards else 0.0,

            "success_marginal_gain_mean": float(np.mean(success_marginal_gains))
            if success_marginal_gains else 0.0,

            "success_compressed_marginal_gain_mean": float(np.mean(success_compressed_marginal_gains))
            if success_compressed_marginal_gains else 0.0,

            "success_avg_edge_bc": float(np.mean(success_avg_bc))
            if success_avg_bc else 0.0,

            "success_overlap_mean": float(np.mean(success_overlaps))
            if success_overlaps else 0.0,
            "success_node_overlap_mean": float(np.mean(success_node_overlaps))
            if success_node_overlaps else 0.0,

            "success_new_internal_nodes_mean": float(np.mean(success_new_internal_nodes))
            if success_new_internal_nodes else 0.0,

            "success_negative_marginal_rate": float(np.mean(success_negative_marginal_flags))
            if success_negative_marginal_flags else 0.0,

            "success_selection_score_mean": float(np.mean(success_selection_scores))
            if success_selection_scores else 0.0,
        }

    def train_epoch(
            self,
            env: CriticalPathEnv,
            tasks: List[TaskPair],
    ) -> Tuple[Dict[str, float], List[EpisodeResult]]:

        batch, rollout_stats, episode_results = self.collect_rollouts(env, tasks)

        if len(batch["state_vecs"]) == 0:
            stats = {}
            stats.update(rollout_stats)
            stats.update({
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            })
            return stats, episode_results

        update_stats = self.update(batch)

        stats = {}
        stats.update(rollout_stats)
        stats.update(update_stats)
        return stats, episode_results