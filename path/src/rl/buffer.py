from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class RolloutBuffer:
    states: List[torch.Tensor] = field(default_factory=list)
    action_feats: List[torch.Tensor] = field(default_factory=list)
    action_indices: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def clear(self) -> None:
        self.states.clear()
        self.action_feats.clear()
        self.action_indices.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()
        self.values.clear()

    def add(
        self,
        state_vec: torch.Tensor,
        action_feats: torch.Tensor,
        action_idx: int,
        reward: float,
        done: bool,
        logprob: float,
        value: float,
    ) -> None:
        self.states.append(state_vec.detach().cpu().float())
        self.action_feats.append(action_feats.detach().cpu().float())
        self.action_indices.append(int(action_idx))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.logprobs.append(float(logprob))
        self.values.append(float(value))

    def compute_gae(self, gamma: float, lam: float) -> Dict[str, object]:
        n = len(self.rewards)
        if n == 0:
            raise ValueError("RolloutBuffer is empty.")

        values = self.values + [0.0]
        advantages = [0.0] * n
        gae = 0.0
        for t in reversed(range(n)):
            nonterminal = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae

        returns = [advantages[t] + self.values[t] for t in range(n)]

        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        return {
            "states": self.states,
            "action_feats": self.action_feats,
            "action_indices": torch.tensor(self.action_indices, dtype=torch.long),
            "old_logprobs": torch.tensor(self.logprobs, dtype=torch.float32),
            "returns": torch.tensor(returns, dtype=torch.float32),
            "advantages": adv_tensor,
        }
