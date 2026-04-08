from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPolicyNet(nn.Module):
    """
    Action scoring network.
    Input: action_feats [num_actions, Da]
    Output: logits [num_actions]
    """

    def __init__(self, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, action_feats: torch.Tensor) -> torch.Tensor:
        if action_feats.ndim != 2:
            raise ValueError(f"action_feats must be 2D, got shape={tuple(action_feats.shape)}")
        logits = self.net(action_feats).squeeze(-1)
        return logits
