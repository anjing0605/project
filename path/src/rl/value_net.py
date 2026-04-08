from __future__ import annotations

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """
    State value estimator.
    Input: state_vec [Ds] or [B, Ds]
    Output: scalar value or [B]
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if state_vec.ndim == 1:
            state_vec = state_vec.unsqueeze(0)
            squeeze = True
        value = self.net(state_vec).squeeze(-1)
        return value.squeeze(0) if squeeze else value
