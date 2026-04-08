from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except Exception:  # pragma: no cover
    GCNConv = None

from path.src.core.types import GraphDataBundle


class _GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_classes: int) -> None:
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric is required for GCN encoder.")
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        z = self.conv2(h, edge_index)
        logits = self.classifier(z)
        return z, logits


class FrozenGNNEncoder:
    """
    Produce fixed node embeddings for RL.

    Priority:
      1. load cached embeddings if available
      2. if PyG GCN is available, train a small supervised GCN and export hidden embeddings
      3. otherwise, return standardized input features (with optional zero-padding / truncation)
    """

    @staticmethod
    def _feature_fallback(bundle: GraphDataBundle, out_dim: int) -> torch.Tensor:
        x = bundle.x
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.detach().float().cpu()
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        if x.shape[1] == out_dim:
            return x
        if x.shape[1] > out_dim:
            return x[:, :out_dim]
        pad = torch.zeros(x.shape[0], out_dim - x.shape[1], dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    @staticmethod
    def fit_or_load(
        bundle: GraphDataBundle,
        ckpt_path: str,
        hidden_dim: int = 128,
        out_dim: int = 64,
        epochs: int = 150,
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Returns:
            node_embeddings: torch.Tensor [N, out_dim]
        """
        ckpt = Path(ckpt_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)

        if ckpt.exists():
            obj = torch.load(ckpt, map_location="cpu")
            if isinstance(obj, dict) and "embeddings" in obj:
                return obj["embeddings"].detach().cpu().float()
            if torch.is_tensor(obj):
                return obj.detach().cpu().float()

        if GCNConv is None:
            emb = FrozenGNNEncoder._feature_fallback(bundle, out_dim=out_dim)
            torch.save({"embeddings": emb}, ckpt)
            return emb

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        x = bundle.x
        y = bundle.y
        edge_index = bundle.edge_index

        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        if not torch.is_tensor(edge_index):
            edge_index = torch.tensor(edge_index)

        x = x.float().to(device)
        y = y.long().to(device)
        edge_index = edge_index.long().to(device)

        num_classes = int(torch.unique(y).numel())
        model = _GCNEncoder(
            in_dim=int(x.shape[1]),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            num_classes=max(num_classes, 2),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Try to use train_mask if available; otherwise use all nodes.
        train_mask = None
        if hasattr(bundle, "metadata") and isinstance(bundle.metadata, dict):
            tm = bundle.metadata.get("train_mask")
            if tm is not None:
                train_mask = tm
        if hasattr(bundle, "x") and hasattr(bundle, "y"):
            # no-op, just keep path consistent
            pass

        if train_mask is None and hasattr(bundle, "train_mask"):
            train_mask = bundle.train_mask

        if train_mask is not None:
            if not torch.is_tensor(train_mask):
                train_mask = torch.tensor(train_mask)
            train_mask = train_mask.to(device).bool()
        else:
            train_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)

        best_state = None
        best_loss = float("inf")

        for _ in range(int(epochs)):
            model.train()
            z, logits = model(x, edge_index)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if float(loss.item()) < best_loss:
                best_loss = float(loss.item())
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            z, _ = model(x, edge_index)
        z = z.detach().cpu().float()
        torch.save({"embeddings": z, "best_loss": best_loss}, ckpt)
        return z
