from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import torch


class StateEncoder:
    """
    Convert env state dict to fixed-size tensors.

    state_vec = [curr_emb, target_emb, scalar features]
    action_feat = [curr_emb, action_emb, target_emb, scalar features]

    Stronger target-relative action features:
        - d(curr, target)
        - d(action, target)
        - d(curr, target) - d(action, target)
        - 1[action == target]
        - 1[d(action, target) < d(curr, target)]
    """

    @staticmethod
    def _safe_shortest_dist(graph: nx.Graph, u: int, v: int, fallback: float = 1e6) -> float:
        try:
            return float(nx.shortest_path_length(graph, u, v))
        except Exception:
            return float(fallback)

    @staticmethod
    def encode_state(state: Dict[str, object]) -> torch.Tensor:
        curr_emb = state["curr_emb"]
        target_emb = state["target_emb"]
        if not torch.is_tensor(curr_emb):
            curr_emb = torch.tensor(curr_emb)
        if not torch.is_tensor(target_emb):
            target_emb = torch.tensor(target_emb)

        scalars = torch.tensor(
            [
                float(state["path_length"]),
                float(state["avg_node_importance"]),
                float(state["avg_edge_bc"]),
                float(state["cross_comm_ratio"]),
                float(state["dist_to_target"]),
            ],
            dtype=torch.float32,
        )
        return torch.cat(
            [curr_emb.float().flatten(), target_emb.float().flatten(), scalars],
            dim=0,
        )

    @staticmethod
    def encode_actions(
        state: Dict[str, object],
        node_embeddings: torch.Tensor,
        importance,
        edge_bc,
        graph: nx.Graph,
        community=None,
    ) -> Tuple[torch.Tensor, List[int]]:
        curr_node = int(state["curr_node"])
        target_node = int(state["target_node"])
        valid_actions = [int(a) for a in state["valid_actions"]]
        selected_internal_nodes = set(int(x) for x in state.get("selected_internal_nodes", []))
        path_nodes = set(int(x) for x in state.get("path_nodes", []))

        curr_emb = node_embeddings[curr_node].float()
        target_emb = node_embeddings[target_node].float()

        d_curr_to_target = float(state["dist_to_target"])

        action_rows = []
        for a in valid_actions:
            a_emb = node_embeddings[a].float()

            edge_key = tuple(sorted((curr_node, a)))
            edge_bc_val = float(edge_bc.get(edge_key, edge_bc.get((curr_node, a), 0.0)))
            cross_comm_val = float(0.0 if community is None else int(community[curr_node] != community[a]))

            # ===== target-relative action features =====
            d_a_to_target = StateEncoder._safe_shortest_dist(
                graph=graph,
                u=a,
                v=target_node,
                fallback=float(d_curr_to_target + 1.0),
            )
            delta_dist = float(d_curr_to_target - d_a_to_target)
            is_target = float(a == target_node)
            is_progress = float(d_a_to_target < d_curr_to_target)
            is_selected_internal = float(a in selected_internal_nodes)
            is_in_current_path = float(a in path_nodes)

            row = torch.cat(
                [
                    curr_emb,
                    a_emb,
                    target_emb,
                    torch.tensor(
                        [
                            # original local structural features
                            float(importance[a]),
                            edge_bc_val,
                            cross_comm_val,
                            float(state["path_length"]),

                            # original current-target distance
                            d_curr_to_target,

                            # new target-relative action features
                            d_a_to_target,
                            delta_dist,
                            is_target,
                            is_progress,
                            # 新增：让策略知道这个动作是否会复用已有集合内部节点
                            is_selected_internal,
                            is_in_current_path,
                        ],
                        dtype=torch.float32,
                    ),
                ],
                dim=0,
            )
            action_rows.append(row)

        if not action_rows:
            # 3 * emb_dim + 11 scalar features
            feat_dim = int(curr_emb.numel() * 3 + 11)
            return torch.zeros(0, feat_dim, dtype=torch.float32), valid_actions

        return torch.stack(action_rows, dim=0), valid_actions