from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx
import pandas as pd

from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.path_generator import PathGenerator
from path.src.core.types import GraphDataBundle, TaskPair


class PathRankingDatasetBuilder:

    # =============================
    # helper functions
    # =============================

    @staticmethod
    def path_to_edges(nodes):
        return [
            tuple(sorted((int(nodes[i]), int(nodes[i + 1]))))
            for i in range(len(nodes) - 1)
        ]

    @staticmethod
    def path_to_internal_nodes(nodes):
        """
        Rank / set-gain / evaluator 统一使用“删除路径内部节点”：
            P = [s, ..., t]
            remove P[1:-1]
        """
        if nodes is None or len(nodes) <= 2:
            return set()
        return {int(n) for n in nodes[1:-1]}

    @staticmethod
    def edge_jaccard(a, b):
        ea = set(PathRankingDatasetBuilder.path_to_edges(a))
        eb = set(PathRankingDatasetBuilder.path_to_edges(b))
        return len(ea & eb) / max(len(ea | eb), 1)

    @staticmethod
    def internal_node_jaccard(a, b):
        va = PathRankingDatasetBuilder.path_to_internal_nodes(a)
        vb = PathRankingDatasetBuilder.path_to_internal_nodes(b)
        return len(va & vb) / max(len(va | vb), 1)

    @staticmethod
    def shared_internal_nodes(a, b):
        return len(
            PathRankingDatasetBuilder.path_to_internal_nodes(a)
            & PathRankingDatasetBuilder.path_to_internal_nodes(b)
        )

    # =============================
    # main builder
    # =============================

    @classmethod
    def build_path_samples(
            cls,
            bundle: GraphDataBundle,
            tasks: List[TaskPair],
            path_k: int,
            max_hops: int,
            delta: int,
            raw_k_multiplier: int = 5,
            raw_k_min_extra: int = 20,
            final_k: Optional[int] = None,
            max_internal_overlap: float = 0.80,
            fallback_relax_overlap: float = 0.95,
            fallback_extra_hops: int = 2,
            fragility_weights: Optional[Dict[str, float]] = None,
            fragility_mode: str = "hybrid",
            cache_path: Optional[str] = None,
            exact_every_n_tasks: int = 20,
            exact_top_ranks: int = 1,
            exact_max_path_len: int = 4,
            progress_every: int = 10,
            label_mode: str = "marginal",
            debug: bool = False,
    ) -> pd.DataFrame:

        fragility_weights = fragility_weights or {
            "lambda_E": 0.4,
            "lambda_LCC": 0.4,
            "lambda_ASP": 0.2,
            "lambda_red": 0.2,
        }

        frag_eval_weights = {
            "lambda_E": float(fragility_weights.get("lambda_E", 0.4)),
            "lambda_LCC": float(fragility_weights.get("lambda_LCC", 0.4)),
            "lambda_ASP": float(fragility_weights.get("lambda_ASP", 0.2)),
        }

        fragility_evaluator = FragilityEvaluator(**frag_eval_weights)
        base_metrics = fragility_evaluator.compute_base_metrics(bundle.nx_graph)

        rows: List[Dict[str, object]] = []

        effective_final_k = int(final_k) if final_k is not None else int(path_k)
        effective_raw_k = max(
            int(effective_final_k * raw_k_multiplier),
            int(effective_final_k + raw_k_min_extra),
        )

        if debug:
            print("[DEBUG] candidate generation params:")
            print(f"    path_k={path_k}")
            print(f"    effective_final_k={effective_final_k}")
            print(f"    effective_raw_k={effective_raw_k}")
            print(f"    max_hops={max_hops}")
            print(f"    delta={delta}")
            print(f"    max_internal_overlap={max_internal_overlap}")
            print(f"    fallback_relax_overlap={fallback_relax_overlap}")
            print(f"    fallback_extra_hops={fallback_extra_hops}")

        # =============================
        # generate path samples
        # =============================

        for task_idx, task in enumerate(tasks):

            paths = PathGenerator.diversified_k_shortest_simple_paths(
                G=bundle.nx_graph,
                source=task.source,
                target=task.target,
                raw_k=effective_raw_k,
                final_k=effective_final_k,
                max_hops=max_hops,
                delta=delta,
                max_internal_overlap=max_internal_overlap,
            )

            # fallback: 如果候选太少，则放宽约束再试一次
            min_expected = max(3, effective_final_k // 2)
            if len(paths) < min_expected:
                if debug:
                    print(
                        f"[DEBUG][fallback] task=({task.source},{task.target}) "
                        f"initial_paths={len(paths)} < min_expected={min_expected}, retry with relaxed constraints"
                    )

                relaxed_paths = PathGenerator.diversified_k_shortest_simple_paths(
                    G=bundle.nx_graph,
                    source=task.source,
                    target=task.target,
                    raw_k=max(effective_raw_k * 2, effective_final_k + 30),
                    final_k=effective_final_k,
                    max_hops=max_hops + int(fallback_extra_hops),
                    delta=delta + 1,
                    max_internal_overlap=float(fallback_relax_overlap),
                )

                if len(relaxed_paths) > len(paths):
                    paths = relaxed_paths

            if debug and ((task_idx + 1) % max(progress_every, 1) == 0):
                print(
                    f"[DEBUG] processed task {task_idx + 1}/{len(tasks)} "
                    f"| task=({task.source},{task.target}) | num_paths={len(paths)}"
                )

            for rank_idx, path in enumerate(paths, start=1):
                shortest_len = nx.shortest_path_length(
                    bundle.nx_graph,
                    source=task.source,
                    target=task.target,
                )

                feats = PathFeatureExtractor.extract_features(
                    path=path,
                    importance=bundle.importance,
                    community=bundle.community,
                    edge_bc=bundle.edge_bc,
                    shortest_len=shortest_len,
                    source=task.source,
                    target=task.target,
                )

                frag = fragility_evaluator.compute_fragility(
                    G=bundle.nx_graph,
                    path=path,
                    base_metrics=base_metrics,
                    num_nodes=bundle.num_nodes,
                )

                row = {
                    "source": int(task.source),
                    "target": int(task.target),
                    "path_nodes": json.dumps([int(n) for n in path]),
                    "candidate_rank": int(rank_idx),
                }

                row.update(feats)
                row.update(frag)

                row["y_fragility"] = float(frag["fragility_score"])

                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # =========================================================
        # TRUE marginal gain labels, using INTERNAL-NODE removal.
        # This must be consistent with FragilityEvaluator.compute_fragility()
        # and MethodEvaluator.evaluate_topk_damage().
        # =========================================================

        print("[INFO] computing TRUE marginal gain labels by removing internal nodes...")

        lambda_red = float(fragility_weights.get("lambda_red", 0.2))

        def compute_F(node_set):
            """
            F(S): structural damage after removing a SET of internal nodes.
            """
            H = bundle.nx_graph.copy()
            H.remove_nodes_from([int(n) for n in node_set if H.has_node(int(n))])

            E0 = base_metrics["global_efficiency"]
            L0 = base_metrics["lcc_ratio"]
            A0 = base_metrics["avg_shortest_path_lcc"]

            E1 = fragility_evaluator.global_efficiency_approx(H)
            L1 = fragility_evaluator.lcc_ratio(H, bundle.num_nodes)
            A1 = fragility_evaluator.avg_shortest_path_of_lcc_approx(H)

            return (
                    fragility_evaluator.lambda_E * max(E0 - E1, 0.0)
                    + fragility_evaluator.lambda_LCC * max(L0 - L1, 0.0)
                    + fragility_evaluator.lambda_ASP * max(A1 - A0, 0.0)
            )

        grouped = df.groupby(["source", "target"], sort=False)

        y_marginal = [0.0] * len(df)
        index_map = {idx: i for i, idx in enumerate(df.index)}

        for (src, tgt), group in grouped:
            # 用贪心过程生成“集合边际收益标签”
            # 每一步选当前边际收益最大的路径，并把该路径在该状态下的 gain 作为 y_marginal。
            remaining = list(group.index)

            selected_node_set = set()
            selected_paths = []
            F_S = compute_F(selected_node_set)

            while remaining:
                best_idx = None
                best_gain = -float("inf")
                best_F = None
                best_nodes = None

                for idx in remaining:
                    row = df.loc[idx]
                    nodes = json.loads(row["path_nodes"])
                    cand_nodes = cls.path_to_internal_nodes(nodes)

                    merged_nodes = selected_node_set | cand_nodes
                    F_union = compute_F(merged_nodes)

                    if selected_paths:
                        red_values = []
                        for prev_nodes in selected_paths:
                            overlap = cls.edge_jaccard(nodes, prev_nodes)
                            shared = cls.shared_internal_nodes(nodes, prev_nodes)
                            red_values.append(overlap + 0.1 * shared)
                        redundancy = float(sum(red_values) / len(red_values))
                    else:
                        redundancy = 0.0

                    gain = (F_union - F_S) - lambda_red * redundancy

                    if gain > best_gain:
                        best_gain = float(gain)
                        best_idx = idx
                        best_F = float(F_union)
                        best_nodes = nodes

                if best_idx is None:
                    break

                y_marginal[index_map[best_idx]] = float(best_gain)

                selected_node_set |= cls.path_to_internal_nodes(best_nodes)
                selected_paths.append(best_nodes)
                F_S = float(best_F)

                remaining.remove(best_idx)

        # 拆分标签
        df["y_single"] = df["fragility_score"].astype(float)
        df["y_marginal"] = y_marginal

        # 保留旧字段名，兼容已有代码
        df["y_fragility"] = df["y_single"]
        df["y_gain"] = df["y_marginal"]

        label_mode = str(label_mode).lower().strip()
        if label_mode in ("single", "fragility", "y_single"):
            df["y"] = df["y_single"]
        elif label_mode in ("marginal", "gain", "y_marginal"):
            df["y"] = df["y_marginal"]
        else:
            raise ValueError(
                f"Unsupported label_mode={label_mode}. "
                "Use 'single' or 'marginal'."
            )

        print(
            "[INFO] y_gain stats:",
            "mean=", float(df["y_gain"].mean()),
            "min=", float(df["y_gain"].min()),
            "max=", float(df["y_gain"].max())
        )

        # =============================
        # 离散化标签与严格排序 (For XGBRanker NDCG)
        # =============================

        print("[INFO] Discretizing continuous labels for NDCG ranking...")

        # 1. 必须先按照任务 (source, target) 以及刚刚算出的 y_gain 进行降序排序
        df = df.sort_values(
            by=["source", "target", "y"],
            ascending=[True, True, False]
        ).reset_index(drop=True)

        # 2. 将连续的 y_gain 离散化为 0-4 档的 relevance (相关度)
        def assign_relevance(group):
            # 如果组内分数有区分度，则分为 5 个等级 (0, 1, 2, 3, 4)
            if group['y'].nunique() > 1:
                try:
                    group['relevance'] = pd.qcut(
                        group['y'],
                        q=5,
                        labels=False,
                        duplicates='drop'
                    )
                except ValueError:
                    group['relevance'] = 0
            else:
                # 分数全一样，退化为0
                group['relevance'] = 0
            return group

        # 应用 relevance 分档 (使用 group_keys=False 防止引发 Pandas 的 MultiIndex 警告)
        df = df.groupby(['source', 'target'], group_keys=False).apply(assign_relevance).reset_index(drop=True)

        # 3. 分组 apply 后可能会打乱原本的排序，因此必须再次强行按 Query 排序！这是 LTR 的生命线。
        df = df.sort_values(
            by=["source", "target", "y"],
            ascending=[True, True, False]
        ).reset_index(drop=True)

        print(
            "[INFO] Relevance tiers stats:\\n",
            df["relevance"].value_counts().sort_index()
        )

        return df

    # =============================
    # feature columns
    # =============================

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> List[str]:
        """
        Automatically infer model features.

        Important:
        Do NOT include labels, post-hoc damage metrics, relevance tiers,
        predicted scores, or candidate-generation artifacts.
        """
        excluded = {
            # identifiers / non-features
            "source",
            "target",
            "path_nodes",
            "method",
            "backend",

            # labels
            "y",
            "y_single",
            "y_marginal",
            "y_fragility",
            "y_gain",
            "relevance",

            # target leakage: these are computed from deletion damage
            "delta_E",
            "delta_LCC",
            "delta_ASP",
            "fragility_score",

            # post-model fields
            "pred_score",

            # candidate-generation artifact
            "candidate_rank",
        }

        return [
            c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

    # =============================
    # split
    # =============================

    @staticmethod
    def split_by_task(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        task_df = df[["source", "target"]].drop_duplicates().reset_index(drop=True)
        n_train = max(1, int(len(task_df) * train_ratio))

        train_tasks = task_df.iloc[:n_train]
        train_keys = set(zip(train_tasks["source"], train_tasks["target"]))

        is_train = df.apply(
            lambda r: (int(r["source"]), int(r["target"])) in train_keys,
            axis=1
        )

        train_df = df[is_train].reset_index(drop=True)
        test_df = df[~is_train].reset_index(drop=True)

        return train_df, test_df