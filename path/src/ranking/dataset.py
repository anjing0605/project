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
        return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

    @staticmethod
    def edge_jaccard(a, b):
        ea = set(PathRankingDatasetBuilder.path_to_edges(a))
        eb = set(PathRankingDatasetBuilder.path_to_edges(b))
        return len(ea & eb) / max(len(ea | eb), 1)

    @staticmethod
    def shared_internal_nodes(a, b):
        return len(set(a[1:-1]) & set(b[1:-1]))

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
        debug: bool = False,
    ) -> pd.DataFrame:

        fragility_weights = fragility_weights or {
            "lambda_E": 0.4,
            "lambda_LCC": 0.4,
            "lambda_ASP": 0.2,
        }

        fragility_evaluator = FragilityEvaluator(**fragility_weights)
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
        # TRUE marginal gain (Δ(p|S), union-edge)
        # =========================================================

        print("[INFO] computing TRUE marginal gain labels...")

        lambda_red = float(fragility_weights.get("lambda_red", 0.2))

        def compute_F(edge_set):
            H = bundle.nx_graph.copy()
            H.remove_edges_from(edge_set)

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

        y_gain = [0.0] * len(df)
        index_map = {idx: i for i, idx in enumerate(df.index)}

        for (src, tgt), group in grouped:

            group = group.sort_values("fragility_score", ascending=False)

            top1 = group.iloc[0]
            top1_nodes = json.loads(top1["path_nodes"])

            S_edges = set(cls.path_to_edges(top1_nodes))
            F_S = compute_F(S_edges)

            for idx, row in group.iterrows():
                nodes = json.loads(row["path_nodes"])
                cand_edges = set(cls.path_to_edges(nodes))

                merged_edges = S_edges | cand_edges
                F_union = compute_F(merged_edges)

                overlap = cls.edge_jaccard(nodes, top1_nodes)
                shared = cls.shared_internal_nodes(nodes, top1_nodes)

                redundancy = overlap + 0.1 * shared

                gain = (F_union - lambda_red * redundancy) - F_S

                y_gain[index_map[idx]] = float(gain)

        df["y_gain"] = y_gain
        df["y"] = df["y_gain"]

        print(
            "[INFO] y_gain stats:",
            "mean=", float(df["y_gain"].mean()),
            "min=", float(df["y_gain"].min()),
            "max=", float(df["y_gain"].max())
        )

        # =============================
        # sort
        # =============================

        df = df.sort_values(
            by=["source", "target", "candidate_rank", "fragility_score"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

        return df

    # =============================
    # feature columns
    # =============================

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> List[str]:
        excluded = {
            "source",
            "target",
            "path_nodes",
            "method",
            "y_fragility",
            "y_gain",
            "y",
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