from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.path_generator import PathGenerator
from path.src.core.types import GraphDataBundle, TaskPair


class PathRankingDatasetBuilder:
    """
    Build candidate-path samples for supervised ranking/regression.

    Each row corresponds to one candidate path between a task pair.

    Acceleration features:
    1. in-memory cache for exact fragility
    2. optional disk cache (json)
    3. approx fragility mode
    4. hybrid mode: exact for selected paths, approx for the rest
    5. progress logging
    """

    # -----------------------------
    # cache helpers
    # -----------------------------
    @staticmethod
    def _cache_key(
        bundle: GraphDataBundle,
        task: TaskPair,
        path: List[int],
        fragility_weights: Dict[str, float],
    ) -> str:
        payload = {
            "dataset": getattr(bundle, "name", "unknown"),
            "num_nodes": int(bundle.num_nodes),
            "source": int(task.source),
            "target": int(task.target),
            "path": [int(x) for x in path],
            "weights": {
                "lambda_E": float(fragility_weights.get("lambda_E", 0.4)),
                "lambda_LCC": float(fragility_weights.get("lambda_LCC", 0.4)),
                "lambda_ASP": float(fragility_weights.get("lambda_ASP", 0.2)),
            },
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _load_disk_cache(cache_path: Optional[str], debug: bool = False) -> Dict[str, Dict[str, float]]:
        if not cache_path:
            return {}
        p = Path(cache_path)
        if not p.exists():
            if debug:
                print(f"[DEBUG] cache file not found, start with empty cache: {p}")
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if debug:
                print(f"[DEBUG] loaded disk cache: {p}, entries={len(obj)}")
            return obj
        except Exception as e:
            if debug:
                print(f"[WARN] failed to load disk cache {p}: {e}")
            return {}

    @staticmethod
    def _save_disk_cache(
        cache_path: Optional[str],
        cache_obj: Dict[str, Dict[str, float]],
        debug: bool = False,
    ) -> None:
        if not cache_path:
            return
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cache_obj, f, ensure_ascii=False, indent=2)
            if debug:
                print(f"[DEBUG] saved disk cache: {p}, entries={len(cache_obj)}")
        except Exception as e:
            if debug:
                print(f"[WARN] failed to save disk cache {p}: {e}")

    # -----------------------------
    # approx fragility
    # -----------------------------
    @staticmethod
    def _clip01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return float(x)

    @classmethod
    def _approx_fragility_from_features(
        cls,
        feats: Dict[str, float],
        task: TaskPair,
        path: List[int],
    ) -> Dict[str, float]:
        """
        A cheap surrogate for fragility.

        It does NOT modify the graph.
        It approximates path importance using:
        - avg_edge_bc
        - cross_comm_ratio
        - internal_node_importance
        - avg_node_importance
        - relative path stretch penalty

        Returned keys match compute_fragility() output format.
        """
        avg_edge_bc = float(feats.get("avg_edge_bc", 0.0))
        cross_comm_ratio = float(feats.get("cross_comm_ratio", 0.0))
        internal_node_importance = float(feats.get("internal_node_importance", 0.0))
        avg_node_importance = float(feats.get("avg_node_importance", 0.0))
        path_length = float(feats.get("path_length", len(path)))
        shortest_len = max(1.0, float(task.shortest_len))

        # relative stretch: 越接近最短路越好
        stretch = path_length / (shortest_len + 1.0)
        stretch_bonus = cls._clip01(1.0 / stretch)

        # 是否跨社团任务，对跨社团路径给一点额外偏置
        cross_task_bonus = 0.15 if not bool(task.same_community) else 0.0

        # surrogate score in [roughly 0,1+]
        surrogate = (
            0.40 * avg_edge_bc
            + 0.20 * cross_comm_ratio
            + 0.20 * internal_node_importance
            + 0.10 * avg_node_importance
            + 0.10 * stretch_bonus
            + cross_task_bonus
        )

        surrogate = max(0.0, float(surrogate))

        # 注意：下面三项只是“伪分解”，便于保持接口兼容
        # 它们不是严格物理量，只是近似代理
        delta_E = 0.50 * surrogate
        delta_LCC = 0.30 * surrogate
        delta_ASP = 0.20 * surrogate

        return {
            "delta_E": float(delta_E),
            "delta_LCC": float(delta_LCC),
            "delta_ASP": float(delta_ASP),
            "fragility_score": float(surrogate),
        }

    # -----------------------------
    # exact / approx selector
    # -----------------------------
    @staticmethod
    def _should_use_exact(
        task_idx: int,
        candidate_rank: int,
        path: List[int],
        exact_every_n_tasks: int,
        exact_top_ranks: int,
        exact_max_path_len: int,
    ) -> bool:
        """
        For hybrid mode:
        - exact for top candidate ranks
        - exact every n tasks
        - exact for short paths
        """
        if candidate_rank <= int(exact_top_ranks):
            return True
        if exact_every_n_tasks > 0 and ((task_idx + 1) % exact_every_n_tasks == 0):
            return True
        if len(path) <= int(exact_max_path_len):
            return True
        return False

    @classmethod
    def _path_row(
        cls,
        bundle: GraphDataBundle,
        task: TaskPair,
        path: List[int],
        fragility_evaluator: FragilityEvaluator,
        base_metrics: Dict[str, float],
        candidate_rank: int,
        task_idx: int,
        fragility_weights: Dict[str, float],
        fragility_mode: str,
        cache_mem: Dict[str, Dict[str, float]],
        cache_disk: Dict[str, Dict[str, float]],
        exact_every_n_tasks: int,
        exact_top_ranks: int,
        exact_max_path_len: int,
        debug: bool,
    ) -> Dict[str, object]:
        feats = PathFeatureExtractor.extract_features(
            path=path,
            importance=bundle.importance,
            community=bundle.community,
            edge_bc=bundle.edge_bc,
        )

        key = cls._cache_key(
            bundle=bundle,
            task=task,
            path=path,
            fragility_weights=fragility_weights,
        )

        frag: Dict[str, float]
        frag_mode_used: str

        # -------------------------
        # exact
        # -------------------------
        if fragility_mode == "exact":
            frag = fragility_evaluator.compute_fragility(
                G=bundle.nx_graph,
                path=path,
                base_metrics=base_metrics,
                num_nodes=bundle.num_nodes,
            )
            frag_mode_used = "exact"

        # -------------------------
        # cached
        # -------------------------
        elif fragility_mode == "cached":
            if key in cache_mem:
                frag = cache_mem[key]
                frag_mode_used = "cache_mem"
            elif key in cache_disk:
                frag = cache_disk[key]
                cache_mem[key] = frag
                frag_mode_used = "cache_disk"
            else:
                frag = fragility_evaluator.compute_fragility(
                    G=bundle.nx_graph,
                    path=path,
                    base_metrics=base_metrics,
                    num_nodes=bundle.num_nodes,
                )
                cache_mem[key] = frag
                cache_disk[key] = frag
                frag_mode_used = "exact_cached"

        # -------------------------
        # approx
        # -------------------------
        elif fragility_mode == "approx":
            frag = cls._approx_fragility_from_features(
                feats=feats,
                task=task,
                path=path,
            )
            frag_mode_used = "approx"

        # -------------------------
        # hybrid
        # -------------------------
        elif fragility_mode == "hybrid":
            if key in cache_mem:
                frag = cache_mem[key]
                frag_mode_used = "cache_mem"
            elif key in cache_disk:
                frag = cache_disk[key]
                cache_mem[key] = frag
                frag_mode_used = "cache_disk"
            else:
                use_exact = cls._should_use_exact(
                    task_idx=task_idx,
                    candidate_rank=candidate_rank,
                    path=path,
                    exact_every_n_tasks=exact_every_n_tasks,
                    exact_top_ranks=exact_top_ranks,
                    exact_max_path_len=exact_max_path_len,
                )
                if use_exact:
                    frag = fragility_evaluator.compute_fragility(
                        G=bundle.nx_graph,
                        path=path,
                        base_metrics=base_metrics,
                        num_nodes=bundle.num_nodes,
                    )
                    cache_mem[key] = frag
                    cache_disk[key] = frag
                    frag_mode_used = "exact_hybrid"
                else:
                    frag = cls._approx_fragility_from_features(
                        feats=feats,
                        task=task,
                        path=path,
                    )
                    frag_mode_used = "approx_hybrid"
        else:
            raise ValueError(
                f"Unsupported fragility_mode: {fragility_mode}. "
                f"Expected one of ['exact', 'cached', 'approx', 'hybrid']"
            )

        row: Dict[str, object] = {
            "source": int(task.source),
            "target": int(task.target),
            "shortest_len": int(task.shortest_len),
            "same_community": int(task.same_community),
            "pair_score": float(task.pair_score),
            "candidate_rank": int(candidate_rank),
            "path_nodes": json.dumps([int(n) for n in path]),
            "path_length_int": int(len(path)),
            "method": "candidate",
            "fragility_mode_used": frag_mode_used,
        }
        row.update({k: float(v) for k, v in feats.items()})
        row.update({k: float(v) for k, v in frag.items()})
        row["y_fragility"] = float(frag["fragility_score"])

        if debug:
            print(
                f"[DEBUG] row built | task=({task.source}->{task.target}) "
                f"| rank={candidate_rank} | len={len(path)} "
                f"| fragility_mode_used={frag_mode_used} "
                f"| y_fragility={row['y_fragility']:.6f}"
            )

        return row

    @classmethod
    def build_path_samples(
        cls,
        bundle: GraphDataBundle,
        tasks: List[TaskPair],
        path_k: int,
        max_hops: int,
        delta: int,
        fragility_weights: Optional[Dict[str, float]] = None,
        fragility_mode: str = "hybrid",
        cache_path: Optional[str] = None,
        exact_every_n_tasks: int = 20,
        exact_top_ranks: int = 1,
        exact_max_path_len: int = 4,
        progress_every: int = 10,
        debug: bool = False,
    ) -> pd.DataFrame:
        """
        Build a path-level supervised dataset.

        Returns a DataFrame with at least these columns:
        [source, target, path_nodes, avg_node_importance, internal_node_importance,
         avg_edge_bc, cross_comm_ratio, path_length, num_edges,
         delta_E, delta_LCC, delta_ASP, fragility_score, y_fragility]

        Parameters
        ----------
        fragility_mode:
            - 'exact'  : always recompute fragility
            - 'cached' : exact + memory/disk cache
            - 'approx' : surrogate only
            - 'hybrid' : recommended, exact for selected paths and cached; approx for others

        cache_path:
            optional disk cache json file path
        """
        fragility_weights = fragility_weights or {
            "lambda_E": 0.4,
            "lambda_LCC": 0.4,
            "lambda_ASP": 0.2,
        }

        fragility_evaluator = FragilityEvaluator(**fragility_weights)
        base_metrics = fragility_evaluator.compute_base_metrics(bundle.nx_graph)

        rows: List[Dict[str, object]] = []

        cache_mem: Dict[str, Dict[str, float]] = {}
        cache_disk = cls._load_disk_cache(cache_path, debug=debug)

        total_tasks = len(tasks)
        total_paths = 0
        total_exact = 0
        total_approx = 0
        total_cache_mem = 0
        total_cache_disk = 0
        failed_tasks = 0
        failed_paths = 0

        if debug:
            print("[DEBUG] build_path_samples start")
            print(f"[DEBUG] dataset = {getattr(bundle, 'name', None)}")
            print(f"[DEBUG] total_tasks = {total_tasks}")
            print(f"[DEBUG] path_k = {path_k}, max_hops = {max_hops}, delta = {delta}")
            print(f"[DEBUG] fragility_mode = {fragility_mode}")
            print(f"[DEBUG] cache_path = {cache_path}")
            print(f"[DEBUG] exact_every_n_tasks = {exact_every_n_tasks}")
            print(f"[DEBUG] exact_top_ranks = {exact_top_ranks}")
            print(f"[DEBUG] exact_max_path_len = {exact_max_path_len}")

        t_global_start = time.perf_counter()

        for task_idx, task in enumerate(tasks):
            if progress_every > 0 and ((task_idx + 1) % progress_every == 0 or task_idx == 0):
                elapsed = time.perf_counter() - t_global_start
                print(
                    f"[PROGRESS] task {task_idx + 1}/{total_tasks} "
                    f"| rows={len(rows)} | total_paths={total_paths} "
                    f"| elapsed={elapsed:.2f}s"
                )

            # -------------------------
            # path generation
            # -------------------------
            try:
                t_paths = time.perf_counter()
                paths = PathGenerator.k_shortest_simple_paths(
                    bundle.nx_graph,
                    source=task.source,
                    target=task.target,
                    k=path_k,
                    max_hops=max_hops,
                    delta=delta,
                )
                if debug:
                    print(
                        f"[DEBUG] generated {len(paths)} candidate paths "
                        f"for task ({task.source}->{task.target}) "
                        f"in {time.perf_counter() - t_paths:.4f}s"
                    )
            except Exception as e:
                failed_tasks += 1
                print(
                    f"[ERROR] path generation failed for task "
                    f"({task.source},{task.target}): {e}"
                )
                continue

            # -------------------------
            # path rows
            # -------------------------
            for rank_idx, path in enumerate(paths, start=1):
                total_paths += 1
                t0 = time.perf_counter()

                try:
                    row = cls._path_row(
                        bundle=bundle,
                        task=task,
                        path=path,
                        fragility_evaluator=fragility_evaluator,
                        base_metrics=base_metrics,
                        candidate_rank=rank_idx,
                        task_idx=task_idx,
                        fragility_weights=fragility_weights,
                        fragility_mode=fragility_mode,
                        cache_mem=cache_mem,
                        cache_disk=cache_disk,
                        exact_every_n_tasks=exact_every_n_tasks,
                        exact_top_ranks=exact_top_ranks,
                        exact_max_path_len=exact_max_path_len,
                        debug=debug,
                    )
                    rows.append(row)

                    mode_used = row["fragility_mode_used"]
                    if mode_used in {"exact", "exact_cached", "exact_hybrid"}:
                        total_exact += 1
                    elif mode_used in {"approx", "approx_hybrid"}:
                        total_approx += 1
                    elif mode_used == "cache_mem":
                        total_cache_mem += 1
                    elif mode_used == "cache_disk":
                        total_cache_disk += 1

                except Exception as e:
                    failed_paths += 1
                    print(f"[ERROR] _path_row failed for path={path}: {e}")
                    continue

                dt = time.perf_counter() - t0
                if debug and dt > 0.1:
                    print(
                        f"[WARN] slow path row: {dt:.4f}s "
                        f"| task=({task.source}->{task.target}) "
                        f"| rank={rank_idx} | path_len={len(path)}"
                    )

        total_time = time.perf_counter() - t_global_start

        # save disk cache
        cls._save_disk_cache(cache_path, cache_disk, debug=debug)

        print("[SUMMARY] build_path_samples finished")
        print(f"[SUMMARY] total_tasks      = {total_tasks}")
        print(f"[SUMMARY] failed_tasks     = {failed_tasks}")
        print(f"[SUMMARY] total_paths      = {total_paths}")
        print(f"[SUMMARY] failed_paths     = {failed_paths}")
        print(f"[SUMMARY] total_rows       = {len(rows)}")
        print(f"[SUMMARY] total_exact      = {total_exact}")
        print(f"[SUMMARY] total_approx     = {total_approx}")
        print(f"[SUMMARY] total_cache_mem  = {total_cache_mem}")
        print(f"[SUMMARY] total_cache_disk = {total_cache_disk}")
        print(f"[SUMMARY] total_time_sec   = {total_time:.4f}")
        if total_paths > 0:
            print(f"[SUMMARY] avg_time_per_path = {total_time / total_paths:.6f}s")

        df = pd.DataFrame(rows)
        if df.empty:
            print("[WARN] resulting dataframe is empty")
            return df

        # deterministic ordering for split/reproducibility
        df = df.sort_values(
            by=["source", "target", "candidate_rank", "fragility_score"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

        print(f"[SUMMARY] final df shape = {df.shape}")
        return df

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> List[str]:
        excluded = {
            "source",
            "target",
            "path_nodes",
            "method",
            "y_fragility",
            "fragility_mode_used",
        }
        return [
            c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

    @staticmethod
    def split_by_task(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df.copy(), df.copy()

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