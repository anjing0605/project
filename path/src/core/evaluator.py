from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Sequence
import time
import networkx as nx

from path.src.core.fragility import FragilityEvaluator
from path.src.core.path_features import PathFeatureExtractor
from path.src.core.types import PathRecord


class MethodEvaluator:
    """Evaluate top-k cumulative damage curves for different path-selection methods."""

    def __init__(
        self,
        lambda_E: float = 0.4,
        lambda_LCC: float = 0.4,
        lambda_ASP: float = 0.2,
    ) -> None:
        self.fragility_evaluator = FragilityEvaluator(
            lambda_E=lambda_E,
            lambda_LCC=lambda_LCC,
            lambda_ASP=lambda_ASP,
        )

    @staticmethod
    def _sorted_k_list(k_list: Sequence[int]) -> List[int]:
        cleaned = sorted({int(k) for k in k_list if int(k) > 0})
        if not cleaned:
            raise ValueError("k_list must contain at least one positive integer.")
        return cleaned

    @staticmethod
    def _unique_edges_from_records(path_records: Sequence[PathRecord]) -> List[tuple[int, int]]:
        seen = set()
        ordered_edges: List[tuple[int, int]] = []
        for record in path_records:
            for u, v in PathFeatureExtractor.path_to_edges(record.nodes):
                edge = tuple(sorted((int(u), int(v))))
                if edge not in seen:
                    seen.add(edge)
                    ordered_edges.append(edge)
        return ordered_edges

    def evaluate_topk_damage(
            self,
            G: nx.Graph,
            selected_paths: Sequence[PathRecord],
            k_list: Sequence[int],
            mode: str = "approx",
            early_stop: bool = False,
            tol: float = 1e-4,
            debug: bool = True,
            shared_base_metrics: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:

        ks = self._sorted_k_list(k_list)


        if shared_base_metrics is None:
            base_metrics = self.fragility_evaluator.compute_base_metrics(G)
        else:
            base_metrics = dict(shared_base_metrics)
        max_k = len(selected_paths)
        ks = [k for k in ks if k <= max_k]

        if not ks:
            return {
                "k_list": [],
                "delta_E_curve": [],
                "delta_LCC_curve": [],
                "delta_ASP_curve": [],
                "fragility_score_curve": [],
                "num_removed_nodes": [],
                "base_metrics": {k: float(v) for k, v in base_metrics.items()},
            }

        num_nodes = int(G.number_of_nodes())

        print("[compare] building prefix internal-node cache...")

        prefix_nodes: Dict[int, List[int]] = {}
        seen_nodes = set()
        curr_nodes = []

        for i, record in enumerate(selected_paths):
            for n in record.nodes[1:-1]:
                n = int(n)
                if n not in seen_nodes:
                    seen_nodes.add(n)
                    curr_nodes.append(n)
            prefix_nodes[i + 1] = list(curr_nodes)

        print(f"[compare] prefix internal-node cache built: max_nodes={len(curr_nodes)}")

        delta_E_curve = []
        delta_LCC_curve = []
        delta_ASP_curve = []
        fragility_score_curve = []
        num_removed_nodes = []

        metrics_cache: Dict[int, tuple] = {}

        for k in ks:
            nodes = prefix_nodes.get(k, [])

            t_step = time.perf_counter()

            H = G.copy()
            H.remove_nodes_from(nodes)

            if k in metrics_cache:
                E1, ASP1, LCC1 = metrics_cache[k]
                cache_hit = True
            else:
                cache_hit = False
                if mode == "approx":
                    E1 = self.fragility_evaluator.global_efficiency_approx(H)
                    ASP1 = self.fragility_evaluator.avg_shortest_path_of_lcc_approx(H)
                    LCC1 = self.fragility_evaluator.lcc_ratio(H, num_nodes)
                elif mode == "exact":
                    E1 = self.fragility_evaluator.global_efficiency_exact(H)
                    ASP1 = self.fragility_evaluator.avg_shortest_path_of_lcc_exact(H)
                    LCC1 = self.fragility_evaluator.lcc_ratio(H, num_nodes)
                elif mode == "hybrid":
                    if k <= 3:
                        E1 = self.fragility_evaluator.global_efficiency_exact(H)
                        ASP1 = self.fragility_evaluator.avg_shortest_path_of_lcc_exact(H)
                    else:
                        E1 = self.fragility_evaluator.global_efficiency_approx(H)
                        ASP1 = self.fragility_evaluator.avg_shortest_path_of_lcc_approx(H)
                    LCC1 = self.fragility_evaluator.lcc_ratio(H, num_nodes)
                else:
                    raise ValueError(f"unknown mode={mode}")

                metrics_cache[k] = (E1, ASP1, LCC1)

            delta_E = max(0.0, float(base_metrics["global_efficiency"]) - E1)
            delta_LCC = max(0.0, float(base_metrics["lcc_ratio"]) - LCC1)
            delta_ASP = max(0.0, ASP1 - float(base_metrics["avg_shortest_path_lcc"]))

            fragility_score = (
                    self.fragility_evaluator.lambda_E * delta_E
                    + self.fragility_evaluator.lambda_LCC * delta_LCC
                    + self.fragility_evaluator.lambda_ASP * delta_ASP
            )

            dt = time.perf_counter() - t_step

            print(
                f"[compare][k={k}] "
                f"nodes={len(nodes)} "
                f"time={dt:.3f}s "
                f"fragility={fragility_score:.6f} "
                f"{'(cache)' if cache_hit else ''}"
            )

            delta_E_curve.append(float(delta_E))
            delta_LCC_curve.append(float(delta_LCC))
            delta_ASP_curve.append(float(delta_ASP))
            fragility_score_curve.append(float(fragility_score))
            num_removed_nodes.append(len(nodes))

        return {
            "k_list": ks[:len(fragility_score_curve)],
            "delta_E_curve": delta_E_curve,
            "delta_LCC_curve": delta_LCC_curve,
            "delta_ASP_curve": delta_ASP_curve,
            "fragility_score_curve": fragility_score_curve,
            "num_removed_nodes": num_removed_nodes,
            "base_metrics": {k: float(v) for k, v in base_metrics.items()},
        }

    def evaluate_fixed_node_budget_damage(
            self,
            G: nx.Graph,
            selected_paths: Sequence[PathRecord],
            node_budget_list: Sequence[int],
            mode: str = "exact",
            shared_base_metrics: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """
        Fixed internal-node budget evaluation.

        与 top-k path evaluation 不同：
        - top-k 是固定路径条数；
        - fixed-node-budget 是固定删除的 unique internal nodes 数量。

        删除顺序：
        按 selected_paths 的顺序依次收集路径内部节点 P[1:-1]，
        对每个 budget B，只删除前 B 个 unique internal nodes。
        """
        budgets = sorted({int(b) for b in node_budget_list if int(b) > 0})
        if not budgets:
            raise ValueError("node_budget_list must contain at least one positive integer.")

        if shared_base_metrics is None:
            base_metrics = self.fragility_evaluator.compute_base_metrics(G)
        else:
            base_metrics = dict(shared_base_metrics)

        num_nodes = int(G.number_of_nodes())

        ordered_unique_nodes: List[int] = []
        seen_nodes = set()

        for record in selected_paths:
            for n in record.nodes[1:-1]:
                n = int(n)
                if n not in seen_nodes:
                    seen_nodes.add(n)
                    ordered_unique_nodes.append(n)

        if not ordered_unique_nodes:
            return {
                "node_budget_list": [],
                "delta_E_curve": [],
                "delta_LCC_curve": [],
                "delta_ASP_curve": [],
                "fragility_score_curve": [],
                "num_removed_nodes": [],
                "base_metrics": {k: float(v) for k, v in base_metrics.items()},
            }

        effective_budgets = [b for b in budgets if b <= len(ordered_unique_nodes)]

        # 如果某个方法 unique internal nodes 少于最小 budget，
        # 仍然保留它的最大可删除节点数，避免整条曲线为空。
        if not effective_budgets:
            effective_budgets = [len(ordered_unique_nodes)]

        delta_E_curve: List[float] = []
        delta_LCC_curve: List[float] = []
        delta_ASP_curve: List[float] = []
        fragility_score_curve: List[float] = []
        num_removed_nodes: List[int] = []

        for budget in effective_budgets:
            nodes = ordered_unique_nodes[:budget]

            H = G.copy()
            H.remove_nodes_from(nodes)

            if mode == "approx":
                E1 = self.fragility_evaluator.global_efficiency_approx(H)
                ASP1 = self.fragility_evaluator.avg_shortest_path_of_lcc_approx(H)
                LCC1 = self.fragility_evaluator.lcc_ratio(H, num_nodes)

            elif mode == "exact":
                E1 = self.fragility_evaluator.global_efficiency_exact(H)
                ASP1 = self.fragility_evaluator.avg_shortest_path_of_lcc_exact(H)
                LCC1 = self.fragility_evaluator.lcc_ratio(H, num_nodes)

            elif mode == "hybrid":
                # fixed-budget 建议前几个预算 exact，后面 approx。
                # 如果你想完全严格，可直接把 hybrid 也改成 exact。
                if budget <= 10:
                    E1 = self.fragility_evaluator.global_efficiency_exact(H)
                    ASP1 = self.fragility_evaluator.avg_shortest_path_of_lcc_exact(H)
                else:
                    E1 = self.fragility_evaluator.global_efficiency_approx(H)
                    ASP1 = self.fragility_evaluator.avg_shortest_path_of_lcc_approx(H)
                LCC1 = self.fragility_evaluator.lcc_ratio(H, num_nodes)

            else:
                raise ValueError(f"unknown mode={mode}")

            delta_E = max(0.0, float(base_metrics["global_efficiency"]) - float(E1))
            delta_LCC = max(0.0, float(base_metrics["lcc_ratio"]) - float(LCC1))
            delta_ASP = max(0.0, float(ASP1) - float(base_metrics["avg_shortest_path_lcc"]))

            fragility_score = (
                    self.fragility_evaluator.lambda_E * delta_E
                    + self.fragility_evaluator.lambda_LCC * delta_LCC
                    + self.fragility_evaluator.lambda_ASP * delta_ASP
            )

            delta_E_curve.append(float(delta_E))
            delta_LCC_curve.append(float(delta_LCC))
            delta_ASP_curve.append(float(delta_ASP))
            fragility_score_curve.append(float(fragility_score))
            num_removed_nodes.append(int(len(nodes)))

        return {
            "node_budget_list": effective_budgets,
            "delta_E_curve": delta_E_curve,
            "delta_LCC_curve": delta_LCC_curve,
            "delta_ASP_curve": delta_ASP_curve,
            "fragility_score_curve": fragility_score_curve,
            "num_removed_nodes": num_removed_nodes,
            "base_metrics": {k: float(v) for k, v in base_metrics.items()},
        }
    def compare_methods(
            self,
            result_dict: Dict[str, Sequence[PathRecord]],
            G: nx.Graph,
            k_list: Sequence[int],
            mode: str = "approx",
            early_stop: bool = False,
            tol: float = 1e-4,
            debug: bool = True,
            shared_base_metrics: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """Evaluate multiple methods under the same top-k cumulative deletion protocol."""
        if shared_base_metrics is None:
            t0 = time.perf_counter()
            shared_base_metrics = self.fragility_evaluator.compute_base_metrics(G)
            stage_msg = f"[compare_methods] shared_base_metrics computed in {time.perf_counter() - t0:.2f}s"
            if debug:
                print(stage_msg)
        else:
            if debug:
                print("[compare_methods] using provided shared_base_metrics")
        comparison: Dict[str, Any] = {}

        for method_name, records in result_dict.items():
            if debug:
                print(
                    f"[compare_methods] method={method_name}, "
                    f"num_records={len(records)}, mode={mode}, "
                    f"early_stop={early_stop}, tol={tol}"
                )

            curves = self.evaluate_topk_damage(
                G=G,
                selected_paths=list(records),
                k_list=k_list,
                mode=mode,
                early_stop=early_stop,
                tol=tol,
                debug=debug,
                shared_base_metrics=shared_base_metrics,
            )

            comparison[method_name] = {
                "num_paths": int(len(records)),
                "eval_mode": mode,
                "eval_early_stop": bool(early_stop),
                "eval_tol": float(tol),
                **curves,
            }

        return {
            "shared_base_metrics": {k: float(v) for k, v in shared_base_metrics.items()},
            "methods": comparison,
        }

    def compare_methods_fixed_node_budget(
            self,
            result_dict: Dict[str, Sequence[PathRecord]],
            G: nx.Graph,
            node_budget_list: Sequence[int],
            mode: str = "exact",
            shared_base_metrics: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """
        Compare methods under the same removed-internal-node budget.
        """
        if shared_base_metrics is None:
            shared_base_metrics = self.fragility_evaluator.compute_base_metrics(G)

        comparison: Dict[str, Any] = {}

        for method_name, records in result_dict.items():
            curves = self.evaluate_fixed_node_budget_damage(
                G=G,
                selected_paths=list(records),
                node_budget_list=node_budget_list,
                mode=mode,
                shared_base_metrics=shared_base_metrics,
            )

            comparison[method_name] = {
                "num_paths": int(len(records)),
                "eval_mode": mode,
                **curves,
            }

        return {
            "shared_base_metrics": {
                k: float(v) for k, v in shared_base_metrics.items()
            },
            "methods": comparison,
        }
    @staticmethod
    def summarize_top_paths(records: Sequence[PathRecord], top_n: int = 10) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for idx, record in enumerate(records[:top_n], start=1):
            summary.append(
                {
                    "rank": idx,
                    "source": int(record.source),
                    "target": int(record.target),
                    "nodes": [int(n) for n in record.nodes],
                    "length": int(len(record.nodes)),
                    "score": None if record.score is None else float(record.score),
                    "method": record.method,
                    "features": {
                        k: float(v) for k, v in (record.features or {}).items()
                    },
                    "fragility": {
                        k: float(v) for k, v in (record.fragility or {}).items()
                    },
                    "metadata": dict(record.metadata or {}),
                }
            )
        return summary

    @staticmethod
    def attach_rank(records: Sequence[PathRecord]) -> List[PathRecord]:
        ranked: List[PathRecord] = []
        for idx, record in enumerate(records, start=1):
            metadata = dict(record.metadata or {})
            metadata["rank"] = idx
            ranked.append(replace(record, metadata=metadata))
        return ranked
