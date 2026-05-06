from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
'''读取 rule / rank / rl_eval 的 metrics json
汇总成统一 dataframe
读取 paths json，生成路径质量表
提供保存 csv 的接口
'''

class MetricsReporter:
    """
    Utility functions for:
    1) loading experiment metrics json
    2) building unified comparison table
    3) building path quality table
    4) saving report tables
    """

    @staticmethod
    def load_json(path: str | Path) -> Dict[str, Any]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            return default

    def _extract_curve_rows(
            dataset: str,
            source_file: str,
            experiment: str,
            comparison: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        if isinstance(comparison, dict) and "methods" in comparison:
            method_dict = comparison.get("methods", {})
        else:
            method_dict = comparison

        for method, metrics in method_dict.items():
            if not isinstance(metrics, dict):
                continue

            if "delta_E_curve" not in metrics:
                continue

            delta_E_curve = metrics.get("delta_E_curve", [])
            delta_LCC_curve = metrics.get("delta_LCC_curve", [])
            delta_ASP_curve = metrics.get("delta_ASP_curve", [])
            fragility_score_curve = metrics.get("fragility_score_curve", [])
            num_removed_nodes = metrics.get("num_removed_nodes", [])
            k_list = metrics.get("k_list", list(range(1, len(delta_E_curve) + 1)))

            n = min(len(k_list), len(delta_E_curve), len(delta_LCC_curve), len(delta_ASP_curve))

            for i in range(n):
                rows.append(
                    {
                        "dataset": dataset,
                        "source_file": source_file,
                        "experiment": experiment,
                        "method": method,
                        "k": int(k_list[i]),
                        "delta_E": MetricsReporter._safe_float(delta_E_curve[i]),
                        "delta_LCC": MetricsReporter._safe_float(delta_LCC_curve[i]),
                        "delta_ASP": MetricsReporter._safe_float(delta_ASP_curve[i]),
                        "fragility_score": MetricsReporter._safe_float(
                            fragility_score_curve[i] if i < len(fragility_score_curve) else None
                        ),
                        "num_removed_nodes": MetricsReporter._safe_float(
                            num_removed_nodes[i] if i < len(num_removed_nodes) else None
                        ),
                    }
                )

        return rows

    @staticmethod
    def _extract_budget_rows(
            dataset: str,
            source_file: str,
            experiment: str,
            comparison: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        if isinstance(comparison, dict) and "methods" in comparison:
            method_dict = comparison.get("methods", {})
        else:
            method_dict = comparison

        for method, metrics in method_dict.items():
            if not isinstance(metrics, dict):
                continue

            if "delta_E_curve" not in metrics:
                continue

            delta_E_curve = metrics.get("delta_E_curve", [])
            delta_LCC_curve = metrics.get("delta_LCC_curve", [])
            delta_ASP_curve = metrics.get("delta_ASP_curve", [])
            fragility_score_curve = metrics.get("fragility_score_curve", [])
            num_removed_nodes = metrics.get("num_removed_nodes", [])
            budget_list = metrics.get(
                "node_budget_list",
                list(range(1, len(delta_E_curve) + 1)),
            )

            n = min(
                len(budget_list),
                len(delta_E_curve),
                len(delta_LCC_curve),
                len(delta_ASP_curve),
            )

            for i in range(n):
                rows.append(
                    {
                        "dataset": dataset,
                        "source_file": source_file,
                        "experiment": experiment,
                        "method": method,
                        "budget": int(budget_list[i]),
                        "delta_E": MetricsReporter._safe_float(delta_E_curve[i]),
                        "delta_LCC": MetricsReporter._safe_float(delta_LCC_curve[i]),
                        "delta_ASP": MetricsReporter._safe_float(delta_ASP_curve[i]),
                        "fragility_score": MetricsReporter._safe_float(
                            fragility_score_curve[i]
                            if i < len(fragility_score_curve)
                            else None
                        ),
                        "num_removed_nodes": MetricsReporter._safe_float(
                            num_removed_nodes[i]
                            if i < len(num_removed_nodes)
                            else None
                        ),
                    }
                )

        return rows
    @staticmethod
    def parse_rule_metrics(metrics_path: str | Path) -> List[Dict[str, Any]]:
        obj = MetricsReporter.load_json(metrics_path)
        dataset = obj["dataset"]["name"]
        comparison = obj.get("comparison", {})
        return MetricsReporter._extract_curve_rows(
            dataset=dataset,
            source_file=Path(metrics_path).name,
            experiment="rule_compare",
            comparison=comparison,
        )

    @staticmethod
    def parse_fixed_node_budget_metrics(metrics_path: str | Path) -> List[Dict[str, Any]]:
        obj = MetricsReporter.load_json(metrics_path)
        dataset = obj["dataset"]["name"]
        comparison = obj.get("fixed_node_budget_comparison", {})

        if not comparison:
            return []

        return MetricsReporter._extract_budget_rows(
            dataset=dataset,
            source_file=Path(metrics_path).name,
            experiment="fixed_node_budget_compare",
            comparison=comparison,
        )

    @staticmethod
    def parse_rank_metrics(metrics_path: str | Path) -> List[Dict[str, Any]]:
        obj = MetricsReporter.load_json(metrics_path)
        dataset = obj["dataset"]["name"]
        comparison = obj.get("comparison", {})

        ranking = obj.get("ranking", {}) or {}
        mode = str(ranking.get("mode", "rank")).strip()
        selector = str(ranking.get("selector", "none")).strip()

        if mode == "pure_rank":
            method_alias = "rank_pure"
        elif mode == "rank_set" and selector != "none":
            method_alias = f"rank_set_{selector}"
        else:
            method_alias = f"rank_{mode}"

        rows = MetricsReporter._extract_curve_rows(
            dataset=dataset,
            source_file=Path(metrics_path).name,
            experiment="rank_compare",
            comparison=comparison,
        )

        for r in rows:
            r["method"] = method_alias

        return rows

    @staticmethod
    def parse_rl_eval_metrics(metrics_path: str | Path) -> List[Dict[str, Any]]:
        obj = MetricsReporter.load_json(metrics_path)
        dataset = obj["dataset"]["name"]
        comparison = obj.get("comparison", {})
        return MetricsReporter._extract_curve_rows(
            dataset=dataset,
            source_file=Path(metrics_path).name,
            experiment="rl_eval_compare",
            comparison=comparison,
        )

    @staticmethod
    def build_method_comparison_table(metrics_dir: str | Path) -> pd.DataFrame:
        metrics_dir = Path(metrics_dir)
        rows: List[Dict[str, Any]] = []

        for path in metrics_dir.glob("*_rule_metrics.json"):
            rows.extend(MetricsReporter.parse_rule_metrics(path))

        for path in metrics_dir.glob("*rank*metrics.json"):
            name = path.name
            if "_rl_eval_" in name:
                continue
            if "_rule_" in name:
                continue
            rows.extend(MetricsReporter.parse_rank_metrics(path))

        for path in metrics_dir.glob("*_rl_eval_metrics.json"):
            rows.extend(MetricsReporter.parse_rl_eval_metrics(path))

        if not rows:
            return pd.DataFrame(
                columns=[
                    "dataset",
                    "source_file",
                    "experiment",
                    "method",
                    "k",
                    "delta_E",
                    "delta_LCC",
                    "delta_ASP",
                ]
            )

        df = pd.DataFrame(rows)
        return df.sort_values(by=["dataset", "experiment", "method", "k"]).reset_index(drop=True)

    @staticmethod
    def build_fixed_node_budget_table(metrics_dir: str | Path) -> pd.DataFrame:
        metrics_dir = Path(metrics_dir)
        rows: List[Dict[str, Any]] = []

        for path in metrics_dir.glob("*_rule_metrics.json"):
            rows.extend(MetricsReporter.parse_fixed_node_budget_metrics(path))

        if not rows:
            return pd.DataFrame(
                columns=[
                    "dataset",
                    "source_file",
                    "experiment",
                    "method",
                    "budget",
                    "delta_E",
                    "delta_LCC",
                    "delta_ASP",
                    "fragility_score",
                    "num_removed_nodes",
                ]
            )

        df = pd.DataFrame(rows)
        return df.sort_values(
            by=["dataset", "method", "budget"]
        ).reset_index(drop=True)
    @staticmethod
    def _flatten_path_records(
        dataset: str,
        method: str,
        records: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for rank_idx, r in enumerate(records[:top_n], start=1):
            features = r.get("features", {}) or {}
            fragility = r.get("fragility", {}) or {}

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "rank": rank_idx,
                    "source": r.get("source"),
                    "target": r.get("target"),
                    "nodes": " -> ".join(map(str, r.get("nodes", []))),
                    "num_nodes_in_path": len(r.get("nodes", [])),
                    "score": MetricsReporter._safe_float(r.get("score")),
                    "path_length": MetricsReporter._safe_float(
                        features.get("path_length"),
                        len(r.get("nodes", [])),
                    ),
                    "cross_comm_ratio": MetricsReporter._safe_float(features.get("cross_comm_ratio")),
                    "avg_node_importance": MetricsReporter._safe_float(features.get("avg_node_importance")),
                    "avg_edge_bc": MetricsReporter._safe_float(features.get("avg_edge_bc")),
                    "delta_E": MetricsReporter._safe_float(fragility.get("delta_E")),
                    "delta_LCC": MetricsReporter._safe_float(fragility.get("delta_LCC")),
                    "delta_ASP": MetricsReporter._safe_float(fragility.get("delta_ASP")),
                    "fragility_score": MetricsReporter._safe_float(fragility.get("fragility_score")),
                }
            )
        return rows

    @staticmethod
    def build_path_quality_table(paths_dir: str | Path, top_n: int = 10) -> pd.DataFrame:
        paths_dir = Path(paths_dir)
        rows: List[Dict[str, Any]] = []

        # rule paths: dict[method -> records]
        for path in paths_dir.glob("*_rule_paths.json"):
            obj = MetricsReporter.load_json(path)
            dataset = path.stem.replace("_rule_paths", "")
            for method, records in obj.items():
                if isinstance(records, list):
                    rows.extend(MetricsReporter._flatten_path_records(dataset, method, records, top_n))

        # rank paths: list[records]
        for path in paths_dir.glob("*rank*paths.json"):
            name = path.name
            if "_rl_eval_" in name:
                continue
            if "_rule_" in name:
                continue

            records = MetricsReporter.load_json(path)

            if "pure_rank" in name:
                method = "rank_pure"
            elif "pred_score_selector" in name:
                method = "rank_set_pred_score_selector"
            elif "submodular_greedy" in name:
                method = "rank_set_submodular_greedy"
            else:
                method = "rank"

            stem = path.stem
            dataset_raw = stem.split("_rank_")[0]

            dataset_map = {
                "cora": "Cora",
                "citeseer": "Citeseer",
                "pubmed": "Pubmed",
                "computers": "Computers",
                "photo": "Photo",
                "cs": "CS",
                "physics": "Physics",
            }

            dataset = dataset_map.get(dataset_raw.lower(), dataset_raw)
            if isinstance(records, list):
                rows.extend(
                    MetricsReporter._flatten_path_records(
                        dataset=dataset,
                        method=method,
                        records=records,
                        top_n=top_n,
                    )
                )

        # rl eval paths: dict[method -> records]
        for path in paths_dir.glob("*_rl_eval_paths.json"):
            obj = MetricsReporter.load_json(path)
            dataset = path.stem.replace("_rl_eval_paths", "")
            for method, records in obj.items():
                if isinstance(records, list):
                    rows.extend(MetricsReporter._flatten_path_records(dataset, method, records, top_n))

        if not rows:
            return pd.DataFrame(
                columns=[
                    "dataset",
                    "method",
                    "rank",
                    "source",
                    "target",
                    "nodes",
                    "num_nodes_in_path",
                    "score",
                    "path_length",
                    "cross_comm_ratio",
                    "avg_node_importance",
                    "avg_edge_bc",
                    "delta_E",
                    "delta_LCC",
                    "delta_ASP",
                    "fragility_score",
                ]
            )

        df = pd.DataFrame(rows)
        return df.sort_values(by=["dataset", "method", "rank"]).reset_index(drop=True)

    @staticmethod
    def save_table(df: pd.DataFrame, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")