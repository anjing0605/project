from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
'''从 base yaml 生成消融 yaml
修改输出文件名，避免覆盖
提供构造消融配置的函数
读取消融指标结果，形成表格'''


class AblationManager:
    """
    Build, save, and summarize ablation configurations/results.
    """

    @staticmethod
    def load_yaml(path: str | Path) -> Dict[str, Any]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def dump_yaml(path: str | Path, obj: Dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def set_output_tag(cfg: Dict[str, Any], suffix: str) -> Dict[str, Any]:
        cfg = copy.deepcopy(cfg)

        if "output" not in cfg:
            return cfg

        for k, v in cfg["output"].items():
            if isinstance(v, str):
                p = Path(v)
                stem = p.stem
                suffixes = "".join(p.suffixes)
                parent = str(p.parent).replace("\\", "/")
                cfg["output"][k] = f"{parent}/{stem}_{suffix}{suffixes}" if parent != "." else f"{stem}_{suffix}{suffixes}"

        return cfg

    @staticmethod
    def build_rule_ablations(base_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        out: List[Tuple[str, Dict[str, Any]]] = []

        # 1) 去掉 fragility_score
        cfg = copy.deepcopy(base_cfg)
        cfg["rule_weights"]["fragility_score"] = 0.0
        out.append(("no_fragility_score", AblationManager.set_output_tag(cfg, "no_fragility_score")))

        # 2) 去掉 cross_comm_ratio
        cfg = copy.deepcopy(base_cfg)
        cfg["rule_weights"]["cross_comm_ratio"] = 0.0
        out.append(("no_cross_comm_ratio", AblationManager.set_output_tag(cfg, "no_cross_comm_ratio")))

        # 3) 去掉 avg_edge_bc
        cfg = copy.deepcopy(base_cfg)
        cfg["rule_weights"]["avg_edge_bc"] = 0.0
        out.append(("no_avg_edge_bc", AblationManager.set_output_tag(cfg, "no_avg_edge_bc")))

        # 4) 改 topk
        for k in [20, 50]:
            cfg = copy.deepcopy(base_cfg)
            cfg["keynode"]["k"] = k
            out.append((f"topk_{k}", AblationManager.set_output_tag(cfg, f"topk_{k}")))

        # 5) 改 max_pairs
        for max_pairs in [100, 300]:
            cfg = copy.deepcopy(base_cfg)
            cfg["keynode"]["max_pairs"] = max_pairs
            out.append((f"maxpairs_{max_pairs}", AblationManager.set_output_tag(cfg, f"maxpairs_{max_pairs}")))

        # 6) 改 top_q
        for top_q in [5, 20]:
            cfg = copy.deepcopy(base_cfg)
            cfg["paths"]["top_q"] = top_q
            out.append((f"topq_{top_q}", AblationManager.set_output_tag(cfg, f"topq_{top_q}")))

        return out

    @staticmethod
    def load_json(path: str | Path) -> Dict[str, Any]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def build_ablation_summary(metrics_dir: str | Path) -> pd.DataFrame:
        metrics_dir = Path(metrics_dir)
        rows: List[Dict[str, Any]] = []

        for path in metrics_dir.glob("*_rule_metrics_*.json"):
            obj = AblationManager.load_json(path)
            dataset = obj["dataset"]["name"]
            comparison = obj.get("comparison", {})

            if "rule" not in comparison:
                continue

            metrics = comparison["rule"]
            if "delta_E_curve" not in metrics:
                continue

            delta_E_curve = metrics.get("delta_E_curve", [])
            delta_LCC_curve = metrics.get("delta_LCC_curve", [])
            delta_ASP_curve = metrics.get("delta_ASP_curve", [])
            k_list = metrics.get("k_list", list(range(1, len(delta_E_curve) + 1)))

            # 从文件名中剥离消融名
            # 例如: cora_rule_metrics_no_fragility_score.json
            stem = path.stem
            marker = "_rule_metrics_"
            if marker in stem:
                ablation_name = stem.split(marker, 1)[1]
            else:
                ablation_name = stem

            n = min(len(k_list), len(delta_E_curve), len(delta_LCC_curve), len(delta_ASP_curve))
            for i in range(n):
                rows.append(
                    {
                        "dataset": dataset,
                        "ablation": ablation_name,
                        "method": "rule",
                        "k": int(k_list[i]),
                        "delta_E": float(delta_E_curve[i]),
                        "delta_LCC": float(delta_LCC_curve[i]),
                        "delta_ASP": float(delta_ASP_curve[i]),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=["dataset", "ablation", "method", "k", "delta_E", "delta_LCC", "delta_ASP"]
            )

        df = pd.DataFrame(rows)
        return df.sort_values(by=["dataset", "ablation", "k"]).reset_index(drop=True)