from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class TBLogger:
    """
    Lightweight TensorBoard logger wrapper.

    Features:
    - unified log_dir creation
    - scalar / scalars / text logging
    - config dump
    - local artifact saving (json / text)
    - safe no-op if tensorboard is unavailable
    """

    def __init__(
        self,
        log_root: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and (SummaryWriter is not None)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if run_name is None:
            final_run_name = timestamp
        else:
            final_run_name = f"{run_name}_{timestamp}"

        self.log_dir = str(Path(log_root) / experiment_name / final_run_name)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir) if self.enabled else None

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        if self.writer is None:
            return
        if value is None:
            return
        try:
            self.writer.add_scalar(tag, float(value), step)
        except Exception:
            pass

    def add_scalars(self, scalar_dict: Dict[str, Any], step: int, prefix: str = "") -> None:
        if not scalar_dict:
            return
        for k, v in scalar_dict.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.add_scalar(tag, v, step)

    def add_text(self, tag: str, text_string: str, step: int = 0) -> None:
        if self.writer is None:
            return
        try:
            self.writer.add_text(tag, text_string, step)
        except Exception:
            pass

    def add_config(self, config: Dict[str, Any], tag: str = "config") -> None:
        """
        Save config both to TensorBoard text and local json file.
        """
        try:
            text = json.dumps(config, indent=2, ensure_ascii=False)
        except Exception:
            text = str(config)

        self.add_text(tag, f"```json\n{text}\n```", step=0)
        self.save_json("config.json", config)

    def add_histogram(self, tag: str, values: Any, step: int) -> None:
        if self.writer is None:
            return
        try:
            self.writer.add_histogram(tag, values, global_step=step)
        except Exception:
            pass

    def add_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        if self.writer is None:
            return

        safe_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                safe_hparams[k] = v
            else:
                safe_hparams[k] = str(v)

        safe_metrics = {}
        for k, v in metrics.items():
            try:
                safe_metrics[k] = float(v)
            except Exception:
                continue

        try:
            self.writer.add_hparams(safe_hparams, safe_metrics)
        except Exception:
            pass

    def save_json(self, filename: str, obj: Dict[str, Any]) -> None:
        out_path = Path(self.log_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def save_text(self, filename: str, text: str) -> None:
        out_path = Path(self.log_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten nested dict:
    {"a": {"b": 1}} -> {"a.b": 1}
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, parent_key=new_key, sep=sep))
        else:
            items[new_key] = v
    return items