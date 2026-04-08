from __future__ import annotations

import json
from pathlib import Path

from path.src.analysis.plotting import Plotter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    logs_dir = PATH_ROOT / "outputs" / "logs"
    save_dir = PATH_ROOT / "outputs" / "figures" / "rl_curves"

    log_files = list(logs_dir.glob("*_rl_train_log.json"))
    if not log_files:
        raise RuntimeError(f"No RL train log found under: {logs_dir}")

    for log_path in log_files:
        tag = log_path.stem.replace("_rl_train_log", "")
        log_obj = load_json(log_path)
        Plotter.plot_rl_training_curves_from_log(
            log_obj=log_obj,
            tag=tag,
            save_dir=save_dir,
        )
        print(f"[DONE] plotted RL curves for: {tag}")

    print(f"[DONE] all RL figures saved under: {save_dir}")


if __name__ == "__main__":
    main()