from __future__ import annotations

import subprocess
from pathlib import Path

from path.src.analysis.ablation import AblationManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PATH_ROOT / "configs"
ABLATION_DIR = CONFIG_DIR / "ablation"


def run_command(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    base_rule_yamls = list(CONFIG_DIR.glob("core_rule.yaml"))
    if not base_rule_yamls:
        raise RuntimeError(f"No base rule yaml found under: {CONFIG_DIR}")

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    for base_yaml in base_rule_yamls:
        base_cfg = AblationManager.load_yaml(base_yaml)
        dataset_tag = base_yaml.stem.replace("_rule", "")

        ablations = AblationManager.build_rule_ablations(base_cfg)

        for ablation_name, ablation_cfg in ablations:
            ablation_yaml_path = ABLATION_DIR / f"{dataset_tag}_{ablation_name}.yaml"
            AblationManager.dump_yaml(ablation_yaml_path, ablation_cfg)

            run_command(
                [
                    "python",
                    "-m",
                    "path.scripts.run_rule",
                    "--config",
                    str(ablation_yaml_path),
                ]
            )

    print(f"[DONE] all ablation runs finished. configs saved under: {ABLATION_DIR}")


if __name__ == "__main__":
    main()