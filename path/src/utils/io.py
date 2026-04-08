from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
    path = ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def resolve_under_root(root: str | Path, path_str: str | Path) -> str:
    """
    If path_str is relative, resolve it under root.
    If already absolute, return as-is.
    """
    root = Path(root)
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(root / p)