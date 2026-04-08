from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class SimpleMemoryCache:
    def __init__(self):
        self._store: dict[str, Any] = {}

    @staticmethod
    def _normalize_key(key: Any) -> str:
        if isinstance(key, (str, int, float, bool)):
            return str(key)
        return json.dumps(key, ensure_ascii=False, sort_keys=True, default=str)

    def has(self, key: Any) -> bool:
        return self._normalize_key(key) in self._store

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        return self._store.get(self._normalize_key(key), default)

    def set(self, key: Any, value: Any) -> None:
        self._store[self._normalize_key(key)] = value

    def clear(self) -> None:
        self._store.clear()


class SimpleDiskCache:
    """
    key -> md5(key).json
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_key(key: Any) -> str:
        if isinstance(key, (str, int, float, bool)):
            raw = str(key)
        else:
            raw = json.dumps(key, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _key_to_path(self, key: Any) -> Path:
        return self.cache_dir / f"{self._normalize_key(key)}.json"

    def has(self, key: Any) -> bool:
        return self._key_to_path(key).exists()

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        path = self._key_to_path(key)
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: Any, value: Any) -> None:
        path = self._key_to_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        for p in self.cache_dir.glob("*.json"):
            p.unlink()