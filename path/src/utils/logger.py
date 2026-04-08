from __future__ import annotations

from datetime import datetime


class SimpleLogger:
    def __init__(self, name: str = "LOG"):
        self.name = name

    def _fmt(self, level: str, msg: str) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{ts}] [{self.name}] [{level}] {msg}"

    def info(self, msg: str) -> None:
        print(self._fmt("INFO", msg))

    def warning(self, msg: str) -> None:
        print(self._fmt("WARN", msg))

    def error(self, msg: str) -> None:
        print(self._fmt("ERROR", msg))