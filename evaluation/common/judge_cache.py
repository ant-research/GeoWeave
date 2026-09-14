"""Append-only JSONL cache for judge calls.

Key = sha256(model + system + user + sorted image sha256s). Value = response (str or dict).
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Optional


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def make_key(
    *,
    model: str,
    system: str,
    user: str,
    image_paths: Optional[list[str]] = None,
    extra: str = "",
) -> str:
    parts = [model, system, user, extra]
    for p in sorted(image_paths or []):
        parts.append(_hash_file(p))
    blob = "".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class JudgeCache:
    """File-backed K/V cache. One jsonl per shard (by key prefix) to bound file size."""

    def __init__(self, root: str, shards: int = 16):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shards = shards
        self._mem: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._loaded = False

    def _shard_path(self, key: str) -> Path:
        idx = int(key[:2], 16) % self.shards
        return self.root / f"cache_{idx:02d}.jsonl"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            for p in self.root.glob("cache_*.jsonl"):
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            self._mem[rec["key"]] = rec["value"]
                        except Exception:
                            continue
            self._loaded = True

    def get(self, key: str) -> Optional[Any]:
        self._ensure_loaded()
        with self._lock:
            return self._mem.get(key)

    def put(self, key: str, value: Any) -> None:
        self._ensure_loaded()
        with self._lock:
            if key in self._mem:
                return
            self._mem[key] = value
            with open(self._shard_path(key), "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
