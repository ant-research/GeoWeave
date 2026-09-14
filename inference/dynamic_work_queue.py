"""Small cross-process FIFO counter for single-node inference launchers.

Workers are independent subprocesses, so a file protected by ``fcntl.flock``
provides a lightweight shared queue without introducing a manager/server
process.  The launcher resets the counter before spawning workers; each worker
atomically claims the next benchmark index after finishing its previous item.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path


def initialize_dynamic_queue(path: str | os.PathLike[str]) -> Path:
    """Reset ``path`` to the first task and return its absolute path."""
    queue_path = Path(path).expanduser().resolve()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = queue_path.parent / f".{queue_path.name}.tmp.{os.getpid()}"
    tmp_path.write_text("0\n", encoding="utf-8")
    tmp_path.replace(queue_path)
    return queue_path


class DynamicWorkQueue:
    """Atomically hand out monotonically increasing benchmark indices."""

    def __init__(self, path: str | os.PathLike[str], total_items: int) -> None:
        self.path = Path(path).expanduser().resolve()
        self.total_items = int(total_items)
        if self.total_items < 0:
            raise ValueError(f"total_items must be non-negative, got {total_items}")

    def claim(self) -> int | None:
        """Return the next index, or ``None`` after all indices are claimed."""
        with self.path.open("r+", encoding="utf-8") as queue_file:
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
            raw = queue_file.read().strip()
            next_index = int(raw or "0")
            if next_index >= self.total_items:
                return None

            queue_file.seek(0)
            queue_file.write(f"{next_index + 1}\n")
            queue_file.truncate()
            queue_file.flush()
            return next_index

    def iter_indices(self):
        while True:
            index = self.claim()
            if index is None:
                return
            yield index
