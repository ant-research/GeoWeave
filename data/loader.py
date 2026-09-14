"""
Data loader for geo_aux_bench.
Supports two data formats:
  - Level 1-4 (geo-aux): JSONL with images/query/thinking/answer
"""
import json
from pathlib import Path
from typing import List, Optional


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, data: List[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_annotation_json(path: str, num_samples: int = -1) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if num_samples > 0:
        data = data[:num_samples]
    return data


def filter_by_task_type(samples: List[dict], task_types: List[str]) -> List[dict]:
    return [s for s in samples if s.get("problem_version") in task_types]


def filter_by_source(samples: List[dict], sources: List[str]) -> List[dict]:
    return [s for s in samples if s.get("source") in sources]
