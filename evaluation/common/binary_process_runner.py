"""Shared runner for binary process diagnostics (perception/reasoning)."""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import yaml
from tqdm import tqdm

from data.loader import load_jsonl
from evaluation.common.judge_client import load_judge_clients
from evaluation.common.parsing import aux_image_path, parse_output, strip_image_tags


def _pred_id(p: dict) -> str:
    return str(p.get("id", p.get("index", "")))


def _parent_id(p: dict) -> str:
    return str(p.get("parent_id", _pred_id(p)))


def _resolve(path: str, base: Path) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    anchor = "geo-aux-bench/"
    idx = path.rfind(anchor)
    if idx >= 0:
        candidate = Path(__file__).resolve().parents[2] / path[idx + len(anchor):]
        if candidate.exists():
            return str(candidate)
    candidates = [base / path, base / p.name, base / "aux_images" / p.name]
    for c in candidates:
        if c.exists():
            return str(c)
    for hit in base.rglob(p.name):
        return str(hit)
    return str(p)


def _images_from_prediction(p: dict, bench_item: dict | None, base: Path, bench_base: Path | None = None) -> tuple[list[str], list[str]]:
    raw_input = p.get("input_images") or p.get("image") or []
    if isinstance(raw_input, str):
        raw_input = [raw_input]
    original = [_resolve(str(raw_input[0]), base)] if raw_input else []
    if (not original or not Path(original[0]).exists()) and bench_item:
        images = bench_item.get("images") or []
        if images:
            original = [_resolve(str(images[0]), bench_base or base)]
    raw_aux = p.get("images") or []
    if isinstance(raw_aux, str):
        raw_aux = [raw_aux]
    parsed = parse_output(str(p.get("text", "")))
    aux: list[str] = []
    indices = parsed.aux_indices or list(range(1, len(raw_aux) + 1))
    for k in indices:
        path = aux_image_path(raw_aux, k)
        if path:
            resolved = _resolve(str(path), base)
            if Path(resolved).exists():
                aux.append(resolved)
    return original, aux


def _question(p: dict, bench_item: dict | None) -> str:
    if bench_item:
        return str(bench_item.get("query", ""))
    return str(p.get("query", p.get("question", "")))


def _reasoning(p: dict) -> str:
    # Keep the complete model response, including text after </think>. The
    # judges are instructed to ignore the final-answer correctness itself, while
    # retaining the full response prevents us from dropping valid process text.
    text = str(p.get("text", ""))
    parsed = parse_output(text)
    combined = parsed.think
    if parsed.final.strip():
        combined += "\n" + parsed.final
    return strip_image_tags(combined).strip()


def run_binary(
    *, predictions_path: str, output_dir: str, config_path: str,
    bench_path: str | None, metric_name: str,
    judge_fn: Callable, use_aux_images: bool,
    qps: float | None = None, workers: int | None = None,
) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    judge_cfg = cfg.get("judge", {})
    effective_qps = float(judge_cfg.get("qps", 0) if qps is None else qps)
    configured_workers = judge_cfg.get("workers")
    effective_workers = int(workers if workers is not None else (
        configured_workers if configured_workers is not None
        else (max(1, math.ceil(effective_qps)) if effective_qps > 0 else 1)))
    if effective_qps < 0 or effective_workers < 1:
        raise ValueError("qps must be >= 0 and workers must be >= 1")

    preds = load_jsonl(predictions_path)
    bench = load_jsonl(bench_path) if bench_path else []
    gt = {str(x.get("id")): x for x in bench}
    base = Path(predictions_path).parent
    bench_base = Path(bench_path).parent if bench_path else base
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = str(out / "judge_cache")
    vision_judge, _text_judge, cache = load_judge_clients(
        config_path, cache_dir_override=cache_dir, qps_override=effective_qps)

    def work(p: dict) -> dict:
        pid = _pred_id(p)
        item = gt.get(_parent_id(p))
        original, aux = _images_from_prediction(p, item, base, bench_base)
        reasoning = _reasoning(p)
        if not original or not Path(original[0]).exists():
            return {"id": pid, "skipped": True, "skip_reason": "original image missing"}
        if not reasoning:
            return {"id": pid, "skipped": True, "skip_reason": "reasoning missing"}
        try:
            if metric_name == "perception_correct":
                verdict = judge_fn(original_image=original[0], question=_question(p, item),
                                   reasoning=reasoning, vision_judge=vision_judge, cache=cache)
            else:
                verdict = judge_fn(images=original + (aux if use_aux_images else []),
                                   question=_question(p, item), reasoning=reasoning,
                                   vision_judge=vision_judge, cache=cache)
            return {"id": pid, "skipped": False, **verdict}
        except Exception as exc:
            return {"id": pid, "skipped": True, "skip_reason": f"judge error: {str(exc)[:300]}"}

    executor = None
    if effective_workers == 1:
        results = map(work, preds)
    else:
        executor = ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix=metric_name)
        results = executor.map(work, preds)
    n_total = n_judged = n_skipped = n_correct = 0
    error_types: dict[str, int] = {}
    path = out / "per_item.jsonl"
    try:
        with path.open("w", encoding="utf-8") as f:
            for row in tqdm(results, total=len(preds), desc=metric_name, unit="item", dynamic_ncols=True):
                n_total += 1
                if row.get("skipped"):
                    n_skipped += 1
                else:
                    n_judged += 1
                    n_correct += int(row.get(metric_name, 0))
                    for error in row.get("errors", []):
                        if isinstance(error, dict):
                            typ = str(error.get("error_type", "unknown"))
                            error_types[typ] = error_types.get(typ, 0) + 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    summary = {
        "metric": metric_name, "n_total": n_total, "n_judged": n_judged,
        "n_skipped": n_skipped, "n_correct": n_correct,
        "accuracy": n_correct / n_judged if n_judged else 0.0,
        "error_type_counts": error_types, "qps": effective_qps, "workers": effective_workers,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--bench", required=False)
    ap.add_argument("--output-dir", required=False)
    ap.add_argument("--qps", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
