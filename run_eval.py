#!/usr/bin/env python3
"""Unified evaluation driver for the five GeoVAD-Bench levels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _find_predictions(base_dir: str) -> str:
    path = Path(base_dir) / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {base_dir}")
    return str(path)


def _file_checksum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _print_summary(level: str, summary: dict):
    print(f"\n{'=' * 60}\n  Level {level} Summary\n{'=' * 60}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()


def _load_cached_summary(level: int, predictions: str, output_dir: str) -> dict | None:
    """Return a completed result when the output matches this predictions file."""
    per_item = Path(output_dir) / "per_item.jsonl"
    summary_path = Path(output_dir) / "summary.json"
    checksum_path = Path(output_dir) / ".predictions_checksum"
    if not (per_item.exists() and summary_path.exists() and checksum_path.exists()):
        return None
    if checksum_path.read_text(encoding="utf-8").strip() != _file_checksum(predictions):
        return None
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    print(f"[L{level}] Skipping — results unchanged: {per_item}")
    return summary


def _save_predictions_checksum(predictions: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / ".predictions_checksum").write_text(
        _file_checksum(predictions) + "\n", encoding="utf-8"
    )


def _run_binary(predictions, config, bench, output_dir, judge_fn, metric_name, use_aux_images, label, qps, workers):
    from evaluation.common.binary_process_runner import run_binary

    print(f"[L{label}] Starting — output: {output_dir}")
    t0 = time.time()
    summary = run_binary(
        predictions_path=predictions,
        output_dir=output_dir,
        config_path=config,
        bench_path=bench,
        metric_name=metric_name,
        judge_fn=judge_fn,
        use_aux_images=use_aux_images,
        qps=qps,
        workers=workers,
    )
    print(f"[L{label}] Done in {time.time() - t0:.1f}s")
    return summary


def run_level1(predictions, config, bench, output_dir, qps, workers):
    cached = _load_cached_summary(1, predictions, output_dir)
    if cached is not None:
        return cached
    from evaluation.level1.judge import judge_perception
    summary = _run_binary(predictions, config, bench, output_dir, judge_perception,
                          "perception_correct", False, 1, qps, workers)
    _save_predictions_checksum(predictions, output_dir)
    return summary


def run_level2(predictions, config, output_dir, qps, workers):
    cached = _load_cached_summary(2, predictions, output_dir)
    if cached is not None:
        return cached
    from evaluation.level2.runner import run
    print(f"[L2] Starting — output: {output_dir}")
    t0 = time.time()
    summary = run(predictions_path=predictions, output_dir=output_dir,
                  config_path=config, qps=qps, workers=workers)
    _save_predictions_checksum(predictions, output_dir)
    print(f"[L2] Done in {time.time() - t0:.1f}s")
    return summary


def run_level3(predictions, config, output_dir, l2_dir, qps, workers):
    cached = _load_cached_summary(3, predictions, output_dir)
    if cached is not None:
        return cached
    from evaluation.level3.runner import run
    print(f"[L3] Starting — output: {output_dir}")
    t0 = time.time()
    summary = run(predictions_path=predictions, output_dir=output_dir,
                  config_path=config, l2_dir=l2_dir, qps=qps, workers=workers)
    _save_predictions_checksum(predictions, output_dir)
    print(f"[L3] Done in {time.time() - t0:.1f}s")
    return summary


def run_level4(predictions, config, bench, output_dir, qps, workers):
    cached = _load_cached_summary(4, predictions, output_dir)
    if cached is not None:
        return cached
    from evaluation.level4.judge import judge_reasoning
    summary = _run_binary(predictions, config, bench, output_dir, judge_reasoning,
                          "reasoning_correct", True, 4, qps, workers)
    _save_predictions_checksum(predictions, output_dir)
    return summary


def run_level5(predictions, config, bench, output_dir, qps, workers):
    from evaluation.level5.runner import run

    cached = _load_cached_summary(5, predictions, output_dir)
    if cached is not None:
        return cached

    print(f"[L5] Starting — output: {output_dir}")
    t0 = time.time()
    summary = run(predictions_path=predictions, bench_path=bench,
                  output_dir=output_dir, config_path=config,
                  use_llm_fallback=True, qps=qps, workers=workers)
    _save_predictions_checksum(predictions, output_dir)
    print(f"[L5] Done in {time.time() - t0:.1f}s")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Unified GeoVAD-Bench evaluation (L1–L5)")
    ap.add_argument("--predictions", required=True, help="Directory containing predictions.jsonl and aux_images/")
    ap.add_argument("--config", required=True, help="YAML config")
    ap.add_argument("--bench", required=True, help="Path to bench.jsonl")
    ap.add_argument("--output-dir", required=True, help="Root output directory for results")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    choices=[1, 2, 3, 4, 5], help="Which levels to run")
    ap.add_argument("--include-perception-reasoning", "--with-process-metrics",
                    dest="include_process_metrics", action="store_true",
                    help="Also run Level 1 and Level 4 process-quality evaluation")
    ap.add_argument("--qps", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.qps is not None and args.qps < 0:
        ap.error("--qps must be >= 0")
    if args.workers is not None and args.workers < 1:
        ap.error("--workers must be >= 1")

    levels = set(args.levels)
    if args.include_process_metrics:
        levels.update({1, 4})
    if 3 in levels:
        levels.add(2)
    levels = sorted(levels)

    predictions_dir = os.path.abspath(args.predictions)
    predictions_file = _find_predictions(predictions_dir)
    config = os.path.abspath(args.config)
    bench = os.path.abspath(args.bench)
    out_root = os.path.abspath(args.output_dir)
    os.makedirs(out_root, exist_ok=True)

    print(f"Predictions : {predictions_dir}\nConfig      : {config}\nBench       : {bench}")
    print(f"Output      : {out_root}\nLevels      : {levels}")

    dirs = {level: os.path.join(out_root, f"level{level}") for level in range(1, 6)}
    results = {}
    t_all = time.time()
    for level in levels:
        if level == 1:
            results[1] = run_level1(predictions_file, config, bench, dirs[1], args.qps, args.workers)
        elif level == 2:
            results[2] = run_level2(predictions_file, config, dirs[2], args.qps, args.workers)
        elif level == 3:
            results[3] = run_level3(predictions_file, config, dirs[3], dirs[2], args.qps, args.workers)
        elif level == 4:
            results[4] = run_level4(predictions_file, config, bench, dirs[4], args.qps, args.workers)
        elif level == 5:
            results[5] = run_level5(predictions_file, config, bench, dirs[5], args.qps, args.workers)
        _print_summary(str(level), results[level])

    combined_path = os.path.join(out_root, "all_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({f"level{k}": v for k, v in results.items()}, f, indent=2, ensure_ascii=False)
    print(f"All done — {len(results)} level(s) in {time.time() - t_all:.1f}s")
    print(f"Combined summary: {combined_path}")


if __name__ == "__main__":
    main()
