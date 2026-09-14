"""Reasoning-process factual correctness evaluation."""
from __future__ import annotations
import argparse, json
from evaluation.common.binary_process_runner import add_common_args, run_binary
from evaluation.level4.judge import judge_reasoning

def main():
    ap = argparse.ArgumentParser(description="Reasoning process correctness evaluation")
    add_common_args(ap); args = ap.parse_args()
    if args.qps is not None and args.qps < 0: ap.error("--qps must be >= 0")
    if args.workers is not None and args.workers < 1: ap.error("--workers must be >= 1")
    with open(args.config, encoding="utf-8") as f: import yaml; cfg = yaml.safe_load(f) or {}
    output = args.output_dir or cfg.get("level4", {}).get("output_dir", "./results/level4")
    print(json.dumps(run_binary(predictions_path=args.predictions, output_dir=output,
        config_path=args.config, bench_path=args.bench, metric_name="reasoning_correct",
        judge_fn=judge_reasoning, use_aux_images=True, qps=args.qps, workers=args.workers),
        indent=2, ensure_ascii=False))
