"""Level 5 runner: read predictions + bench GT, score answers, write outputs.

Rule scoring and optional LLM fallback run concurrently across samples. All LLM
requests share one process-local QPS limiter.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm

from data.loader import load_jsonl
from evaluation.common.judge_client import load_judge_clients
from evaluation.common.parsing import parse_output
from evaluation.level5.answer_extractor import extract_pred
from evaluation.level5.scorer import score as score_one


def _index_bench(bench: list[dict]) -> dict[str, dict]:
    return {str(sample["id"]): sample for sample in bench}


def _pred_to_id(p: dict) -> str:
    """Prefer ``id``; otherwise fall back to ``index``."""
    if "id" in p:
        return str(p["id"])
    return str(p.get("index", ""))


def _pred_to_parent_id(p: dict) -> str:
    """Map a candidate prediction back to the original benchmark sample."""
    return str(p.get("parent_id", _pred_to_id(p)))


def _evaluate_one(p: dict, *, gt_by_id: dict[str, dict], text_judge, cache) -> tuple[str, str, dict | None]:
    """Evaluate one prediction. Safe to call from a worker thread."""
    pid = _pred_to_id(p)
    parent_id = _pred_to_parent_id(p)
    gt_item = gt_by_id.get(parent_id)
    if gt_item is None:
        return "missing_gt", pid, None

    gt = str(gt_item.get("answer", "")).strip()
    if not gt:
        return "no_gt", pid, None

    text = p.get("text", "")
    parsed = parse_output(text)
    # A closed </think> makes the following text authoritative.  Extract the
    # complete final answer there (including any boxed answer) rather than
    # preferring an intermediate boxed expression from the reasoning trace.
    malformed_think = bool(
        re.search(r"<think\b", text, flags=re.IGNORECASE)
        and not re.search(r"</think>", text, flags=re.IGNORECASE)
    )
    if parsed.final.strip():
        pred_raw = extract_pred(parsed.final)
    elif malformed_think:
        # An unclosed think block is a truncated/malformed generation, not a
        # final answer.  Do not guess from its reasoning or ask an extractor
        # model to guess from the truncated trace.
        pred_raw = ""
    else:
        pred_raw = extract_pred(parsed.think)

    result = score_one(
        pred_raw=pred_raw,
        gt=gt,
        text_judge=text_judge,
        cache=cache,
        question=str(gt_item.get("query", "")),
        response_text="" if malformed_think else text,
    )
    result["id"] = pid
    result["parent_id"] = parent_id
    result["candidate_index"] = int(p.get("candidate_index", 0))
    result["pass_n"] = int(p.get("pass_n", 1))
    result["has_aux"] = len(parsed.aux_indices) > 0
    return "scored", pid, result


def run(
    *,
    predictions_path: str,
    bench_path: str,
    output_dir: str,
    config_path: str,
    use_llm_fallback: bool = True,
    cache_dir: str | None = None,
    qps: float | None = None,
    workers: int | None = None,
) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    judge_cfg = cfg.get("judge", {})
    effective_qps = float(judge_cfg.get("qps", 0) if qps is None else qps)
    configured_workers = judge_cfg.get("workers")
    effective_workers = int(
        workers if workers is not None else (
            configured_workers if configured_workers is not None
            else (max(1, math.ceil(effective_qps)) if effective_qps > 0 else 1)
        )
    )
    if effective_qps < 0:
        raise ValueError(f"qps must be >= 0, got {effective_qps}")
    if effective_workers < 1:
        raise ValueError(f"workers must be >= 1, got {effective_workers}")

    preds = load_jsonl(predictions_path)
    bench = load_jsonl(bench_path)
    gt_by_id = _index_bench(bench)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = str(Path(output_dir) / "judge_cache")

    text_judge = None
    cache = None
    if use_llm_fallback:
        _, text_judge, cache = load_judge_clients(
            config_path,
            cache_dir_override=cache_dir,
            qps_override=effective_qps,
        )

    def work(p: dict) -> tuple[str, str, dict | None]:
        return _evaluate_one(p, gt_by_id=gt_by_id, text_judge=text_judge, cache=cache)

    if effective_workers == 1:
        results_iter = map(work, preds)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="level5")
        results_iter = executor.map(work, preds)

    n_correct = n_total = 0
    method_counts: dict[str, int] = {}
    extraction_counts: dict[str, int] = {}
    missing_gt: list[str] = []
    parent_results: dict[str, list[dict]] = {}
    per_path = Path(output_dir) / "per_item.jsonl"

    try:
        with per_path.open("w", encoding="utf-8") as pf:
            pbar = tqdm(results_iter, total=len(preds), desc="level5", unit="item", dynamic_ncols=True)
            for status, pid, result in pbar:
                if status == "missing_gt":
                    missing_gt.append(pid)
                    continue
                if status == "no_gt":
                    continue
                assert result is not None
                pf.write(json.dumps(result, ensure_ascii=False) + "\n")
                pf.flush()
                parent_results.setdefault(result["parent_id"], []).append(result)

                n_total += 1
                n_correct += int(result["correct"])
                method = result["method"]
                method_counts[method] = method_counts.get(method, 0) + 1
                extraction_method = result.get("extraction_method", "rule")
                extraction_counts[extraction_method] = extraction_counts.get(extraction_method, 0) + 1
                acc = n_correct / n_total if n_total else 0.0
                pbar.set_postfix(acc=f"{acc:.3f}", correct=n_correct, total=n_total)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    parent_rows: list[dict] = []
    declared_pass_n = 1
    n_parent_passed = 0
    n_incomplete_parents = 0
    for parent_id, candidates in parent_results.items():
        candidates.sort(key=lambda row: (row.get("candidate_index", 0), row["id"]))
        expected_n = max(int(row.get("pass_n", 1)) for row in candidates)
        declared_pass_n = max(declared_pass_n, expected_n)
        correct_candidates = [row for row in candidates if row["correct"]]
        passed = bool(correct_candidates)
        n_parent_passed += int(passed)
        n_incomplete_parents += int(len(candidates) < expected_n)
        parent_rows.append({
            "id": parent_id,
            "pass": passed,
            "pass_n": expected_n,
            "n_candidates": len(candidates),
            "n_correct_candidates": len(correct_candidates),
            "candidate_ids": [row["id"] for row in candidates],
            "correct_candidate_indices": [
                row.get("candidate_index", 0) for row in correct_candidates
            ],
        })
    parent_rows.sort(key=lambda row: row["id"])
    parent_path = Path(output_dir) / "per_parent.jsonl"
    with parent_path.open("w", encoding="utf-8") as pf:
        for row in parent_rows:
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_parent_total = len(parent_rows)
    pass_at_n = n_parent_passed / n_parent_total if n_parent_total else 0.0

    summary = {
        # Candidate-level fields retained for backward compatibility.
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_total) if n_total else 0.0,
        "n_candidates_total": n_total,
        "n_candidates_correct": n_correct,
        "candidate_accuracy": (n_correct / n_total) if n_total else 0.0,
        # Parent-level pass@N: a sample passes when any candidate is correct.
        "pass_n": declared_pass_n,
        "n_parent_total": n_parent_total,
        "n_parent_passed": n_parent_passed,
        "pass_at_n": pass_at_n,
        "n_incomplete_parents": n_incomplete_parents,
        "per_parent_path": str(parent_path),
        "method_counts": method_counts,
        "extraction_counts": extraction_counts,
        "missing_gt": missing_gt,
        "qps": effective_qps,
        "workers": effective_workers,
    }
    with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--bench", required=False, help="Override bench.jsonl path; defaults to config.level5.dataset_path")
    ap.add_argument("--output-dir", required=False, help="Override output dir; defaults to config.level5.output_dir")
    ap.add_argument("--no-llm-fallback", action="store_true")
    ap.add_argument("--qps", type=float, default=None, help="Override config judge.qps; 0 disables limiting.")
    ap.add_argument("--workers", type=int, default=None, help="Override config judge.workers.")
    args = ap.parse_args()

    if args.qps is not None and args.qps < 0:
        ap.error("--qps must be >= 0")
    if args.workers is not None and args.workers < 1:
        ap.error("--workers must be >= 1")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    bench_path = args.bench or cfg.get("level5", {}).get("dataset_path", "")
    output_dir = args.output_dir or cfg.get("level5", {}).get("output_dir", "./results/level5")
    if not bench_path:
        raise SystemExit("bench path is required (set level5.dataset_path or pass --bench)")

    summary = run(
        predictions_path=args.predictions,
        bench_path=bench_path,
        output_dir=output_dir,
        config_path=args.config,
        use_llm_fallback=not args.no_llm_fallback,
        qps=args.qps,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
