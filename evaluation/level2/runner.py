"""Level 2 runner: extract instructions, judge each aux image, aggregate per item.

Supports sample-level thread concurrency. Text and vision judge requests share one
process-local QPS limiter, so ``qps`` caps their combined request start rate.
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm

from data.loader import load_jsonl
from evaluation.common.judge_client import load_judge_clients
from evaluation.common.parsing import aux_image_path, parse_output
from evaluation.level2.aux_image_judge import judge_aux_image
from evaluation.level2.instruction_extractor import extract_instruction


def _pred_to_id(p: dict) -> str:
    if "id" in p:
        return str(p["id"])
    return str(p.get("index", ""))


def _repo_root() -> Path:
    """The geo-aux-bench repo root: <repo>/evaluation/level2/runner.py → <repo>."""
    return Path(__file__).resolve().parents[2]


_REPO_ANCHOR = "geo-aux-bench/"


def _remap_to_local_repo(path: str) -> str | None:
    idx = path.rfind(_REPO_ANCHOR)
    if idx < 0:
        return None
    tail = path[idx + len(_REPO_ANCHOR):]
    candidate = _repo_root() / tail
    return str(candidate) if candidate.exists() else None


def _resolve(path: str, base: Path) -> str:
    """Resolve an image path, tolerating stale absolute paths from another machine."""
    p = Path(path)
    if p.exists():
        return str(p)
    remapped = _remap_to_local_repo(path)
    if remapped:
        return remapped
    name = p.name
    candidates = [base / path, base / name, base / "aux_images" / name]
    for c in candidates:
        if c.exists():
            return str(c)
    for hit in base.rglob(name):
        return str(hit)
    return str(p)


def _evaluate_one(
    p: dict,
    *,
    base: Path,
    vision_judge,
    text_judge,
    cache,
) -> dict:
    """Evaluate one prediction. Safe to call from a worker thread."""
    pid = _pred_to_id(p)
    text = p.get("text", "")
    input_images = p.get("input_images") or p.get("image") or []
    if isinstance(input_images, str):
        input_images = [input_images]
    aux_images = p.get("images") or []

    parsed = parse_output(text)
    # Editing-converter fallback: one or more image paths but no explicit tags.
    if not parsed.aux_indices and aux_images:
        parsed.aux_indices = list(range(1, len(aux_images) + 1))

    if not aux_images:
        return {
            "id": pid,
            "no_aux": True,
            "skipped": True,
            "skip_reason": "no auxiliary image generated",
            "per_image": [],
            "avg_l1": None,
        }
    if not input_images:
        return {
            "id": pid,
            "no_aux": False,
            "skipped": True,
            "skip_reason": "original image missing",
            "per_image": [],
            "avg_l1": None,
        }

    orig = _resolve(input_images[0], base)
    per_image_scores: list[dict] = []
    item_sum = 0.0
    item_n = 0

    for k in parsed.aux_indices:
        aux_rel = aux_image_path(aux_images, k)
        if aux_rel is None:
            continue
        aux = _resolve(aux_rel, base)
        if not Path(aux).exists() or not Path(orig).exists():
            per_image_scores.append({"idx": k, "skipped": True, "reason": "image file missing"})
            continue
        try:
            instr = extract_instruction(
                think=parsed.think,
                tag_index=k,
                text_judge=text_judge,
                cache=cache,
            )
        except Exception as e:
            per_image_scores.append({
                "idx": k,
                "skipped": True,
                "reason": f"instruction extraction error: {str(e)[:200]}",
            })
            continue

        instruction = str(instr.get("instruction", "")).strip()
        instruction_found = instr.get("found") is True and bool(instruction)
        if not instruction_found:
            # L1's instruction-consistency axis is undefined without an explicit
            # construction instruction. Do not ask the vision judge to invent one
            # from the images; exclude this auxiliary image from all L1 averages.
            per_image_scores.append({
                "idx": k,
                "skipped": True,
                "reason": "construction instruction not extracted",
                "instruction": instruction,
                "instruction_found": False,
            })
            continue

        try:
            judged = judge_aux_image(
                original_image=orig,
                aux_image=aux,
                instruction=instruction,
                vision_judge=vision_judge,
                cache=cache,
            )
        except Exception as e:
            per_image_scores.append({
                "idx": k,
                "skipped": True,
                "reason": f"judge error: {str(e)[:200]}",
                "instruction": instruction,
                "instruction_found": True,
            })
            continue

        ep = judged["element_preservation"]
        ic = judged["instruction_consistency"]
        ep_norm = ep / 2.0
        ic_norm = ic / 2.0
        score = (ep_norm + ic_norm) / 2.0
        per_image_scores.append({
            "idx": k,
            "instruction": instruction,
            "instruction_found": True,
            "element_preservation": ep,
            "element_preservation_norm": ep_norm,
            "element_preservation_reason": judged["element_preservation_reason"],
            "instruction_consistency": ic,
            "instruction_consistency_norm": ic_norm,
            "instruction_consistency_reason": judged["instruction_consistency_reason"],
            "score": score,
        })
        item_sum += score
        item_n += 1

    return {
        "id": pid,
        "no_aux": False,
        "skipped": item_n == 0,
        "skip_reason": "no auxiliary image could be judged" if item_n == 0 else None,
        "per_image": per_image_scores,
        "avg_l1": (item_sum / item_n) if item_n else None,
    }


def run(
    *,
    predictions_path: str,
    output_dir: str,
    config_path: str,
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
        workers
        if workers is not None
        else (configured_workers if configured_workers is not None else (max(1, math.ceil(effective_qps)) if effective_qps > 0 else 1))
    )
    if effective_qps < 0:
        raise ValueError(f"qps must be >= 0, got {effective_qps}")
    if effective_workers < 1:
        raise ValueError(f"workers must be >= 1, got {effective_workers}")

    preds = load_jsonl(predictions_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base = Path(predictions_path).parent

    if cache_dir is None:
        cache_dir = str(Path(output_dir) / "judge_cache")
    vision_judge, text_judge, cache = load_judge_clients(
        config_path,
        cache_dir_override=cache_dir,
        qps_override=effective_qps,
    )

    def work(p: dict) -> dict:
        return _evaluate_one(
            p,
            base=base,
            vision_judge=vision_judge,
            text_judge=text_judge,
            cache=cache,
        )

    if effective_workers == 1:
        results_iter = map(work, preds)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="level2")
        # executor.map runs items concurrently but yields in input order, keeping
        # per_item.jsonl deterministic and aligned with predictions.jsonl.
        results_iter = executor.map(work, preds)

    n_total = n_with_aux = n_judged_items = n_judged_images = 0
    n_skipped_images = n_instruction_missing_images = 0
    n_items_with_instruction_missing = 0
    sum_l1 = sum_ep = sum_ic = 0.0
    per_path = Path(output_dir) / "per_item.jsonl"

    try:
        with per_path.open("w", encoding="utf-8") as pf:
            pbar = tqdm(results_iter, total=len(preds), desc="level2", unit="item", dynamic_ncols=True)
            for row in pbar:
                n_total += 1
                if not row["no_aux"]:
                    n_with_aux += 1
                judged = [x for x in row["per_image"] if not x.get("skipped") and "score" in x]
                skipped_images = [x for x in row["per_image"] if x.get("skipped")]
                instruction_missing = [
                    x for x in skipped_images
                    if x.get("reason") == "construction instruction not extracted"
                ]
                n_skipped_images += len(skipped_images)
                n_instruction_missing_images += len(instruction_missing)
                n_items_with_instruction_missing += int(bool(instruction_missing))
                if judged:
                    n_judged_items += 1
                    n_judged_images += len(judged)
                    sum_l1 += float(row["avg_l1"])
                    sum_ep += sum(x["element_preservation_norm"] for x in judged)
                    sum_ic += sum(x["instruction_consistency_norm"] for x in judged)
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                pf.flush()
                mean = (sum_l1 / n_judged_items) if n_judged_items else 0.0
                pbar.set_postfix(mean=f"{mean:.3f}", with_aux=n_with_aux, judged=n_judged_items, n=n_total)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    mean_l1 = (sum_l1 / n_judged_items) if n_judged_items else 0.0
    summary = {
        "n_total": n_total,
        "n_with_aux": n_with_aux,
        "n_without_aux": n_total - n_with_aux,
        "aux_generation_rate": (n_with_aux / n_total) if n_total else 0.0,
        "n_judged_items": n_judged_items,
        "n_judged_images": n_judged_images,
        "n_skipped_images": n_skipped_images,
        "n_instruction_missing_images": n_instruction_missing_images,
        "n_items_with_instruction_missing": n_items_with_instruction_missing,
        "n_failed_with_aux": n_with_aux - n_judged_items,
        "n_skipped_items": n_total - n_judged_items,
        # Canonical L1 excludes valid Mode-A direct-answer samples that did not
        # generate an auxiliary image. Judge/file failures are skipped as well.
        "mean_l1": mean_l1,
        "mean_l1_no_skip": mean_l1,
        "mean_ep": (sum_ep / n_judged_images) if n_judged_images else 0.0,
        "mean_ic": (sum_ic / n_judged_images) if n_judged_images else 0.0,
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
    ap.add_argument("--output-dir", required=False)
    ap.add_argument(
        "--qps",
        type=float,
        default=None,
        help="Override config judge.qps; 0 disables process-local limiting.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override config judge.workers for concurrent sample evaluation.",
    )
    args = ap.parse_args()

    if args.qps is not None and args.qps < 0:
        ap.error("--qps must be >= 0")
    if args.workers is not None and args.workers < 1:
        ap.error("--workers must be >= 1")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = args.output_dir or cfg.get("level2", {}).get("output_dir", "./results/level2")

    summary = run(
        predictions_path=args.predictions,
        output_dir=output_dir,
        config_path=args.config,
        qps=args.qps,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
