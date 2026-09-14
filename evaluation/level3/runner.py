"""Level 3 runner: judge auxiliary-line usage per item.

Supports sample-level thread concurrency. ``qps`` caps vision-judge request
starts through the shared process-local limiter.
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
from evaluation.common.parsing import aux_image_path, parse_output, strip_image_tags
from evaluation.level2.instruction_extractor import extract_instruction
from evaluation.level3.usage_judge import judge_aux_usage


def _pred_to_id(p: dict) -> str:
    if "id" in p:
        return str(p["id"])
    return str(p.get("index", ""))


def _repo_root() -> Path:
    """The geo-aux-bench repo root: <repo>/evaluation/level3/runner.py → <repo>."""
    return Path(__file__).resolve().parents[2]


_REPO_ANCHOR = "geo-aux-bench/"


def _remap_to_local_repo(path: str) -> str | None:
    """Re-root a stale absolute path under the current repository, if possible."""
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
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    for hit in base.rglob(name):
        return str(hit)
    return str(p)


def _load_l2_instructions(l2_dir: str | None) -> dict[str, dict[int, str]]:
    """Load successful instruction extractions from a prior L2 run, if any."""
    if not l2_dir:
        return {}
    path = Path(l2_dir) / "per_item.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[int, str]] = {}
    for item in load_jsonl(str(path)):
        per: dict[int, str] = {}
        for entry in item.get("per_image", []):
            instruction = str(entry.get("instruction", "")).strip()
            if entry.get("idx") is not None and entry.get("instruction_found") is True and instruction:
                per[int(entry["idx"])] = instruction
        out[str(item.get("id", ""))] = per
    return out


def _evaluate_one(
    p: dict,
    *,
    base: Path,
    vision_judge,
    text_judge,
    cache,
    prior_instructions: dict[int, str] | None = None,
) -> dict:
    """Evaluate one prediction. Safe to call from a worker thread."""
    pid = _pred_to_id(p)
    text = p.get("text", "")
    input_images = p.get("input_images") or p.get("image") or []
    if isinstance(input_images, str):
        input_images = [input_images]
    aux_images = p.get("images") or []

    parsed = parse_output(text)
    if not parsed.aux_indices and aux_images:
        parsed.aux_indices = list(range(1, len(aux_images) + 1))

    if not aux_images:
        return {
            "id": pid,
            "no_aux": True,
            "skipped": True,
            "aux_usage": None,
            "new_elements_referenced": [],
            "reason": "no auxiliary image generated",
            "_with_aux": False,
            "_judge_ok": False,
        }
    if not input_images:
        return {
            "id": pid,
            "no_aux": False,
            "skipped": True,
            "aux_usage": None,
            "new_elements_referenced": [],
            "reason": "original image missing",
            "_with_aux": True,
            "_judge_ok": False,
        }

    orig = _resolve(input_images[0], base)
    aux_paths: list[str] = []
    instructions: list[str] = []
    instruction_found: list[bool] = []
    aux_indices: list[int] = []

    for k in parsed.aux_indices:
        aux_rel = aux_image_path(aux_images, k)
        if aux_rel is None:
            continue
        aux_path = _resolve(aux_rel, base)
        if not Path(aux_path).exists():
            continue
        aux_paths.append(aux_path)
        aux_indices.append(k)
        # L2 uses the same text-based extraction as L1 when possible. An
        # extraction failure is deliberately non-fatal: the vision judge will
        # infer the visual delta for this image as the fallback.
        instruction = str((prior_instructions or {}).get(k, "")).strip()
        found = bool(instruction)
        if not found:
            try:
                instr = extract_instruction(
                    think=parsed.think,
                    tag_index=k,
                    text_judge=text_judge,
                    cache=cache,
                )
                instruction = str(instr.get("instruction", "")).strip()
                found = instr.get("found") is True and bool(instruction)
            except Exception:
                instruction = ""
                found = False
        instructions.append(instruction if found else "")
        instruction_found.append(found)

    if not aux_paths:
        return {
            "id": pid,
            "no_aux": False,
            "skipped": True,
            "aux_usage": None,
            "new_elements_referenced": [],
            "reason": "aux image files missing",
            "_with_aux": True,
            "_judge_ok": False,
        }

    reasoning_text = strip_image_tags(parsed.think).strip()
    judge_ok = True
    try:
        verdict = judge_aux_usage(
            original_image=orig,
            aux_images=aux_paths,
            instructions=instructions,
            reasoning=reasoning_text,
            vision_judge=vision_judge,
            cache=cache,
        )
    except Exception as exc:
        judge_ok = False
        verdict = {
            "aux_usage": 0,
            "element_reference": 0,
            "element_reference_reason": "",
            "reasoning_integration": 0,
            "reasoning_integration_reason": "",
            "new_elements_referenced": [],
            "reason": f"judge error: {str(exc)[:200]}",
        }

    er = float(verdict.get("element_reference", verdict.get("aux_usage", 0)))
    ri = float(verdict.get("reasoning_integration", verdict.get("aux_usage", 0)))
    er_norm = er / 2.0
    ri_norm = ri / 2.0
    l2_score = (er_norm + ri_norm) / 2.0
    return {
        "id": pid,
        "no_aux": False,
        "aux_usage": verdict["aux_usage"],
        "element_reference": verdict.get("element_reference", verdict["aux_usage"]),
        "element_reference_norm": er_norm,
        "element_reference_reason": verdict.get("element_reference_reason", ""),
        "reasoning_integration": verdict.get("reasoning_integration", verdict["aux_usage"]),
        "reasoning_integration_norm": ri_norm,
        "reasoning_integration_reason": verdict.get("reasoning_integration_reason", ""),
        "new_elements_referenced": verdict["new_elements_referenced"],
        "reason": verdict["reason"],
        "score": l2_score,
        "aux_indices": aux_indices,
        "instructions": instructions,
        "instruction_found": instruction_found,
        "analysis_basis": verdict.get("analysis_basis", "construction_instruction_with_visual_fallback"),
        "skipped": not judge_ok,
        "_with_aux": True,
        "_judge_ok": judge_ok,
    }


def run(
    *,
    predictions_path: str,
    output_dir: str,
    config_path: str,
    l2_dir: str | None = None,
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
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base = Path(predictions_path).parent

    if cache_dir is None:
        cache_dir = str(Path(output_dir) / "judge_cache")
    vision_judge, text_judge, cache = load_judge_clients(
        config_path,
        cache_dir_override=cache_dir,
        qps_override=effective_qps,
    )
    # Reuse successful L2 extractions when available; otherwise extract from the
    # prediction text here. Either way, image comparison is only the fallback for
    # auxiliary images with no usable instruction.
    l2_instructions = _load_l2_instructions(l2_dir)

    def work(p: dict) -> dict:
        return _evaluate_one(
            p,
            base=base,
            vision_judge=vision_judge,
            text_judge=text_judge,
            cache=cache,
            prior_instructions=l2_instructions.get(_pred_to_id(p), {}),
        )

    if effective_workers == 1:
        results_iter = map(work, preds)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="level3")
        results_iter = executor.map(work, preds)

    n_total = n_with_aux = n_judged_items = 0
    sum_l2 = 0.0
    sum_element_reference = sum_reasoning_integration = 0.0
    per_path = Path(output_dir) / "per_item.jsonl"

    try:
        with per_path.open("w", encoding="utf-8") as pf:
            pbar = tqdm(results_iter, total=len(preds), desc="level3", unit="item", dynamic_ncols=True)
            for row in pbar:
                with_aux = bool(row.pop("_with_aux"))
                judge_ok = bool(row.pop("_judge_ok"))
                n_total += 1
                n_with_aux += int(with_aux)
                if judge_ok:
                    score = float(row["score"])
                    n_judged_items += 1
                    sum_l2 += score
                    sum_element_reference += float(row["element_reference_norm"])
                    sum_reasoning_integration += float(row["reasoning_integration_norm"])
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                pf.flush()
                mean = (sum_l2 / n_judged_items) if n_judged_items else 0.0
                pbar.set_postfix(mean=f"{mean:.3f}", with_aux=n_with_aux, judged=n_judged_items, n=n_total)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    mean_l2 = (sum_l2 / n_judged_items) if n_judged_items else 0.0
    summary = {
        "n_total": n_total,
        "n_with_aux": n_with_aux,
        "n_without_aux": n_total - n_with_aux,
        "aux_generation_rate": (n_with_aux / n_total) if n_total else 0.0,
        "n_judged_items": n_judged_items,
        "n_failed_with_aux": n_with_aux - n_judged_items,
        "n_skipped_items": n_total - n_judged_items,
        # Canonical L2 is conditional on an auxiliary image being generated and
        # successfully judged; optional direct-answer samples are not zeroes.
        "mean_l2": mean_l2,
        "mean_l2_no_skip": mean_l2,
        "mean_element_reference": (sum_element_reference / n_judged_items) if n_judged_items else 0.0,
        "mean_reasoning_integration": (sum_reasoning_integration / n_judged_items) if n_judged_items else 0.0,
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
    ap.add_argument("--l2-dir", required=False, help="Backward-compatible option; L2 extracts instructions from prediction text directly.")
    ap.add_argument("--qps", type=float, default=None, help="Override config judge.qps; 0 disables limiting.")
    ap.add_argument("--workers", type=int, default=None, help="Override config judge.workers.")
    args = ap.parse_args()

    if args.qps is not None and args.qps < 0:
        ap.error("--qps must be >= 0")
    if args.workers is not None and args.workers < 1:
        ap.error("--workers must be >= 1")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    output_dir = args.output_dir or cfg.get("level3", {}).get("output_dir", "./results/level3")
    l2_dir = args.l2_dir or cfg.get("level2", {}).get("output_dir")

    summary = run(
        predictions_path=args.predictions,
        output_dir=output_dir,
        config_path=args.config,
        l2_dir=l2_dir,
        qps=args.qps,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
