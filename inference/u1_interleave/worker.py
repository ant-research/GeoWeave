"""Single-rank worker for the GeoWeave interleave geo-aux inference pipeline.

Loads ``bench.jsonl``, runs ``interleave_gen`` per item, and writes a
per-rank predictions/trace file.  In dynamic mode all ranks claim benchmark
indices from a shared FIFO, so a fast GPU immediately picks up another item
instead of waiting after finishing a fixed rank/world-size slice.
Launched by ``run_inference.py``; can also be run standalone for
debugging a single GPU.

Output schema matches the rest of the bench so L1/L2/L3 evaluators consume
it without modification:
    {"id", "text", "input_images": [orig_abs], "images": [aux_abs, ...]}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from data.loader import load_jsonl
from inference.dynamic_work_queue import DynamicWorkQueue

from .engine import (
    DEFAULT_RESOLUTION,
    SUPPORTED_RESOLUTIONS,
    SenseNovaU1Interleave,
    resolve_image_size,
)
from .prompts import build_user_prompt, problem_from_item


AR_ENTROPY_DEFINITION = (
    "Mean next-token entropy in nats from softmax(raw model logits), before "
    "repetition penalty/temperature/top-k/top-p; excludes selected padding, "
    "EOS/stop, image start/end/context placeholder tokens, and all "
    "diffusion/image-latent steps."
)


def _abs(path: str, bench_path: str, image_root: str | None = None) -> str:
    """Resolve an image without relying on machine-specific filesystem paths."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    bench_rel = (Path(bench_path).expanduser().resolve().parent / p).resolve()
    if bench_rel.exists():
        return str(bench_rel)
    if image_root:
        rooted = (Path(image_root).expanduser().resolve() / p).resolve()
        if rooted.exists():
            return str(rooted)
    return str(bench_rel)


def _safe_id(s: str) -> str:
    return s.replace("/", "_").replace("#", "_").replace(":", "_")


def _candidate_id(parent_id: str, candidate_index: int, pass_n: int) -> str:
    """Keep legacy ids for N=1; make every pass@N trajectory uniquely resumable."""
    if pass_n == 1:
        return parent_id
    return f"{parent_id}::candidate_{candidate_index:04d}"


def _candidate_seed(base_seed: int, parent_id: str, candidate_index: int) -> int:
    """Derive a stable, rank-independent seed for one candidate trajectory."""
    payload = f"{base_seed}\0{parent_id}\0{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _request_at(items: list[dict], request_index: int, pass_n: int) -> tuple[dict, int]:
    return items[request_index // pass_n], request_index % pass_n


def _read_done_ids(path: Path) -> set[str]:
    """Re-read an existing shard predictions file to find ids already written."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except Exception:
                continue
    return done


def _read_all_done_ids(out_root: Path) -> set[str]:
    """Read completed ids globally so resume is independent of GPU count."""
    done: set[str] = set()
    merged = out_root / "predictions.jsonl"
    if merged.exists():
        done.update(_read_done_ids(merged))
    for path in out_root.glob("predictions.shard*.jsonl"):
        done.update(_read_done_ids(path))
    return done


def _parse_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _shard(items: list, rank: int, world_size: int) -> list:
    return items[rank::world_size]


_IMAGE_TAG_RE = re.compile(r"<image\d+>")
_BARE_IMAGE_TAG_RE = re.compile(r"<image>(?!\d)")


def _number_bare_generated_image_tags(text: str) -> str:
    """Convert official interleave placeholders from <image> to <image1>...

    Training/evaluation traces use numbered auxiliary-image tags, while the
    upstream interleave_gen implementation inserts a bare <image> each time it
    actually generates an image. Numbering here keeps inference rows compatible
    with both the SFT format and L1/L2 parsers without changing model behavior.
    """
    idx = 0

    def repl(_: re.Match) -> str:
        nonlocal idx
        idx += 1
        return f"<image{idx}>"

    return _BARE_IMAGE_TAG_RE.sub(repl, text or "")


def _item_images(item: dict) -> list[str]:
    images = item.get("images")
    if images is None:
        images = item.get("image", [])
    return list(images or [])


def run_worker(
    *,
    config_path: str,
    bench_path: str,
    output_dir: str,
    rank: int,
    world_size: int,
    limit: int | None,
    resume: bool,
    dynamic_queue_path: str | None = None,
    model_path_override: str | None = None,
    lora_path_override: str | None = None,
    pass_n: int = 1,
    temperature_override: float | None = None,
    top_p_override: float | None = None,
    top_k_override: int | None = None,
    max_new_tokens_override: int | None = None,
) -> dict:
    if pass_n < 1:
        raise ValueError(f"pass_n must be >= 1, got {pass_n}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    icfg = cfg["inference"]
    if model_path_override:
        icfg["model_path"] = model_path_override
    if lora_path_override is not None:
        # "" disables, any other value sets the LoRA path explicitly.
        icfg["lora_path"] = lora_path_override or None

    bench_all = load_jsonl(bench_path)
    if limit is not None:
        bench_all = bench_all[:limit]
    n_requests = len(bench_all) * pass_n
    if dynamic_queue_path:
        queue = DynamicWorkQueue(dynamic_queue_path, n_requests)
        requests = (
            _request_at(bench_all, request_index, pass_n)
            for request_index in queue.iter_indices()
        )
        scheduling = "dynamic"
    else:
        request_indices = range(rank, n_requests, world_size)
        requests = (_request_at(bench_all, request_index, pass_n) for request_index in request_indices)
        scheduling = "static"

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    aux_dir = out_root / "aux_images"
    aux_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_root / f"predictions.shard{rank:02d}.jsonl"
    trace_path = out_root / f"trace.shard{rank:02d}.jsonl"

    done_ids = _read_all_done_ids(out_root) if resume else set()
    pred_mode = "a" if (resume and pred_path.exists()) else "w"
    trace_mode = "a" if (resume and trace_path.exists()) else "w"
    if not resume:
        # Truncate any stale shard files so we don't double-count rows.
        if pred_path.exists():
            pred_path.unlink()
        if trace_path.exists():
            trace_path.unlink()

    # ── Build engine ────────────────────────────────────────────────────────
    dtype = _parse_dtype(icfg.get("dtype", "bfloat16"))
    # Each worker runs on a single visible GPU (set by parent via
    # CUDA_VISIBLE_DEVICES). "cuda" -> "cuda:0" inside this process.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = SenseNovaU1Interleave(
        model_path=icfg["model_path"],
        device=device,
        dtype=dtype,
        attn_backend=icfg.get("attn_backend", "auto"),
        vram_mode=icfg.get("vram_mode", "full"),
        lora_path=icfg.get("lora_path") or None,
    )

    fallback_res = icfg.get("resolution", DEFAULT_RESOLUTION)
    fallback_size = SUPPORTED_RESOLUTIONS.get(fallback_res, SUPPORTED_RESOLUTIONS[DEFAULT_RESOLUTION])

    cfg_scale = float(icfg.get("cfg_scale", 4.0))
    img_cfg_scale = float(icfg.get("img_cfg_scale", 1.0))
    timestep_shift = float(icfg.get("timestep_shift", 3.0))
    num_steps = int(icfg.get("num_steps", 50))
    max_new_tokens = int(
        max_new_tokens_override
        if max_new_tokens_override is not None
        else icfg.get("max_new_tokens", 8192)
    )
    repetition_penalty = icfg.get("repetition_penalty")
    if repetition_penalty is not None:
        repetition_penalty = float(repetition_penalty)
    min_pixels = icfg.get("min_pixels")
    min_pixels = int(min_pixels) if min_pixels is not None else None
    target_pixels = icfg.get("target_pixels")
    target_pixels = int(target_pixels) if target_pixels is not None else None
    think_mode = bool(icfg.get("interleave_think_mode", icfg.get("think_mode", True)))
    seed = int(icfg.get("seed", 42))
    image_root = icfg.get("image_root")
    do_sample = pass_n > 1
    temperature = float(
        temperature_override if temperature_override is not None else icfg.get("temperature", 1.0)
    )
    top_p = float(top_p_override if top_p_override is not None else icfg.get("top_p", 0.6))
    top_k = int(top_k_override if top_k_override is not None else icfg.get("top_k", 0))
    cfg_interval = tuple(icfg.get("cfg_interval", (0.0, 1.0)))

    # Interleave inference uses no model-dependent system prompt.
    system_message = ""
    print(
        f"[worker rank={rank}] model_path={icfg['model_path']} "
        f"think_mode={think_mode} "
    )

    # ── Main loop ───────────────────────────────────────────────────────────
    n_total = 0 if scheduling == "dynamic" else len(range(rank, n_requests, world_size))
    n_skipped_resume = 0
    n_ok = 0
    n_fail = 0
    ar_entropy_sum_raw = 0.0
    ar_entropy_token_count = 0
    ar_entropy_excluded_token_count = 0
    t0 = time.time()

    pf = pred_path.open(pred_mode, encoding="utf-8")
    tf = trace_path.open(trace_mode, encoding="utf-8")
    desc = f"rank{rank}/{world_size} interleave {scheduling}"
    pbar = tqdm(
        requests,
        total=None if scheduling == "dynamic" else n_total,
        desc=desc,
        unit="item",
        dynamic_ncols=True,
        position=rank,
    )
    try:
        for item, candidate_index in pbar:
            if scheduling == "dynamic":
                n_total += 1
            parent_id = str(item["id"])
            item_id = _candidate_id(parent_id, candidate_index, pass_n)
            # Preserve the historical N=1 seed exactly. For pass@N each
            # candidate gets a stable seed independent of worker/rank order.
            request_seed = (
                seed if pass_n == 1 else _candidate_seed(seed, parent_id, candidate_index)
            )
            if item_id in done_ids:
                n_skipped_resume += 1
                pbar.set_postfix(ok=n_ok, fail=n_fail, resume=n_skipped_resume)
                continue

            images = _item_images(item)
            if not images:
                tf.write(json.dumps({"id": item_id, "stage": "no_images"}, ensure_ascii=False) + "\n")
                tf.flush()
                n_fail += 1
                pbar.set_postfix(ok=n_ok, fail=n_fail, resume=n_skipped_resume)
                continue

            orig_path = _abs(images[0], bench_path, image_root=image_root)
            problem = problem_from_item(item)

            img_paths = [orig_path]
            user_prompt = build_user_prompt(problem)

            # ── Load images ─────────────────────────────────────────────
            try:
                pil_images = [Image.open(p).convert("RGB") for p in img_paths]
            except Exception as e:
                tf.write(json.dumps({
                    "id": item_id, "stage": "load_image",
                    "path": img_paths, "error": str(e)[:300],
                }, ensure_ascii=False) + "\n")
                tf.flush()
                n_fail += 1
                pbar.set_postfix(ok=n_ok, fail=n_fail, resume=n_skipped_resume)
                continue

            # ── Inference ───────────────────────────────────────────────
            try:
                w, h = resolve_image_size(
                    pil_images,
                    fallback=fallback_size,
                    min_pixels=min_pixels,
                    max_pixels=target_pixels,
                )
                text, out_imgs, generation_stats = engine.generate(
                    user_prompt,
                    input_images=pil_images,
                    image_size=(w, h),
                    cfg_scale=cfg_scale,
                    img_cfg_scale=img_cfg_scale,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=num_steps,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    think_mode=think_mode,
                    system_message=system_message,
                    seed=request_seed,
                    min_pixels=min_pixels,
                    max_pixels=target_pixels,
                )
            except Exception as e:
                tf.write(json.dumps({
                    "id": item_id, "stage": "generate",
                    "error": str(e)[:500],
                    "traceback": traceback.format_exc()[-1500:],
                }, ensure_ascii=False) + "\n")
                tf.flush()
                n_fail += 1
                pbar.set_postfix(ok=n_ok, fail=n_fail, resume=n_skipped_resume)
                continue

            # Save generated aux images and assemble predictions row.
            aux_paths: list[str] = []
            stem = _safe_id(item_id)
            for k, img in enumerate(out_imgs):
                p = aux_dir / f"{stem}_aux{k + 1}.png"
                img.save(p)
                aux_paths.append(str(p))

            text = _number_bare_generated_image_tags(text)
            out_images = aux_paths

            row = {
                "id": item_id,
                "parent_id": parent_id,
                "candidate_index": candidate_index,
                "pass_n": pass_n,
                "seed": request_seed,
                "think_mode": think_mode,
                "user_prompt": user_prompt,
                "text": text,
                "input_images": [orig_path],
                "images": out_images,
                "ar_entropy_raw": generation_stats["ar_entropy_raw"],
                "ar_entropy_sum_raw": generation_stats["ar_entropy_sum_raw"],
                "ar_entropy_token_count": generation_stats["ar_entropy_token_count"],
                "ar_entropy_excluded_token_count": generation_stats[
                    "ar_entropy_excluded_token_count"
                ],
            }
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
            pf.flush()
            tf.write(json.dumps({
                "id": item_id, "stage": "ok",
                "parent_id": parent_id, "candidate_index": candidate_index,
                "seed": request_seed, "think_mode": think_mode,
                "n_aux": len(aux_paths), "size": [w, h],
                "ar_entropy_raw": generation_stats["ar_entropy_raw"],
                "ar_entropy_token_count": generation_stats["ar_entropy_token_count"],
                "ar_entropy_excluded_token_count": generation_stats[
                    "ar_entropy_excluded_token_count"
                ],
            }, ensure_ascii=False) + "\n")
            tf.flush()
            ar_entropy_sum_raw += float(generation_stats["ar_entropy_sum_raw"])
            ar_entropy_token_count += int(generation_stats["ar_entropy_token_count"])
            ar_entropy_excluded_token_count += int(
                generation_stats["ar_entropy_excluded_token_count"]
            )
            n_ok += 1
            pbar.set_postfix(ok=n_ok, fail=n_fail, resume=n_skipped_resume)
    finally:
        pf.close()
        tf.close()

    elapsed = time.time() - t0
    summary = {
        "rank": rank,
        "world_size": world_size,
        "mode": "interleave",
        "scheduling": scheduling,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_skipped_resume": n_skipped_resume,
        "elapsed_sec": round(elapsed, 1),
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "pass_n": pass_n,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "think_mode": think_mode,
        "ar_entropy_raw": (
            ar_entropy_sum_raw / ar_entropy_token_count
            if ar_entropy_token_count > 0 else None
        ),
        "ar_entropy_sum_raw": ar_entropy_sum_raw,
        "ar_entropy_token_count": ar_entropy_token_count,
        "ar_entropy_excluded_token_count": ar_entropy_excluded_token_count,
        "ar_entropy_definition": AR_ENTROPY_DEFINITION,
        "predictions_shard": str(pred_path),
        "trace_shard": str(trace_path),
    }
    with open(out_root / f"summary.shard{rank:02d}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--bench", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip ids already present in predictions files.")
    p.add_argument(
        "--dynamic-queue",
        default=None,
        help="Shared FIFO counter created by the launcher. If omitted, use static sharding.",
    )
    p.add_argument("--model-path", default=None,
                   help="Override `inference.model_path` from the yaml. Useful for "
                        "swapping in a finetuned HF checkpoint (e.g. .../hf_avg/).")
    p.add_argument("--lora-path", default=None,
                   help="Override `inference.lora_path`. Pass an empty string to disable.")
    p.add_argument("--pass-n", type=int, default=1,
                   help="Generate N independent candidates per benchmark sample.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_worker(
        config_path=args.config,
        bench_path=args.bench,
        output_dir=args.output_dir,
        rank=args.rank,
        world_size=args.world_size,
        limit=args.limit,
        resume=args.resume,
        dynamic_queue_path=args.dynamic_queue,
        model_path_override=args.model_path,
        lora_path_override=args.lora_path,
        pass_n=args.pass_n,
        temperature_override=args.temperature,
        top_p_override=args.top_p,
        top_k_override=args.top_k,
        max_new_tokens_override=args.max_new_tokens,
    )
    # Last line of stdout is parseable by the parent; non-fatal if missed.
    print(json.dumps(summary, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
