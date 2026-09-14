"""Top-level launcher for GeoWeave interleave inference.

Forks N data-parallel workers, each pinned to one GPU, and merges their
per-rank outputs into predictions, trace, and summary files.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from inference.dynamic_work_queue import initialize_dynamic_queue
from inference.run_manifest import write_run_manifest

AR_ENTROPY_DEFINITION = (
    "Mean next-token entropy in nats from softmax(raw model logits), before "
    "repetition penalty/temperature/top-k/top-p; excludes selected padding, "
    "EOS/stop, image start/end/context placeholder tokens, and all "
    "diffusion/image-latent steps."
)


def _worker_library_path() -> str:
    """Prefer this venv's CUDA-12 cuDNN and discard incompatible CUDA paths.

    The host currently exposes CUDA-13 cuDNN globally while this job uses a
    torch 2.8 + CUDA 12.8 build.  Worker processes must resolve libcudnn from
    the active virtual environment before importing torch.
    """
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cudnn_lib = Path(sys.prefix) / "lib" / pyver / "site-packages" / "nvidia" / "cudnn" / "lib"
    inherited = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    cleaned = [
        entry for entry in inherited
        if entry
        and "/cuda/compat" not in entry.rstrip("/")
        and "/cuda-11.1/" not in entry
    ]
    if cudnn_lib.is_dir():
        cleaned.insert(0, str(cudnn_lib))
    return os.pathsep.join(dict.fromkeys(cleaned))


def _parse_gpu_list(icfg: dict) -> list[int]:
    """`inference.gpus` (string "0,1,2") wins over `inference.num_workers` (int)."""
    gpus = icfg.get("gpus")
    if gpus:
        if isinstance(gpus, str):
            ids = [int(x) for x in gpus.split(",") if x.strip()]
        else:
            ids = [int(x) for x in gpus]
        if not ids:
            raise SystemExit("`inference.gpus` is set but parsed to an empty list")
        return ids
    n = int(icfg.get("num_workers", 1))
    if n < 1:
        raise SystemExit("`inference.num_workers` must be >= 1")
    return list(range(n))


def _merge_jsonl_shards(
    shard_paths: list[Path],
    out_path: Path,
    *,
    preserve_existing: bool = False,
    dedupe_key: str | None = None,
) -> int:
    """Merge JSONL sources while preserving prior merged output on resume.

    ``dedupe_key="id"`` keeps one prediction per sample id, with later shard
    rows replacing older merged rows. Without a key, exact duplicate lines are
    removed while distinct trace events are retained.
    """
    lines: list[str] = []
    if preserve_existing and out_path.exists():
        lines.extend(out_path.read_text(encoding="utf-8").splitlines())
    for shard_path in shard_paths:
        if shard_path.exists():
            lines.extend(shard_path.read_text(encoding="utf-8").splitlines())

    if dedupe_key is not None:
        keyed: dict[str, str] = {}
        unkeyed: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = str(row[dedupe_key])
            except (json.JSONDecodeError, KeyError, TypeError):
                unkeyed.append(line)
                continue
            keyed[key] = line
        merged_lines = [*keyed.values(), *unkeyed]
    else:
        seen: set[str] = set()
        merged_lines = []
        for line in lines:
            if not line.strip() or line in seen:
                continue
            seen.add(line)
            merged_lines.append(line)

    with out_path.open("w", encoding="utf-8") as out:
        for line in merged_lines:
            out.write(line.rstrip("\n") + "\n")
    return len(merged_lines)


def _aggregate_ar_entropy(predictions_path: Path) -> dict:
    """Compute a token-weighted entropy from merged prediction rows.

    Reading predictions rather than shard summaries keeps the metric correct
    when rows from different workers have very different generated lengths.
    It also allows old rows without entropy fields to coexist safely.
    """
    entropy_sum = 0.0
    token_count = 0
    excluded_token_count = 0
    sample_count = 0
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    row_sum = row.get("ar_entropy_sum_raw")
                    row_count = row.get("ar_entropy_token_count")
                    if row_sum is None or row_count is None:
                        continue
                    entropy_sum += float(row_sum)
                    token_count += int(row_count)
                    excluded_token_count += int(
                        row.get("ar_entropy_excluded_token_count", 0)
                    )
                    sample_count += 1
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    return {
        "ar_entropy_raw": entropy_sum / token_count if token_count > 0 else None,
        "ar_entropy_sum_raw": entropy_sum,
        "ar_entropy_token_count": token_count,
        "ar_entropy_excluded_token_count": excluded_token_count,
        "ar_entropy_sample_count": sample_count,
        "ar_entropy_definition": AR_ENTROPY_DEFINITION,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config (uses `inference:` block).")
    ap.add_argument("--bench", required=True, help="Path to bench.jsonl")
    ap.add_argument("--output-dir", required=True, help="Where shards + merged outputs go.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run on the first N items of bench (sharded across workers).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip ids already present in predictions files.")
    ap.add_argument(
        "--scheduling",
        choices=["dynamic", "static"],
        default=None,
        help="Work assignment policy. Default: inference.scheduling or dynamic. "
             "Dynamic lets each free GPU claim the next benchmark item; static uses "
             "the legacy items[rank::world_size] split.",
    )
    ap.add_argument("--world-size", type=int, default=None,
                    help="Override worker count. Defaults to len(inference.gpus) "
                         "or inference.num_workers from the config.")
    ap.add_argument("--gpus", default=None,
                    help='Override `inference.gpus` (comma-separated CUDA ids, '
                         'e.g. "0,1,2,3").')
    ap.add_argument(
        "--model-path",
        default=None,
        help="Override `inference.model_path`. Use a directly loadable Hugging Face "
             "checkpoint directory or model identifier.",
    )
    ap.add_argument("--lora-path", default=None,
                    help="Override `inference.lora_path`. Pass an empty string to "
                         "disable a LoRA configured in the yaml.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the worker commands and exit without launching.")
    ap.add_argument(
        "--pass-n",
        type=int,
        default=1,
        help="Generate N independent candidates per sample. N>1 enables sampling.",
    )
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Text sampling temperature for pass@N (default: 1.0).")
    ap.add_argument("--top-p", type=float, default=0.6,
                    help="Nucleus sampling probability for pass@N (default: 0.6).")
    ap.add_argument("--top-k", type=int, default=0,
                    help="Top-k text sampling; 0 disables it (default: 0).")
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override text token cap. Default: 2048 when --pass-n > 1; otherwise YAML.",
    )
    ap.add_argument("--keep-shards", action="store_true",
                    help="Keep per-rank predictions.shardNN.jsonl / trace.shardNN.jsonl / "
                         "summary.shardNN.json after merging. By default they are deleted "
                         "once the merged outputs are written.")
    args = ap.parse_args()

    if args.pass_n < 1:
        ap.error("--pass-n must be >= 1")
    if args.temperature <= 0:
        ap.error("--temperature must be > 0")
    if not 0 < args.top_p <= 1:
        ap.error("--top-p must be in (0, 1]")
    if args.top_k < 0:
        ap.error("--top-k must be >= 0")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        ap.error("--max-new-tokens must be >= 1")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    icfg = cfg["inference"]
    if args.gpus is not None:
        icfg["gpus"] = args.gpus
    scheduling = args.scheduling or str(icfg.get("scheduling", "dynamic"))
    if scheduling not in {"dynamic", "static"}:
        raise SystemExit(
            f"Invalid inference.scheduling={scheduling!r}; expected 'dynamic' or 'static'"
        )

    requested_model_path = args.model_path or icfg.get("model_path")
    if not requested_model_path:
        ap.error("Set inference.model_path in the YAML or pass --model-path")
    local_model_path = Path(requested_model_path).expanduser()
    effective_model_path = (
        str(local_model_path.resolve()) if local_model_path.exists() else requested_model_path
    )

    gpu_ids = _parse_gpu_list(icfg)
    if args.world_size is not None:
        if args.world_size > len(gpu_ids):
            raise SystemExit(
                f"--world-size {args.world_size} > available gpus {len(gpu_ids)} ({gpu_ids})"
            )
        gpu_ids = gpu_ids[: args.world_size]
    world_size = len(gpu_ids)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    resolved_cfg = copy.deepcopy(cfg)
    resolved_icfg = resolved_cfg["inference"]
    resolved_icfg["model_path"] = effective_model_path
    if args.lora_path is not None:
        resolved_icfg["lora_path"] = args.lora_path or None
    effective_think_mode = bool(
        resolved_icfg.get("interleave_think_mode", resolved_icfg.get("think_mode", True))
    )
    resolved_icfg["interleave_think_mode"] = effective_think_mode
    effective_max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else (2048 if args.pass_n > 1 else int(resolved_icfg.get("max_new_tokens", 8192)))
    )
    resolved_icfg["pass_n"] = args.pass_n
    resolved_icfg["do_sample"] = args.pass_n > 1
    resolved_icfg["temperature"] = args.temperature
    resolved_icfg["top_p"] = args.top_p
    resolved_icfg["top_k"] = args.top_k
    resolved_icfg["max_new_tokens"] = effective_max_new_tokens
    resolved_icfg["gpus"] = gpu_ids
    resolved_icfg["num_workers"] = world_size
    resolved_icfg["scheduling"] = scheduling
    manifest_path = write_run_manifest(
        output_dir=out_root,
        launcher="run_inference.py",
        cli_args=vars(args),
        source_config_path=args.config,
        resolved_config=resolved_cfg,
        bench_path=args.bench,
        model_path=resolved_icfg.get("model_path"),
        lora_path=resolved_icfg.get("lora_path"),
        gpu_ids=gpu_ids,
        world_size=world_size,
        repo_root=Path(__file__).resolve().parent,
    )
    print(f"[launcher] saved run manifest: {manifest_path}")

    log_dir = out_root / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent
    py = sys.executable
    # Use the same venv this launcher was invoked from. Workers spawn as
    # `python -m inference.u1_interleave.worker` so package imports resolve.
    base_cmd = [
        py, "-m", "inference.u1_interleave.worker",
        "--config", str(Path(args.config).resolve()),
        "--bench", str(Path(args.bench).resolve()),
        "--output-dir", str(out_root.resolve()),
        "--world-size", str(world_size),
    ]
    if args.limit is not None:
        base_cmd += ["--limit", str(args.limit)]
    if args.resume:
        base_cmd += ["--resume"]
    # Always pass the resolved local path (or unchanged Hugging Face model id).
    base_cmd += ["--model-path", effective_model_path]
    if args.lora_path is not None:
        # Pass empty string through too so workers can see "disable lora".
        base_cmd += ["--lora-path", args.lora_path]
    base_cmd += [
        "--pass-n", str(args.pass_n),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--top-k", str(args.top_k),
        "--max-new-tokens", str(effective_max_new_tokens),
    ]

    dynamic_queue_path: Path | None = None
    if scheduling == "dynamic":
        dynamic_queue_path = out_root / ".u1_dynamic_queue"
        base_cmd += ["--dynamic-queue", str(dynamic_queue_path.resolve())]

    print(
        f"[launcher] mode=interleave scheduling={scheduling} pass_n={args.pass_n} "
        f"world_size={world_size}, gpu_ids={gpu_ids}, "
        f"think_mode={effective_think_mode}, "
    )
    print(f"[launcher] output_dir={out_root}")

    if dynamic_queue_path is not None and not args.dry_run:
        initialize_dynamic_queue(dynamic_queue_path)

    procs: list[subprocess.Popen] = []
    log_files = []
    t0 = time.time()
    try:
        for rank, gpu in enumerate(gpu_ids):
            cmd = base_cmd + ["--rank", str(rank)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["LD_LIBRARY_PATH"] = _worker_library_path()
            log_path = log_dir / f"rank{rank:02d}.log"
            print(f"[launcher] rank={rank} gpu={gpu} -> {log_path}")
            if args.dry_run:
                print("  ", " ".join(cmd))
                continue
            lf = log_path.open("w", encoding="utf-8")
            log_files.append(lf)
            p = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
            procs.append(p)

        if args.dry_run:
            return

        # Wait. On Ctrl-C, propagate SIGINT to all workers so each one can
        # flush its shard files before exiting.
        rc_list: list[int] = []
        try:
            for p in procs:
                rc_list.append(p.wait())
        except KeyboardInterrupt:
            print("[launcher] SIGINT received, forwarding to workers ...")
            for p in procs:
                try:
                    p.send_signal(signal.SIGINT)
                except Exception:
                    pass
            for p in procs:
                try:
                    rc_list.append(p.wait(timeout=30))
                except Exception:
                    p.kill()
                    rc_list.append(-9)
            raise
    finally:
        for lf in log_files:
            try:
                lf.close()
            except Exception:
                pass

    elapsed = time.time() - t0
    print(f"[launcher] all workers exited in {elapsed:.1f}s; return codes={rc_list}")

    # ── Merge shards ────────────────────────────────────────────────────────
    # On resume, merge every historical shard because world_size may differ
    # from the interrupted run. Also preserve a prior merged file in case its
    # shards were already cleaned after a successful launch.
    if args.resume:
        pred_shards = sorted(out_root.glob("predictions.shard*.jsonl"))
        trace_shards = sorted(out_root.glob("trace.shard*.jsonl"))
    else:
        pred_shards = [out_root / f"predictions.shard{r:02d}.jsonl" for r in range(world_size)]
        trace_shards = [out_root / f"trace.shard{r:02d}.jsonl" for r in range(world_size)]
    predictions_path = out_root / "predictions.jsonl"
    n_pred = _merge_jsonl_shards(
        pred_shards,
        predictions_path,
        preserve_existing=args.resume,
        dedupe_key="id",
    )
    n_trace = _merge_jsonl_shards(
        trace_shards,
        out_root / "trace.jsonl",
        preserve_existing=args.resume,
    )
    ar_entropy_summary = _aggregate_ar_entropy(predictions_path)

    # Aggregate per-shard summaries.
    shard_summaries: list[dict] = []
    n_ok = n_fail = n_skipped_resume = n_total = 0
    for r in range(world_size):
        sp = out_root / f"summary.shard{r:02d}.json"
        if not sp.exists():
            continue
        with sp.open("r", encoding="utf-8") as f:
            s = json.load(f)
        shard_summaries.append(s)
        n_ok += s.get("n_ok", 0)
        n_fail += s.get("n_fail", 0)
        n_skipped_resume += s.get("n_skipped_resume", 0)
        n_total += s.get("n_total", 0)

    summary = {
        "mode": "u1_interleave",
        "scheduling": scheduling,
        "pass_n": args.pass_n,
        "do_sample": args.pass_n > 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": effective_max_new_tokens,
        "think_mode": effective_think_mode,
        "world_size": world_size,
        "gpu_ids": gpu_ids,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_skipped_resume": n_skipped_resume,
        "n_pred_lines": n_pred,
        "n_trace_lines": n_trace,
        "elapsed_sec": round(elapsed, 1),
        "return_codes": rc_list,
        "predictions_path": str(predictions_path),
        **ar_entropy_summary,
        "run_manifest_path": str(out_root / "run_manifest.json"),
        "resolved_config_path": str(out_root / "resolved_config.yaml"),
        "shards": shard_summaries,
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "shards"},
                     indent=2, ensure_ascii=False))

    # ── Clean up shard files ────────────────────────────────────────────────
    # Only delete shards when every rank succeeded; on partial failure we keep
    # them so the user can --resume without losing the completed work.
    all_ok = bool(rc_list) and all(rc == 0 for rc in rc_list)
    if not args.keep_shards and all_ok:
        removed = 0
        for pattern in (
            "predictions.shard*.jsonl",
            "trace.shard*.jsonl",
            "summary.shard*.json",
        ):
            for shard_path in out_root.glob(pattern):
                shard_path.unlink()
                removed += 1
        print(f"[launcher] cleaned up {removed} shard files (pass --keep-shards to retain)")
    elif not all_ok:
        print("[launcher] keeping shard files because some workers failed; "
              "rerun with --resume to continue")

    if dynamic_queue_path is not None and dynamic_queue_path.exists():
        dynamic_queue_path.unlink()

    if any(rc != 0 for rc in rc_list):
        sys.exit(1)


if __name__ == "__main__":
    main()
