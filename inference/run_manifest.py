"""Utilities for recording reproducible inference-run provenance.

Launchers call :func:`write_run_manifest` before spawning workers.  The files
are intentionally written into the result directory so predictions can always
be traced back to the exact CLI, resolved YAML, bench and model locations.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_SMALL_FILE_HASH_LIMIT = 16 * 1024 * 1024
_MODEL_FILE_SUFFIXES = {
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt",
    ".json", ".yaml", ".yml", ".txt", ".model",
}
_SECRET_CONFIG_KEY_RE = re.compile(
    r"(?:^|_)(?:api_key|access_token|auth_token|password|secret)$",
    re.IGNORECASE,
)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _resolved_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None or str(value) == "":
        return None
    return str(Path(value).expanduser().resolve())


def _file_info(path: Path, *, hash_content: bool = True) -> dict[str, Any]:
    st = path.stat()
    info: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "mtime_ns": st.st_mtime_ns,
    }
    if hash_content:
        info["sha256"] = _sha256_file(path)
    return info


def _artifact_info(value: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    """Describe a model/LoRA path without hashing multi-GB weight shards.

    Every relevant file's relative path, size and nanosecond mtime is recorded,
    and those records receive a deterministic snapshot SHA256. Small metadata
    files are content-hashed. This is fast enough for normal launches while
    still revealing checkpoint replacement in nearly all practical cases.
    """
    resolved = _resolved_path(value)
    if resolved is None:
        return None
    path = Path(resolved)
    result: dict[str, Any] = {
        "provided_path": str(value),
        "resolved_path": resolved,
        "exists": path.exists(),
    }
    if not path.exists():
        return result
    if path.is_file():
        result.update({"kind": "file", **_file_info(path, hash_content=True)})
        return result

    files: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*")):
        if not child.is_file() or child.suffix.lower() not in _MODEL_FILE_SUFFIXES:
            continue
        st = child.stat()
        row: dict[str, Any] = {
            "relative_path": str(child.relative_to(path)),
            "size_bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
        if st.st_size <= _SMALL_FILE_HASH_LIMIT:
            row["sha256"] = _sha256_file(child)
        files.append(row)

    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result.update({
        "kind": "directory",
        "artifact_file_count": len(files),
        "artifact_total_size_bytes": sum(x["size_bytes"] for x in files),
        "snapshot_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "note": (
            "snapshot_sha256 covers relative path, size, mtime_ns and small-file hashes; "
            "large model weights are not content-hashed to avoid expensive startup I/O"
        ),
    })
    return result


def _git_info(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    status = run("status", "--porcelain=v1")
    diff = run("diff", "--no-ext-diff", "--no-textconv", "HEAD")
    return {
        "repo_root": str(repo_root.resolve()),
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": bool(status) if status is not None else None,
        "status_porcelain": status,
        "tracked_diff_sha256": (
            hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff is not None else None
        ),
    }


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _redact_config_secrets(value: Any) -> Any:
    """Recursively redact credential values while retaining env-var names.

    Keys such as ``api_key_env`` are intentionally preserved because they are
    provenance, not credentials. Direct ``api_key``/token/password values are
    never copied into a closed-model run directory.
    """
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if _SECRET_CONFIG_KEY_RE.search(key_text):
                out[key_text] = "<REDACTED>" if child else child
            else:
                out[key_text] = _redact_config_secrets(child)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact_config_secrets(child) for child in value]
    return value


def write_run_manifest(
    *,
    output_dir: str | os.PathLike[str],
    launcher: str,
    cli_args: Mapping[str, Any],
    source_config_path: str | os.PathLike[str],
    resolved_config: Mapping[str, Any],
    bench_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str] | None,
    lora_path: str | os.PathLike[str] | None,
    gpu_ids: Sequence[int],
    world_size: int,
    repo_root: str | os.PathLike[str],
    redact_config_secrets: bool = False,
    inference_target: Mapping[str, Any] | None = None,
) -> Path:
    """Write latest and timestamped provenance files into ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = out / "run_manifest_history"
    history.mkdir(parents=True, exist_ok=True)

    source_config = Path(source_config_path).expanduser().resolve()
    bench = Path(bench_path).expanduser().resolve()
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-pid{os.getpid()}"

    # Human-friendly config copies plus effective values after CLI overrides.
    # Closed-model launchers request redaction so credentials directly embedded
    # in YAML are never copied into result directories.
    source_copy = history / f"{run_id}.source_config{source_config.suffix or '.yaml'}"
    manifest_config: Any = resolved_config
    if redact_config_secrets:
        source_data = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        source_copy.write_text(
            yaml.safe_dump(
                _jsonable(_redact_config_secrets(source_data)),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        manifest_config = _redact_config_secrets(resolved_config)
    else:
        shutil.copy2(source_config, source_copy)
    resolved_text = yaml.safe_dump(
        _jsonable(manifest_config), allow_unicode=True, sort_keys=False
    )
    resolved_history = history / f"{run_id}.resolved_config.yaml"
    resolved_history.write_text(resolved_text, encoding="utf-8")
    shutil.copy2(source_copy, out / "source_config.yaml")
    (out / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "launcher": launcher,
        "command": [sys.executable, *sys.argv],
        "command_shell": " ".join(__import__("shlex").quote(x) for x in [sys.executable, *sys.argv]),
        "cwd": str(Path.cwd().resolve()),
        "cli_args": _jsonable(cli_args),
        "runtime": {
            "world_size": world_size,
            "gpu_ids": list(gpu_ids),
            "cuda_visible_devices_at_launcher": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "paths": {
            "output_dir": str(out.resolve()),
            "source_config": _file_info(source_config),
            "source_config_backup": str(source_copy.resolve()),
            "resolved_config_backup": str(resolved_history.resolve()),
            "bench": _file_info(bench),
            "model": _artifact_info(model_path),
            "lora": _artifact_info(lora_path),
        },
        "resolved_config": _jsonable(manifest_config),
        "config_secrets_redacted": redact_config_secrets,
        "inference_target": _jsonable(inference_target),
        "code": _git_info(Path(repo_root).resolve()),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "packages": _package_versions(
                [
                    "torch", "transformers", "flash-attn", "sensenova-u1",
                    "PyYAML", "Pillow", "requests", "openai",
                ]
            ),
        },
    }

    history_path = history / f"{run_id}.json"
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    history_path.write_text(text, encoding="utf-8")
    (out / "run_manifest.json").write_text(text, encoding="utf-8")
    return history_path
