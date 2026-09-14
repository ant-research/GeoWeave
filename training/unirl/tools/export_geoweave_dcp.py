"""Export a GeoWeave UniRL DCP checkpoint to an inference-ready HF folder.

GeoWeave RL recipes train ``bundle.transformer == model.language_model``.
Consequently, a UniRL ``checkpoint_format=dcp`` checkpoint stores Qwen/LLM
keys below DCP's ``model`` entry (for example ``model.layers.0...``), while a
full NEOChatModel Hugging Face snapshot names the same tensor
``language_model.model.layers.0...``.  Vision and FM modules are frozen and are
not part of that trainable module, so they must be retained from the exact HF
base snapshot used to start RL.

This exporter streams the DCP tensors one at a time, casts them to the base
snapshot's inference dtype, and rewrites the original HF safetensor shards.
Peak RAM is approximately one output shard plus one source tensor; it never
materializes the full model or optimizer state.

Example::

    python -m unirl.tools.export_geoweave_dcp \\
      --checkpoint /path/to/checkpoint-200 \\
      --base /path/to/U1_stage3_0713 \\
      --output /path/to/hf_checkpoint-200

Use ``--dry-run`` first to validate all keys, shapes, and files without reading
large tensor payloads or writing output weights.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_INDEX = "model.safetensors.index.json"
DCP_METADATA = ".metadata"
APP_METADATA = "metadata.pt"
DCP_CONTAINER_PREFIX = "model."
HF_LANGUAGE_PREFIX = "language_model."

# All non-weight files in the HF base are copied.  Explicitly exclude common
# weight formats so a stale base model cannot leak into the exported folder.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
_WEIGHT_NAMES = {MODEL_INDEX, "pytorch_model.bin.index.json"}

class _CachedFileSystemReader(dcp.FileSystemReader):
    """Reuse parsed DCP metadata across many one-tensor load plans."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        super().__init__(path)
        self._cached_metadata = super().read_metadata()

    def read_metadata(self, *args, **kwargs):
        # A 16-rank GeoWeave checkpoint has a multi-MB metadata pickle.  The
        # exporter intentionally issues one load plan per tensor for bounded
        # RAM, so reparsing that pickle for every tensor would dominate runtime.
        return self._cached_metadata


_SAFETENSORS_TO_TORCH = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _load_dcp_metadata(checkpoint: Path):
    """Read DCP metadata without opening multi-GB ``*.distcp`` payloads."""
    metadata_path = checkpoint / DCP_METADATA
    if not metadata_path.is_file():
        raise FileNotFoundError(f"DCP metadata not found: {metadata_path}")
    # FileSystemReader.read_metadata() is public, but older torch versions may
    # not expose it consistently.  The file itself is a DCP Metadata pickle.
    try:
        return dcp.FileSystemReader(str(checkpoint)).read_metadata()
    except (AttributeError, TypeError):
        with metadata_path.open("rb") as f:
            return pickle.load(f)


def _load_app_metadata(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / APP_METADATA
    if not path.is_file():
        raise FileNotFoundError(f"UniRL checkpoint metadata not found: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"expected a dict in {path}, got {type(value).__name__}")
    return value


def _is_auxiliary_file(path: Path) -> bool:
    name = path.name
    if name in _WEIGHT_NAMES or name.startswith("model-") or name.startswith("moemodel-"):
        return False
    return not name.endswith(_WEIGHT_SUFFIXES)


def _copy_auxiliary_files(base: Path, output: Path) -> None:
    for src in sorted(base.iterdir()):
        if not src.is_file() or not _is_auxiliary_file(src):
            continue
        shutil.copy2(src, output / src.name)


def _copy_file(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:  # guarded by argparse
        raise ValueError(mode)


def _base_tensor_specs(base: Path, weight_map: dict[str, str]):
    """Return key -> (shape, dtype string), reading safetensors headers only."""
    by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        by_file[filename].append(key)

    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    file_metadata: dict[str, dict[str, str] | None] = {}
    for filename, expected_keys in by_file.items():
        path = base / filename
        if not path.is_file():
            raise FileNotFoundError(f"base shard referenced by index is missing: {path}")
        with safe_open(path, framework="pt", device="cpu") as f:
            actual = set(f.keys())
            expected = set(expected_keys)
            if actual != expected:
                raise ValueError(
                    f"base index/shard mismatch for {filename}: "
                    f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
                )
            file_metadata[filename] = f.metadata()
            for key in expected_keys:
                tensor_slice = f.get_slice(key)
                specs[key] = (tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
    return specs, file_metadata, by_file


def _discover_replacements(dcp_metadata, weight_map: dict[str, str]):
    """Map HF keys to DCP inner model keys and validate exact coverage."""
    state_metadata = dcp_metadata.state_dict_metadata
    dcp_model_fqns = {
        key for key, value in state_metadata.items()
        if key.startswith(DCP_CONTAINER_PREFIX) and hasattr(value, "size") and hasattr(value, "properties")
    }
    if not dcp_model_fqns:
        raise ValueError("DCP checkpoint contains no tensor entries below the 'model' state")

    replacements: dict[str, str] = {}
    for fqn in dcp_model_fqns:
        inner_key = fqn[len(DCP_CONTAINER_PREFIX):]
        hf_key = HF_LANGUAGE_PREFIX + inner_key
        if hf_key not in weight_map:
            raise ValueError(
                f"DCP tensor {fqn!r} maps to {hf_key!r}, which is absent from the base HF index. "
                "Use the exact HF checkpoint configured as SENSENOVA_U1_PATH for this RL run."
            )
        replacements[hf_key] = inner_key

    base_language_keys = {key for key in weight_map if key.startswith(HF_LANGUAGE_PREFIX)}
    missing = base_language_keys - set(replacements)
    if missing:
        raise ValueError(
            f"DCP checkpoint is missing {len(missing)} base language_model tensors, for example: "
            f"{sorted(missing)[:8]}. The checkpoint may be adapter-only, incomplete, or from a different base."
        )
    return replacements


def _torch_dtype_from_safe_name(name: str) -> torch.dtype:
    try:
        return _SAFETENSORS_TO_TORCH[name]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype {name!r}") from exc


def _load_one_dcp_tensor(
    checkpoint: Path,
    inner_key: str,
    source_shape: tuple[int, ...],
    source_dtype: torch.dtype,
    *,
    storage_reader: dcp.FileSystemReader | None = None,
) -> torch.Tensor:
    state = {"model": {inner_key: torch.empty(source_shape, dtype=source_dtype, device="cpu")}}
    # no_dist avoids process-group initialization and reads only the chunks
    # required to reconstruct this one full tensor on CPU.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="torch.distributed is disabled.*")
        if storage_reader is None:
            dcp.load(state, checkpoint_id=str(checkpoint), no_dist=True)
        else:
            dcp.load(state, storage_reader=storage_reader, no_dist=True)
    return state["model"][inner_key]


def export(
    checkpoint: str | os.PathLike[str],
    base: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    static_mode: str = "copy",
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    checkpoint = Path(checkpoint).resolve()
    base = Path(base).resolve()
    output = Path(output).resolve()

    if not checkpoint.is_dir():
        raise NotADirectoryError(checkpoint)
    if not base.is_dir():
        raise NotADirectoryError(base)
    if output == base or output == checkpoint:
        raise ValueError("--output must differ from --base and --checkpoint")

    app_metadata = _load_app_metadata(checkpoint)
    save_mode = str(app_metadata.get("save_mode", "full"))
    if save_mode != "full":
        raise ValueError(f"expected save_mode='full', got {save_mode!r}; adapter checkpoints need LoRA merging")

    index_path = base / MODEL_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(f"base HF safetensors index not found: {index_path}")
    with index_path.open() as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty weight_map in {index_path}")

    storage_reader = _CachedFileSystemReader(str(checkpoint))
    dcp_metadata = storage_reader.read_metadata()
    specs, shard_metadata, by_file = _base_tensor_specs(base, weight_map)
    replacements = _discover_replacements(dcp_metadata, weight_map)
    dcp_state_metadata = dcp_metadata.state_dict_metadata

    # Validate shapes before reading any payload.
    for hf_key, inner_key in replacements.items():
        fqn = DCP_CONTAINER_PREFIX + inner_key
        source_shape = tuple(dcp_state_metadata[fqn].size)
        base_shape, _ = specs[hf_key]
        if source_shape != base_shape:
            raise ValueError(f"shape mismatch for {hf_key}: DCP={source_shape}, base={base_shape}")

    retained = set(weight_map) - set(replacements)
    # Keep empty/unindexed compatibility shards too.  GeoWeave's published
    # layout has 16 model files although several empty shards do not appear in
    # weight_map; some inference launchers expect those filenames to exist.
    indexed_shards = set(weight_map.values())
    shard_names = sorted(path.name for path in base.glob("model-*.safetensors"))
    if not indexed_shards.issubset(shard_names):
        raise FileNotFoundError(
            f"base index references missing shards: {sorted(indexed_shards - set(shard_names))}"
        )
    rewritten = [name for name in shard_names if any(k in replacements for k in by_file[name])]
    static = [name for name in shard_names if name not in rewritten]
    print(
        f"validated checkpoint step={app_metadata.get('step')}, save_mode={save_mode}: "
        f"{len(replacements)} RL language tensors + {len(retained)} retained base tensors"
    )
    print(f"HF shards: {len(rewritten)} rewritten, {len(static)} unchanged; base={base}")
    if dry_run:
        print("dry-run complete: no tensor payloads read and no files written")
        return

    if output.exists():
        if any(output.iterdir()) and not overwrite:
            raise FileExistsError(f"output directory is not empty: {output} (pass --overwrite to replace files)")
    output.mkdir(parents=True, exist_ok=True)

    # Remove only files this exporter owns.  This keeps overwrite predictable
    # while avoiding a recursive deletion of an accidentally supplied path.
    if overwrite:
        for name in shard_names + [MODEL_INDEX]:
            path = output / name
            if path.is_file() or path.is_symlink():
                path.unlink()

    _copy_auxiliary_files(base, output)

    for shard_idx, filename in enumerate(shard_names, start=1):
        src_path = base / filename
        dst_path = output / filename
        keys = by_file[filename]
        replacement_keys = [key for key in keys if key in replacements]
        if not replacement_keys:
            _copy_file(src_path, dst_path, static_mode)
            print(f"[{shard_idx}/{len(shard_names)}] retained {filename} ({len(keys)} tensors, {static_mode})")
            continue

        states: dict[str, torch.Tensor] = {}
        # Retained tensors matter for architectures whose shard mixes frozen
        # vision/FM weights with language weights.
        retained_keys = [key for key in keys if key not in replacements]
        if retained_keys:
            with safe_open(src_path, framework="pt", device="cpu") as f:
                for key in retained_keys:
                    states[key] = f.get_tensor(key)

        for tensor_idx, hf_key in enumerate(replacement_keys, start=1):
            inner_key = replacements[hf_key]
            source_meta = dcp_state_metadata[DCP_CONTAINER_PREFIX + inner_key]
            tensor = _load_one_dcp_tensor(
                checkpoint,
                inner_key,
                tuple(source_meta.size),
                source_meta.properties.dtype,
                storage_reader=storage_reader,
            )
            target_dtype = _torch_dtype_from_safe_name(specs[hf_key][1])
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            states[hf_key] = tensor.contiguous()
            if tensor_idx == 1 or tensor_idx == len(replacement_keys) or tensor_idx % 25 == 0:
                print(
                    f"  {filename}: loaded {tensor_idx}/{len(replacement_keys)} RL tensors",
                    flush=True,
                )

        tmp_path = output / f".{filename}.tmp"
        if tmp_path.exists():
            tmp_path.unlink()
        save_file(states, tmp_path, metadata=shard_metadata[filename])
        os.replace(tmp_path, dst_path)
        size_gib = dst_path.stat().st_size / 2**30
        print(
            f"[{shard_idx}/{len(shard_names)}] wrote {filename}: "
            f"{len(replacement_keys)} RL + {len(retained_keys)} base tensors, {size_gib:.2f} GiB"
        )
        del states

    with (output / MODEL_INDEX).open("w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    # Final structural check catches interrupted copies and writer mistakes.
    out_specs, _, _ = _base_tensor_specs(output, weight_map)
    if out_specs != specs:
        raise RuntimeError("exported HF tensor shapes/dtypes differ from the base index")
    print(f"export complete: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="UniRL checkpoint-<step> directory containing .metadata")
    parser.add_argument("--base", required=True, help="exact inference-ready HF model used as RL SENSENOVA_U1_PATH")
    parser.add_argument("--output", required=True, help="destination inference-ready HF directory")
    parser.add_argument(
        "--static-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="how to retain base shards containing no RL tensors (default: %(default)s)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate metadata, keys and shapes without writing")
    parser.add_argument("--overwrite", action="store_true", help="replace exporter-owned files in an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(
        args.checkpoint,
        args.base,
        args.output,
        static_mode=args.static_mode,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
