#!/usr/bin/env python3
"""Convert a UniRL GeoWeave RL DCP checkpoint to Hugging Face format.

The CLI intentionally matches ``tools/revert2hf.py`` so SFT and RL conversion
commands differ only in the script name::

    # SFT
    python tools/revert2hf.py \
        --src RUN/.../4000 \
        --tgt RUN/.../hf4000 \
        --extras-from /path/to/GeoWeave-HF

    # RL
    python tools/revert_rl2hf.py \
        --src /path/to/checkpoint-200 \
        --tgt /path/to/hf_checkpoint-200 \
        --extras-from /path/to/GeoWeave-HF

For RL, ``--extras-from`` must be the exact inference-ready Hugging Face model
used as ``SENSENOVA_U1_PATH`` when the RL run was launched.  Its frozen
vision/FM tensors, tokenizer/config files, and safetensor shard layout are
retained; the DCP language-model tensors from ``--src`` replace the base LLM
weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Add this repository's root to sys.path so the command works even when
# the package is not installed editable in the active virtual environment.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if not (_REPO_ROOT / "unirl").is_dir():
    raise RuntimeError(f"training repository not found: {_REPO_ROOT}")
sys.path.insert(0, str(_REPO_ROOT))

from unirl.tools.export_geoweave_dcp import export  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src",
        required=True,
        help="RL checkpoint-<step> directory containing .metadata, metadata.pt, and *.distcp",
    )
    parser.add_argument(
        "--tgt",
        required=True,
        help="output inference-ready Hugging Face directory",
    )
    parser.add_argument(
        "--extras-from",
        required=True,
        help="exact inference-ready HF base used as SENSENOVA_U1_PATH for the RL run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate checkpoint keys/shapes without reading tensor payloads or writing output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace exporter-owned files if --tgt already exists",
    )
    parser.add_argument(
        "--static-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="how to retain base shards containing no RL tensors (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(
        checkpoint=args.src,
        base=args.extras_from,
        output=args.tgt,
        static_mode=args.static_mode,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
