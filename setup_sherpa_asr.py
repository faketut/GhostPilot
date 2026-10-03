#!/usr/bin/env python3
"""Download sherpa-onnx ASR models for GhostPilot's local speech backend.

Separate from ``pip install sherpa-onnx`` on purpose: the package is 19 MB, the
acoustic models are 74 MB – 1 GB, and they only matter if you set
``ASR_BACKEND=sherpa``. GhostPilot itself downloads on first use
(``SHERPA_AUTODOWNLOAD``); this script exists so the download can happen up
front — before an interview, on a fast connection, with visible progress.

Usage::

    python setup_sherpa_asr.py --list
    python setup_sherpa_asr.py                        # default model
    python setup_sherpa_asr.py --model zipformer-zh-14M
    python setup_sherpa_asr.py --model /path/to/already-extracted-dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import sherpa_models  # noqa: E402


def _progress(done: int, total: int) -> None:
    mb = done / (1 << 20)
    if total:
        pct = done * 100 // total
        print(f"\r  {pct:3d}%  {mb:7.1f} / {total / (1 << 20):.1f} MB", end="", flush=True)
    else:
        print(f"\r  {mb:7.1f} MB", end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download sherpa-onnx ASR models for GhostPilot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default=sherpa_models.DEFAULT_MODEL,
                    help=f"registry name or an already-extracted model directory (default: {sherpa_models.DEFAULT_MODEL})")
    ap.add_argument("--dir", default="", help="model root (default: ~/.ghostpilot/models/sherpa-onnx)")
    ap.add_argument("--list", action="store_true", help="list known models and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if the model is already present")
    args = ap.parse_args()

    if args.list:
        print(sherpa_models.describe())
        return 0

    try:
        import sherpa_onnx  # noqa: F401
    except Exception:
        print("sherpa-onnx is not installed. Run:\n\n    pip install sherpa-onnx\n")
        return 1

    root = args.dir.strip() or None

    # A path is not a registry entry: nothing to download, just verify it.
    if Path(args.model).expanduser().is_dir():
        try:
            files = sherpa_models.resolve(args.model, root=root, allow_download=False)
        except Exception as e:
            print(f"error: {e}")
            return 1
        print(f"Model directory OK ({args.model}):")
        for role, path in sorted(files.items()):
            print(f"  {role:<8} {path}")
        return 0

    try:
        entry = sherpa_models.get(args.model)
    except ValueError as e:
        print(f"error: {e}")
        return 1

    target = sherpa_models.model_dir(entry, root)
    if sherpa_models.is_ready(entry, root) and not args.force:
        print(f"Already present: {target}")
        print("Nothing to do (use --force to re-download).")
        return 0

    missing = sherpa_models.missing_files(entry, root)
    if missing:
        print(f"Missing from {target}: {', '.join(missing)}")

    print(f"{entry.label}")
    print(f"  {entry.size_mb} MB download → {target}")
    print(f"  {entry.url}")
    if entry.note:
        print(f"  note: {entry.note}")
    print()
    try:
        sherpa_models.download(entry, root=root, on_progress=_progress)
    except KeyboardInterrupt:
        print("\nAborted; nothing was extracted.")
        return 130
    except Exception as e:
        print(f"\nerror: {type(e).__name__}: {e}")
        return 1
    print()
    print("Done. Set the backend and restart the app:")
    print()
    print("    ASR_BACKEND=sherpa")
    print(f"    SHERPA_MODEL={entry.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
