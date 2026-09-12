#!/usr/bin/env python3
"""Create the optimized GLM-OCR Ollama model used by GhostPilot's screenshot OCR.

    python setup_glm_ocr.py                 # pull glm-ocr, create glm-ocr-optimized
    python setup_glm_ocr.py --threads 4     # override the CPU thread count
    python setup_glm_ocr.py --name my-ocr   # custom model name (set OCR_MODEL to match)

Sampling parameters (temperature 0, top_k 1, …) and the 16k context are pinned
in ``ocr/GLM-Config`` because GLM-OCR is used for exactly one task — text
recognition — and these values are known-good. Only ``num_thread`` is derived
per machine: the model is far faster when pinned to *physical* cores.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

BASE_MODEL = "glm-ocr"
DEFAULT_NAME = "glm-ocr-optimized"
MODELFILE = Path(__file__).resolve().parent / "ocr" / "GLM-Config"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _find_ollama() -> str | None:
    """Locate the ollama CLI: PATH first, then the default install locations."""
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe",
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
        Path("/opt/homebrew/bin/ollama"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _physical_cores() -> int | None:
    """Best-effort physical (not logical) core count for the current machine."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            cores = [int(n) for n in re.findall(r"\d+", out)]
            if cores:
                return sum(cores)
        elif sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            if out.strip().isdigit():
                return int(out.strip())
        else:
            txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            per_socket = [int(m) for m in re.findall(r"^cpu cores\s*:\s*(\d+)", txt, re.M)]
            sockets = len(re.findall(r"^physical id\s*:", txt, re.M)) or 1
            if per_socket:
                return sum(per_socket) if len(per_socket) == sockets else per_socket[0] * sockets
    except Exception:
        pass
    logical = os.cpu_count() or 0
    # Hyper-threading is the norm; halving is a better guess than the logical count.
    return max(1, logical // 2) if logical else None


def _server_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _run(ollama: str, args: list[str]) -> int:
    print(f"$ ollama {' '.join(args)}")
    return subprocess.call([ollama, *args])


def main() -> int:
    ap = argparse.ArgumentParser(description="Create the optimized GLM-OCR Ollama model.")
    ap.add_argument("--name", default=DEFAULT_NAME, help=f"model name to create (default: {DEFAULT_NAME})")
    ap.add_argument("--base", default=BASE_MODEL, help=f"base model to derive from (default: {BASE_MODEL})")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: detected physical cores)")
    ap.add_argument("--skip-pull", action="store_true", help="do not pull the base model")
    args = ap.parse_args()

    ollama = _find_ollama()
    if not ollama:
        print("[ERROR] ollama CLI not found. Install it from https://ollama.com/download", file=sys.stderr)
        return 1

    if not _server_reachable():
        print(f"[ERROR] No Ollama server at {OLLAMA_HOST}. Start Ollama and re-run.", file=sys.stderr)
        return 1

    if not args.skip_pull:
        if _run(ollama, ["pull", args.base]) != 0:
            print(f"[ERROR] Could not pull {args.base}.", file=sys.stderr)
            return 1

    if not MODELFILE.is_file():
        print(f"[ERROR] Modelfile missing: {MODELFILE}", file=sys.stderr)
        return 1
    text = MODELFILE.read_text(encoding="utf-8")
    text = re.sub(r"^FROM\s+\S+", f"FROM {args.base}", text, count=1, flags=re.M)

    threads = args.threads or _physical_cores()
    if threads:
        if re.search(r"^PARAMETER num_thread\s+\d+", text, re.M):
            text = re.sub(r"^PARAMETER num_thread\s+\d+", f"PARAMETER num_thread {threads}", text, flags=re.M)
        else:
            text += f"\nPARAMETER num_thread {threads}\n"
        print(f"[i] Using {threads} CPU threads (physical cores).")

    # ollama create needs a real file; keep it out of the repo.
    fd, tmp = tempfile.mkstemp(prefix="glm-ocr-", suffix=".Modelfile")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        if _run(ollama, ["create", args.name, "-f", tmp]) != 0:
            print("[ERROR] ollama create failed.", file=sys.stderr)
            return 1
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    print(
        f"\n✅  Model '{args.name}' is ready.\n"
        f"    GhostPilot uses it when OCR_MODEL={args.name} (the default).\n"
        "    Verify in the app: Settings → LLM → Screenshot OCR → test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
