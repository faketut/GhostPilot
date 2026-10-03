#!/usr/bin/env python3
"""Create the optimized GLM-OCR Ollama model used by GhostPilot's screenshot OCR.

    python setup_glm_ocr.py                 # pull glm-ocr, create glm-ocr-optimized
    python setup_glm_ocr.py --threads 4     # override the CPU thread count
    python setup_glm_ocr.py --name my-ocr   # custom model name (set OCR_MODEL to match)

Sampling parameters (temperature 0, top_k 1, …) and the 16k context are pinned
in ``ocr/GLM-Config`` because GLM-OCR is used for exactly one task — text
recognition — and these values are known-good. Only ``num_thread`` is derived
per machine: ``_default_threads`` explains the measurement behind using every
logical core rather than every physical one.

This script also repairs the model's **end-of-generation metadata**, which is
what makes OCR terminate. Ollama's ``glm-ocr`` GGUF ships without
``tokenizer.ggml.eot_token_id``, so ``<|user|>`` — the token the model emits to
end its turn — is never registered as an end-of-generation token and the model
*cannot stop*. It then runs to ``num_predict``, re-emitting the page it has just
transcribed; on Ollama ≥ 0.34.1 the byte-identical replay trips llama.cpp's
token-repeat guard, which is where ``repeat token`` / "limit exceeded" errors
come from. Measured on one synthetic banner image: 2000 tokens / 141 s
unpatched → **21 tokens / 8.6 s** patched, same transcript. On a full code page:
2000 → 334 tokens.

The repair is a metadata edit on a *copy* of the blob (``pip install gguf``),
because neither a prompt change nor ``PARAMETER stop`` can fix it: the token is
a control token, so it never appears in the decoded text a stop string is
matched against. If ``gguf`` is missing the script still builds the model and
says so — the repetition guard in ``src/ocr_client.py`` then has to catch it.
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

# The token GLM-OCR emits to end a turn. Ollama's glm-ocr GGUF leaves it out of
# `tokenizer.ggml.eot_token_id`, which is the whole reason OCR does not stop
# (see the module docstring).
EOT_TOKEN = "<|user|>"
# Two copies of the model exist at once while patching (~2.2 GB each), so the
# repair is skipped rather than half-written on a full disk.
_MIN_FREE_BYTES = 5 * 1024**3


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


def _default_threads() -> int | None:
    """Thread count for the model: every *logical* core.

    Measured on the 4-core/8-thread i7-1165G7 this was developed on, at 1024px
    (min of repeated runs, prompt cache defeated):

    ===============  ==========  ============
    num_thread       prefill     decode
    ===============  ==========  ============
    4 (physical)     23.2 s      18.7 tok/s
    6                31.2 s      23.7 tok/s
    **8 (logical)**  **18.5 s**  **23.9 tok/s**
    10               28.9 s      19.9 tok/s
    12               29.8 s      21.8 tok/s
    ===============  ==========  ============

    So all logical cores win on both phases and oversubscribing beyond them
    loses again. This reverses the earlier assumption that pinning to physical
    cores was faster — that was measured on decode alone, where the effect was
    small enough to dismiss, and prefill was never measured separately.

    The vision encoder is memory-latency-bound, which is why hyperthreads still
    help: the extra threads hide DRAM latency instead of competing for an ALU.
    """
    return os.cpu_count() or None


def _server_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _run(ollama: str, args: list[str]) -> int:
    print(f"$ ollama {' '.join(args)}")
    return subprocess.call([ollama, *args])


def _model_blob(ollama: str, model: str) -> Path | None:
    """The GGUF file a model was created FROM.

    ``ollama show --modelfile`` prints the resolved blob path, which is the only
    handle on the file: the blob has no extension and lives under a content
    hash, so it cannot be found by name.
    """
    try:
        out = subprocess.run(
            [ollama, "show", "--modelfile", model],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    m = re.search(r"^FROM\s+(.+?)\s*$", out.stdout, re.M)
    if not m:
        return None
    path = Path(m.group(1).strip().strip('"'))
    return path if path.is_file() else None


def _free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path.anchor or path).free
    except OSError:
        return 0


def _repair_eot(ollama: str, base: str, dest: Path) -> tuple[str, str]:
    """Write a copy of the base model's GGUF with a working EOG token.

    Returns ``(status, detail)`` with status one of:

    * ``"ok"`` — a repaired GGUF was written to ``dest``; build FROM it.
    * ``"skip"`` — the metadata is already correct; build FROM the base model.
    * ``"failed"`` — the repair could not be applied; build FROM the base model
      and let the repetition guard handle the runaways.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        return "failed", "the `gguf` package is not installed (pip install gguf)"

    blob = _model_blob(ollama, base)
    if blob is None:
        return "failed", f"could not resolve the GGUF behind model '{base}'"

    try:
        reader = GGUFReader(str(blob))
        field = reader.get_field("tokenizer.ggml.tokens")
        tokens = [bytes(field.parts[off]).decode("utf-8", "replace") for off in field.data]
    except Exception as e:  # noqa: BLE001 — any unreadable/odd GGUF lands here
        return "failed", f"could not read {blob.name}: {e}"

    if EOT_TOKEN not in tokens:
        return "failed", f"{EOT_TOKEN} is not in the tokenizer"
    wanted = tokens.index(EOT_TOKEN)

    # `get_field` returns None (not KeyError) when the key is absent — which is
    # the state this repair exists for.
    current_field = reader.get_field("tokenizer.ggml.eot_token_id")
    try:
        current = int(current_field.parts[-1][0]) if current_field is not None else None
    except (AttributeError, IndexError, TypeError, ValueError):
        current = None
    if current == wanted:
        return "skip", f"{EOT_TOKEN} is already registered as end-of-generation (id {wanted})"

    if _free_bytes(dest.parent) < _MIN_FREE_BYTES:
        return "failed", f"less than {_MIN_FREE_BYTES // 1024**3} GB free for the model copy"

    # Rewriting the 2.2 GB GGUF takes ~30 s; the driver is the same module the
    # `gguf-new-metadata` console script wraps, so only the package is required.
    rc = subprocess.call([
        sys.executable, "-m", "gguf.scripts.gguf_new_metadata",
        "--special-token-by-id", "eot", str(wanted), str(blob), str(dest),
    ])
    if rc != 0 or not dest.is_file():
        return "failed", "rewriting the GGUF metadata failed (see output above)"
    return "ok", f"registered {EOT_TOKEN} (id {wanted}) as end-of-generation"


def main() -> int:
    ap = argparse.ArgumentParser(description="Create the optimized GLM-OCR Ollama model.")
    ap.add_argument("--name", default=DEFAULT_NAME, help=f"model name to create (default: {DEFAULT_NAME})")
    ap.add_argument("--base", default=BASE_MODEL, help=f"base model to derive from (default: {BASE_MODEL})")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: all detected logical cores)")
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

    # Repair the end-of-generation metadata first: without it the model cannot
    # stop and OCR runs to the token cap on every screenshot.
    workdir = tempfile.mkdtemp(prefix="glm-ocr-eot-")
    repaired = Path(workdir) / "glm-ocr-eog.gguf"
    status, detail = _repair_eot(ollama, args.base, repaired)
    if status == "ok":
        from_ref = repaired
        print(f"[i] {detail} — the model can now end its turn on its own.")
    else:
        from_ref = Path(args.base)
        if status == "skip":
            print(f"[i] {detail}.")
        else:
            print(f"[!] End-of-generation repair not applied: {detail}", file=sys.stderr)
            print(
                "    The model will run to num_predict and re-emit the page, which is\n"
                "    what surfaces as 'repeat token' / limit errors. Fix with:\n"
                "        pip install gguf   (then re-run this script)",
                file=sys.stderr,
            )
    # Function replacement, not a template: a Windows path is full of
    # backslashes and `re.sub` would read them as escapes (bare `\U…`).
    text = re.sub(r"^FROM\s+\S+", lambda _m: f'FROM "{from_ref}"', text, count=1, flags=re.M)

    if status == "ok":
        # Pointing FROM at a raw GGUF drops everything the `glm-ocr` model
        # definition contributed *by inheritance* — including `RENDERER glm-ocr`
        # / `PARSER glm-ocr`, which is what wires up the image input. Without
        # them the model never sees the screenshot and just continues the prompt
        # ("Output only the text. Output only the text. …"), which looks like a
        # corrupted model but is only a lost renderer. Restate them so the
        # derived model is the base model plus the metadata repair.
        for directive in ("TEMPLATE {{ .Prompt }}", "RENDERER glm-ocr", "PARSER glm-ocr"):
            key = directive.split()[0]
            if not re.search(rf"^{key}\b", text, re.M):
                text += f"\n{directive}\n"

    threads = args.threads or _default_threads()
    if threads:
        if re.search(r"^PARAMETER num_thread\s+\d+", text, re.M):
            text = re.sub(r"^PARAMETER num_thread\s+\d+", f"PARAMETER num_thread {threads}", text, flags=re.M)
        else:
            text += f"\nPARAMETER num_thread {threads}\n"
        print(f"[i] Using {threads} CPU threads (all logical cores).")

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
        # `ollama create` has copied the repaired GGUF into its own blob store,
        # so the working copy is 2.2 GB of dead weight.
        shutil.rmtree(workdir, ignore_errors=True)

    print(
        f"\n✅  Model '{args.name}' is ready.\n"
        f"    GhostPilot uses it when OCR_MODEL={args.name} (the default).\n"
        + ("    End-of-generation token registered — recognition now stops by itself.\n"
           if status == "ok" else "")
        + "    Verify in the app: Settings → LLM → Screenshot OCR → test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
