"""Model registry + downloader for the sherpa-onnx ASR backend.

``pip install sherpa-onnx`` ships the *runtime* only; the acoustic models are
separate downloads (74 MB – 1 GB). This module is the single place that knows
where they come from, which files each one contains and where they live on
disk, so the ASR client, the Settings test button and ``setup_sherpa_asr.py``
cannot disagree about it.

On-disk layout::

    ~/.ghostpilot/models/sherpa-onnx/<registry-name>/
        tokens.txt
        encoder-….onnx
        decoder-….onnx
        joiner-….onnx        (transducer models only)

``SHERPA_MODEL_DIR`` relocates the root. ``SHERPA_MODEL`` may also be a path to
a directory of your own — anything already extracted (e.g. straight from the
release tarball) is detected in place, no re-download.
"""
from __future__ import annotations

import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

# Progress callback: (bytes_done, bytes_total) — total is 0 when the server
# sends no Content-Length.
ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True)
class SherpaModel:
    """One downloadable model. ``files()`` is the authority on what it contains.

    ``size_mb`` is the compressed (download) size, for the "this is a big
    download" messages — the extracted int8 files are smaller for most entries.
    ``languages`` is the *model's* coverage, not a setting: unlike Azure, a
    sherpa-onnx backend cannot be pointed at another language at runtime, so a
    mismatch with ``ASR_LANGUAGE`` is a silent no-transcription, not a slow path.
    """

    name: str    # registry key, i.e. the value of SHERPA_MODEL
    folder: str  # archive basename; also the top-level directory it extracts to
    label: str
    arch: str    # "transducer" | "paraformer"
    size_mb: int
    languages: tuple[str, ...]
    note: str = ""
    # Archive file names, when they do not follow the epoch-99-avg-1 convention
    # (role -> filename). Verified against the model's own release page.
    filenames: tuple[tuple[str, str], ...] = ()

    @property
    def url(self) -> str:
        return f"{_RELEASE}/{self.folder}.tar.bz2"

    def files(self) -> dict[str, str]:
        """role → filename inside the model directory (int8 quantised only).

        int8 rather than fp32 throughout: 3-5x smaller and 2-4x faster on CPU,
        which is the whole point of running this locally next to the OCR model.
        """
        if self.filenames:
            return dict(self.filenames)
        if self.arch == "paraformer":
            return {
                "tokens": "tokens.txt",
                "encoder": "encoder.int8.onnx",
                "decoder": "decoder.int8.onnx",
            }
        return {
            "tokens": "tokens.txt",
            "encoder": "encoder-epoch-99-avg-1.int8.onnx",
            "decoder": "decoder-epoch-99-avg-1.int8.onnx",
            "joiner": "joiner-epoch-99-avg-1.int8.onnx",
        }


# File names verified against the per-model pages at
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/
# and .../online-paraformer/ .
MODELS: dict[str, SherpaModel] = {
    "zipformer-bilingual-zh-en": SherpaModel(
        name="zipformer-bilingual-zh-en",
        folder="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        label="Zipformer · Chinese + English (default)",
        arch="transducer",
        size_mb=511,
        languages=("zh", "en"),
    ),
    "zipformer-small-bilingual-zh-en": SherpaModel(
        name="zipformer-small-bilingual-zh-en",
        folder="sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16",
        label="Zipformer small · Chinese + English",
        arch="transducer",
        size_mb=458,
        languages=("zh", "en"),
        note="half the encoder of the default; 64/ sub-folders trade RTF for latency",
    ),
    "zipformer-zh-14M": SherpaModel(
        name="zipformer-zh-14M",
        folder="sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
        label="Zipformer 14M · Chinese only (smallest)",
        arch="transducer",
        size_mb=74,
        languages=("zh",),
        note="lowest CPU cost of the set; Chinese only",
    ),
    "zipformer-en-20M": SherpaModel(
        name="zipformer-en-20M",
        folder="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        label="Zipformer 20M · English only",
        arch="transducer",
        size_mb=128,
        languages=("en",),
        note="English only — measured on synthesized speech it drops the opening "
             "words of an utterance (see README), so prefer the bilingual or Kroko model",
    ),
    "zipformer-en-kroko": SherpaModel(
        name="zipformer-en-kroko",
        folder="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
        label="Zipformer Kroko · English only (recommended for English)",
        arch="transducer",
        size_mb=54,
        languages=("en",),
        note="2025 model, smallest download of the English set",
        filenames=(
            ("tokens", "tokens.txt"),
            ("encoder", "encoder.onnx"),
            ("decoder", "decoder.onnx"),
            ("joiner", "joiner.onnx"),
        ),
    ),
    "paraformer-bilingual-zh-en": SherpaModel(
        name="paraformer-bilingual-zh-en",
        folder="sherpa-onnx-streaming-paraformer-bilingual-zh-en",
        label="Paraformer · Chinese + English",
        arch="paraformer",
        size_mb=1048,
        languages=("zh", "en"),
        note="non-autoregressive decoder; heavier download, no hotword support",
    ),
}

DEFAULT_MODEL = "zipformer-bilingual-zh-en"

_BYTES_PER_MB = 1 << 20
_CHUNK = 1 << 20


def default_root() -> Path:
    """Model root: ``SHERPA_MODEL_DIR``, else ``~/.ghostpilot/models/sherpa-onnx``.

    Reads ``config`` at call time (not import time) so a Settings change hot-patches
    the location without a restart — same reason ``config._secret`` imports late.
    """
    from src.config import config

    custom = str(getattr(config, "SHERPA_MODEL_DIR", "") or "").strip()
    return Path(custom).expanduser() if custom else Path.home() / ".ghostpilot" / "models" / "sherpa-onnx"


def find(label: str) -> Optional[SherpaModel]:
    """Look up an entry by registry name *or* by its extracted folder name.

    Both are things a user will reasonably put in ``SHERPA_MODEL``: the short
    name from `--list`, or the folder that a manual download left behind.
    """
    key = (label or "").strip()
    if key in MODELS:
        return MODELS[key]
    for entry in MODELS.values():
        if entry.folder == key:
            return entry
    return None


def get(name: str) -> SherpaModel:
    entry = find(name)
    if entry is None:
        raise ValueError(
            f"unknown sherpa-onnx model '{name}'. Known names: {', '.join(MODELS)}"
        )
    return entry


def model_dir(entry: SherpaModel, root: Optional[str | Path] = None) -> Path:
    base = Path(root).expanduser() if root else default_root()
    return base / entry.folder


def missing_files(entry: SherpaModel, root: Optional[str | Path] = None) -> list[str]:
    d = model_dir(entry, root)
    return [name for name in entry.files().values() if not (d / name).is_file()]


def is_ready(entry: SherpaModel, root: Optional[str | Path] = None) -> bool:
    return not missing_files(entry, root)


def _detect_dir(d: Path) -> dict[str, str]:
    """Locate the model files in a user-supplied (already extracted) directory.

    int8 wins over fp32 when both are present, matching what the registry ships.
    """
    if not (d / "tokens.txt").is_file():
        return {}

    def pick(role: str) -> str:
        candidates = sorted(d.glob(f"*{role}*.onnx"))
        for c in candidates:
            if ".int8." in c.name:
                return c.name
        return candidates[0].name if candidates else ""

    out = {"tokens": "tokens.txt"}
    for role in ("encoder", "decoder", "joiner"):
        found = pick(role)
        if found:
            out[role] = found
    return out


def _require(files: dict[str, str], where: str) -> None:
    if not files.get("tokens") or not files.get("encoder") or not files.get("decoder"):
        raise RuntimeError(
            f"no sherpa-onnx model found in '{where}': need tokens.txt, an "
            "*encoder*.onnx and a *decoder*.onnx (see "
            f"{_RELEASE.rsplit('/', 3)[0]}/releases/tag/asr-models)."
        )


def resolve(
    model: str = "",
    *,
    root: Optional[str | Path] = None,
    allow_download: bool = True,
    on_progress: Optional[ProgressFn] = None,
) -> dict[str, str]:
    """Return absolute paths by role for ``model`` (name, path, or "" for default).

    Downloads first when the model is a registry entry that is not on disk yet
    and ``allow_download`` is set; otherwise raises with the exact command to run.
    """
    name = (model or "").strip() or DEFAULT_MODEL
    as_path = Path(name).expanduser()
    if as_path.is_dir():
        files = _detect_dir(as_path)
        _require(files, str(as_path))
        return {role: str((as_path / fname).resolve()) for role, fname in files.items()}

    entry = get(name)
    d = model_dir(entry, root)
    if not is_ready(entry, root):
        if not allow_download:
            raise RuntimeError(_download_hint(entry))
        logger.info(
            "sherpa-onnx model '%s' missing — downloading ~%d MB (one time) …",
            entry.name, entry.size_mb,
        )
        d = download(entry, root=root, on_progress=on_progress)
    files = entry.files()
    return {role: str(d / fname) for role, fname in files.items()}


def _download_hint(entry: SherpaModel) -> str:
    return (
        f"model '{entry.name}' is not downloaded — run: "
        f"python setup_sherpa_asr.py --model {entry.name} "
        "(or set SHERPA_MODEL_DIR / point SHERPA_MODEL at an already-extracted directory)"
    )


def _fetch(url: str, dest: Path, *, on_progress: Optional[ProgressFn] = None) -> int:
    """Stream ``url`` to ``dest``. Returns bytes written."""
    done = 0
    next_pct = 5
    # Generous read timeout: a slow GitHub release download must not be killed
    # mid-file, while a stalled connection still fails in finite time.
    timeout = httpx.Timeout(30.0, read=120.0)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(_CHUNK):
                f.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total)
                elif total and done * 100 // total >= next_pct:
                    logger.info("  … %d%% of %d MB", done * 100 // total, total // _BYTES_PER_MB)
                    next_pct += 5
    logger.info("  downloaded %d MB", done // _BYTES_PER_MB)
    return done


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract into ``dest``, refusing anything that escapes it.

    Trusted upstream (k2-fsa release artefacts), but a path-escaping or symlink
    member would write outside the model directory, so it is checked rather than
    assumed. The explicit ``filter="data"`` is the same policy for Python ≥3.12,
    which otherwise warns about the changing default.
    """
    for member in tf.getmembers():
        parts = PurePosixPath(member.name).parts
        if member.name.startswith(("/", "\\")) or ".." in parts or ":" in (parts[0] if parts else ""):
            raise RuntimeError(f"refusing to extract unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"refusing to extract link member: {member.name}")
        try:
            tf.extract(member, dest, filter="data")
        except TypeError:  # Python < 3.12 has no `filter` parameter
            tf.extract(member, dest)


def download(
    entry: SherpaModel,
    *,
    root: Optional[str | Path] = None,
    on_progress: Optional[ProgressFn] = None,
) -> Path:
    """Fetch + extract ``entry`` into the model root. Returns its directory.

    Extraction happens into a staging directory that is moved into place only
    after the whole archive unpacked, so an interrupted download can never leave
    a directory that ``is_ready()`` would accept.
    """
    base = Path(root).expanduser() if root else default_root()
    base.mkdir(parents=True, exist_ok=True)
    target = base / entry.folder
    archive = base / f"{entry.folder}.tar.bz2"
    staging = base / f".{entry.folder}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        _fetch(entry.url, archive, on_progress=on_progress)
        staging.mkdir()
        with tarfile.open(archive, "r:bz2") as tf:
            _safe_extract(tf, staging)
        extracted = staging / entry.folder
        if not extracted.is_dir():
            raise RuntimeError(
                f"unexpected archive layout: no '{entry.folder}/' directory in {entry.url}"
            )
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(extracted), str(target))
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
    logger.info("sherpa-onnx model '%s' ready at %s", entry.name, target)
    return target


def describe() -> str:
    """Human-readable registry listing (``setup_sherpa_asr.py --list``)."""
    lines = ["sherpa-onnx models (set SHERPA_MODEL to one of these names):", ""]
    for m in MODELS.values():
        mark = " *" if m.name == DEFAULT_MODEL else "  "
        lines.append(f"{mark} {m.name:<32} {m.size_mb:>5} MB  {m.label}")
        covers = " / ".join(m.languages)
        detail = f"covers: {covers}"
        if m.note:
            detail += f" — {m.note}"
        lines.append(f"{'':<42}{detail}")
    lines += ["", f"* = default.  Model directory: {default_root()}"]
    return "\n".join(lines)
