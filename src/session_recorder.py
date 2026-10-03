"""Session recorder (Phase 5.4).

Writes per-session JSONL files to ~/Documents/GhostPilot/recordings/<ts>/:
- transcript.jsonl: ASR partials/finals
- llm.jsonl:        LLM Q/A + usage
- meta.json:        start/end + config snapshot with secrets redacted

On stop the directory is zipped and the original folder removed.

Audio tee + screenshots are intentionally omitted in this first cut.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_SECRET_NAMES = {"OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "AZURE_SPEECH_KEY"}


def _recordings_root() -> Path:
    return Path.home() / "Documents" / "GhostPilot" / "recordings"


def _redact(d: dict) -> dict:
    return {k: ("***" if k in _SECRET_NAMES else v) for k, v in d.items()}


# ── Retention ─────────────────────────────────────────────────────────────
#
# Recordings accumulate otherwise: `stop` writes a `.zip` and nothing ever
# removes one. Two independent caps, applied in a single oldest-first pass and
# both disabled by 0:
#
#   RECORDING_KEEP_LAST     how many recordings to keep
#   RECORDING_MAX_TOTAL_MB  how much disk they may occupy in total
#
# Only entries directly inside the recordings root are considered, so files a
# user keeps elsewhere are never touched, and the recording just written is
# never a candidate — deleting the session the user just stopped would be the
# one unacceptable failure here.


def _recording_entries() -> list[Path]:
    """Recordings under the root, newest first.

    Both shapes count: the `.zip` a completed recording becomes, and the bare
    directory left behind when zipping failed.
    """
    try:
        entries = [p for p in _recordings_root().iterdir() if p.is_dir() or p.suffix == ".zip"]
    except OSError:
        return []
    try:
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    return entries


def _entry_bytes(path: Path) -> int:
    """Size of a recording, following a directory when the zip failed."""
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def _remove_recording(path: Path, size: int) -> bool:
    # `size` is passed in rather than measured here: it has to be read *before*
    # the removal, or the log line reports 0 MB for every deletion.
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as e:
        logger.warning("Retention could not remove %s: %s", path, e)
        return False
    logger.info("Retention removed %s (%.1f MB)", path.name, size / 1048576)
    return True


def prune_recordings(
    *, keep_last: int | None = None, max_total_mb: int | None = None,
    protect: Path | None = None,
) -> list[Path]:
    """Apply the retention policy; return the recordings that were removed.

    ``protect`` (the recording just written) is always kept and still counts
    toward both caps, so ``stop`` cannot delete what it just produced even when
    that recording alone exceeds the size cap.

    The caps default to ``RECORDING_KEEP_LAST`` / ``RECORDING_MAX_TOTAL_MB``;
    passing either argument overrides it, which is what keeps this testable
    without patching global config.
    """
    if keep_last is None or max_total_mb is None:
        from src.config import config
        if keep_last is None:
            keep_last = int(getattr(config, "RECORDING_KEEP_LAST", 10) or 0)
        if max_total_mb is None:
            max_total_mb = int(getattr(config, "RECORDING_MAX_TOTAL_MB", 2048) or 0)

    if keep_last <= 0 and max_total_mb <= 0:
        return []                       # both caps disabled

    cap_bytes = max_total_mb * 1024 * 1024
    removed: list[Path] = []
    kept = 0
    used = 0

    for entry in _recording_entries():
        size = _entry_bytes(entry)
        is_protected = protect is not None and entry == protect
        over_count = keep_last > 0 and kept >= keep_last
        over_size = max_total_mb > 0 and used + size > cap_bytes
        if not is_protected and (over_count or over_size):
            if _remove_recording(entry, size):
                removed.append(entry)
            continue
        kept += 1
        used += size

    if removed:
        logger.info(
            "Recording retention: kept %d (%.1f MB), removed %d "
            "(RECORDING_KEEP_LAST=%d, RECORDING_MAX_TOTAL_MB=%d)",
            kept, used / 1048576, len(removed), keep_last, max_total_mb,
        )
    return removed


class SessionRecorder:
    def __init__(self):
        self._lock = Lock()
        self._dir: Path | None = None
        self._transcript_f = None
        self._llm_f = None
        self._meta: dict[str, Any] = {}
        self._started_at: float = 0.0
        self._screenshot_idx: int = 0

    def is_recording(self) -> bool:
        return self._dir is not None

    def start(self) -> Path | None:
        with self._lock:
            if self._dir is not None:
                return self._dir
            try:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                d = _recordings_root() / stamp
                d.mkdir(parents=True, exist_ok=True)
                self._transcript_f = (d / "transcript.jsonl").open("w", encoding="utf-8")
                self._llm_f = (d / "llm.jsonl").open("w", encoding="utf-8")
                self._dir = d
                self._started_at = time.time()
                self._screenshot_idx = 0
                # Snapshot config minus secrets.
                try:
                    from src.config import config
                    snapshot = {k: getattr(config, k) for k in dir(config) if k.isupper()}
                    self._meta = {
                        "started_at": self._started_at,
                        "config": _redact(snapshot),
                    }
                except Exception:
                    self._meta = {"started_at": self._started_at}
                logger.info("Recording started: %s", d)
                return d
            except Exception as e:
                logger.warning("Failed to start recording: %s", e)
                self._cleanup()
                return None

    def stop(self) -> Path | None:
        """Stop, write meta, zip and remove dir. Returns the .zip path."""
        with self._lock:
            if self._dir is None:
                return None
            d = self._dir
            try:
                self._meta["ended_at"] = time.time()
                self._meta["duration_sec"] = self._meta["ended_at"] - self._started_at
                (d / "meta.json").write_text(
                    json.dumps(self._meta, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.warning("meta.json write failed: %s", e)
            self._cleanup()
            try:
                zip_path = shutil.make_archive(str(d), "zip", root_dir=d)
                shutil.rmtree(d, ignore_errors=True)
                logger.info("Recording saved: %s", zip_path)
                result = Path(zip_path)
            except Exception as e:
                logger.warning("Recording zip failed: %s", e)
                result = d
            # Retention runs here, the one point where a recording is complete:
            # `stop` is the only place a recording is produced, so this is what
            # keeps the old ones bounded. The new file is protected.
            try:
                prune_recordings(protect=result)
            except Exception as e:
                logger.warning("Recording retention failed: %s", e)
            return result

    def _cleanup(self) -> None:
        for f in (self._transcript_f, self._llm_f):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._transcript_f = None
        self._llm_f = None
        self._dir = None

    # ── Logging API ─────────────────────────────────────────────────────
    def log_transcript(self, kind: str, text: str) -> None:
        if self._transcript_f is None:
            return
        try:
            self._transcript_f.write(json.dumps(
                {"t": time.time(), "type": kind, "text": text}, ensure_ascii=False,
            ) + "\n")
            self._transcript_f.flush()
        except Exception:
            pass

    def log_llm(self, role: str, content: str, *, tokens_in: int = 0, tokens_out: int = 0,
                cost_usd: float | None = None, model: str = "") -> None:
        if self._llm_f is None:
            return
        try:
            self._llm_f.write(json.dumps(
                {"t": time.time(), "role": role, "content": content,
                 "tokens_in": tokens_in, "tokens_out": tokens_out,
                 "cost_usd": cost_usd, "model": model},
                ensure_ascii=False,
            ) + "\n")
            self._llm_f.flush()
        except Exception:
            pass

    def log_user_turn(self, question: str, *, q_type: str = "", kind: str = "text") -> None:
        """Log a ``{role:'user'}`` row immediately before the matching LLM
        request. Used by replay to pair each user question with the
        ``{role:'assistant'}`` row that follows it in ``llm.jsonl``.
        """
        if self._llm_f is None:
            return
        try:
            self._llm_f.write(json.dumps(
                {"t": time.time(), "role": "user", "content": question,
                 "q_type": q_type, "kind": kind},
                ensure_ascii=False,
            ) + "\n")
            self._llm_f.flush()
        except Exception:
            pass

    def log_screenshot(self, data: bytes, *, ext: str = "jpg") -> Path | None:
        """Persist a screenshot under the active session dir and log a pointer
        to it in ``llm.jsonl``. Returns the saved path, or None if not recording.
        """
        if self._dir is None:
            return None
        try:
            self._screenshot_idx += 1
            assets = self._dir / "screenshots"
            assets.mkdir(parents=True, exist_ok=True)
            out = assets / f"{self._screenshot_idx:04d}.{ext}"
            out.write_bytes(data)
            if self._llm_f is not None:
                self._llm_f.write(json.dumps(
                    {"t": time.time(), "role": "screenshot",
                     # Always forward-slash so replays load on every OS.
                     "path": out.relative_to(self._dir).as_posix()},
                    ensure_ascii=False,
                ) + "\n")
                self._llm_f.flush()
            return out
        except Exception as e:
            logger.warning("log_screenshot failed: %s", e)
            return None


# Singleton
recorder = SessionRecorder()
