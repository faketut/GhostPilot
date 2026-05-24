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
                return Path(zip_path)
            except Exception as e:
                logger.warning("Recording zip failed: %s", e)
                return d

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
                     "path": str(out.relative_to(self._dir))},
                    ensure_ascii=False,
                ) + "\n")
                self._llm_f.flush()
            return out
        except Exception as e:
            logger.warning("log_screenshot failed: %s", e)
            return None


# Singleton
recorder = SessionRecorder()
