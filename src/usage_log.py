"""Append-only JSONL usage log.

Best-effort: every failure is swallowed so logging never breaks a request.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def default_path() -> Path:
    custom = os.environ.get("USAGE_LOG_PATH", "").strip()
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".ghostpilot" / "usage.jsonl"


def log_usage(event: dict, *, path: Optional[Path] = None, enabled: bool = True) -> bool:
    """Append `event` as a single JSON line. Returns True on success."""
    if not enabled:
        return False
    target = path or default_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), **event}
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("usage_log write failed: %s", e)
        return False


def read_usage(path: Optional[Path] = None) -> list[dict]:
    """Read all logged events; empty list if missing/invalid."""
    target = path or default_path()
    if not target.exists():
        return []
    out: list[dict] = []
    try:
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:  # noqa: BLE001
        logger.warning("usage_log read failed: %s", e)
    return out
