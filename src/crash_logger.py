"""
Crash / error logging to disk.

On Windows: %LOCALAPPDATA%/GhostPilot/logs/crash.log
On macOS:   ~/Library/Logs/GhostPilot/crash.log
Otherwise:  ~/.local/share/GhostPilot/logs/crash.log

Installs both `sys.excepthook` and a Qt message handler. The asyncio loop's
exception handler is wired separately in main.py because it needs the loop
instance.

The installed `sys.excepthook` is also what keeps PyQt slot bugs survivable:
with the stock excepthook, an unhandled exception raised inside a slot invoked
from C++ makes PyQt6 call qFatal()/abort(), killing the process with exit code
0xC0000409 and no traceback (silent under pythonw). With a custom excepthook it
is reported and the app keeps running. So `install()` MUST run before any Qt
object is created — main.py calls it early in `main()` for exactly that reason.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

_logger = logging.getLogger(__name__)
_installed = False


def _log_dir() -> Path:
    sysname = platform.system()
    if sysname == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "GhostPilot" / "logs"
    if sysname == "Darwin":
        return Path.home() / "Library" / "Logs" / "GhostPilot"
    return Path.home() / ".local" / "share" / "GhostPilot" / "logs"


def crash_log_path() -> Path:
    return _log_dir() / "crash.log"


def install(level: int = logging.WARNING) -> Path | None:
    """Attach the rotating file handler + excepthook. Idempotent.

    ``level`` is the file handler's threshold; background (console-less) runs
    pass INFO so the service still leaves a log trail.
    """
    global _installed
    if _installed:
        return crash_log_path()

    path = crash_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )
        root = logging.getLogger()
        root.addHandler(handler)
    except Exception as e:
        # Never let crash logger setup itself crash the app.
        _logger.warning(f"Failed to attach crash log handler at {path}: {e}")
        return None

    prev_hook = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        try:
            logging.getLogger("crash").critical(
                "Uncaught exception:\n" + "".join(traceback.format_exception(exc_type, exc, tb))
            )
        finally:
            prev_hook(exc_type, exc, tb)

    sys.excepthook = _excepthook
    _installed = True
    return path
