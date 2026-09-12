"""Best-effort start of a local Ollama server.

Screenshot OCR needs Ollama, but nothing else in the app touches it: without
this, launching GhostPilot while Ollama happens to be down means every
screenshot fails with a raw `ConnectError` and the only remedy the user is told
about is nothing at all.

Ollama's own installer registers a Startup-folder shortcut, so after a normal
login it is already running and this module's probe succeeds immediately. It
matters for the cases that shortcut does not cover: Ollama was quit or crashed,
its autostart was disabled, or GhostPilot comes up before Ollama has finished
booting.

Only a *local* endpoint is ever started. A configured remote
``OLLAMA_BASE_URL`` means someone else's server — launching a process for that
address would be both useless and surprising.

Disable with ``OLLAMA_AUTOSTART=0``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# A reachable verdict is cached this long, so a screenshot does not pay a probe
# every time while still noticing a server that died mid-session.
_REACHABLE_TTL_SEC = 15.0

# Readiness deadline for a freshly started server. The HTTP port opens well
# before a model is loaded, so this covers process start, not model load.
_START_TIMEOUT_SEC = 45.0
_POLL_INTERVAL_SEC = 0.5

_reachable_until: float = 0.0
_start_lock: asyncio.Lock | None = None
_inflight: asyncio.Task | None = None


def _autostart_enabled() -> bool:
    return (os.getenv("OLLAMA_AUTOSTART", "1") or "1").strip().lower() not in {"0", "false", "no", ""}


def is_local_endpoint(url: str) -> bool:
    """True when `url` points at this machine."""
    try:
        host = (urlparse(url).hostname or "").strip().lower()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def probe(root: str, timeout: float = 2.0) -> bool:
    """True when an Ollama HTTP API answers at `root`. Blocking — call via a thread."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{root.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _windows_candidates() -> list[Path]:
    local = os.getenv("LOCALAPPDATA") or ""
    out = []
    if local:
        out.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
    out.append(Path("C:/Program Files/Ollama/ollama.exe"))
    return out


def _other_candidates() -> list[Path]:
    return [
        Path("/usr/local/bin/ollama"),
        Path("/opt/homebrew/bin/ollama"),
        Path("/usr/bin/ollama"),
        Path("/Applications/Ollama.app/Contents/Resources/ollama"),
    ]


def find_launcher() -> list[str] | None:
    """Command that starts a local server, or None when Ollama isn't installed.

    On Windows the tray application is preferred over ``ollama serve``: it is
    what the installer's Startup entry runs, and it puts the familiar tray icon
    back. ``ollama serve`` is the fallback everywhere and starts the server
    without a UI.
    """
    exe = shutil.which("ollama")
    exe_path = Path(exe) if exe else next(
        (p for p in (*_windows_candidates(), *_other_candidates()) if p.is_file()),
        None,
    )
    if exe_path is None:
        return None
    if sys.platform == "win32":
        tray = exe_path.with_name("ollama app.exe")
        if tray.is_file():
            return [str(tray)]
    return [str(exe_path), "serve"]


def _spawn_detached(cmd: list[str]) -> bool:
    """Start `cmd` in its own process group so it outlives / detaches from us.

    Windows picks up the existing tray instance if one is running, so this is
    safe to call when a server is already starting.
    """
    try:
        kwargs: dict = {"close_fds": True}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(cmd, **kwargs)  # noqa: S603 - fixed, chosen command
        return True
    except OSError as e:
        logger.warning("Could not start Ollama (%s): %s", cmd[0], e)
        return False


def _probe_cached(root: str) -> bool:
    global _reachable_until
    now = time.monotonic()
    if now < _reachable_until:
        return True
    if probe(root):
        _reachable_until = now + _REACHABLE_TTL_SEC
        return True
    return False


async def ensure_available(root: str, *, timeout: float = _START_TIMEOUT_SEC) -> bool:
    """Make sure a local Ollama answers at `root`; start it if it does not.

    Returns True when the endpoint is reachable. Never raises: a failure here
    must not take down the pipeline, it only means OCR will report its own
    error when used.
    """
    global _start_lock, _inflight

    # Cheap path: a recent successful probe, or a non-local endpoint we must not
    # manage.
    if not is_local_endpoint(root):
        return True
    if _probe_cached(root):
        return True

    # A start already in flight: join it rather than spawning a second server.
    if _inflight is not None and not _inflight.done():
        try:
            return await asyncio.shield(_inflight)
        except Exception:  # noqa: BLE001
            return False

    if _start_lock is None:
        _start_lock = asyncio.Lock()

    async def _start_and_wait() -> bool:
        cmd = find_launcher()
        if cmd is None:
            logger.warning(
                "Ollama is not reachable at %s and no ollama executable was found; "
                "screenshot OCR will not work until it is installed and started.",
                root,
            )
            return False
        logger.info("Ollama not reachable at %s — starting %s", root, " ".join(cmd))
        if not _spawn_detached(cmd):
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL_SEC)
            if await asyncio.to_thread(probe, root):
                global _reachable_until
                _reachable_until = time.monotonic() + _REACHABLE_TTL_SEC
                logger.info("Ollama is up at %s.", root)
                return True
        logger.warning("Ollama did not answer at %s within %.0fs.", root, timeout)
        return False

    async with _start_lock:
        # Re-check: another waiter may have completed the start while we queued.
        if _probe_cached(root):
            return True
        _inflight = asyncio.ensure_future(_start_and_wait())
    try:
        return await asyncio.shield(_inflight)
    except Exception as e:  # noqa: BLE001
        logger.warning("Ollama start failed: %s", e)
        return False


async def ensure_available_for_config() -> bool:
    """`ensure_available` for the configured OCR endpoint, honouring the toggle."""
    if not _autostart_enabled():
        return True
    from src.config import config
    from src.ocr_client import ollama_root

    if not is_local_endpoint(getattr(config, "OLLAMA_BASE_URL", "")):
        return True
    root = ollama_root(getattr(config, "OLLAMA_BASE_URL", ""))
    return await ensure_available(root)
