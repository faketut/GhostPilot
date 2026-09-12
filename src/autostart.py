"""
Windows login autostart — the practical "background service" for GhostPilot.

Registers a per-user entry under
``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run`` so the copilot comes
up with the session, tray-only, with no console window and no taskbar button.

Why not a real Windows service: Session 0 isolation gives services no desktop,
so the overlays, global hotkeys, screenshot capture and WASAPI loopback capture
are all unavailable there. The GUI must live in the interactive user session,
and a tray-resident process is the correct shape for it.

Non-Windows platforms are a no-op (``is_enabled`` → False, ``set_enabled`` →
False) and the tray action is disabled.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "GhostPilot"
LAUNCHER_NAME = "GhostPilot.pyw"


def _app_root() -> Path:
    """Directory holding ``main.py`` / the frozen executable."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def launch_command() -> str | None:
    """Command line Windows should run at login, or None when unavailable.

    Frozen builds point at the (windowed) executable; source checkouts run the
    launcher under ``pythonw.exe`` so the login entry never opens a console.
    """
    if sys.platform != "win32":
        return None
    if getattr(sys, "frozen", False):
        return f'"{Path(sys.executable).resolve()}"'
    pythonw = Path(sys.executable).resolve().with_name("pythonw.exe")
    if not pythonw.exists():
        logger.warning("pythonw.exe not found next to %s; autostart unavailable.", sys.executable)
        return None
    launcher = _app_root() / LAUNCHER_NAME
    entry = launcher if launcher.exists() else _app_root() / "main.py"
    return f'"{pythonw}" "{entry}"'


def is_enabled() -> bool:
    """True when the HKCU Run entry is present."""
    if sys.platform != "win32":
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, VALUE_NAME)
        return bool(value)
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.warning("Could not read the autostart entry: %s", e)
        return False


def set_enabled(enabled: bool) -> bool:
    """Create (True) or delete (False) the login entry. Returns success."""
    if sys.platform != "win32":
        return False
    try:
        import winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            if enabled:
                command = launch_command()
                if not command:
                    return False
                winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)
                logger.info("Autostart enabled: %s", command)
            else:
                try:
                    winreg.DeleteValue(key, VALUE_NAME)
                    logger.info("Autostart disabled.")
                except FileNotFoundError:
                    pass
        return is_enabled() == bool(enabled)
    except OSError as e:
        logger.warning("Could not update the autostart entry: %s", e)
        return False
