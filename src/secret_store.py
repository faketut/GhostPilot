"""
Secret storage with OS keyring + plain-text fallback.

Reads (priority):
    1. OS keyring (Windows Credential Manager / macOS Keychain / SecretService)
    2. config.json key (legacy, still used as fallback so old installs keep working)
    3. environment variable (handled by callers; this module is keyring-only)

Writes:
    - Try keyring first; on failure, return False so caller can fall back to
      config.json (which keeps the install working on locked-down systems).

`SECRET_KEYS` is the canonical list of config attributes that should be treated
as secrets. settings_ui.py uses it to know which fields to migrate / strip.
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)

_SERVICE = "GhostPilot"

# All config attribute names that should live in the keyring rather than disk.
SECRET_KEYS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "AZURE_SPEECH_KEY",
)

try:
    import keyring as _kr
    _AVAILABLE = True
except Exception as e:  # pragma: no cover — keyring is optional
    _kr = None
    _AVAILABLE = False
    logger.info(f"keyring unavailable, secrets will fall back to config.json: {e}")


def available() -> bool:
    return _AVAILABLE


def get(name: str) -> str | None:
    """Return the secret or None. Never raises."""
    if not _AVAILABLE:
        return None
    try:
        return _kr.get_password(_SERVICE, name)
    except Exception as e:
        logger.warning(f"keyring read failed for {name}: {e}")
        return None


def set(name: str, value: str) -> bool:
    """Store secret. Returns True on success, False on failure."""
    if not _AVAILABLE:
        return False
    try:
        if value:
            _kr.set_password(_SERVICE, name, value)
        else:
            delete(name)
        return True
    except Exception as e:
        logger.warning(f"keyring write failed for {name}: {e}")
        return False


def delete(name: str) -> bool:
    if not _AVAILABLE:
        return False
    try:
        _kr.delete_password(_SERVICE, name)
        return True
    except Exception:
        return False


def migrate_from_dict(d: dict, keys: Iterable[str] = SECRET_KEYS) -> list[str]:
    """
    Move plain-text secrets from `d` into the keyring (in-place).
    Returns the list of keys actually migrated so the caller can rewrite
    config.json without them.
    """
    migrated: list[str] = []
    if not _AVAILABLE:
        return migrated
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v and set(k, v):
            d.pop(k, None)
            migrated.append(k)
    if migrated:
        logger.info(f"Migrated secrets to keyring: {', '.join(migrated)}")
    return migrated
