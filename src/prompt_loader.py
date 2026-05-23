"""
PromptLoader
------------
Loads system prompts from the `prompts/` directory at startup and caches them.
Prompts are plain Markdown files — users can edit them directly without touching code.

File → question type mapping:
    prompts/algorithm.md  →  "algorithm"
    prompts/technical.md  →  "technical"
    prompts/behavioral.md →  "behavioral"
    prompts/vision.md     →  "vision"

If a file is missing, a sensible inline fallback is used so the app never crashes.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# When frozen by PyInstaller, bundled data lives under sys._MEIPASS; otherwise
# use the repo root (two levels up from this file).
_BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.parent))
_PROMPT_DIR = _BASE_DIR / "prompts"

# ── Inline fallbacks (used when .md file is absent) ──────────────────────
_FALLBACKS: dict[str, str] = {
    "algorithm": (
        "You are an algorithms coach. Output: (1) one line [U][M][P] tight clauses; "
        "(2) line with [I] then a fenced code block, no comments; "
        "(3) one line [R] with T(n)/S(n) + sanity. "
        "Chinese unless the question is clearly English."
    ),
    "technical": (
        "You are a senior SWE coach. Output exactly ONE line: "
        "[R]... then TAB, [E]..., [A]..., [C]..., [T]... using real TAB characters. "
        "Short clauses only. Chinese unless the question is clearly English."
    ),
    "behavioral": (
        "You are a behavioral coach. Exactly four lines: either STAR [S][T][A][R] for a past story, "
        "or WYEC [W][Y][E][C] for motivation/why-us/fit/self-pitch. "
        "Chinese unless the question is clearly English."
    ),
    "vision": (
        "You analyze interview screenshots. First line: 题目类型: behavioral|technical|algorithm|other. "
        "Then use the same letter format as that type (behavioral: STAR or WYEC four lines; technical: one tabbed line; "
        "algorithm: UMP + [I] + code + [R]). Chinese unless the question is clearly English."
    ),
}

_cache: dict[str, str] = {}


def _load(q_type: str) -> str:
    """Load and cache a single prompt file."""
    if q_type in _cache:
        return _cache[q_type]

    path = _PROMPT_DIR / f"{q_type}.md"
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8").strip()
            _cache[q_type] = text
            logger.info(f"Loaded prompt: {path.name} ({len(text)} chars)")
            return text
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e} — using fallback")

    fallback = _FALLBACKS.get(q_type, "You are a helpful interview assistant.")
    _cache[q_type] = fallback
    logger.warning(f"Using inline fallback for prompt type '{q_type}'")
    return fallback


def get_prompt(q_type: str) -> str:
    """Return the system prompt for the given question type."""
    return _load(q_type)


def list_prompts() -> list[str]:
    """Known prompt names (matches files we'll surface in the UI)."""
    return list(_FALLBACKS.keys())


def prompt_path(q_type: str) -> Path:
    """Filesystem path where `q_type`'s prompt would live."""
    return _PROMPT_DIR / f"{q_type}.md"


def fallback_text(q_type: str) -> str:
    return _FALLBACKS.get(q_type, "")


def reload(q_type: str) -> str:
    """Drop the cached value for one prompt and reload from disk."""
    _cache.pop(q_type, None)
    return _load(q_type)


def save(q_type: str, text: str) -> Path:
    """Write `text` to the prompt file and refresh the cache. Returns the path."""
    path = prompt_path(q_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _cache[q_type] = text.strip()
    logger.info(f"Saved prompt: {path.name} ({len(text)} chars)")
    return path


# Pre-load at import time so first LLM call has zero disk I/O
def _preload():
    for q_type in _FALLBACKS:
        _load(q_type)

_preload()
