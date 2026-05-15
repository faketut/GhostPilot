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
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"

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
        "You are a behavioral coach. Output exactly four lines: "
        "[S]..., [T]..., [A]..., [R]... — one short line each, no extra headers. "
        "Chinese unless the question is clearly English."
    ),
    "vision": (
        "You analyze interview screenshots. First line: 题目类型: behavioral|technical|algorithm|other. "
        "Then use the same letter format as that type (four S/T/A/R lines; one tabbed R/E/A/C/T line; "
        "or UMP line + [I] + code + [R]). Chinese unless the question is clearly English."
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


def reload_all():
    """Force-reload all cached prompts from disk (useful after in-app edits)."""
    _cache.clear()
    for q_type in _FALLBACKS:
        _load(q_type)
    logger.info("All prompts reloaded from disk.")


# Pre-load at import time so first LLM call has zero disk I/O
def _preload():
    for q_type in _FALLBACKS:
        _load(q_type)

_preload()
