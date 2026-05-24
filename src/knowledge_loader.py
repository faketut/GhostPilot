from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# When frozen by PyInstaller, bundled resources live under sys._MEIPASS.
# In dev, fall back to the repo root (one level up from src/).
_BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def _resolve_root(root_dir: str) -> Path:
    """Try the path as-is, then relative to the bundle/repo base."""
    p = Path(root_dir)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    return _BASE_DIR / root_dir


def _iter_files(root: Path, patterns: list[str]) -> Iterable[Path]:
    for pat in patterns:
        yield from root.rglob(pat)


def list_knowledge_files(
    root_dir: str,
    *,
    patterns: list[str] | None = None,
    max_files: int = 500,
) -> list[Path]:
    """Return absolute paths of knowledge files that ``load_knowledge_dir`` would read."""
    patterns = patterns or ["*.md", "*.txt"]
    root = _resolve_root(root_dir)
    if not root.exists():
        return []
    files = [p for p in _iter_files(root, patterns) if p.is_file()]
    return sorted(set(files))[:max_files]


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    """
    Simple character-based chunker with overlap.
    Keeps it dependency-free and robust for mixed Markdown/text files.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Normalise whitespace (keeps newlines, avoids huge blank runs)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)

        # Try to cut at paragraph boundary if possible
        cut = text.rfind("\n\n", i, j)
        if cut != -1 and cut > i + int(chunk_chars * 0.6):
            j = cut

        chunk = text[i:j].strip()
        if chunk:
            out.append(chunk)

        if j >= n:
            break
        i = max(j - overlap_chars, j)

    return out


def load_knowledge_dir(
    root_dir: str,
    *,
    patterns: list[str] | None = None,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    max_files: int = 500,
) -> list[str]:
    """
    Load local knowledge base from disk and return chunked documents for RAG.

    - Reads .md/.txt by default
    - Adds a short source header per chunk for traceability
    """
    patterns = patterns or ["*.md", "*.txt"]
    root = _resolve_root(root_dir)
    if not root.exists():
        logger.warning(f"Knowledge dir not found: {root.resolve()}")
        return []

    files = []
    for p in _iter_files(root, patterns):
        if p.is_file():
            files.append(p)

    files = sorted(set(files))[:max_files]
    if not files:
        logger.warning(f"Knowledge dir is empty: {root.resolve()}")
        return []

    docs: list[str] = []
    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to read knowledge file {fp}: {e}")
            continue

        rel = fp.relative_to(root).as_posix()
        chunks = _chunk_text(raw, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        for idx, c in enumerate(chunks):
            header = f"[kb:{rel}#{idx+1}]"
            docs.append(f"{header}\n{c}")

    logger.info(f"Knowledge loaded: {len(files)} files → {len(docs)} chunks")
    return docs

