"""Replay a recorded session through a (possibly tweaked) LLMEngine.

A session directory (the unzipped form of what ``SessionRecorder`` produces)
contains ``llm.jsonl`` with interleaved rows:

    {"role": "user",       "content": "...question...", "q_type": "...", "kind": "text"|"vision"}
    {"role": "screenshot", "path":    "screenshots/0001.jpg"}              # vision only
    {"role": "assistant",  "content": "...answer...",  "tokens_in":..., "tokens_out":...}

``iter_turns`` walks this stream and emits one :class:`Turn` per
user-question, attaching the screenshot bytes (when present) and the
recorded assistant answer.

``replay_turn`` re-runs a turn through an ``LLMEngine`` instance with the
*current* prompt templates loaded and returns the new answer text plus
timing, leaving comparison/diffing to the caller.

This module is intentionally engine-agnostic — it only relies on the
``generate_answer_stream`` / ``generate_vision_answer_stream`` contracts so
the unit tests can stub them out.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    kind: str  # "text" | "vision"
    question: str
    q_type: str
    original_answer: str
    screenshot_path: Optional[Path] = None
    screenshot_bytes: Optional[bytes] = None
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


@dataclass
class ReplayResult:
    turn: Turn
    new_answer: str
    new_total_ms: int
    error: Optional[str] = None
    new_provider: str = ""
    info: list[dict] = field(default_factory=list)


# ── Loading ───────────────────────────────────────────────────────────


def _load_rows(session_dir: Path) -> list[dict]:
    f = session_dir / "llm.jsonl"
    if not f.exists():
        return []
    out: list[dict] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def open_session(path: Path) -> Path:
    """Return a usable session directory.

    Accepts either a directory (returned as-is) or a ``.zip`` produced by
    ``SessionRecorder.stop`` (extracted to a sibling temp dir and returned).
    """
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffix == ".zip" and path.exists():
        out = path.with_suffix("")  # strip .zip
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out)
        return out
    raise FileNotFoundError(f"Not a session dir or zip: {path}")


def iter_turns(session_dir: Path) -> Iterator[Turn]:
    """Walk ``llm.jsonl`` and yield one :class:`Turn` per user question.

    Pairing rule: a ``user`` row consumes the first following ``assistant``
    row (with optional ``screenshot`` rows in between). Stray rows are
    skipped — the recorder writes them in order but be defensive.
    """
    rows = _load_rows(session_dir)
    pending: Optional[dict] = None
    pending_shot: Optional[dict] = None
    for r in rows:
        role = r.get("role")
        if role == "user":
            # If a previous user turn never got an assistant, drop it.
            pending = r
            pending_shot = None
        elif role == "screenshot" and pending is not None:
            pending_shot = r
        elif role == "assistant" and pending is not None:
            shot_path: Optional[Path] = None
            shot_bytes: Optional[bytes] = None
            if pending_shot is not None:
                p = session_dir / pending_shot.get("path", "")
                if p.exists():
                    shot_path = p
                    try:
                        shot_bytes = p.read_bytes()
                    except Exception:
                        shot_bytes = None
            yield Turn(
                kind=str(pending.get("kind") or "text"),
                question=str(pending.get("content") or ""),
                q_type=str(pending.get("q_type") or ""),
                original_answer=str(r.get("content") or ""),
                screenshot_path=shot_path,
                screenshot_bytes=shot_bytes,
                tokens_in=int(r.get("tokens_in") or 0),
                tokens_out=int(r.get("tokens_out") or 0),
                model=str(r.get("model") or ""),
            )
            pending = None
            pending_shot = None


# ── Replay ────────────────────────────────────────────────────────────


async def _drain_queue_to_text(queue: asyncio.Queue, stop: asyncio.Event,
                                info_sink: list[dict]) -> str:
    parts: list[str] = []
    while True:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            if stop.is_set() and queue.empty():
                break
            continue
        if msg.get("type") == "token":
            parts.append(str(msg.get("text") or ""))
        elif msg.get("type") == "info":
            info_sink.append(msg)
    return "".join(parts)


async def replay_turn(engine, turn: Turn) -> ReplayResult:
    """Re-run ``turn`` through ``engine`` and return the new answer.

    ``engine`` must implement ``generate_answer_stream(question, ui_queue, q_type=)``
    for text turns and ``generate_vision_answer_stream(image_bytes, ui_queue)``
    for vision turns. The UI queue receives the same shape the live app uses,
    so this works against the real ``LLMEngine`` *and* against test stubs.
    """
    queue: asyncio.Queue = asyncio.Queue()
    stop = asyncio.Event()
    info: list[dict] = []
    t0 = time.monotonic()
    err: Optional[str] = None

    async def producer():
        try:
            if turn.kind == "vision":
                if turn.screenshot_bytes is None:
                    raise RuntimeError("vision turn missing screenshot bytes")
                await engine.generate_vision_answer_stream(turn.screenshot_bytes, queue)
            else:
                await engine.generate_answer_stream(
                    turn.question, queue, q_type=turn.q_type or None,
                )
        finally:
            stop.set()

    consumer = asyncio.create_task(_drain_queue_to_text(queue, stop, info))
    try:
        await producer()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        stop.set()
    new_answer = await consumer
    dt_ms = int((time.monotonic() - t0) * 1000)
    new_provider = ""
    for m in info:
        if m.get("provider"):
            new_provider = str(m["provider"])
            break
    return ReplayResult(
        turn=turn,
        new_answer=new_answer,
        new_total_ms=dt_ms,
        error=err,
        new_provider=new_provider,
        info=info,
    )


__all__ = ["Turn", "ReplayResult", "iter_turns", "replay_turn", "open_session"]
