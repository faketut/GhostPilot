"""Tests for src.session_replay (iter_turns + replay_turn)."""
from __future__ import annotations

import json

import pytest

from src import session_replay as sr


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_iter_turns_pairs_user_and_assistant(tmp_path):
    _write_jsonl(tmp_path / "llm.jsonl", [
        {"role": "user", "content": "q1", "q_type": "technical", "kind": "text"},
        {"role": "assistant", "content": "a1", "tokens_in": 2, "tokens_out": 3},
        {"role": "user", "content": "q2", "q_type": "behavioral", "kind": "text"},
        {"role": "assistant", "content": "a2"},
    ])
    turns = list(sr.iter_turns(tmp_path))
    assert [t.question for t in turns] == ["q1", "q2"]
    assert [t.original_answer for t in turns] == ["a1", "a2"]
    assert turns[0].q_type == "technical" and turns[0].tokens_out == 3


def test_iter_turns_attaches_screenshot(tmp_path):
    shot_dir = tmp_path / "screenshots"
    shot_dir.mkdir()
    (shot_dir / "0001.jpg").write_bytes(b"\xff\xd8\xff")
    _write_jsonl(tmp_path / "llm.jsonl", [
        {"role": "user", "content": "what's on screen?", "q_type": "technical", "kind": "vision"},
        {"role": "screenshot", "path": "screenshots/0001.jpg"},
        {"role": "assistant", "content": "I see a code editor."},
    ])
    turns = list(sr.iter_turns(tmp_path))
    assert len(turns) == 1
    t = turns[0]
    assert t.kind == "vision"
    assert t.screenshot_path == shot_dir / "0001.jpg"
    assert t.screenshot_bytes == b"\xff\xd8\xff"


def test_iter_turns_drops_orphan_user_when_next_user_appears(tmp_path):
    """Defensive: if a user row never got an assistant, the next user row replaces it."""
    _write_jsonl(tmp_path / "llm.jsonl", [
        {"role": "user", "content": "orphan", "kind": "text"},
        {"role": "user", "content": "real", "kind": "text"},
        {"role": "assistant", "content": "answer-to-real"},
    ])
    turns = list(sr.iter_turns(tmp_path))
    assert [t.question for t in turns] == ["real"]


def test_iter_turns_skips_invalid_lines(tmp_path):
    (tmp_path / "llm.jsonl").write_text(
        '{"role":"user","content":"q","kind":"text"}\nnot-json\n'
        '{"role":"assistant","content":"a"}\n',
        encoding="utf-8",
    )
    turns = list(sr.iter_turns(tmp_path))
    assert len(turns) == 1 and turns[0].original_answer == "a"


def test_iter_turns_empty_when_no_file(tmp_path):
    assert list(sr.iter_turns(tmp_path)) == []


def test_open_session_extracts_zip(tmp_path):
    src_dir = tmp_path / "20260524-120000"
    src_dir.mkdir()
    _write_jsonl(src_dir / "llm.jsonl", [
        {"role": "user", "content": "hi", "kind": "text"},
        {"role": "assistant", "content": "hello"},
    ])
    import shutil
    zip_path = shutil.make_archive(str(src_dir), "zip", root_dir=src_dir)
    shutil.rmtree(src_dir)
    out = sr.open_session(zip_path)
    assert out.is_dir()
    assert list(sr.iter_turns(out))[0].original_answer == "hello"


# ── replay_turn ────────────────────────────────────────────────────


class _StubEngine:
    """Minimal engine that streams a canned reply through the ui_queue."""

    def __init__(self, reply: str = "stub-reply", provider: str = "stub", raises=None):
        self.reply = reply
        self.provider = provider
        self.raises = raises
        self.calls: list[tuple] = []

    async def generate_answer_stream(self, question, ui_queue, *, q_type=None):
        self.calls.append(("text", question, q_type))
        if self.raises:
            raise self.raises
        await ui_queue.put({"type": "info", "provider": self.provider, "rag_hits": 0})
        for ch in self.reply:
            await ui_queue.put({"type": "token", "text": ch})

    async def generate_vision_answer_stream(self, image_bytes, ui_queue):
        self.calls.append(("vision", len(image_bytes)))
        await ui_queue.put({"type": "info", "provider": self.provider})
        await ui_queue.put({"type": "token", "text": self.reply})


@pytest.mark.asyncio
async def test_replay_turn_text_streams_new_answer():
    engine = _StubEngine(reply="hello", provider="openai")
    turn = sr.Turn(kind="text", question="q1", q_type="technical", original_answer="old")
    result = await sr.replay_turn(engine, turn)
    assert result.new_answer == "hello"
    assert result.error is None
    assert result.new_provider == "openai"
    assert engine.calls == [("text", "q1", "technical")]


@pytest.mark.asyncio
async def test_replay_turn_vision_uses_screenshot_bytes():
    engine = _StubEngine(reply="visual-answer")
    turn = sr.Turn(
        kind="vision", question="what?", q_type="technical",
        original_answer="old", screenshot_bytes=b"\xff\xd8\xff\xaa",
    )
    result = await sr.replay_turn(engine, turn)
    assert result.new_answer == "visual-answer"
    assert engine.calls == [("vision", 4)]


@pytest.mark.asyncio
async def test_replay_turn_vision_missing_bytes_records_error():
    engine = _StubEngine()
    turn = sr.Turn(kind="vision", question="?", q_type="x", original_answer="")
    result = await sr.replay_turn(engine, turn)
    assert result.new_answer == ""
    assert result.error and "screenshot bytes" in result.error


@pytest.mark.asyncio
async def test_replay_turn_captures_engine_error():
    engine = _StubEngine(raises=RuntimeError("boom"))
    turn = sr.Turn(kind="text", question="q", q_type="", original_answer="")
    result = await sr.replay_turn(engine, turn)
    assert result.error == "RuntimeError: boom"
