"""Unit tests for SessionRecorder (isolated to a tmp recordings root)."""
from __future__ import annotations

import json
import zipfile

import pytest

from src import session_recorder as sr


@pytest.fixture
def isolated_recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "_recordings_root", lambda: tmp_path)
    rec = sr.SessionRecorder()
    yield rec
    # Best-effort cleanup if a test left it open.
    if rec.is_recording():
        rec.stop()


def test_redact_masks_known_secret_names():
    out = sr._redact({
        "OPENAI_API_KEY": "sk-real",
        "GEMINI_API_KEY": "g-real",
        "TEXT_MODEL": "gpt-4o-mini",
    })
    assert out["OPENAI_API_KEY"] == "***"
    assert out["GEMINI_API_KEY"] == "***"
    assert out["TEXT_MODEL"] == "gpt-4o-mini"


def test_log_before_start_is_noop(isolated_recorder):
    # Should not raise and should not create any files.
    isolated_recorder.log_transcript("partial", "hello")
    isolated_recorder.log_llm("assistant", "hi")
    assert not isolated_recorder.is_recording()


def test_start_creates_dir_and_files(isolated_recorder, tmp_path):
    d = isolated_recorder.start()
    assert d is not None
    assert d.exists()
    assert (d / "transcript.jsonl").exists()
    assert (d / "llm.jsonl").exists()
    assert isolated_recorder.is_recording()


def test_start_is_idempotent(isolated_recorder):
    d1 = isolated_recorder.start()
    d2 = isolated_recorder.start()
    assert d1 == d2


def test_log_and_stop_produces_zip_with_records(isolated_recorder):
    d = isolated_recorder.start()
    assert d is not None
    isolated_recorder.log_transcript("final", "hello world")
    isolated_recorder.log_llm("assistant", "hi there", tokens_in=10, tokens_out=5,
                              cost_usd=0.0001, model="gpt-4o-mini")
    zip_path = isolated_recorder.stop()
    assert zip_path is not None
    assert zip_path.suffix == ".zip"
    assert zip_path.exists()
    # Original dir should be removed.
    assert not d.exists()
    # Zip should contain the three artefacts.
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "transcript.jsonl" in names
        assert "llm.jsonl" in names
        assert "meta.json" in names
        tr = zf.read("transcript.jsonl").decode("utf-8").strip().splitlines()
        llm = zf.read("llm.jsonl").decode("utf-8").strip().splitlines()
        meta = json.loads(zf.read("meta.json"))
    assert len(tr) == 1 and json.loads(tr[0])["text"] == "hello world"
    assert len(llm) == 1 and json.loads(llm[0])["tokens_in"] == 10
    assert "started_at" in meta and "ended_at" in meta
    # Secrets in snapshot should be redacted (if present at all).
    cfg = meta.get("config", {})
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "AZURE_SPEECH_KEY"):
        if k in cfg:
            assert cfg[k] == "***"


def test_stop_without_start_returns_none(isolated_recorder):
    assert isolated_recorder.stop() is None
