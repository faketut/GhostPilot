"""Tests for the append-only JSONL usage log."""
from __future__ import annotations

import json


from src.usage_log import log_usage, read_usage, summarize


def test_log_usage_appends_json_lines(tmp_path):
    p = tmp_path / "usage.jsonl"
    assert log_usage({"in": 1, "out": 2}, path=p) is True
    assert log_usage({"in": 3, "out": 4}, path=p) is True

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["in"] == 1 and rec2["in"] == 3
    assert "ts" in rec1 and isinstance(rec1["ts"], float)


def test_log_usage_disabled_writes_nothing(tmp_path):
    p = tmp_path / "usage.jsonl"
    assert log_usage({"x": 1}, path=p, enabled=False) is False
    assert not p.exists()


def test_log_usage_creates_parent_directory(tmp_path):
    p = tmp_path / "nested" / "deep" / "usage.jsonl"
    assert log_usage({"x": 1}, path=p) is True
    assert p.exists()


def test_log_usage_swallows_write_errors(tmp_path):
    # Pass a directory as the file path — open() will fail.
    bad = tmp_path  # an existing directory
    assert log_usage({"x": 1}, path=bad) is False


def test_read_usage_returns_events(tmp_path):
    p = tmp_path / "usage.jsonl"
    log_usage({"a": 1}, path=p)
    log_usage({"a": 2}, path=p)
    events = read_usage(path=p)
    assert [e["a"] for e in events] == [1, 2]


def test_read_usage_missing_file_returns_empty(tmp_path):
    assert read_usage(path=tmp_path / "nope.jsonl") == []


def test_read_usage_skips_invalid_lines(tmp_path):
    p = tmp_path / "usage.jsonl"
    p.write_text('{"good": 1}\nnot-json\n{"good": 2}\n', encoding="utf-8")
    events = read_usage(path=p)
    assert [e["good"] for e in events] == [1, 2]


def test_summarize_empty_returns_zeros():
    s = summarize([])
    assert s["n"] == 0
    assert s["total_in"] == 0
    assert s["total_out"] == 0
    assert s["p50_ms"] == 0
    assert s["p95_ms"] == 0
    assert s["by_provider"] == {}


def test_summarize_aggregates_tokens_and_latency():
    events = [
        {"in": 10, "out": 5, "total_ms": 100, "provider": "openai"},
        {"in": 20, "out": 8, "total_ms": 200, "provider": "openai"},
        {"in": 0, "out": 0, "total_ms": 400, "provider": "deepseek"},
    ]
    s = summarize(events)
    assert s["n"] == 3
    assert s["total_in"] == 30
    assert s["total_out"] == 13
    assert s["p50_ms"] == 200  # middle of [100, 200, 400]
    assert s["p95_ms"] == 400

    openai = s["by_provider"]["openai"]
    assert openai["n"] == 2
    assert openai["in"] == 30 and openai["out"] == 13
    # Nearest-rank p50 on [100, 200] picks the lower → 100
    assert openai["p50_ms"] in (100, 200)
    assert openai["p95_ms"] == 200

    deepseek = s["by_provider"]["deepseek"]
    assert deepseek["n"] == 1
    assert deepseek["p50_ms"] == 400


def test_summarize_handles_missing_fields():
    """Vision latency events lack in/out tokens — must not crash."""
    events = [
        {"total_ms": 700, "provider": "gemini", "kind": "vision"},
        {"provider": "gemini"},  # no latency either
    ]
    s = summarize(events)
    assert s["n"] == 2
    assert s["total_in"] == 0 and s["total_out"] == 0
    assert s["p50_ms"] == 700
