"""SherpaOnnxASRClient tests — fake recognizer, no model download, no audio device.

The observable contract is the one ``main.py``'s ``asr_router`` consumes:
``{"type": "partial"|"final", "text", "speaker"}`` into ``text_queue``.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from src import sherpa_asr
from src import sherpa_models as sm
from src.config import config
from src.sherpa_asr import SherpaOnnxASRClient, pcm16_to_float


def _silence(seconds: float = 0.16) -> bytes:
    return b"\x00\x00" * int(seconds * 16000)


# ── fake recognizer exposing the same shape as sherpa_onnx.OnlineRecognizer ──


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text


class _Stream:
    def __init__(self, recognizer: _FakeRecognizer) -> None:
        self._rec = recognizer
        # The result the recognizer reports for the audio most recently fed —
        # exactly like the real binding, where is_endpoint/get_result describe
        # the current stream state rather than consuming a new result.
        self.current: tuple[str, bool] = ("", False)

    def accept_waveform(self, sample_rate: float, samples) -> None:
        self.current = self._rec.next_entry()


class _FakeRecognizer:
    """Replays one scripted (text, endpoint) pair per fed chunk."""

    def __init__(self, script: list[tuple[str, bool]]) -> None:
        self.script = list(script)
        self.streams = 0
        self.resets = 0
        self.decodes = 0

    def next_entry(self) -> tuple[str, bool]:
        return self.script.pop(0) if self.script else ("", False)

    def create_stream(self):
        self.streams += 1
        return _Stream(self)

    def is_ready(self, stream) -> bool:  # one decode step per chunk is enough
        return self.decodes < self.streams

    def decode_stream(self, stream) -> None:
        self.decodes += 1

    def get_result_all(self, stream) -> _Result:
        return _Result(stream.current[0])

    def is_endpoint(self, stream) -> bool:
        return stream.current[1]

    def reset(self, stream) -> None:
        self.resets += 1
        stream.current = ("", False)


async def _drive(client, script: list[tuple[str, bool]], chunks: int) -> tuple[list[dict], _FakeRecognizer | None]:
    """Run the client against a fake recognizer; return (events, recognizer)."""
    fake = _FakeRecognizer(script)
    client._recognizer = fake
    aq: asyncio.Queue = asyncio.Queue()
    tq: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    task = loop.create_task(client.start_streaming(aq, tq, loop))
    for _ in range(chunks):
        await aq.put(_silence())
    # Deterministic rendezvous: the script is consumed exactly once per chunk
    # that reached the worker, so an empty script means every chunk was decoded.
    # A fixed sleep here would be a flake waiting to happen.
    for _ in range(2000):
        if not fake.script:
            break
        await asyncio.sleep(0.005)
    client.stop()
    await aq.put(b"")  # wake the pending get() so the loop can observe the flag
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()

    events: list[dict] = []
    while not tq.empty():
        events.append(tq.get_nowait())
    return events, (fake if client._recognizer is fake else None)


# ── event contract ───────────────────────────────────────────────────────


def test_partials_emitted_on_change_and_final_at_endpoint():
    client = SherpaOnnxASRClient(model="zipformer-bilingual-zh-en")
    # (text, endpoint) per chunk: growing partials, then an endpoint.
    script = [("你", False), ("你好", False), ("你好", False), ("你好", True)]

    events, fake = asyncio.run(_drive(client, script, chunks=4))

    assert events == [
        {"type": "partial", "text": "你", "speaker": "Unknown"},
        {"type": "partial", "text": "你好", "speaker": "Unknown"},
        {"type": "final", "text": "你好", "speaker": "Unknown"},
    ]
    # The utterance boundary resets the stream exactly once, when the endpoint fires.
    assert fake is not None and fake.resets == 1


def test_unchanged_partial_is_not_re_emitted():
    """Repainting the same string restarts main.py's finalize timer on every frame."""
    client = SherpaOnnxASRClient(model="zipformer-bilingual-zh-en")
    script = [("hello", False)] * 6

    events, _ = asyncio.run(_drive(client, script, chunks=6))

    assert events == [{"type": "partial", "text": "hello", "speaker": "Unknown"}]


def test_endpoint_without_text_resets_but_emits_nothing():
    """rule1 ends pure silence; an empty final would make main.py answer ""."""
    client = SherpaOnnxASRClient(model="zipformer-bilingual-zh-en")
    script = [("", False), ("", True), ("hi", False)]

    events, fake = asyncio.run(_drive(client, script, chunks=3))

    assert events == [{"type": "partial", "text": "hi", "speaker": "Unknown"}]
    assert fake is not None and fake.resets == 1


def test_partial_after_endpoint_is_not_swallowed():
    """Two utterances in one session must produce two independents finals."""
    client = SherpaOnnxASRClient(model="zipformer-bilingual-zh-en")
    script = [("one", True), ("two", True)]

    events, _ = asyncio.run(_drive(client, script, chunks=2))

    assert events == [
        {"type": "partial", "text": "one", "speaker": "Unknown"},
        {"type": "final", "text": "one", "speaker": "Unknown"},
        {"type": "partial", "text": "two", "speaker": "Unknown"},
        {"type": "final", "text": "two", "speaker": "Unknown"},
    ]


def test_missing_recognizer_disables_the_backend_without_raising(monkeypatch):
    client = SherpaOnnxASRClient(model="zipformer-bilingual-zh-en")

    def _boom():
        raise RuntimeError("sherpa-onnx is not installed.")

    monkeypatch.setattr(client, "_get_recognizer", _boom)

    async def scenario():
        aq: asyncio.Queue = asyncio.Queue()
        tq: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(client.start_streaming(aq, tq, loop), timeout=2.0)
        return tq.empty()

    assert asyncio.run(scenario()) is True


# ── configuration hazards ────────────────────────────────────────────────


def test_rule2_above_router_timer_warns_of_duplicate_answers(monkeypatch, caplog):
    """Both finalize the same utterance; if the router wins, the answer repeats."""
    monkeypatch.setattr(config, "ASR_PARTIAL_SILENCE_MS", 300)
    with caplog.at_level(logging.WARNING, logger="src.sherpa_asr"):
        SherpaOnnxASRClient(model="zipformer-bilingual-zh-en", rule2_min_trailing_silence=0.5)
    assert any("duplicate answer" in r.message for r in caplog.records)


def test_rule2_below_router_timer_stays_quiet(monkeypatch, caplog):
    monkeypatch.setattr(config, "ASR_PARTIAL_SILENCE_MS", 650)
    with caplog.at_level(logging.WARNING, logger="src.sherpa_asr"):
        SherpaOnnxASRClient(model="zipformer-bilingual-zh-en", rule2_min_trailing_silence=0.5)
    assert not caplog.records


def test_hotwords_force_a_decoding_method_that_honours_them():
    """hotwords_score is ignored under greedy_search — biasing would do nothing."""
    client = SherpaOnnxASRClient(model="zipformer-bilingual-zh-en", hotwords_file="hot.txt")
    assert client.decoding_method == "modified_beam_search"
    assert SherpaOnnxASRClient(model="zipformer-bilingual-zh-en").decoding_method == "greedy_search"


# ── language intent vs model coverage ────────────────────────────────────
#
# ASR_LANGUAGE is an Azure setting. A sherpa-onnx model's language is fixed at
# download time, so a mismatch transcribes nothing while every status line still
# looks healthy — the user debugs the LLM instead of the model choice.


def test_language_match_is_reported_but_not_warned(monkeypatch, caplog):
    monkeypatch.setattr(config, "ASR_LANGUAGE", "en-US", raising=False)
    with caplog.at_level(logging.INFO, logger="src.sherpa_asr"):
        SherpaOnnxASRClient(model="zipformer-bilingual-zh-en")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ASR_LANGUAGE" in r.message for r in caplog.records)


def test_english_audio_against_a_chinese_only_model_warns(monkeypatch, caplog):
    monkeypatch.setattr(config, "ASR_LANGUAGE", "en-US", raising=False)
    with caplog.at_level(logging.WARNING, logger="src.sherpa_asr"):
        SherpaOnnxASRClient(model="zipformer-zh-14M")
    warned = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("trained on zh" in w and "will not transcribe" in w for w in warned), warned
    # The message has to name the fix, not just the problem.
    assert any("SHERPA_MODEL" in w for w in warned)


def test_language_coverage_is_recognised_from_a_model_directory_path(monkeypatch, caplog):
    """SHERPA_MODEL may be a path; the folder name still identifies the model."""
    monkeypatch.setattr(config, "ASR_LANGUAGE", "en-US", raising=False)
    path = str(sm.Path("/models") / sm.get("zipformer-zh-14M").folder)
    with caplog.at_level(logging.WARNING, logger="src.sherpa_asr"):
        SherpaOnnxASRClient(model=path)
    assert any("trained on zh" in r.message for r in caplog.records if r.levelno >= logging.WARNING)


def test_unknown_model_directory_makes_no_language_claim(monkeypatch, caplog):
    """A hand-made directory carries no coverage metadata — guessing would be
    worse than saying nothing."""
    monkeypatch.setattr(config, "ASR_LANGUAGE", "en-US", raising=False)
    with caplog.at_level(logging.DEBUG, logger="src.sherpa_asr"):
        client = SherpaOnnxASRClient(model=str(sm.Path("/models/my-own-model")))
    assert client._model_languages() == ()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_english_first_model_is_available_for_english_interviews():
    """zipformer-en-20M is the smallest model that covers English."""
    entry = sm.get("zipformer-en-20M")
    assert "en" in entry.languages
    assert entry.size_mb < sm.get(sm.DEFAULT_MODEL).size_mb


# ── audio conversion ─────────────────────────────────────────────────────


def test_pcm16_scales_to_unit_range():
    out = pcm16_to_float(b"\x00\x00\xff\x7f\x00\x80")
    assert out.dtype == np.float32
    assert out[0] == 0.0
    assert out[1] == pytest.approx(1.0, abs=1e-4)
    assert out[2] == pytest.approx(-1.0, abs=1e-4)


def test_odd_length_chunk_does_not_raise():
    """A trailing half sample would otherwise raise on every following chunk."""
    assert pcm16_to_float(b"\x01\x02\x03").shape == (1,)
    assert pcm16_to_float(b"").shape == (0,)


def test_empty_chunk_is_ignored_by_the_loop():
    client = SherpaOnnxASRClient(model="zipformer-bilingual-zh-en")
    fake = _FakeRecognizer([("x", False)])
    client._recognizer = fake

    async def scenario():
        aq: asyncio.Queue = asyncio.Queue()
        tq: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        task = loop.create_task(client.start_streaming(aq, tq, loop))
        await aq.put(b"")
        await asyncio.sleep(0.02)
        client.stop()
        await aq.put(b"")
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
        return fake.decodes, tq.empty()

    decodes, empty = asyncio.run(scenario())
    assert decodes == 0 and empty


def test_sample_rate_matches_the_capture_pipeline():
    """AudioCapture resamples to this rate; a mismatch decodes as garbage."""
    assert sherpa_asr.SAMPLE_RATE == 16000
    assert config.SAMPLE_RATE == sherpa_asr.SAMPLE_RATE
