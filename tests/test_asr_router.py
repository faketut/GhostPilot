"""The ASR routing state machine, driven directly.

This is the only code that decides *when a question is complete*. It has two
finalization paths that race — the backend's ``final`` event and the
partial-silence timer — and a race between them answers one utterance twice.
Both are exercised here.

Also pins the ``speaker`` contract: it is display metadata. A backend that
cannot diarize (sherpa-onnx, faster-whisper) reports ``"Unknown"`` and the
pipeline must behave exactly as it does for Azure's ``"Guest-1"``.
"""
from __future__ import annotations

import asyncio

import pytest

import main
from src.config import config

# ── doubles ──────────────────────────────────────────────────────────────


class _UI:
    def __init__(self) -> None:
        self.blocks: list[str] = []
        self.statuses: list[str] = []
        self.thinking: list[str] = []
        self.streaming: list[bool] = []

    def append_block(self, text: str) -> None:
        self.blocks.append(text)

    def set_status(self, text: str) -> None:
        self.statuses.append(text)

    def show_thinking(self, q_type: str = "") -> None:
        self.thinking.append(q_type)

    def set_streaming(self, flag: bool) -> None:
        self.streaming.append(bool(flag))


class _Engine:
    def __init__(self) -> None:
        self.questions: list[str] = []
        self.q_types: list[str] = []

    async def classify_question_llm(self, question: str) -> str:
        return "behavioral"

    async def generate_answer_stream(self, question, ui_queue, *, q_type=None):
        self.questions.append(question)
        self.q_types.append(q_type)
        await ui_queue.put({"type": "token", "text": "answer"})


def _make(monkeypatch, *, silence_ms: int, punctuation: bool):
    """The real router, with the two timing knobs shortened for the test."""
    monkeypatch.setattr(config, "ASR_PARTIAL_SILENCE_MS", silence_ms, raising=False)
    monkeypatch.setattr(config, "ASR_PUNCTUATION_FINALIZE", punctuation, raising=False)

    ui, engine = _UI(), _Engine()
    tq: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    task = loop.create_task(main.make_asr_router(
        text_queue=tq, ui=ui, ui_queue=asyncio.Queue(), llm_engine=engine, loop=loop,
    )())
    return task, ui, engine, tq


async def _settle(task, engine, *, want: int, steps: int = 300) -> int:
    """Let the router work; return how many questions were answered."""
    try:
        for _ in range(steps):
            if len(engine.questions) >= want:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return len(engine.questions)


# ── the speaker contract ─────────────────────────────────────────────────


@pytest.mark.parametrize("speaker", ["Unknown", "Guest-1", "1", "讲师"])
@pytest.mark.asyncio
async def test_any_speaker_label_routes_identically(monkeypatch, speaker):
    """Whether a backend diarizes must not change what the pipeline does.

    ``Unknown`` is what sherpa-onnx and faster-whisper always report, so in this
    app it is the common case, not an edge case.
    """
    task, ui, engine, tq = _make(monkeypatch, silence_ms=10_000, punctuation=False)

    await tq.put({"type": "final", "text": "tell me about a time", "speaker": speaker})
    assert await _settle(task, engine, want=1) == 1

    assert engine.questions == ["tell me about a time"]
    assert engine.q_types == ["behavioral"]
    # The label is carried into the block header verbatim — display only.
    assert ui.blocks == [f"[{speaker}] tell me about a time\nA: "]


@pytest.mark.asyncio
async def test_a_missing_speaker_key_still_routes(monkeypatch):
    """The router defaults the key, so it can never be load-bearing."""
    task, ui, engine, tq = _make(monkeypatch, silence_ms=10_000, punctuation=False)

    await tq.put({"type": "final", "text": "no speaker field"})
    assert await _settle(task, engine, want=1) == 1

    assert engine.questions == ["no speaker field"]
    assert ui.blocks == ["[Unknown] no speaker field\nA: "]


# ── the two finalization paths ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_punctuation_finalizes_a_partial_without_waiting(monkeypatch):
    """An Azure partial ending in '?' is answered immediately. A backend that
    emits no punctuation (sherpa-onnx) never reaches this path."""
    task, ui, engine, tq = _make(monkeypatch, silence_ms=10_000, punctuation=True)

    await tq.put({"type": "partial", "text": "what is a deadlock?", "speaker": "Unknown"})
    assert await _settle(task, engine, want=1) == 1

    assert engine.questions == ["what is a deadlock?"]
    # The live partial was shown before finalization cleared the status.
    assert "[Unknown] what is a deadlock?" in ui.statuses


@pytest.mark.asyncio
async def test_silence_timer_finalizes_a_partial(monkeypatch):
    """The path a non-punctuating backend depends on entirely."""
    task, ui, engine, tq = _make(monkeypatch, silence_ms=20, punctuation=False)

    await tq.put({"type": "partial", "text": "explain indexes", "speaker": "Unknown"})
    assert await _settle(task, engine, want=1) == 1

    assert engine.questions == ["explain indexes"]


@pytest.mark.asyncio
async def test_new_partials_restart_the_timer(monkeypatch):
    """Only the last partial of an utterance may be answered."""
    task, _ui, engine, tq = _make(monkeypatch, silence_ms=60, punctuation=False)

    for text in ("exp", "explain", "explain ind", "explain indexes"):
        await tq.put({"type": "partial", "text": text, "speaker": "Unknown"})
        await asyncio.sleep(0.01)   # well inside the 60 ms silence budget

    assert await _settle(task, engine, want=1) == 1
    assert engine.questions == ["explain indexes"], "only the final partial is a question"


@pytest.mark.asyncio
async def test_a_final_supersedes_a_pending_partial(monkeypatch):
    """The dedupe path, and the race that makes SHERPA_RULE2_SILENCE_SEC need to
    sit below ASR_PARTIAL_SILENCE_MS: a ``final`` must clear the pending partial
    so an already-armed timer cannot answer the same utterance again."""
    task, _ui, engine, tq = _make(monkeypatch, silence_ms=50, punctuation=False)

    await tq.put({"type": "partial", "text": "tell me about a time", "speaker": "Unknown"})
    await asyncio.sleep(0.01)
    # The backend's own endpoint fires before the silence timer would.
    await tq.put({"type": "final", "text": "tell me about a time", "speaker": "Unknown"})
    await asyncio.sleep(0.15)   # long past the 50 ms timer
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert engine.questions == ["tell me about a time"], (
        f"the utterance was answered {len(engine.questions)}x — a partial timer "
        "survived the final and duplicated the answer"
    )


@pytest.mark.asyncio
async def test_two_utterances_produce_two_independent_answers(monkeypatch):
    task, ui, engine, tq = _make(monkeypatch, silence_ms=10_000, punctuation=False)

    await tq.put({"type": "final", "text": "first question", "speaker": "Unknown"})
    assert await _settle(task, engine, want=1, steps=100) == 1

    # A second utterance needs a live router, so re-arm it on the same queue.
    loop = asyncio.get_running_loop()
    task2 = loop.create_task(main.make_asr_router(
        text_queue=tq, ui=ui, ui_queue=asyncio.Queue(), llm_engine=engine, loop=loop,
    )())
    await tq.put({"type": "final", "text": "second question", "speaker": "Unknown"})
    assert await _settle(task2, engine, want=2, steps=100) == 2

    assert engine.questions == ["first question", "second question"]
    assert len(ui.blocks) == 2
