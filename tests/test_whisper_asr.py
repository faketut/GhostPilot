"""Unit tests for FasterWhisperASRClient (model is stubbed)."""
import asyncio

import pytest

from src.whisper_asr import FasterWhisperASRClient


def _silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * 16000)


def test_lazy_load_failure_surfaces_runtime_error(monkeypatch):
    client = FasterWhisperASRClient(model_name="tiny")
    # Force the optional import to fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            raise ImportError("simulated missing dep")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
        client._get_model()


def test_window_size_respects_floor():
    # 0.1s requested but floor is 1s of audio (32000 bytes).
    c = FasterWhisperASRClient(window_sec=0.1)
    assert c.window_bytes == 16000 * 2


def test_start_streaming_emits_final_after_window_fills():
    """Buffer fills → executor.run is called → final event is emitted."""
    client = FasterWhisperASRClient(window_sec=1.0)
    # Stub the model entirely so no faster_whisper import / download happens.
    client._model = object()
    client._transcribe = lambda pcm: "hello world"  # type: ignore[assignment]

    async def scenario():
        aq: asyncio.Queue = asyncio.Queue()
        tq: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        task = loop.create_task(client.start_streaming(aq, tq, loop))

        # Push 1.5s of silence (>= 1.0s window) split into 3 chunks.
        for _ in range(3):
            await aq.put(_silence(0.5))

        evt = await asyncio.wait_for(tq.get(), timeout=2.0)
        client.stop()
        # Unblock the queue.get() in the loop so the task can exit.
        await aq.put(b"")
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
        return evt

    evt = asyncio.run(scenario())
    assert evt == {"type": "final", "text": "hello world", "speaker": "Unknown"}
