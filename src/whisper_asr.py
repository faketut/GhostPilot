"""
Local ASR backend using faster-whisper.

Slots in behind the same duck-typed surface as ``ASRClient``
(``start_streaming(audio_queue, text_queue, loop)`` + ``stop()``) so
``main.py`` can swap implementations via ``config.ASR_BACKEND``.

Whisper is not a streaming model; we accumulate a small audio window
(``WHISPER_WINDOW_SEC``) and run inference each time the window fills.
Each pass emits a single ``final`` event — no speculative partials, which
keeps the overlay readable at the cost of higher latency than Azure.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 16 kHz mono int16 — same format produced by AudioCapture.
_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2


class FasterWhisperASRClient:
    def __init__(
        self,
        *,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        language: Optional[str] = None,
        window_sec: float = 2.5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language or None
        self.window_bytes = max(int(window_sec * _SAMPLE_RATE * _BYTES_PER_SAMPLE), _SAMPLE_RATE * _BYTES_PER_SAMPLE)
        self._is_running = False
        self._model = None  # lazy

    # ── lazy model load ──────────────────────────────────────────────────
    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise RuntimeError(
                "faster-whisper is not installed. Install with `pip install faster-whisper`."
            ) from e
        logger.info(
            f"Loading faster-whisper model '{self.model_name}' on {self.device} ({self.compute_type})..."
        )
        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        return self._model

    # ── inference (runs in executor) ─────────────────────────────────────
    def _transcribe(self, pcm: bytes) -> str:
        import numpy as np

        model = self._get_model()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""
        segments, _info = model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip())

    # ── public interface (duck-types ASRClient) ──────────────────────────
    async def start_streaming(
        self,
        audio_queue: asyncio.Queue,
        text_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._is_running = True
        logger.info("faster-whisper ASR streaming starting...")
        buf = bytearray()
        while self._is_running:
            try:
                chunk = await audio_queue.get()
            except asyncio.CancelledError:
                break
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) < self.window_bytes:
                continue

            window = bytes(buf)
            buf.clear()
            try:
                text = await loop.run_in_executor(None, self._transcribe, window)
            except Exception as e:
                logger.error(f"whisper inference failed: {e}")
                continue
            if text:
                text_queue.put_nowait({"type": "final", "text": text, "speaker": "Unknown"})

    def stop(self) -> None:
        self._is_running = False
