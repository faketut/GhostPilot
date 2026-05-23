"""Cross-platform audio capture using `sounddevice` (PortAudio).

Captures the default *microphone* — NOT system audio. Use this on macOS/Linux
for dev/testing. For real system-audio loopback you need:
  - macOS: install BlackHole and route system output to it
  - Linux: select a Pulseaudio "monitor" source

Mirrors the public API of `AudioCapture` so `main.py` can swap backends.
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

TARGET_RATE = 16000


class AudioCaptureSD:
    def __init__(self, sample_rate: int = TARGET_RATE, chunk_size: int = 2560):
        import sounddevice as sd  # lazy import — keeps module importable without dep
        self._sd = sd
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        self._is_running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_callback_ts = 0.0
        self._last_error: str | None = None
        self._native_rate = sample_rate

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @staticmethod
    def _resample(samples_f32: np.ndarray, src_rate: int, dst_rate: int) -> bytes:
        if src_rate == dst_rate:
            return (samples_f32 * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
        num_src = len(samples_f32)
        num_dst = int(num_src * dst_rate / src_rate)
        if num_dst == 0:
            return b""
        src_indices = np.linspace(0, num_src - 1, num_dst)
        resampled = np.interp(src_indices, np.arange(num_src), samples_f32)
        return (resampled * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()

    def start(self, output_queue: asyncio.Queue, *, device_name_contains: str = "") -> None:
        sd = self._sd
        try:
            device_idx = None
            if device_name_contains:
                needle = device_name_contains.lower()
                for i, info in enumerate(sd.query_devices()):
                    if info.get("max_input_channels", 0) > 0 and needle in info.get("name", "").lower():
                        device_idx = i
                        break

            try:
                default_in = sd.default.device[0] if device_idx is None else device_idx
                info = sd.query_devices(default_in)
                self._native_rate = int(info.get("default_samplerate") or TARGET_RATE)
                dev_name = info.get("name", "?")
            except Exception:
                self._native_rate = TARGET_RATE
                dev_name = "default"

            blocksize = max(1, int(self._native_rate * 0.16))
            logger.info(
                f"[sounddevice] Capturing mic '{dev_name}' "
                f"(native={self._native_rate}Hz -> {TARGET_RATE}Hz). "
                "NOTE: this is microphone input, not system audio loopback."
            )

            def callback(indata, frames, time_info, status):
                self._last_callback_ts = time.time()
                if status:
                    logger.debug(f"[sounddevice] status: {status}")
                # indata is float32 shape (frames, channels)
                mono = indata[:, 0] if indata.ndim > 1 else indata
                pcm = self._resample(mono.astype(np.float32), self._native_rate, TARGET_RATE)
                if self._loop and not self._loop.is_closed():
                    try:
                        self._loop.call_soon_threadsafe(output_queue.put_nowait, pcm)
                    except Exception:
                        pass

            self.stream = sd.InputStream(
                samplerate=self._native_rate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                device=device_idx,
                callback=callback,
            )
            self.stream.start()
            self._is_running = True
            self._last_error = None
            logger.info("[sounddevice] Audio capture started.")
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"[sounddevice] Failed to start capture: {e}")

    def stop(self) -> None:
        if self._is_running and self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self._is_running = False
                logger.info("[sounddevice] Audio capture stopped.")

    def restart(self, output_queue: asyncio.Queue, *, device_name_contains: str = "") -> None:
        try:
            if self.stream is not None:
                try:
                    self.stream.stop()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
            self.stream = None
            self._is_running = False
        except Exception:
            pass
        self.start(output_queue, device_name_contains=device_name_contains)

    def last_callback_age_sec(self) -> float:
        if not self._last_callback_ts:
            return 0.0
        return max(0.0, time.time() - self._last_callback_ts)

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def last_error(self) -> str | None:
        return self._last_error
