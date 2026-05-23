import asyncio
import logging
import pyaudiowpatch as pyaudio
import numpy as np

logger = logging.getLogger(__name__)

TARGET_RATE = 16000  # Azure Speech requires 16kHz
BITS_PER_SAMPLE = 16


class AudioCapture:
    def __init__(self, sample_rate=TARGET_RATE, chunk_size=2560):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # 160ms = 2560 samples at 16kHz
        self.p = pyaudio.PyAudio()
        self.stream = None
        self._is_running = False
        self._loop = None
        self._last_callback_ts = 0.0
        self._last_error: str | None = None

    def _find_loopback_device(self, *, name_contains: str = ""):
        """Finds the default WASAPI loopback device, with clear error messages."""
        try:
            wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            raise RuntimeError("WASAPI not available. Please run on Windows with PyAudioWPatch.")

        # If user specified a preferred device, try it first.
        if name_contains:
            name_contains_l = name_contains.lower()
            for loopback in self.p.get_loopback_device_info_generator():
                if name_contains_l in loopback.get("name", "").lower():
                    return loopback

        default_output_idx = wasapi_info.get("defaultOutputDevice")
        if default_output_idx is None or default_output_idx < 0:
            raise RuntimeError("No default output device found.")

        default_speakers = self.p.get_device_info_by_index(default_output_idx)

        if not default_speakers.get("isLoopbackDevice"):
            for loopback in self.p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    return loopback
            raise RuntimeError(
                f"Could not find a WASAPI loopback device matching '{default_speakers['name']}'. "
                "Ensure 'Stereo Mix' or a virtual cable is enabled in Windows Sound settings."
            )

        return default_speakers

    @staticmethod
    def _resample(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
        """Resamples raw PCM int16 audio from src_rate to dst_rate using linear interpolation."""
        if src_rate == dst_rate:
            return pcm_bytes
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        num_src = len(samples)
        num_dst = int(num_src * dst_rate / src_rate)
        if num_dst == 0:
            return b''
        src_indices = np.linspace(0, num_src - 1, num_dst)
        resampled = np.interp(src_indices, np.arange(num_src), samples).astype(np.int16)
        return resampled.tobytes()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Injects the running asyncio loop so the callback can schedule coroutines."""
        self._loop = loop

    def start(self, output_queue: asyncio.Queue, *, device_name_contains: str = ""):
        """Starts capturing audio from the default WASAPI loopback device."""
        try:
            device = self._find_loopback_device(name_contains=device_name_contains)
            native_rate = int(device["defaultSampleRate"])
            # Buffer: 160ms of audio at the native rate
            native_chunk = max(1, int(native_rate * 0.16))

            logger.info(
                f"Using loopback device: '{device['name']}' "
                f"(native rate={native_rate}Hz -> resampling to {TARGET_RATE}Hz)"
            )

            def callback(in_data, frame_count, time_info, status):
                import time
                self._last_callback_ts = time.time()
                if in_data and len(in_data) > 0:
                    pcm = self._resample(in_data, native_rate, TARGET_RATE)
                else:
                    # Silence padding: maintain ASR heartbeat when system audio is silent
                    silence_samples = int(TARGET_RATE * 0.16)
                    pcm = b'\x00' * (silence_samples * 2)  # 16-bit = 2 bytes/sample

                if self._loop and not self._loop.is_closed():
                    try:
                        self._loop.call_soon_threadsafe(output_queue.put_nowait, pcm)
                    except Exception:
                        pass
                return (in_data, pyaudio.paContinue)

            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=native_rate,
                input=True,
                frames_per_buffer=native_chunk,
                input_device_index=device["index"],
                stream_callback=callback,
            )
            self._is_running = True
            self._last_error = None
            logger.info("Audio capture started successfully.")
        except RuntimeError as e:
            self._last_error = str(e)
            logger.error(f"Audio capture setup failed: {e}")
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Unexpected error starting audio capture: {e}")

    def stop(self):
        if self._is_running and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self._is_running = False
            logger.info("Audio capture stopped.")

    def last_callback_age_sec(self) -> float:
        import time
        # If start() failed or we haven't been started yet, report 0 so the
        # watchdog doesn't busy-loop attempting restarts on the same broken state.
        # The watchdog separately checks `last_error` / `_is_running` to decide.
        if not self._last_callback_ts:
            return 0.0
        return max(0.0, time.time() - self._last_callback_ts)

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def restart(self, output_queue: asyncio.Queue, *, device_name_contains: str = ""):
        """Best-effort restart on device change/driver hiccup."""
        try:
            if self.stream:
                try:
                    self.stream.stop_stream()
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
        # Recreate PyAudio instance (helps after hotplug on some systems)
        try:
            self.p.terminate()
        except Exception:
            pass
        self.p = pyaudio.PyAudio()
        self.start(output_queue, device_name_contains=device_name_contains)
