"""Tests for the audio_capture backend factory."""
from __future__ import annotations

import sys
from unittest.mock import patch


def test_factory_explicit_sounddevice():
    """Explicit backend='sounddevice' returns AudioCaptureSD without importing pyaudiowpatch."""
    from src.audio_capture import make_audio_capture
    from src.audio_capture_sounddevice import AudioCaptureSD

    # AudioCaptureSD.__init__ imports sounddevice — patch it to a stub.
    with patch.dict(sys.modules, {"sounddevice": _stub_sd()}):
        cap = make_audio_capture(backend="sounddevice")
    assert isinstance(cap, AudioCaptureSD)
    assert cap.sample_rate == 16000


def test_factory_env_var(monkeypatch):
    """AUDIO_BACKEND env var picks the backend."""
    from src.audio_capture import make_audio_capture
    from src.audio_capture_sounddevice import AudioCaptureSD

    monkeypatch.setenv("AUDIO_BACKEND", "sounddevice")
    with patch.dict(sys.modules, {"sounddevice": _stub_sd()}):
        cap = make_audio_capture()
    assert isinstance(cap, AudioCaptureSD)


def test_factory_explicit_pyaudiowpatch_imports_lazily(monkeypatch):
    """backend='pyaudiowpatch' goes to AudioCapture which lazy-imports pyaudiowpatch."""
    from src.audio_capture import AudioCapture, make_audio_capture

    fake_pa = _stub_pyaudio()
    with patch.dict(sys.modules, {"pyaudiowpatch": fake_pa}):
        cap = make_audio_capture(backend="pyaudiowpatch")
    assert isinstance(cap, AudioCapture)


def test_factory_default_non_windows(monkeypatch):
    """On non-win32 with no env var, factory falls back to sounddevice."""
    from src.audio_capture import make_audio_capture
    from src.audio_capture_sounddevice import AudioCaptureSD

    monkeypatch.delenv("AUDIO_BACKEND", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    with patch.dict(sys.modules, {"sounddevice": _stub_sd()}):
        cap = make_audio_capture()
    assert isinstance(cap, AudioCaptureSD)


# ── helpers ──────────────────────────────────────────────────────────────


def _stub_sd():
    import types

    sd = types.SimpleNamespace()
    sd.default = types.SimpleNamespace(device=(0, 0))
    sd.query_devices = lambda *_a, **_k: {"name": "stub", "default_samplerate": 48000, "max_input_channels": 1}
    sd.InputStream = lambda **_k: types.SimpleNamespace(
        start=lambda: None, stop=lambda: None, close=lambda: None
    )
    return sd


def _stub_pyaudio():
    import types

    pa = types.SimpleNamespace()
    pa.paWASAPI = 1
    pa.paInt16 = 8
    pa.paContinue = 0
    pa.PyAudio = lambda: types.SimpleNamespace(
        get_host_api_info_by_type=lambda _t: {"defaultOutputDevice": -1},
        terminate=lambda: None,
    )
    return pa
