"""Speech (ASR backend) settings surface.

The contract: picking a backend reveals exactly that backend's settings — a
field the user cannot reach is a feature they cannot configure — and the
backend's connection test reports what to do next.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src import settings_tests  # noqa: E402
from src.config import config  # noqa: E402
from src.settings_ui import _ASR_BACKEND_OPTIONS, SettingsUI  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def _visible(dlg: SettingsUI, keys: set[str]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for layout, row, key in dlg._scoped_rows:
        if key in keys:
            out[key] = layout.isRowVisible(row)
    return out


AZURE_KEYS = {"AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT", "ASR_LANGUAGE"}
SHERPA_KEYS = {"SHERPA_MODEL", "SHERPA_MODEL_DIR", "SHERPA_RULE2_SILENCE_SEC", "SHERPA_NUM_THREADS"}


def _select_backend(dlg: SettingsUI, backend: str) -> None:
    combo = dlg._fields["ASR_BACKEND"][1]
    for i in range(combo.count()):
        if combo.itemData(i) == backend:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"{backend} is not offered by the ASR backend selector")


@pytest.mark.parametrize("backend", ["azure", "sherpa"])
def test_each_cloud_backend_reveals_only_its_own_settings(qt_app, monkeypatch, tmp_path, backend):
    monkeypatch.setattr("src.settings_ui.CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(config, "ASR_BACKEND", backend, raising=False)
    dlg = SettingsUI()
    try:
        rows = _visible(dlg, AZURE_KEYS | SHERPA_KEYS)
        assert all(rows[k] for k in (AZURE_KEYS if backend == "azure" else SHERPA_KEYS)), rows
        assert not any(rows[k] for k in (SHERPA_KEYS if backend == "azure" else AZURE_KEYS)), rows
    finally:
        dlg.deleteLater()


def test_switching_backend_moves_the_visible_settings(qt_app, monkeypatch, tmp_path):
    """The selector must drive the panel, not just record a value."""
    monkeypatch.setattr("src.settings_ui.CONFIG_FILE", str(tmp_path / "config.json"))
    dlg = SettingsUI()
    try:
        _select_backend(dlg, "sherpa")
        assert _visible(dlg, SHERPA_KEYS)["SHERPA_MODEL"] is True
        assert _visible(dlg, AZURE_KEYS)["AZURE_SPEECH_KEY"] is False

        _select_backend(dlg, "azure")
        assert _visible(dlg, SHERPA_KEYS)["SHERPA_MODEL"] is False
        assert _visible(dlg, AZURE_KEYS)["AZURE_SPEECH_KEY"] is True
    finally:
        dlg.deleteLater()


def test_sherpa_settings_are_collected_for_saving(qt_app, monkeypatch, tmp_path):
    """Unregistered fields silently never reach config.json."""
    monkeypatch.setattr("src.settings_ui.CONFIG_FILE", str(tmp_path / "config.json"))
    dlg = SettingsUI()
    try:
        saved = dlg._collect()
        assert saved["SHERPA_MODEL"]
        assert "SHERPA_RULE2_SILENCE_SEC" in saved
    finally:
        dlg.deleteLater()


def test_the_selector_offers_every_backend_the_factory_builds():
    """main._make_asr_client branches on these values."""
    assert {code for _label, code in _ASR_BACKEND_OPTIONS} == {"azure", "sherpa", "whisper"}


# ── backend connection test (Settings ✗/✓) ───────────────────────────────


def test_sherpa_test_names_the_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)  # makes `import sherpa_onnx` raise
    ok, msg = settings_tests.test_sherpa({})
    assert ok is False
    assert "pip install sherpa-onnx" in msg


def test_sherpa_test_names_the_download_command_for_a_missing_model(monkeypatch, tmp_path):
    pytest.importorskip("sherpa_onnx")
    ok, msg = settings_tests.test_sherpa(
        {"SHERPA_MODEL": "zipformer-en-20M", "SHERPA_MODEL_DIR": str(tmp_path)}
    )
    assert ok is False
    assert "setup_sherpa_asr.py --model zipformer-en-20M" in msg


def test_sherpa_test_lists_known_names_for_an_unknown_model(monkeypatch, tmp_path):
    pytest.importorskip("sherpa_onnx")
    ok, msg = settings_tests.test_sherpa(
        {"SHERPA_MODEL": "my-imaginary-model", "SHERPA_MODEL_DIR": str(tmp_path)}
    )
    assert ok is False
    assert "zipformer-bilingual-zh-en" in msg


def test_sherpa_test_accepts_an_already_extracted_directory(monkeypatch, tmp_path):
    pytest.importorskip("sherpa_onnx")
    model = tmp_path / "hand-made"
    model.mkdir()
    (model / "tokens.txt").write_text("a 1\n", encoding="utf-8")
    (model / "encoder-epoch-99-avg-1.int8.onnx").write_bytes(b"onnx")
    (model / "decoder-epoch-99-avg-1.int8.onnx").write_bytes(b"onnx")

    ok, msg = settings_tests.test_sherpa({"SHERPA_MODEL": str(model)})
    assert ok is True, msg
    assert "hand-made" in msg
