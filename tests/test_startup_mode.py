"""Launch-workflow contract: which overlays start, and the ASR opt-out.

`vision` mode exists so the user can run only the visual overlay without the
loopback capture device and the speech service; `resolve_mode` decides it from
the CLI, the saved config, or by asking.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QCheckBox, QDialog  # noqa: E402

from src import startup_mode  # noqa: E402
from src.config import config  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


# ── Mode resolution ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["--overlay", "vision"], "vision"),
        (["--overlay=vision"], "vision"),
        (["--overlay", "VISION"], "vision"),
        (["--overlay", "both"], "both"),
        ([], None),
        (["--other", "vision"], None),
    ],
)
def test_parse_cli_mode(argv, expected):
    assert startup_mode.parse_cli_mode(argv) == expected


def test_cli_overrides_the_saved_mode(monkeypatch):
    monkeypatch.setattr(config, "STARTUP_OVERLAY_MODE", "both", raising=False)
    assert startup_mode.resolve_mode(["--overlay", "vision"]) == "vision"


def test_saved_mode_is_used_when_no_flag(monkeypatch):
    monkeypatch.setattr(config, "STARTUP_OVERLAY_MODE", "vision", raising=False)
    assert startup_mode.resolve_mode([]) == "vision"


def test_unknown_values_fall_back_to_asking(monkeypatch):
    """A typo must not leave the app with no overlay at all."""
    monkeypatch.setattr(config, "STARTUP_OVERLAY_MODE", "visoin", raising=False)
    assert startup_mode.resolve_mode([]) == "ask"
    monkeypatch.setattr(config, "STARTUP_OVERLAY_MODE", "both", raising=False)
    assert startup_mode.resolve_mode(["--overlay", "nope"]) == "both"


# ── Chooser ──────────────────────────────────────────────────────────────


def _open_with(fake_result):
    """Run the chooser with `exec` stubbed, capturing the dialog it built."""
    captured: dict = {}

    def fake_exec(self):
        captured["dlg"] = self
        return fake_result

    return captured, fake_exec


def test_chooser_defaults_to_both_overlays(monkeypatch, qt_app):
    captured, fake_exec = _open_with(QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QDialog, "exec", fake_exec)

    assert startup_mode.choose_overlay_mode() == "both"
    assert "dlg" in captured  # the dialog really was constructed


def test_chooser_returns_none_when_dismissed(monkeypatch, qt_app):
    """Cancel must mean "quit", not "start anyway"."""
    _, fake_exec = _open_with(QDialog.DialogCode.Rejected)
    monkeypatch.setattr(QDialog, "exec", fake_exec)

    assert startup_mode.choose_overlay_mode() is None


def test_remember_choice_persists_the_picked_mode(monkeypatch, qt_app):
    saved: dict = {}
    monkeypatch.setattr("src.settings_ui.save_config_value",
                        lambda k, v: saved.__setitem__(k, v) or True)

    def exec_and_tick(self):
        self.findChild(QCheckBox).setChecked(True)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", exec_and_tick)

    assert startup_mode.choose_overlay_mode() == "both"
    assert saved.get("STARTUP_OVERLAY_MODE") == "both"


def test_chooser_offers_vision_without_asr():
    """The whole point of the chooser: a Vision-only launch is one click away."""
    modes = [mode for mode, _title, _desc in startup_mode.CHOICES]
    assert startup_mode.MODE_VISION in modes
    assert startup_mode.MODE_BOTH in modes
    assert startup_mode.MODE_ASR in modes


# ── Which pipelines start ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode,asr,vision",
    [
        ("both", True, True),
        ("vision", False, True),
        ("asr", True, False),
    ],
)
def test_pipeline_flags(mode, asr, vision):
    """The contract the launch workflow rests on: `vision` must not start ASR."""
    assert startup_mode.pipeline_flags(mode) == (asr, vision)


def test_vision_mode_never_starts_the_asr_half():
    start_asr, start_vision = startup_mode.pipeline_flags(startup_mode.MODE_VISION)
    assert start_asr is False and start_vision is True


def test_ask_is_not_a_runnable_mode():
    """`ask` must be resolved by the chooser, never interpreted as a pipeline set."""
    with pytest.raises(ValueError):
        startup_mode.pipeline_flags(startup_mode.MODE_ASK)


def test_every_chooser_entry_is_runnable():
    for mode, _title, _desc in startup_mode.CHOICES:
        assert startup_mode.pipeline_flags(mode) != (False, False)


# ── Settings tab ─────────────────────────────────────────────────────────


def test_settings_exposes_the_launch_mode(qt_app, monkeypatch, tmp_path):
    """The tab must register the key for the generic save path, or the choice
    silently never reaches config.json."""
    from src.settings_ui import SettingsUI

    monkeypatch.setattr("src.settings_ui.CONFIG_FILE", str(tmp_path / "config.json"))
    dlg = SettingsUI()
    try:
        kind, widget = dlg._fields["STARTUP_OVERLAY_MODE"]
        assert kind == "combo"
        assert widget.currentData() == getattr(config, "STARTUP_OVERLAY_MODE", "ask")
    finally:
        dlg.deleteLater()


def test_save_config_value_merges_instead_of_clobbering(monkeypatch, tmp_path):
    """`Save the mode` must not wipe the overlay geometry already in the file."""
    from src import settings_ui

    path = tmp_path / "config.json"
    path.write_text('{"OVERLAY_ASR_GEOMETRY": [1, 2, 3, 4]}', encoding="utf-8")
    monkeypatch.setattr(settings_ui, "CONFIG_FILE", str(path))
    monkeypatch.setattr(config, "STARTUP_OVERLAY_MODE", "ask", raising=False)

    assert settings_ui.save_config_value("STARTUP_OVERLAY_MODE", "vision") is True
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "OVERLAY_ASR_GEOMETRY": [1, 2, 3, 4],
        "STARTUP_OVERLAY_MODE": "vision",
    }
    assert config.STARTUP_OVERLAY_MODE == "vision"
