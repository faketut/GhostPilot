"""Startup overlay-mode selection.

GhostPilot brings up two independent pipelines — the ASR overlay (WASAPI
loopback → speech recognition → text LLM) and the Vision overlay (screenshot →
local OCR → text LLM). The ASR side owns the microphone/loopback device and the
speech service, and is pointless when the user only wants the visual overlay.

This module decides, at launch, which overlays to start:

* :func:`resolve_mode` — precedence ``--overlay <mode>`` > ``STARTUP_OVERLAY_MODE``
  env/config > ``"ask"``. It never touches Qt and is safe to unit test.
* :func:`choose_overlay_mode` — the Qt chooser shown when the resolved mode is
  ``"ask"``. Returns the picked mode, or ``None`` when the user quits.

Modes:
    ``both``   — ASR + Vision overlays (the pre-existing behaviour)
    ``vision`` — Vision only; no audio capture, no ASR service
    ``asr``    — ASR only; no screenshot/vision overlay
    ``ask``    — show the chooser (only meaningful as a configured value)
"""

from __future__ import annotations

import logging
import sys

from src.config import config

logger = logging.getLogger(__name__)

MODE_ASK = "ask"
MODE_BOTH = "both"
MODE_VISION = "vision"
MODE_ASR = "asr"

# Modes that actually start something (everything except the "ask" sentinel).
RUN_MODES = (MODE_BOTH, MODE_VISION, MODE_ASR)
_ALL_MODES = (MODE_ASK, *RUN_MODES)

#: Chooser entries: (mode, title, one-line explanation).
CHOICES = (
    (MODE_BOTH, "Both overlays", "ASR (speech) + Vision (screenshot) — the full pipeline."),
    (MODE_VISION, "Vision overlay only", "Screenshot → local OCR → LLM. No audio capture, no ASR service."),
    (MODE_ASR, "ASR overlay only", "Speech → LLM. No screenshot/vision overlay."),
)


def parse_cli_mode(argv: list[str]) -> str | None:
    """Extract ``--overlay <mode>`` / ``--overlay=<mode>`` from ``argv``.

    Returns the lower-cased raw value (validation happens in
    :func:`resolve_mode`) or ``None`` when the flag is absent.
    """
    for i, arg in enumerate(argv):
        if arg == "--overlay" and i + 1 < len(argv):
            return argv[i + 1].strip().lower()
        if arg.startswith("--overlay="):
            return arg.split("=", 1)[1].strip().lower()
    return None


def resolve_mode(argv: list[str] | None = None) -> str:
    """Resolve the launch mode: CLI flag > configured value > ``"ask"``.

    Unknown values (from either source) are logged and ignored, so a typo can
    never leave the app with no overlays.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    cli = parse_cli_mode(args)
    if cli is not None:
        if cli in _ALL_MODES:
            return cli
        logger.warning("Ignoring unknown --overlay value %r.", cli)

    configured = str(getattr(config, "STARTUP_OVERLAY_MODE", MODE_ASK) or MODE_ASK).strip().lower()
    if configured in _ALL_MODES:
        return configured
    if configured:
        logger.warning("Ignoring unknown STARTUP_OVERLAY_MODE %r.", configured)
    return MODE_ASK


def pipeline_flags(mode: str) -> tuple[bool, bool]:
    """Map a run mode to ``(start_asr, start_vision)``.

    The whole point of the ``vision`` mode is that the ASR half — loopback
    capture, watchdog, speech service, transcript router — never starts. That
    promise lives here and nowhere else, so it can be asserted directly.
    """
    if mode == MODE_ASK:  # defensive: "ask" is resolved to a run mode before this
        raise ValueError("'ask' is not a run mode; resolve it first")
    return (
        mode in (MODE_BOTH, MODE_ASR),
        mode in (MODE_BOTH, MODE_VISION),
    )


def choose_overlay_mode() -> str | None:
    """Show the modal overlay chooser. Returns the chosen mode, or ``None`` to quit.

    Tick "remember my choice" to persist the pick to ``config.json`` (so a
    later launch, including a login/autostart one, goes straight to the
    pipelines). Qt is imported locally so importing this module never requires
    a working PyQt6 install.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QButtonGroup,
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QLabel,
        QRadioButton,
        QVBoxLayout,
    )

    from src import icons
    from src.settings_ui import _dark_stylesheet, save_config_value

    dlg = QDialog()
    dlg.setWindowTitle("GhostPilot — overlay")
    dlg.setMinimumWidth(440)
    dlg.setStyleSheet(_dark_stylesheet())
    dlg.setWindowFlags(
        Qt.WindowType.Dialog
        | Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.Tool
    )
    app_icon = icons.app_icon()
    if not app_icon.isNull():
        dlg.setWindowIcon(app_icon)

    root = QVBoxLayout(dlg)
    root.setContentsMargins(16, 16, 16, 16)
    root.setSpacing(10)

    heading = QLabel("Which overlay should GhostPilot open?")
    heading.setStyleSheet("font-size: 14px; font-weight: bold;")
    root.addWidget(heading)

    group = QButtonGroup(dlg)
    buttons: dict[str, QRadioButton] = {}
    for mode, title, desc in CHOICES:
        btn = QRadioButton(title)
        btn.setToolTip(desc)
        group.addButton(btn)
        root.addWidget(btn)
        hint = QLabel(desc)
        hint.setStyleSheet("color: #9aa; font-size: 11px; margin-left: 22px;")
        hint.setWordWrap(True)
        root.addWidget(hint)
        buttons[mode] = btn
    buttons[MODE_BOTH].setChecked(True)

    remember = QCheckBox("Remember my choice (skip this dialog next launch)")
    root.addWidget(remember)
    root.addSpacing(4)

    box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    box.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
    box.button(QDialogButtonBox.StandardButton.Cancel).setText("Quit")
    box.accepted.connect(dlg.accept)
    box.rejected.connect(dlg.reject)
    root.addWidget(box)

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None

    for mode, btn in buttons.items():
        if btn.isChecked():
            chosen = mode
            break
    else:
        chosen = MODE_BOTH

    if remember.isChecked():
        if save_config_value("STARTUP_OVERLAY_MODE", chosen):
            logger.info("Startup overlay mode remembered: %s", chosen)
        else:
            logger.warning("Could not persist STARTUP_OVERLAY_MODE=%s", chosen)
    return chosen
