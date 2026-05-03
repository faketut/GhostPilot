import json
import os
import logging
from typing import Callable, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QMessageBox,
    QHBoxLayout, QLabel, QComboBox, QGroupBox,
)
from PyQt6.QtCore import Qt
from src.config import config

logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"

_DARK_STYLE = """
QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 4px;
    color: #89b4fa;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
}
QLabel { color: #cdd6f4; }
QLineEdit, QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #cdd6f4;
}
QLineEdit:focus, QComboBox:focus { border-color: #89b4fa; }
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 5px;
    padding: 6px 18px;
    font-weight: bold;
}
QPushButton:hover { background-color: #b4befe; }
QPushButton#cancelBtn {
    background-color: #45475a;
    color: #cdd6f4;
}
QPushButton#cancelBtn:hover { background-color: #585b70; }
"""


class SettingsUI(QDialog):
    def __init__(self, parent=None, *, on_saved: Optional[Callable[[dict], None]] = None):
        super().__init__(parent)
        self.setWindowTitle("GhostPilot Copilot — Settings")
        self.setMinimumWidth(460)
        self.setStyleSheet(_DARK_STYLE)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )

        self._on_saved = on_saved
        self.saved_config = self._load_config()
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Azure Speech ──────────────────────────────────────────────────
        azure_group = QGroupBox("🎙️ Azure Speech Service")
        azure_form = QFormLayout(azure_group)

        self.azure_key_input = self._secret_field("AZURE_SPEECH_KEY", config.AZURE_SPEECH_KEY)
        azure_form.addRow("Subscription Key:", self.azure_key_input)

        self.azure_region_input = self._plain_field("AZURE_SPEECH_REGION", config.AZURE_SPEECH_REGION)
        azure_form.addRow("Region (e.g. eastus):", self.azure_region_input)

        self.asr_language_combo = QComboBox()
        languages = [
            ("zh-CN — 中文（普通话）", "zh-CN"),
            ("en-US — English (US)", "en-US"),
            ("en-GB — English (UK)", "en-GB"),
            ("ja-JP — 日本語", "ja-JP"),
            ("ko-KR — 한국어", "ko-KR"),
            ("de-DE — Deutsch", "de-DE"),
            ("fr-FR — Français", "fr-FR"),
        ]
        current_lang = self.saved_config.get("ASR_LANGUAGE", config.ASR_LANGUAGE)
        for label, code in languages:
            self.asr_language_combo.addItem(label, code)
        for i, (_, code) in enumerate(languages):
            if code == current_lang:
                self.asr_language_combo.setCurrentIndex(i)
                break
        azure_form.addRow("ASR Language:", self.asr_language_combo)
        root.addWidget(azure_group)

        # ── LLM Providers ─────────────────────────────────────────────────
        llm_group = QGroupBox("🤖 LLM Providers")
        llm_form = QFormLayout(llm_group)

        self.openai_key_input = self._secret_field("OPENAI_API_KEY", config.OPENAI_API_KEY)
        llm_form.addRow("OpenAI API Key:", self.openai_key_input)

        self.deepseek_key_input = self._secret_field("DEEPSEEK_API_KEY", config.DEEPSEEK_API_KEY)
        llm_form.addRow("DeepSeek API Key:", self.deepseek_key_input)

        self.text_model_input = self._plain_field("TEXT_MODEL", config.TEXT_MODEL)
        llm_form.addRow("Text Model:", self.text_model_input)

        self.vision_model_input = self._plain_field("VISION_MODEL", config.VISION_MODEL)
        llm_form.addRow("Vision Model:", self.vision_model_input)
        root.addWidget(llm_group)

        # ── Hotkeys ───────────────────────────────────────────────────────
        hk_group = QGroupBox("⌨️ Hotkeys")
        hk_form = QFormLayout(hk_group)

        self.screenshot_hk_input = self._plain_field("SCREENSHOT_HOTKEY", config.SCREENSHOT_HOTKEY)
        hk_form.addRow("Screenshot (Vision):", self.screenshot_hk_input)

        self.interaction_hk_input = self._plain_field("INTERACTION_HOTKEY", config.INTERACTION_HOTKEY)
        hk_form.addRow("Toggle Interaction:", self.interaction_hk_input)
        root.addWidget(hk_group)

        # ── Language ─────────────────────────────────────────────────────
        lang_group = QGroupBox("🌐 Language / 输出语言")
        lang_form = QFormLayout(lang_group)

        self.response_lang_combo = QComboBox()
        self.response_lang_combo.addItem("Auto (跟随问题语言)", "auto")
        self.response_lang_combo.addItem("中文 (Chinese)", "zh")
        self.response_lang_combo.addItem("English", "en")
        current_lang = self.saved_config.get("RESPONSE_LANGUAGE", getattr(config, "RESPONSE_LANGUAGE", "auto"))
        for i in range(self.response_lang_combo.count()):
            if self.response_lang_combo.itemData(i) == current_lang:
                self.response_lang_combo.setCurrentIndex(i)
                break
        lang_form.addRow("LLM response language:", self.response_lang_combo)
        root.addWidget(lang_group)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        save_btn = QPushButton("💾 Save")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancelBtn")
        save_btn.clicked.connect(self._save)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        root.addLayout(btn_row)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _plain_field(self, key: str, fallback: str) -> QLineEdit:
        field = QLineEdit()
        field.setText(self.saved_config.get(key, fallback))
        return field

    def _secret_field(self, key: str, fallback: str) -> QLineEdit:
        field = self._plain_field(key, fallback)
        field.setEchoMode(QLineEdit.EchoMode.Password)
        return field

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config.json: {e}")
        return {}

    def _save(self):
        new_cfg = {
            "AZURE_SPEECH_KEY": self.azure_key_input.text().strip(),
            "AZURE_SPEECH_REGION": self.azure_region_input.text().strip(),
            "ASR_LANGUAGE": self.asr_language_combo.currentData(),
            "OPENAI_API_KEY": self.openai_key_input.text().strip(),
            "DEEPSEEK_API_KEY": self.deepseek_key_input.text().strip(),
            "TEXT_MODEL": self.text_model_input.text().strip(),
            "VISION_MODEL": self.vision_model_input.text().strip(),
            "SCREENSHOT_HOTKEY": self.screenshot_hk_input.text().strip(),
            "INTERACTION_HOTKEY": self.interaction_hk_input.text().strip(),
            "RESPONSE_LANGUAGE": self.response_lang_combo.currentData(),
        }

        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(new_cfg, f, indent=4, ensure_ascii=False)

            # Hot-patch the in-memory config
            for k, v in new_cfg.items():
                if hasattr(config, k):
                    setattr(config, k, v)

            if self._on_saved:
                try:
                    self._on_saved(new_cfg)
                except Exception as e:
                    logger.error(f"Settings on_saved callback failed: {e}")

            QMessageBox.information(
                self, "Saved",
                "Settings saved.\nMost changes apply immediately."
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
            logger.error(f"Failed to save settings: {e}")


def apply_saved_config():
    """Call at startup to override env/defaults with anything stored in config.json."""
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for k, v in saved.items():
            if hasattr(config, k):
                setattr(config, k, v)
        logger.info(f"Loaded {len(saved)} settings from {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Failed to apply {CONFIG_FILE}: {e}")
