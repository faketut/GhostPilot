"""
Settings UI
-----------
Tabbed Qt dialog with show/hide toggles on every secret field.  Persists to
config.json and hot-patches the in-memory `config` so most changes apply
without a restart.
"""

import json
import os
import logging
from typing import Callable, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QTabWidget, QWidget,
    QLineEdit, QPushButton, QMessageBox, QLabel, QToolButton,
    QHBoxLayout, QComboBox, QPlainTextEdit, QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from src.config import config
from src import theme
from src import settings_tests
from src import prompt_loader

logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"


def _dark_stylesheet() -> str:
    pal = theme.palette()
    return f"""
QDialog {{
    background-color: {pal['dialog_bg']};
    color: {pal['text_primary']};
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {pal['dialog_border']};
    border-radius: 6px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {pal['text_dim']};
    padding: 6px 14px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar::tab:selected {{
    background: {pal['dialog_field']};
    color: {pal['accent_blue']};
    border-color: {pal['dialog_border']};
}}
QLabel {{ color: {pal['text_primary']}; }}
QLineEdit, QComboBox {{
    background-color: {pal['dialog_field']};
    border: 1px solid {pal['dialog_border']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {pal['text_primary']};
}}
QLineEdit:focus, QComboBox:focus {{ border-color: {pal['accent_blue']}; }}
QPushButton {{
    background-color: {pal['dialog_btn']};
    color: {pal['dialog_bg']};
    border: none;
    border-radius: 5px;
    padding: 6px 18px;
    font-weight: bold;
}}
QPushButton:hover {{ background-color: {pal['dialog_btn_hi']}; }}
QPushButton#cancelBtn {{
    background-color: {pal['dialog_btn2']};
    color: {pal['text_primary']};
}}
QPushButton#cancelBtn:hover {{ background-color: {pal['dialog_btn2_hi']}; }}
"""


# Centralised field schema (issue #20: "adding a new field requires changes in exactly one place").
# Each entry: (config_key, label, kind)
#   kind ∈ {"text", "secret", "combo:<id>"}
_ASR_LANG_OPTIONS = [
    ("zh-CN — 中文（普通话）", "zh-CN"),
    ("en-US — English (US)", "en-US"),
    ("en-GB — English (UK)", "en-GB"),
    ("ja-JP — 日本語", "ja-JP"),
    ("ko-KR — 한국어", "ko-KR"),
    ("de-DE — Deutsch", "de-DE"),
    ("fr-FR — Français", "fr-FR"),
]
_RESPONSE_LANG_OPTIONS = [
    ("Auto (follow question language)", "auto"),
    ("中文 (Chinese)", "zh"),
    ("English", "en"),
]
_TEXT_PROVIDER_OPTIONS = [
    ("Auto (infer from model name)", ""),
    ("OpenAI", "openai"),
    ("DeepSeek", "deepseek"),
    ("Gemini", "gemini"),
    ("Ollama (local)", "ollama"),
]
_VISION_PROVIDER_OPTIONS = [
    ("Auto (infer from model name)", ""),
    ("OpenAI", "openai"),
    ("Gemini", "gemini"),
]
_ASR_BACKEND_OPTIONS = [
    ("Azure Speech (cloud)", "azure"),
    ("faster-whisper (local, offline)", "whisper"),
]


class SettingsUI(QDialog):
    def __init__(self, parent=None, *, on_saved: Optional[Callable[[dict], None]] = None):
        super().__init__(parent)
        self.setWindowTitle("GhostPilot Copilot — Settings")
        self.setMinimumWidth(520)
        self.setStyleSheet(_dark_stylesheet())
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )

        self._on_saved = on_saved
        self.saved_config = self._load_config()
        # Layer keyring values on top so the form pre-fills with current secrets.
        try:
            from src import secret_store
            for k in secret_store.SECRET_KEYS:
                v = secret_store.get(k)
                if v:
                    self.saved_config[k] = v
        except Exception as e:
            logger.warning(f"keyring read for SettingsUI failed: {e}")
        # Maps config key → (kind, widget). Saved/loaded generically.
        self._fields: dict[str, tuple[str, QWidget]] = {}
        # (layout, row_index, config_key) for rows that auto-hide by provider.
        self._scoped_rows: list[tuple[QFormLayout, int, str]] = []

        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        tabs = QTabWidget(self)

        # ── Tab: Azure ──
        azure = QFormLayout()
        azure_w = QWidget()
        azure_w.setLayout(azure)
        azure.addRow("Subscription key:",   self._add_secret("AZURE_SPEECH_KEY", config.AZURE_SPEECH_KEY, test="azure"))
        self._track_scoped(azure, "AZURE_SPEECH_KEY")
        azure.addRow("Region (e.g. eastus):", self._add_text("AZURE_SPEECH_REGION", config.AZURE_SPEECH_REGION))
        self._track_scoped(azure, "AZURE_SPEECH_REGION")
        azure.addRow("Custom endpoint (optional):", self._add_text("AZURE_SPEECH_ENDPOINT", config.AZURE_SPEECH_ENDPOINT))
        self._track_scoped(azure, "AZURE_SPEECH_ENDPOINT")
        azure.addRow("ASR language:", self._add_combo("ASR_LANGUAGE", config.ASR_LANGUAGE, _ASR_LANG_OPTIONS))
        self._track_scoped(azure, "ASR_LANGUAGE")
        azure.addRow("ASR backend:", self._add_combo(
            "ASR_BACKEND",
            getattr(config, "ASR_BACKEND", "azure"),
            _ASR_BACKEND_OPTIONS,
        ))
        tabs.addTab(azure_w, "🎙️ Azure")

        # ── Tab: LLM ──
        llm = QFormLayout()
        llm_w = QWidget()
        llm_w.setLayout(llm)
        llm.addRow("OpenAI API key:",   self._add_secret("OPENAI_API_KEY", config.OPENAI_API_KEY, test="openai"))
        self._track_scoped(llm, "OPENAI_API_KEY")
        llm.addRow("DeepSeek API key:", self._add_secret("DEEPSEEK_API_KEY", config.DEEPSEEK_API_KEY, test="deepseek"))
        self._track_scoped(llm, "DEEPSEEK_API_KEY")
        llm.addRow("Gemini API key:",   self._add_secret("GEMINI_API_KEY", config.GEMINI_API_KEY, test="gemini"))
        self._track_scoped(llm, "GEMINI_API_KEY")
        llm.addRow("Text model:",       self._add_text("TEXT_MODEL", config.TEXT_MODEL))
        llm.addRow("Text provider:",    self._add_combo("TEXT_PROVIDER", getattr(config, "TEXT_PROVIDER", ""), _TEXT_PROVIDER_OPTIONS))
        llm.addRow("Vision model:",     self._add_text("VISION_MODEL", config.VISION_MODEL))
        llm.addRow("Vision provider:",  self._add_combo("VISION_PROVIDER", getattr(config, "VISION_PROVIDER", ""), _VISION_PROVIDER_OPTIONS))
        llm.addRow("Context turns (0=off):", self._add_text("CONTEXT_TURNS", str(getattr(config, "CONTEXT_TURNS", 0))))
        # ── Ollama (local) ──
        ollama_row = QHBoxLayout()
        ollama_field = self._add_text("OLLAMA_BASE_URL", getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        ollama_btn = QToolButton(); ollama_btn.setText("🔍"); ollama_btn.setToolTip("Detect running Ollama instance")
        ollama_status = QLabel(""); ollama_status.setMinimumWidth(140)
        def _detect_ollama():
            import urllib.request, json as _json
            url = (ollama_field.text() or "").rstrip("/")
            base = url.replace("/v1", "")
            try:
                with urllib.request.urlopen(f"{base}/api/tags", timeout=2) as r:
                    data = _json.loads(r.read().decode("utf-8"))
                models = [m.get("name", "?") for m in data.get("models", [])]
                if models:
                    ollama_status.setText(f"✓ {len(models)} models: {', '.join(models[:3])}")
                    ollama_status.setStyleSheet("color: #2e7d32;")
                else:
                    ollama_status.setText("✓ running, no models")
                    ollama_status.setStyleSheet("color: #2e7d32;")
            except Exception as e:
                ollama_status.setText(f"✗ {e}")
                ollama_status.setStyleSheet("color: #c62828;")
        ollama_btn.clicked.connect(_detect_ollama)
        ollama_row.addWidget(ollama_field, 1); ollama_row.addWidget(ollama_btn); ollama_row.addWidget(ollama_status)
        ollama_w = QWidget(); ollama_w.setLayout(ollama_row)
        llm.addRow("Ollama base URL:", ollama_w)
        self._track_scoped(llm, "OLLAMA_BASE_URL")

        # ── Master test (selected text + vision providers) ──
        test_row = QHBoxLayout()
        test_btn = QPushButton("🧪 Test selected providers")
        test_status = QLabel("")
        test_status.setStyleSheet("color:#9aa; font-size:11px;")
        test_row.addWidget(test_btn); test_row.addWidget(test_status, 1)

        def _resolve_provider(combo_val: str, model: str, is_vision: bool) -> str:
            v = (combo_val or "").strip().lower()
            if v:
                return v
            m = (model or "").lower()
            if not is_vision and (m.startswith("ollama/") or m.startswith("llama") or m.startswith("qwen") or m.startswith("mistral") or m.startswith("phi")):
                return "ollama"
            if "deepseek" in m:
                return "deepseek"
            if "gemini" in m:
                return "gemini"
            return "openai"

        def _run_master_test():
            test_btn.setEnabled(False)
            test_status.setText("… testing text + vision")
            test_status.setStyleSheet("color:#9aa; font-size:11px;")
            params = self._snapshot_for_test()
            text_p = _resolve_provider(params.get("TEXT_PROVIDER", ""), params.get("TEXT_MODEL", ""), is_vision=False)
            vis_p = _resolve_provider(params.get("VISION_PROVIDER", ""), params.get("VISION_MODEL", ""), is_vision=True)
            results: dict[str, tuple[bool, str]] = {}

            def _maybe_done():
                if "text" in results and "vision" in results:
                    test_btn.setEnabled(True)
                    ok_t, msg_t = results["text"]
                    ok_v, msg_v = results["vision"]
                    ok = ok_t and ok_v
                    text = f"text({text_p}): {'✓' if ok_t else '✗'} {msg_t}  ·  vision({vis_p}): {'✓' if ok_v else '✗'} {msg_v}"
                    test_status.setText(text)
                    test_status.setStyleSheet(f"color:{'#6c6' if ok else '#c66'}; font-size:11px;")

            settings_tests.run_test(text_p, params, lambda ok, msg: (results.__setitem__("text", (ok, msg)), _maybe_done()))
            settings_tests.run_test(vis_p, params, lambda ok, msg: (results.__setitem__("vision", (ok, msg)), _maybe_done()))

        test_btn.clicked.connect(_run_master_test)
        test_w = QWidget(); test_w.setLayout(test_row)
        llm.addRow("", test_w)
        tabs.addTab(llm_w, "🤖 LLM")

        # ── Tab: Hotkeys (issue #4) ──
        hk = QFormLayout()
        hk_w = QWidget()
        hk_w.setLayout(hk)
        hk.addRow("Screenshot (Vision):",        self._add_text("SCREENSHOT_HOTKEY", config.SCREENSHOT_HOTKEY))
        hk.addRow("Both overlays (primary):",    self._add_text("ASR_INTERACTION_HOTKEY", config.ASR_INTERACTION_HOTKEY))
        hk.addRow("Both overlays (backup):",     self._add_text("ASR_INTERACTION_HOTKEY_BACKUP", config.ASR_INTERACTION_HOTKEY_BACKUP))
        hk.addRow("Vision overlay (primary):",   self._add_text("VISION_INTERACTION_HOTKEY", config.VISION_INTERACTION_HOTKEY))
        hk.addRow("Vision overlay (backup):",    self._add_text("VISION_INTERACTION_HOTKEY_BACKUP", config.VISION_INTERACTION_HOTKEY_BACKUP))
        hk.addRow("Force stealth (primary):",    self._add_text("FORCE_STEALTH_HOTKEY", config.FORCE_STEALTH_HOTKEY))
        hk.addRow("Force stealth (backup):",     self._add_text("FORCE_STEALTH_HOTKEY_BACKUP", config.FORCE_STEALTH_HOTKEY_BACKUP))
        hk.addRow("Both overlays (alias):",      self._add_text("INTERACTION_HOTKEY", config.INTERACTION_HOTKEY))
        tabs.addTab(hk_w, "⌨️ Hotkeys")

        # ── Tab: Language ──
        lang = QFormLayout()
        lang_w = QWidget()
        lang_w.setLayout(lang)
        lang.addRow("LLM response language:", self._add_combo(
            "RESPONSE_LANGUAGE",
            getattr(config, "RESPONSE_LANGUAGE", "auto"),
            _RESPONSE_LANG_OPTIONS,
        ))
        tabs.addTab(lang_w, "🌐 Language")

        # ── Tab: Prompts (issue #10 in plan) ──
        tabs.addTab(self._build_prompts_tab(), "📝 Prompts")

        root.addWidget(tabs, 1)

        # ── Provider-scoped row visibility ──
        self._wire_visibility()
        self._refresh_visibility()

        # ── Buttons ──
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

    # ── Provider-scoped row visibility ───────────────────────────────────

    def _track_scoped(self, layout: QFormLayout, key: str) -> None:
        """Remember the row just added so it can be hidden when its provider is inactive."""
        self._scoped_rows.append((layout, layout.rowCount() - 1, key))

    def _wire_visibility(self) -> None:
        # Recompute whenever the user changes any provider hint.
        for key in ("TEXT_PROVIDER", "VISION_PROVIDER", "ASR_BACKEND"):
            w = self._fields.get(key, (None, None))[1]
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(lambda _i: self._refresh_visibility())
        for key in ("TEXT_MODEL", "VISION_MODEL"):
            w = self._fields.get(key, (None, None))[1]
            if isinstance(w, QLineEdit):
                w.textChanged.connect(lambda _t: self._refresh_visibility())

    def _resolve_text_provider(self) -> str:
        combo = self._fields.get("TEXT_PROVIDER", (None, None))[1]
        v = (combo.currentData() if isinstance(combo, QComboBox) else "") or ""
        v = v.strip().lower()
        if v:
            return v
        m_field = self._fields.get("TEXT_MODEL", (None, None))[1]
        m = (m_field.text() if isinstance(m_field, QLineEdit) else "").lower()
        if m.startswith(("ollama/", "llama", "qwen", "mistral", "phi")):
            return "ollama"
        if "deepseek" in m:
            return "deepseek"
        if "gemini" in m:
            return "gemini"
        return "openai"

    def _resolve_vision_provider(self) -> str:
        combo = self._fields.get("VISION_PROVIDER", (None, None))[1]
        v = (combo.currentData() if isinstance(combo, QComboBox) else "") or ""
        v = v.strip().lower()
        if v:
            return v
        m_field = self._fields.get("VISION_MODEL", (None, None))[1]
        m = (m_field.text() if isinstance(m_field, QLineEdit) else "").lower()
        if "gemini" in m:
            return "gemini"
        return "openai"

    def _resolve_asr_backend(self) -> str:
        combo = self._fields.get("ASR_BACKEND", (None, None))[1]
        v = (combo.currentData() if isinstance(combo, QComboBox) else "") or "azure"
        return v.strip().lower() or "azure"

    def _refresh_visibility(self) -> None:
        t = self._resolve_text_provider()
        v = self._resolve_vision_provider()
        a = self._resolve_asr_backend()
        show = {
            "OPENAI_API_KEY":      t == "openai" or v == "openai",
            "DEEPSEEK_API_KEY":    t == "deepseek",
            "GEMINI_API_KEY":      t == "gemini" or v == "gemini",
            "OLLAMA_BASE_URL":     t == "ollama",
            "AZURE_SPEECH_KEY":    a == "azure",
            "AZURE_SPEECH_REGION": a == "azure",
            "AZURE_SPEECH_ENDPOINT": a == "azure",
            "ASR_LANGUAGE":        a == "azure",
        }
        for layout, row, key in self._scoped_rows:
            if key in show:
                try:
                    layout.setRowVisible(row, show[key])
                except AttributeError:
                    # Qt < 6.4 fallback: leave visible.
                    pass

    # ── Prompts tab ──────────────────────────────────────────────────────

    def _build_prompts_tab(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        self._prompt_list = QListWidget()
        self._prompt_list.setMaximumWidth(140)
        for name in prompt_loader.list_prompts():
            self._prompt_list.addItem(QListWidgetItem(name))
        h.addWidget(self._prompt_list)

        right = QVBoxLayout()
        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setStyleSheet("font-family: 'Menlo','Consolas',monospace; font-size:12px;")
        right.addWidget(self._prompt_edit, 1)

        btn_row = QHBoxLayout()
        self._prompt_status = QLabel("")
        self._prompt_status.setStyleSheet("color:#9aa; font-size:11px;")
        btn_row.addWidget(self._prompt_status, 1)
        restore_btn = QPushButton("Restore default")
        save_prompt_btn = QPushButton("💾 Save prompt")
        restore_btn.clicked.connect(self._on_prompt_restore)
        save_prompt_btn.clicked.connect(self._on_prompt_save)
        btn_row.addWidget(restore_btn)
        btn_row.addWidget(save_prompt_btn)
        right.addLayout(btn_row)

        rw = QWidget(); rw.setLayout(right)
        h.addWidget(rw, 1)

        self._prompt_list.currentRowChanged.connect(self._on_prompt_selected)
        if self._prompt_list.count() > 0:
            self._prompt_list.setCurrentRow(0)
        return w

    def _current_prompt_name(self) -> str | None:
        it = self._prompt_list.currentItem()
        return it.text() if it else None

    def _on_prompt_selected(self, _row: int) -> None:
        name = self._current_prompt_name()
        if not name:
            return
        self._prompt_edit.setPlainText(prompt_loader.get_prompt(name))
        self._prompt_status.setText(str(prompt_loader.prompt_path(name)))

    def _on_prompt_save(self) -> None:
        name = self._current_prompt_name()
        if not name:
            return
        try:
            path = prompt_loader.save(name, self._prompt_edit.toPlainText())
            self._flash_prompt_status(f"Saved → {path.name}", ok=True)
        except Exception as e:
            self._flash_prompt_status(f"Save failed: {e}", ok=False)

    def _on_prompt_restore(self) -> None:
        name = self._current_prompt_name()
        if not name:
            return
        self._prompt_edit.setPlainText(prompt_loader.fallback_text(name))
        self._flash_prompt_status("Restored to bundled default (not yet saved)", ok=True)

    def _flash_prompt_status(self, msg: str, *, ok: bool) -> None:
        color = "#6c6" if ok else "#c66"
        self._prompt_status.setText(msg)
        self._prompt_status.setStyleSheet(f"color:{color}; font-size:11px;")
        QTimer.singleShot(5000, lambda: (
            self._prompt_status.setText(""),
            self._prompt_status.setStyleSheet("color:#9aa; font-size:11px;"),
        ))

    # ── Field builders ───────────────────────────────────────────────────

    def _add_text(self, key: str, fallback: str) -> QLineEdit:
        field = QLineEdit()
        field.setText(self.saved_config.get(key, fallback))
        self._fields[key] = ("text", field)
        return field

    def _add_secret(self, key: str, fallback: str, *, test: str | None = None) -> QWidget:
        field = QLineEdit()
        field.setText(self.saved_config.get(key, fallback))
        field.setEchoMode(QLineEdit.EchoMode.Password)
        # Show/hide eye action (issue #20)
        action = QAction("👁", field)
        action.setToolTip("Show / hide")
        action.setCheckable(True)

        def _toggle(checked: bool) -> None:
            field.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)
            action.setText("🙈" if checked else "👁")

        action.toggled.connect(_toggle)
        field.addAction(action, QLineEdit.ActionPosition.TrailingPosition)
        self._fields[key] = ("secret", field)
        if test is None:
            return field
        return self._wrap_with_test(field, test, key)

    # ── Connection-test row ──────────────────────────────────────────────

    def _wrap_with_test(self, field: QLineEdit, provider: str, key: str) -> QWidget:
        """Wrap a secret field with a 🧪 test button and inline status label."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(field, 1)

        btn = QToolButton()
        btn.setText("🧪")
        btn.setToolTip(f"Test {provider} connection")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        h.addWidget(btn)

        status = QLabel("")
        status.setStyleSheet("color:#9aa; font-size:11px;")
        status.setMinimumWidth(110)
        h.addWidget(status)

        def _on_click():
            btn.setEnabled(False)
            status.setText("… testing")
            status.setStyleSheet("color:#9aa; font-size:11px;")
            # Snapshot live form values rather than saved config.
            params = self._snapshot_for_test()
            params[key] = field.text().strip()
            settings_tests.run_test(
                provider, params,
                lambda ok, msg: self._show_test_result(btn, status, ok, msg),
            )

        btn.clicked.connect(_on_click)
        return row

    def _snapshot_for_test(self) -> dict:
        out: dict = {}
        for k, (_kind, w) in self._fields.items():
            if isinstance(w, QLineEdit):
                out[k] = w.text().strip()
        return out

    def _show_test_result(self, btn: QToolButton, status: QLabel, ok: bool, msg: str) -> None:
        btn.setEnabled(True)
        if ok:
            status.setText(f"✓ {msg}")
            status.setStyleSheet("color:#6c6; font-size:11px;")
        else:
            status.setText(f"✗ {msg}")
            status.setStyleSheet("color:#c66; font-size:11px;")
        QTimer.singleShot(8000, lambda: (status.setText(""), status.setStyleSheet("color:#9aa; font-size:11px;")))

    def _add_combo(self, key: str, current: str, options: list[tuple[str, str]]) -> QComboBox:
        combo = QComboBox()
        for label, code in options:
            combo.addItem(label, code)
        for i, (_, code) in enumerate(options):
            if code == current:
                combo.setCurrentIndex(i)
                break
        self._fields[key] = ("combo", combo)
        return combo

    # ── Persistence ──────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config.json: {e}")
        return {}

    def _collect(self) -> dict:
        out: dict = {}
        for key, (kind, widget) in self._fields.items():
            if kind == "combo":
                out[key] = widget.currentData()
            else:
                out[key] = widget.text().strip()
        return out

    def _save(self):
        new_cfg = self._collect()
        # Preserve non-form values (e.g. geometry persisted by overlays).
        existing = self._load_config()
        existing.update(new_cfg)

        # Route secrets through the OS keyring; on success, strip them from
        # the dict so they never hit disk in plain text.
        from src import secret_store
        for k in secret_store.SECRET_KEYS:
            if k in existing and isinstance(existing[k], str) and existing[k]:
                if secret_store.set(k, existing[k]):
                    existing.pop(k, None)

        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4, ensure_ascii=False)

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
    """Call at startup to override env/defaults with anything stored in config.json.

    Also performs a one-shot migration of any plain-text secrets in config.json
    into the OS keyring, then rewrites the file without them.
    """
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)

        # One-shot keyring migration
        from src import secret_store
        migrated = secret_store.migrate_from_dict(saved)
        if migrated:
            try:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(saved, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Could not rewrite {CONFIG_FILE} after migration: {e}")

        for k, v in saved.items():
            if hasattr(config, k):
                setattr(config, k, v)

        # Re-apply secrets from keyring (overrides any leftover values).
        for k in secret_store.SECRET_KEYS:
            v = secret_store.get(k)
            if v and hasattr(config, k):
                setattr(config, k, v)

        logger.info(f"Loaded {len(saved)} settings from {CONFIG_FILE}"
                    + (f" (migrated {len(migrated)} secret(s) to keyring)" if migrated else ""))
    except Exception as e:
        logger.error(f"Failed to apply {CONFIG_FILE}: {e}")
