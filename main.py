import sys
import logging
import asyncio
import os

from src.windows_api import set_dpi_awareness
from src.asr_client import ASRClient
from src.rag_manager import RAGManager
from src.llm_engine import LLMEngine
from src.hotkey_manager import HotkeyManager
from src.config import config
from src import crash_logger


def _make_asr_client():
    """Build the ASR client according to ``config.ASR_BACKEND``.

    Always returns something duck-typed compatible with ``ASRClient``
    (``start_streaming``/``stop``) so callers don't branch.
    """
    backend = (getattr(config, "ASR_BACKEND", "azure") or "azure").strip().lower()
    if backend == "whisper":
        try:
            from src.whisper_asr import FasterWhisperASRClient
            return FasterWhisperASRClient(
                model_name=getattr(config, "WHISPER_MODEL", "small"),
                device=getattr(config, "WHISPER_DEVICE", "auto"),
                compute_type=getattr(config, "WHISPER_COMPUTE_TYPE", "int8"),
                language=(getattr(config, "ASR_LANGUAGE", "en-US") or "").split("-")[0] or None,
                window_sec=getattr(config, "WHISPER_WINDOW_SEC", 2.5),
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Whisper backend requested but unavailable ({e}); falling back to Azure."
            )
    return ASRClient(
        config.AZURE_SPEECH_KEY,
        config.AZURE_SPEECH_REGION,
        config.AZURE_SPEECH_ENDPOINT,
    )

# Defer audio_capture import (pyaudiowpatch is Windows-only); on macOS/Linux
# devs can still run UI / RAG / prompts iteration without the audio pipeline.
# Use make_audio_capture() factory to pick the right backend at runtime.
try:
    from src.audio_capture import make_audio_capture
    AUDIO_AVAILABLE = True
    _AUDIO_IMPORT_ERROR: Exception | None = None
except Exception as e:
    make_audio_capture = None  # type: ignore[assignment]
    AUDIO_AVAILABLE = False
    _AUDIO_IMPORT_ERROR = e

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
# Reduce noise from dependency loggers (keep app logs readable)
for name in (
    "httpx",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "torch",
):
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def ui_updater(ui, ui_queue: asyncio.Queue):
    """Consumes the UI queue and updates the PyQt window."""
    while True:
        try:
            msg = await ui_queue.get()
            if msg["type"] == "token":
                ui.update_text(msg["text"], append=True)
            elif msg["type"] == "clear":
                ui.update_text("", append=False)
            elif msg["type"] == "info":
                # Observability strip (provider · rag:N)
                try:
                    ui.set_info_footer(
                        provider=msg.get("provider", ""),
                        rag_hits=msg.get("rag_hits"),
                    )
                except Exception:
                    pass
            elif msg["type"] == "usage":
                # Token / cost footer (Phase 2.3)
                try:
                    from src.llm.pricing import cost_usd, format_cost
                    c = cost_usd(msg.get("model", ""), msg.get("in", 0), msg.get("out", 0))
                    parts = [f"↑{msg.get('in', 0)}", f"↓{msg.get('out', 0)}"]
                    cs = format_cost(c)
                    if cs:
                        parts.append(cs)
                    total_ms = msg.get("total_ms")
                    if isinstance(total_ms, int) and total_ms > 0:
                        parts.append(f"{total_ms}ms")
                    ui.set_usage_footer(" · ".join(parts))
                    # Session recorder (Phase 5.4)
                    from src.session_recorder import recorder as _rec
                    _rec.log_llm("assistant", "", tokens_in=msg.get("in", 0),
                                 tokens_out=msg.get("out", 0), cost_usd=c,
                                 model=msg.get("model", ""))
                except Exception:
                    pass
            elif msg["type"] == "latency":
                # Vision pipeline emits latency separately (no token-usage event).
                try:
                    total_ms = int(msg.get("total_ms") or 0)
                    if total_ms > 0:
                        ui.set_usage_footer(f"{total_ms}ms")
                except Exception:
                    pass
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"UI Updater Error: {e}")


async def run_pipelines(app, loop):
    """Main async task: wires all modules together inside the qasync loop."""
    # Import Qt-dependent modules only after Qt is confirmed importable.
    from src.settings_ui import apply_saved_config
    from src.overlay_ui import OverlayUI
    from src.area_capture import AreaCapture
    apply_saved_config()

    # ── Initialize modules ────────────────────────────────────────────────
    ui_asr = None
    ui_vision = None

    audio_capture = make_audio_capture(sample_rate=config.SAMPLE_RATE, chunk_size=config.CHUNK_SIZE) if AUDIO_AVAILABLE else None
    if not AUDIO_AVAILABLE:
        logger.warning(
            "Audio capture unavailable on this platform (%s). "
            "ASR/audio pipeline is disabled; UI and Vision still work. Reason: %s",
            sys.platform, _AUDIO_IMPORT_ERROR,
        )
    asr_client = _make_asr_client()
    rag_manager = RAGManager()
    # ── Load local knowledge base into RAG (resume / cheatsheets / notes) ──
    def _kb_params() -> dict:
        patterns = [
            p.strip()
            for p in getattr(config, "KNOWLEDGE_PATTERNS", "*.md,*.txt").split(",")
            if p.strip()
        ]
        return {
            "patterns": patterns,
            "chunk_chars": getattr(config, "KNOWLEDGE_CHUNK_CHARS", 900),
            "overlap_chars": getattr(config, "KNOWLEDGE_OVERLAP_CHARS", 120),
        }

    def rebuild_kb() -> int:
        try:
            n = rag_manager.rebuild_from_dir(
                getattr(config, "KNOWLEDGE_DIR", "knowledge"),
                **_kb_params(),
            )
            logger.info("KB rebuilt: %d chunks", n)
            return n
        except Exception as e:
            logger.warning(f"KB rebuild failed: {e}")
            return 0

    rebuild_kb()
    llm_engine = LLMEngine(rag_manager)
    # Preheat the LLM connection in the background so the first real request
    # doesn't pay the cold-start cost (Phase 2.4).
    try:
        loop.create_task(llm_engine.preheat())
    except Exception:
        pass

    # ── Queues ────────────────────────────────────────────────────────────
    audio_queue: asyncio.Queue = asyncio.Queue()
    text_queue: asyncio.Queue = asyncio.Queue()
    ui_queue_asr: asyncio.Queue = asyncio.Queue()
    ui_queue_vision: asyncio.Queue = asyncio.Queue()

    # ── Start audio → ASR pipeline ────────────────────────────────────────
    # Set event loop before start() so the pyaudio callback thread can call
    # loop.call_soon_threadsafe() safely.
    audio_capture.set_event_loop(loop)
    audio_capture.start(audio_queue, device_name_contains=getattr(config, "AUDIO_DEVICE_CONTAINS", ""))

    async def audio_watchdog():
        """Restart loopback capture if callbacks stall (device hotplug / driver hiccup)."""
        while True:
            try:
                await asyncio.sleep(getattr(config, "AUDIO_RECONNECT_SEC", 2.0))
                if audio_capture.last_callback_age_sec() > getattr(config, "AUDIO_WATCHDOG_SEC", 2.0):
                    logger.warning("Audio watchdog: no callbacks recently; restarting loopback capture.")
                    audio_capture.restart(
                        audio_queue,
                        device_name_contains=getattr(config, "AUDIO_DEVICE_CONTAINS", ""),
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Audio watchdog error: {e}")

    audio_watchdog_task = loop.create_task(audio_watchdog())

    asr_task = None
    ui_task_asr = None
    ui_task_vision = None

    hk_screenshot = None
    hk_interactions = []

    def _apply_overlay_opacity():
        try:
            if ui_asr is not None:
                ui_asr.setWindowOpacity(float(getattr(config, "OVERLAY_OPACITY", 1.0)))
            if ui_vision is not None:
                ui_vision.setWindowOpacity(float(getattr(config, "OVERLAY_OPACITY", 1.0)))
        except Exception as e:
            logger.warning(f"Failed to apply overlay opacity: {e}")

    def _restart_asr():
        nonlocal asr_task, asr_client
        try:
            asr_client.stop()
        except Exception:
            pass
        if asr_task is not None and not asr_task.done():
            asr_task.cancel()
        # Recreate client to pick up new keys/region/endpoint/language/backend
        try:
            asr_client = _make_asr_client()
        except Exception as e:
            logger.error(f"Failed to recreate ASR client: {e}")
            return
        asr_task = loop.create_task(asr_client.start_streaming(audio_queue, text_queue, loop))

    def on_settings_saved(_new_cfg: dict):
        """
        Called after Settings UI writes config.json and hot-patches `config`.
        Apply runtime changes without restart.
        """
        try:
            # Apply lightweight UI changes immediately
            _apply_overlay_opacity()
            if ui_asr is not None:
                ui_asr._max_conversation_blocks = getattr(
                    config, "ASR_OVERLAY_MAX_CONVERSATIONS", 3
                )
                ui_asr._retrim_and_render()

            # LLM clients may need recreation (provider/key/model changes)
            llm_engine.reload_clients()
            # Preheat after a key/model change.
            try:
                loop.create_task(llm_engine.preheat())
            except Exception:
                pass

            # Hotkeys are registered once → rebuild to apply new bindings
            _rebuild_hotkeys()

            # Refresh hotkey hints rendered in each overlay's footer
            for _ov in (ui_asr, ui_vision):
                if _ov is not None and hasattr(_ov, "refresh_footer"):
                    try:
                        _ov.refresh_footer()
                    except Exception:
                        pass

            # ASR language/keys changes require restarting the Azure transcriber
            _restart_asr()
        except Exception as e:
            logger.error(f"Runtime settings apply failed: {e}")

    # Create overlays after we have the callback
    ui_asr = OverlayUI(
        title="GhostPilot · ASR",
        with_tray=True,
        start_y=20,
        accent="ASR",
        on_settings_saved=on_settings_saved,
        max_conversation_blocks=getattr(config, "ASR_OVERLAY_MAX_CONVERSATIONS", 3),
        on_stop=lambda: llm_engine.cancel("text"),
        on_clear=llm_engine.clear_history,
        on_rebuild_kb=rebuild_kb,
    )
    ui_vision = OverlayUI(
        title="GhostPilot · Vision",
        with_tray=False,
        start_y=310,
        accent="Vision",
        on_settings_saved=on_settings_saved,
        on_stop=lambda: llm_engine.cancel("vision"),
    )
    # Start UI updaters and ASR now that UIs exist
    asr_task = loop.create_task(asr_client.start_streaming(audio_queue, text_queue, loop))
    ui_task_asr = loop.create_task(ui_updater(ui_asr, ui_queue_asr))
    ui_task_vision = loop.create_task(ui_updater(ui_vision, ui_queue_vision))

    # ── KB watcher: rebuild RAG when knowledge/ files change ──
    # Prefers QFileSystemWatcher (event-driven, ~zero latency) and falls back
    # to mtime polling if Qt watcher is unavailable. Set KB_WATCH_INTERVAL_SEC=0
    # to disable both.
    _kb_watcher_refs: list = []  # keep watcher + timer alive

    def _install_kb_watcher() -> bool:
        interval_sec = max(0, int(getattr(config, "KB_WATCH_INTERVAL_SEC", 5)))
        if interval_sec == 0:
            return True  # disabled by user
        try:
            from PyQt6.QtCore import QFileSystemWatcher, QTimer
            from src.knowledge_loader import list_knowledge_files
        except Exception as e:
            logger.warning(f"QFileSystemWatcher unavailable, will poll: {e}")
            return False

        kb_dir = getattr(config, "KNOWLEDGE_DIR", "knowledge")
        params = _kb_params()
        try:
            files = [str(p) for p in list_knowledge_files(kb_dir, patterns=params["patterns"])]
        except Exception as e:
            logger.warning(f"KB watcher: listing failed ({e}); falling back to poll")
            return False

        watcher = QFileSystemWatcher()
        # Watch the dir (catches new/deleted files) AND each file (catches edits).
        try:
            from pathlib import Path
            if Path(kb_dir).is_dir():
                watcher.addPath(str(Path(kb_dir).resolve()))
        except Exception:
            pass
        if files:
            watcher.addPaths(files)

        debounce_ms = max(200, min(2000, interval_sec * 200))  # 1s default
        debounce = QTimer()
        debounce.setSingleShot(True)
        debounce.setInterval(debounce_ms)

        def _on_change(_path: str = ""):
            debounce.start()

        def _on_debounce_timeout():
            try:
                n = rag_manager.rebuild_if_stale()
                if n is not None:
                    logger.info("KB auto-rebuilt (event): %d chunks", n)
                # Re-arm watcher on the new file set (catches added/renamed files)
                try:
                    fresh = [str(p) for p in list_knowledge_files(kb_dir, patterns=params["patterns"])]
                    current = set(watcher.files())
                    desired = set(fresh)
                    if desired - current:
                        watcher.addPaths(list(desired - current))
                    if current - desired:
                        watcher.removePaths(list(current - desired))
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"KB watcher rebuild failed: {e}")

        debounce.timeout.connect(_on_debounce_timeout)
        watcher.fileChanged.connect(_on_change)
        watcher.directoryChanged.connect(_on_change)

        _kb_watcher_refs.append(watcher)
        _kb_watcher_refs.append(debounce)
        logger.info(
            "KB watcher armed: dir=%s, %d files, debounce=%dms",
            kb_dir, len(files), debounce_ms,
        )
        return True

    async def _kb_poll_loop():
        """Fallback: mtime polling. Only used if QFileSystemWatcher install fails."""
        interval = max(0, int(getattr(config, "KB_WATCH_INTERVAL_SEC", 5)))
        if interval == 0:
            return
        while True:
            try:
                await asyncio.sleep(interval)
                n = rag_manager.rebuild_if_stale()
                if n is not None:
                    logger.info("KB auto-rebuilt (poll): %d chunks", n)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"KB poll loop error: {e}")
                await asyncio.sleep(interval)

    if not _install_kb_watcher():
        kb_watch_task = loop.create_task(_kb_poll_loop())

    # ── Hotkey: Alt+P — area-select screenshot → Vision LLM ──────────────
    async def on_screenshot():
        logger.info("Alt+P triggered — launching area capture overlay.")
        # Make the overlay show activity immediately.
        try:
            ui_vision.show_thinking("vision")
            await ui_queue_vision.put({"type": "clear"})
            await ui_queue_vision.put({"type": "token", "text": "📸 识图中…\n"})
        except Exception:
            pass
        try:
            image_bytes = await AreaCapture.capture_async()
            if image_bytes is None:
                logger.info("Screenshot capture cancelled by user.")
                return
            ui_vision.set_streaming(True)
            try:
                await asyncio.wait_for(
                    llm_engine.generate_vision_answer_stream(image_bytes, ui_queue_vision),
                    timeout=getattr(config, "VISION_TIMEOUT_SEC", 8.0),
                )
            finally:
                ui_vision.set_streaming(False)
        except asyncio.CancelledError:
            logger.info("Vision stream cancelled by user.")
        except Exception as e:
            logger.error(f"Screenshot pipeline error: {e}")
            await ui_queue_vision.put({"type": "token", "text": f"\n[⚠️ Vision Error: {e}]"})

    # ── Hotkey: Alt+A — toggle click-through / interactive mode ──────────
    def on_toggle_vision_interaction():
        logger.info("Vision interaction hotkey triggered.")
        ui_vision.toggle_interaction()

    def on_toggle_both_overlays_interaction():
        """
        ASR_INTERACTION_HOTKEY / INTERACTION_HOTKEY: move both overlays in sync.
        Uses set_interaction so one key always yields one net state (avoids
        double-toggle bugs when the same combo was registered twice).
        """
        all_interactive = ui_asr.is_interactive and ui_vision.is_interactive
        new_state = not all_interactive
        logger.info(
            "Both overlays → %s (draggable / click-through).",
            "interactive" if new_state else "click-through",
        )
        ui_asr.set_interaction(new_state)
        ui_vision.set_interaction(new_state)

    def on_force_stealth():
        logger.info("Force stealth hotkey triggered (click-through for both overlays).")
        ui_asr.set_interaction(False)
        ui_vision.set_interaction(False)

    def _start_hotkey_pair(
        primary: str,
        backup: str,
        cb,
        *,
        label: str,
        registered: set[str],
        trigger_on_release: bool = False,
    ):
        primary = (primary or "").strip()
        backup = (backup or "").strip()
        hks = []

        def _add_combo(combo: str) -> None:
            if not combo:
                return
            key = combo.lower()
            if key in registered:
                logger.warning(
                    "Skipping duplicate hotkey %r (%s) — already bound; "
                    "leave INTERACTION_HOTKEY empty if it matches ASR_INTERACTION_HOTKEY.",
                    combo,
                    label,
                )
                return
            registered.add(key)
            hks.append(
                HotkeyManager(
                    combo,
                    cb,
                    loop,
                    suppress=False,
                    trigger_on_release=trigger_on_release,
                )
            )

        _add_combo(primary)
        if backup.lower() != primary.lower():
            _add_combo(backup)
        for hk in hks:
            hk.start()
        if not hks and (primary or backup):
            logger.warning(f"No hotkey registered for {label} (empty or duplicate combos).")
        elif not hks:
            logger.warning(f"No hotkey configured for {label}")
        return hks

    def _rebuild_hotkeys():
        nonlocal hk_screenshot, hk_interactions
        try:
            if hk_screenshot is not None:
                hk_screenshot.stop()
        except Exception:
            pass
        try:
            for hk in hk_interactions:
                hk.stop()
        except Exception:
            pass
        hk_interactions = []
        registered_hotkeys: set[str] = set()

        logger.info(
            "Hotkeys effective: "
            f"SCREENSHOT_HOTKEY={config.SCREENSHOT_HOTKEY!r}, "
            f"ASR_INTERACTION_HOTKEY={getattr(config, 'ASR_INTERACTION_HOTKEY', '')!r} (both overlays), "
            f"VISION_INTERACTION_HOTKEY={getattr(config, 'VISION_INTERACTION_HOTKEY', '')!r}, "
            f"INTERACTION_HOTKEY(all)={getattr(config, 'INTERACTION_HOTKEY', '')!r}"
        )

        hk_screenshot = HotkeyManager(config.SCREENSHOT_HOTKEY, on_screenshot, loop, suppress=False)
        hk_screenshot.start()
        _ss = (config.SCREENSHOT_HOTKEY or "").strip().lower()
        if _ss:
            registered_hotkeys.add(_ss)

        # ASR_INTERACTION_* defaults (alt+a): toggles both overlays together (draggable ↔ click-through).
        hk_interactions += _start_hotkey_pair(
            getattr(config, "ASR_INTERACTION_HOTKEY", ""),
            getattr(config, "ASR_INTERACTION_HOTKEY_BACKUP", ""),
            on_toggle_both_overlays_interaction,
            label="Both overlays (ASR_INTERACTION_HOTKEY)",
            registered=registered_hotkeys,
        )
        hk_interactions += _start_hotkey_pair(
            getattr(config, "VISION_INTERACTION_HOTKEY", ""),
            getattr(config, "VISION_INTERACTION_HOTKEY_BACKUP", ""),
            on_toggle_vision_interaction,
            label="Vision interaction",
            registered=registered_hotkeys,
        )

        # Optional second binding same as ASR_INTERACTION_* (leave empty to avoid duplicate).
        hk_interactions += _start_hotkey_pair(
            getattr(config, "INTERACTION_HOTKEY", ""),
            getattr(config, "INTERACTION_HOTKEY_BACKUP", ""),
            on_toggle_both_overlays_interaction,
            label="Global interaction (INTERACTION_HOTKEY)",
            registered=registered_hotkeys,
        )

        hk_interactions += _start_hotkey_pair(
            getattr(config, "FORCE_STEALTH_HOTKEY", ""),
            getattr(config, "FORCE_STEALTH_HOTKEY_BACKUP", ""),
            on_force_stealth,
            label="Force stealth",
            registered=registered_hotkeys,
        )

    # Initial hotkeys registration (also supports runtime reloads)
    _rebuild_hotkeys()

    # ── ASR final transcript → Classifier → LLM ──────────────────────────
    # ASR segmentation: finalize on punctuation or partial-silence timeout.
    pending_partial: dict | None = None
    partial_timer: asyncio.Task | None = None

    def _cancel_partial_timer():
        nonlocal partial_timer
        if partial_timer and not partial_timer.done():
            partial_timer.cancel()
        partial_timer = None

    async def _finalize_from_partial():
        nonlocal pending_partial
        if not pending_partial:
            return
        msg = pending_partial
        pending_partial = None
        try:
            speaker = msg.get("speaker", "Unknown")
            question = msg["text"]
            logger.info(f"ASR Finalize (timeout) [{speaker}]: {question}")
            ui_asr.set_status("")
            q_type = await llm_engine.classify_question_llm(question)
            ui_asr.show_thinking(q_type)
            ui_asr.append_block(f"[{speaker}] {question}\nA: ")
            ui_asr.set_streaming(True)
            try:
                await llm_engine.generate_answer_stream(
                    question, ui_queue_asr, q_type=q_type
                )
            finally:
                ui_asr.set_streaming(False)
        except asyncio.CancelledError:
            ui_asr.set_streaming(False)
            logger.info("Text stream cancelled by user (finalize-from-partial).")
        except Exception as e:
            logger.error(f"ASR finalize error: {e}")

    async def asr_router():
        nonlocal pending_partial, partial_timer
        while True:
            try:
                msg = await text_queue.get()

                if msg["type"] == "final":
                    speaker = msg.get("speaker", "Unknown")
                    question = msg["text"]
                    logger.info(f"ASR Final [{speaker}]: {question}")
                    from src.session_recorder import recorder as _rec
                    _rec.log_transcript("final", f"[{speaker}] {question}")
                    pending_partial = None
                    _cancel_partial_timer()
                    q_type = await llm_engine.classify_question_llm(question)
                    ui_asr.set_status("")
                    ui_asr.show_thinking(q_type)
                    ui_asr.append_block(f"[{speaker}] {question}\nA: ")
                    ui_asr.set_streaming(True)
                    try:
                        await llm_engine.generate_answer_stream(
                            question, ui_queue_asr, q_type=q_type
                        )
                    except asyncio.CancelledError:
                        logger.info("Text stream cancelled by user.")
                    finally:
                        ui_asr.set_streaming(False)

                elif msg["type"] == "partial":
                    speaker = msg.get("speaker", "Unknown")
                    pending_partial = msg
                    ui_asr.set_status(f"[{speaker}] {msg['text']}")
                    _cancel_partial_timer()

                    # finalize on punctuation (quick heuristic)
                    if getattr(config, "ASR_PUNCTUATION_FINALIZE", True) and msg["text"].rstrip().endswith(("?", "？", "。", "！", "!")):
                        await _finalize_from_partial()
                        continue

                    async def _timer():
                        await asyncio.sleep(getattr(config, "ASR_PARTIAL_SILENCE_MS", 650) / 1000.0)
                        await _finalize_from_partial()

                    partial_timer = loop.create_task(_timer())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ASR Router Error: {e}")

    router_task = loop.create_task(asr_router())

    logger.info(
        "GhostPilot is running.\n"
        "  Alt+P  → Area screenshot → Vision LLM\n"
        "  Alt+A  → Toggle interaction / stealth mode\n"
        "  Right-click tray icon → Settings / Quit"
    )

    try:
        # Suspend here; the qasync loop keeps Qt + asyncio alive
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down pipelines...")
        if audio_capture is not None:
            audio_capture.stop()
        asr_client.stop()
        if hk_screenshot:
            hk_screenshot.stop()
        for hk in hk_interactions:
            hk.stop()
        asr_task.cancel()
        ui_task_asr.cancel()
        ui_task_vision.cancel()
        router_task.cancel()
        audio_watchdog_task.cancel()


def main():
    set_dpi_awareness()

    log_path = crash_logger.install()
    if log_path:
        logger.info(f"Crash log: {log_path}")

    try:
        from PyQt6.QtWidgets import QApplication
        from qasync import QEventLoop
    except Exception as e:
        logger.error(
            "PyQt6 failed to import; UI cannot start. "
            "This is usually a broken PyQt6 install or missing VC++ runtime.\n"
            f"Import error: {e}"
        )
        raise SystemExit(1)

    app = QApplication(sys.argv)
    # Keep the process alive when all visible windows are closed (tray-only mode)
    app.setQuitOnLastWindowClosed(False)

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        # Surface unexpected asyncio/qasync errors instead of silently stopping.
        _crash_log = logging.getLogger("crash")

        def _loop_exc_handler(_loop, context):
            msg = context.get("message", "Asyncio exception")
            exc = context.get("exception")
            if exc:
                _crash_log.error(f"{msg}: {exc}", exc_info=exc)
            else:
                _crash_log.error(msg)

        loop.set_exception_handler(_loop_exc_handler)

        def _on_about_to_quit():
            logger.warning("QApplication aboutToQuit fired (UI requested exit).")

        app.aboutToQuit.connect(_on_about_to_quit)

        pipelines_task = loop.create_task(run_pipelines(app, loop))

        def _pipelines_done(t: asyncio.Task):
            try:
                _ = t.result()
                logger.warning("run_pipelines finished unexpectedly (no error).")
            except asyncio.CancelledError:
                logger.warning("run_pipelines was cancelled (loop stopping).")
            except Exception as e:
                logger.error(f"run_pipelines crashed: {e}")

        pipelines_task.add_done_callback(_pipelines_done)
        loop.run_forever()

    logger.info("Application exited cleanly.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
