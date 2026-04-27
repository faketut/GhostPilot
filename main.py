import sys
import logging
import asyncio
import os

from src.windows_api import set_dpi_awareness
from src.audio_capture import AudioCapture
from src.asr_client import ASRClient
from src.rag_manager import RAGManager
from src.llm_engine import LLMEngine, classify_question
from src.hotkey_manager import HotkeyManager
from src.config import config

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
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"UI Updater Error: {e}")


async def run_pipelines(app, loop):
    """Main async task: wires all modules together inside the qasync loop."""
    set_dpi_awareness()
    # Import Qt-dependent modules only after Qt is confirmed importable.
    from src.settings_ui import apply_saved_config
    from src.overlay_ui import OverlayUI
    from src.area_capture import AreaCapture
    apply_saved_config()

    # ── Initialize modules ────────────────────────────────────────────────
    ui_asr = OverlayUI(title="GhostPilot · ASR", with_tray=True, start_y=20, accent="🎙️ ASR")
    ui_vision = OverlayUI(title="GhostPilot · Vision", with_tray=False, start_y=310, accent="📸 Vision")

    audio_capture = AudioCapture(sample_rate=config.SAMPLE_RATE, chunk_size=config.CHUNK_SIZE)
    asr_client = ASRClient(
        config.AZURE_SPEECH_KEY,
        config.AZURE_SPEECH_REGION,
        config.AZURE_SPEECH_ENDPOINT,
    )
    rag_manager = RAGManager()
    # ── Load local knowledge base into RAG (resume / cheatsheets / notes) ──
    try:
        from src.knowledge_loader import load_knowledge_dir

        patterns = [
            p.strip()
            for p in getattr(config, "KNOWLEDGE_PATTERNS", "*.md,*.txt").split(",")
            if p.strip()
        ]
        docs = load_knowledge_dir(
            getattr(config, "KNOWLEDGE_DIR", "knowledge"),
            patterns=patterns,
            chunk_chars=getattr(config, "KNOWLEDGE_CHUNK_CHARS", 900),
            overlap_chars=getattr(config, "KNOWLEDGE_OVERLAP_CHARS", 120),
        )
        if docs:
            rag_manager.load_documents(docs)
    except Exception as e:
        logger.warning(f"Failed to load local knowledge base (RAG): {e}")
    llm_engine = LLMEngine(rag_manager)

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

    asr_task = loop.create_task(asr_client.start_streaming(audio_queue, text_queue, loop))
    ui_task_asr = loop.create_task(ui_updater(ui_asr, ui_queue_asr))
    ui_task_vision = loop.create_task(ui_updater(ui_vision, ui_queue_vision))

    # ── Hotkey: Alt+P — area-select screenshot → Vision LLM ──────────────
    logger.info(
        "Hotkeys effective: "
        f"SCREENSHOT_HOTKEY={config.SCREENSHOT_HOTKEY!r}, "
        f"ASR_INTERACTION_HOTKEY={getattr(config, 'ASR_INTERACTION_HOTKEY', '')!r}, "
        f"VISION_INTERACTION_HOTKEY={getattr(config, 'VISION_INTERACTION_HOTKEY', '')!r}, "
        f"INTERACTION_HOTKEY(all)={getattr(config, 'INTERACTION_HOTKEY', '')!r}"
    )
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
            await asyncio.wait_for(
                llm_engine.generate_vision_answer_stream(image_bytes, ui_queue_vision),
                timeout=getattr(config, "VISION_TIMEOUT_SEC", 8.0),
            )
        except Exception as e:
            logger.error(f"Screenshot pipeline error: {e}")
            await ui_queue_vision.put({"type": "token", "text": f"\n[⚠️ Vision Error: {e}]"})

    hk_screenshot = HotkeyManager(config.SCREENSHOT_HOTKEY, on_screenshot, loop, suppress=False)
    hk_screenshot.start()

    # ── Hotkey: Alt+A — toggle click-through / interactive mode ──────────
    def on_toggle_asr_interaction():
        logger.info("ASR interaction hotkey triggered.")
        ui_asr.toggle_interaction()

    def on_toggle_vision_interaction():
        logger.info("Vision interaction hotkey triggered.")
        ui_vision.toggle_interaction()

    def on_toggle_all_interaction():
        logger.info("Global interaction hotkey triggered (toggle both overlays).")
        ui_asr.toggle_interaction()
        ui_vision.toggle_interaction()

    def on_force_stealth():
        logger.info("Force stealth hotkey triggered (click-through for both overlays).")
        ui_asr.set_interaction(False)
        ui_vision.set_interaction(False)

    def _start_hotkey_pair(primary: str, backup: str, cb, *, label: str):
        primary = (primary or "").strip()
        backup = (backup or "").strip()
        hks = []
        if primary:
            hks.append(
                HotkeyManager(primary, cb, loop, suppress=False, trigger_on_release=True)
            )
        if backup and backup.lower() != primary.lower():
            hks.append(
                HotkeyManager(backup, cb, loop, suppress=False, trigger_on_release=True)
            )
        for hk in hks:
            hk.start()
        if not hks:
            logger.warning(f"No hotkey configured for {label}")
        return hks

    hk_interactions = []

    # Preferred: separate per-overlay hotkeys
    hk_interactions += _start_hotkey_pair(
        getattr(config, "ASR_INTERACTION_HOTKEY", ""),
        getattr(config, "ASR_INTERACTION_HOTKEY_BACKUP", ""),
        on_toggle_asr_interaction,
        label="ASR interaction",
    )
    hk_interactions += _start_hotkey_pair(
        getattr(config, "VISION_INTERACTION_HOTKEY", ""),
        getattr(config, "VISION_INTERACTION_HOTKEY_BACKUP", ""),
        on_toggle_vision_interaction,
        label="Vision interaction",
    )

    # Backward compatible: one hotkey toggles both overlays (if set)
    hk_interactions += _start_hotkey_pair(
        getattr(config, "INTERACTION_HOTKEY", ""),
        getattr(config, "INTERACTION_HOTKEY_BACKUP", ""),
        on_toggle_all_interaction,
        label="Global interaction",
    )

    # Force stealth (recommended safety hotkey)
    hk_interactions += _start_hotkey_pair(
        getattr(config, "FORCE_STEALTH_HOTKEY", ""),
        getattr(config, "FORCE_STEALTH_HOTKEY_BACKUP", ""),
        on_force_stealth,
        label="Force stealth",
    )

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
            q_type = classify_question(question)
            ui_asr.show_thinking(q_type)
            ui_asr.append_block(f"[{speaker}] {question}\nA: ")
            await llm_engine.generate_answer_stream(question, ui_queue_asr)
        except Exception as e:
            logger.error(f"ASR finalize error: {e}")

    async def asr_router():
        while True:
            try:
                msg = await text_queue.get()

                if msg["type"] == "final":
                    speaker = msg.get("speaker", "Unknown")
                    question = msg["text"]
                    logger.info(f"ASR Final [{speaker}]: {question}")
                    pending_partial = None
                    _cancel_partial_timer()
                    # Classify immediately so the badge lights up before the first token
                    q_type = classify_question(question)
                    ui_asr.set_status("")
                    ui_asr.show_thinking(q_type)
                    ui_asr.append_block(f"[{speaker}] {question}\nA: ")
                    await llm_engine.generate_answer_stream(question, ui_queue_asr)

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
        audio_capture.stop()
        asr_client.stop()
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
        def _loop_exc_handler(_loop, context):
            msg = context.get("message", "Asyncio exception")
            exc = context.get("exception")
            if exc:
                logger.error(f"{msg}: {exc}")
            else:
                logger.error(msg)

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
