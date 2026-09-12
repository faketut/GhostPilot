# ruff: noqa: E402 — the bootstrap block below must execute before the src.*
# imports: src.config resolves .env / config.json relative to the working
# directory, which is not the app root for a login (autostart) launch.
import sys
import logging
import asyncio
import os
from pathlib import Path

# ── Bootstrap ──────────────────────────────────────────────────────────────
# GhostPilot is meant to run as a tray-resident background app: launched at
# login (or detached from a shell) it must resolve config.json, .env,
# knowledge/ and prompts/ relative to the install, not to whatever directory
# Windows happened to start it in (C:\Windows\system32 for a Run-key entry).
_APP_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
try:
    os.chdir(_APP_ROOT)
except OSError:
    pass
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))


def _process_name(pid: int) -> str:
    """Image file name for ``pid`` (lower-cased), or "" if it cannot be read."""
    import ctypes
    from ctypes import wintypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wintypes.DWORD(len(buf))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return ""
        return buf.value.rsplit("\\", 1)[-1].lower()
    finally:
        kernel32.CloseHandle(handle)


def _owns_console_window() -> bool:
    """True when this process owns a console window of its own.

    ``python.exe`` opens a console *window* on double-click, whereas an
    interpreter started from a terminal (``cmd``/PowerShell ``python main.py``,
    with or without ``>log``) shares that terminal's console, and a
    ``pythonw.exe`` process, a frozen ``--noconsole`` build and a service-style
    run have no console at all.

    Distinguishing the first case from the second: look at who else is attached
    to the console. Only our own interpreter stack (the venv ``python.exe``
    redirector plus the base interpreter) means we created it; any foreign
    process — most importantly a shell — means it is the user's terminal and
    killing it would close their window.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        if not kernel32.GetConsoleWindow():
            return False
        attached = (ctypes.c_uint * 16)()
        count = kernel32.GetConsoleProcessList(attached, 16)
        if count == 0:
            return False
        me = os.getpid()
        others = [p for p in attached[:count] if p and p != me]
        return all(_process_name(p) in ("python.exe", "pythonw.exe") for p in others)
    except Exception:
        return False


def _relaunch_console_free() -> bool:
    """Detach from an owned console by re-spawning under pythonw.exe.

    Running ``python main.py`` leaves a console window in the taskbar; a
    background service has none. Returns True when the parent should exit.
    Disabled by ``GHOSTPILOT_NO_RELAUNCH=1`` and in frozen builds (already
    windowed).
    """
    if not _owns_console_window():
        return False
    if os.environ.get("GHOSTPILOT_NO_RELAUNCH") == "1":
        return False
    pythonw = Path(sys.executable).resolve().with_name("pythonw.exe")
    if not pythonw.exists():
        return False
    try:
        import subprocess
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        subprocess.Popen(
            [str(pythonw), str(_APP_ROOT / "main.py"), *sys.argv[1:]],
            cwd=str(_APP_ROOT),
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(f"Console-free relaunch failed: {e}")
        return False
    return True


from src.windows_api import set_dpi_awareness
from src.asr_client import ASRClient
from src.rag_manager import RAGManager
from src.llm_engine import LLMEngine
from src.hotkey_manager import HotkeyManager
from src.config import config
from src import crash_logger
from src import startup_mode
from src import ollama_boot


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

# Windowed/background runs (pythonw, PyInstaller --noconsole) have no stderr:
# install no console handler, keep INFO flowing to the rotating crash log.
if sys.stderr is None:
    logging.getLogger().setLevel(logging.INFO)
else:
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
    # Per-task running buffer of the in-progress answer (Phase 5.4+).
    # Reset on 'clear'; flushed to the session recorder on the terminal
    # 'usage' event so the recorded answer matches the visible one.
    answer_buf: list[str] = []
    while True:
        try:
            msg = await ui_queue.get()
            if msg["type"] == "token":
                ui.update_text(msg["text"], append=True)
                answer_buf.append(msg["text"])
            elif msg["type"] == "clear":
                ui.update_text("", append=False)
                answer_buf.clear()
            elif msg["type"] == "status":
                # Transient progress (e.g. local OCR recognizing the screenshot).
                ui.set_status(msg.get("text", ""))
            elif msg["type"] == "answer_start":
                # OCR produced the question for the screenshot pipeline: show it
                # as a Q block, then stream the answer into that same block.
                speaker = "OCR"
                ui.append_block(f"[{speaker}] {msg.get('question', '')}\nA: ")
                ui.show_thinking(msg.get("q_type", ""))
                ui.set_status("")
                answer_buf.clear()
            elif msg["type"] == "info":
                # Observability strip (provider · rag:N · algo:Nc)
                try:
                    ui.set_info_footer(
                        provider=msg.get("provider", ""),
                        rag_hits=msg.get("rag_hits"),
                        algo_chars=msg.get("algo_chars", 0),
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
                    _rec.log_llm("assistant", "".join(answer_buf),
                                 tokens_in=msg.get("in", 0),
                                 tokens_out=msg.get("out", 0), cost_usd=c,
                                 model=msg.get("model", ""))
                    answer_buf.clear()
                except Exception:
                    pass
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"UI Updater Error: {e}")


async def run_pipelines(app, loop, overlay_mode: str):
    """Main async task: wires all modules together inside the qasync loop.

    ``overlay_mode`` (see :mod:`src.startup_mode`) decides which overlays exist
    for this launch: ``both`` / ``vision`` / ``asr``. In ``vision`` mode the
    audio capture device, the ASR client and the transcript router are never
    started — the screenshot/OCR pipeline is fully independent.
    """
    # Import Qt-dependent modules only after Qt is confirmed importable.
    # (`apply_saved_config` already ran in main(), before mode resolution.)
    from src.overlay_ui import OverlayUI
    from src.area_capture import AreaCapture

    enable_asr, enable_vision = startup_mode.pipeline_flags(overlay_mode)
    logger.info("Launching pipelines — overlay mode=%s (asr=%s, vision=%s)",
                overlay_mode, enable_asr, enable_vision)

    # ── Initialize modules ────────────────────────────────────────────────
    ui_asr = None
    ui_vision = None

    audio_capture = (
        make_audio_capture(sample_rate=config.SAMPLE_RATE, chunk_size=config.CHUNK_SIZE)
        if (AUDIO_AVAILABLE and enable_asr)
        else None
    )
    if enable_asr and not AUDIO_AVAILABLE:
        logger.warning(
            "Audio capture unavailable on this platform (%s). "
            "ASR/audio pipeline is disabled; UI and Vision still work. Reason: %s",
            sys.platform, _AUDIO_IMPORT_ERROR,
        )
    elif not enable_asr:
        logger.info("Vision-only launch: audio capture and ASR service are disabled.")
    asr_client = _make_asr_client() if enable_asr else None
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
    if audio_capture is not None:
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

    audio_watchdog_task = loop.create_task(audio_watchdog()) if audio_capture is not None else None

    asr_task = None
    ui_task_asr = None
    ui_task_vision = None
    router_task = None

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
        if asr_client is None:
            # Vision-only launch: there is no ASR client to restart.
            return
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
    def _start_replay(path, *, prompt_overrides=None, on_progress=None,
                      cancel_event=None):
        """Kick off a non-blocking replay of `path` through the live engine.

        Re-runs every recorded turn while a fresh SessionRecorder captures the
        results, so the user can compare old vs new by opening the new
        recording in the Sessions tab.

        Args:
            prompt_overrides: ``{q_type: text}`` forwarded to each
                ``replay_turn`` call so the user can A/B test alternate prompts
                without writing them to disk.
            on_progress: optional ``callable(done, total, status_text)`` invoked
                from the asyncio loop between turns. Qasync shares the Qt event
                loop so the callback may safely touch widgets.
            cancel_event: optional ``asyncio.Event``; checked between turns so
                a long replay can be stopped mid-way (the in-flight turn still
                runs to completion).
        """
        from src import session_replay
        from src.session_recorder import recorder as _rec

        async def _run():
            sdir = session_replay.open_session(path)
            turns = list(session_replay.iter_turns(sdir))
            total = len(turns)
            if total == 0:
                if on_progress:
                    try: on_progress(0, 0, "Empty recording — nothing to replay.")
                    except Exception: pass
                return
            _rec.start()
            try:
                for i, turn in enumerate(turns, 1):
                    if cancel_event is not None and cancel_event.is_set():
                        if on_progress:
                            try: on_progress(i - 1, total, "Cancelled.")
                            except Exception: pass
                        break
                    if on_progress:
                        try: on_progress(i - 1, total, f"Replaying turn {i}/{total}…")
                        except Exception: pass
                    try:
                        await session_replay.replay_turn(
                            llm_engine, turn, prompt_overrides=prompt_overrides,
                        )
                    except Exception as e:
                        logger.warning("Replay turn failed: %s", e)
                else:
                    if on_progress:
                        try: on_progress(total, total, f"Done — {total}/{total} turns replayed.")
                        except Exception: pass
            finally:
                out = _rec.stop()
                logger.info("Replay finished → %s", out)

        loop.create_task(_run())

    if enable_asr:
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
            on_replay_session=_start_replay,
        )
    if enable_vision:
        # The ASR overlay owns the tray when both are up; in vision-only mode
        # the Vision overlay takes over so tray menu / Settings stay reachable.
        ui_vision = OverlayUI(
            title="GhostPilot · Vision",
            with_tray=not enable_asr,
            start_y=310 if enable_asr else 20,
            accent="Vision",
            on_settings_saved=on_settings_saved,
            on_stop=lambda: llm_engine.cancel("vision"),
            on_rebuild_kb=rebuild_kb,
            on_replay_session=_start_replay,
        )

        async def _warm_ocr_backend():
            """Bring a local Ollama up, and load the OCR model, before the first
            screenshot needs either.

            Ollama is normally already running (its installer registers a
            Startup-folder shortcut), so this is usually an instant probe. When
            it is not — quit, crashed, autostart disabled, or we launched before
            it finished booting — starting it now beats discovering that on the
            hotkey; preloading likewise moves the model load off the first
            screenshot instead of onto it mid-interview.
            """
            try:
                if await ollama_boot.ensure_available_for_config():
                    await llm_engine.ocr.preload()
            except Exception as e:  # noqa: BLE001 — warming is best-effort
                logger.info("OCR backend warm-up skipped: %s", e)

        loop.create_task(_warm_ocr_backend())

    # Start UI updaters and ASR now that UIs exist
    if enable_asr and asr_client is not None:
        asr_task = loop.create_task(asr_client.start_streaming(audio_queue, text_queue, loop))
        if ui_asr is not None:
            ui_task_asr = loop.create_task(ui_updater(ui_asr, ui_queue_asr))
    if ui_vision is not None:
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
        loop.create_task(_kb_poll_loop())

    # ── Hotkey: Alt+P — area-select screenshot → local OCR → text LLM ────
    async def on_screenshot():
        if ui_vision is None:
            logger.warning("Screenshot hotkey pressed but this launch has no Vision overlay (mode=%s).", overlay_mode)
            return
        logger.info("Area screenshot hotkey triggered — launching area capture overlay.")
        # Make the overlay show activity immediately.
        try:
            ui_vision.show_thinking("vision")
            await ui_queue_vision.put({"type": "clear"})
            await ui_queue_vision.put({"type": "status", "text": "📸 识图中… (GLM-OCR)"})
        except Exception:
            pass
        try:
            image_bytes = await AreaCapture.capture_async()
            if image_bytes is None:
                logger.info("Screenshot capture cancelled by user.")
                return
            ui_vision.set_streaming(True)
            try:
                # OCR runs on CPU and the answer streams afterwards; only the
                # recognition pass is bounded by OCR_TIMEOUT_SEC.
                await llm_engine.generate_vision_answer_stream(image_bytes, ui_queue_vision)
            finally:
                ui_vision.set_streaming(False)
                ui_vision.set_status("")
        except asyncio.CancelledError:
            logger.info("Screenshot pipeline cancelled by user.")
        except Exception as e:
            logger.error(f"Screenshot pipeline error: {e}")
            await ui_queue_vision.put({"type": "token", "text": f"\n[⚠️ Error: {e}]"})

    # ── Hotkey: Alt+A — toggle click-through / interactive mode ──────────
    def on_toggle_vision_interaction():
        if ui_vision is None:
            return
        logger.info("Vision interaction hotkey triggered.")
        ui_vision.toggle_interaction()

    async def on_screenshot_full():
        if ui_vision is None:
            logger.warning("Full-screen hotkey pressed but this launch has no Vision overlay (mode=%s).", overlay_mode)
            return
        logger.info("Full-screen screenshot hotkey triggered — capturing primary monitor.")
        try:
            ui_vision.show_thinking("vision")
            await ui_queue_vision.put({"type": "clear"})
            await ui_queue_vision.put({"type": "status", "text": "📸 全屏截屏，识图中… (GLM-OCR)"})
        except Exception:
            pass
        try:
            # capture_full_screen is synchronous; run in thread to avoid blocking loop
            image_bytes = await asyncio.to_thread(AreaCapture.capture_full_screen)
            if image_bytes is None:
                logger.info("Full-screen capture returned no data.")
                return
            ui_vision.set_streaming(True)
            try:
                await llm_engine.generate_vision_answer_stream(image_bytes, ui_queue_vision)
            finally:
                ui_vision.set_streaming(False)
                ui_vision.set_status("")
        except asyncio.CancelledError:
            logger.info("Full-screen pipeline cancelled by user.")
        except Exception as e:
            logger.error(f"Full-screen screenshot pipeline error: {e}")
            await ui_queue_vision.put({"type": "token", "text": f"\n[⚠️ Error: {e}]"})

    def _overlays() -> list:
        return [o for o in (ui_asr, ui_vision) if o is not None]

    def on_toggle_both_overlays_interaction():
        """
        ASR_INTERACTION_HOTKEY / INTERACTION_HOTKEY: move every live overlay in
        sync. Uses set_interaction so one key always yields one net state (avoids
        double-toggle bugs when the same combo was registered twice). Degrades
        gracefully to a single overlay in vision-only / ASR-only mode.
        """
        overlays = _overlays()
        if not overlays:
            return
        new_state = not all(o.is_interactive for o in overlays)
        logger.info(
            "%s overlay(s) → %s (draggable / click-through).",
            len(overlays),
            "interactive" if new_state else "click-through",
        )
        for o in overlays:
            o.set_interaction(new_state)

    def on_force_stealth():
        logger.info("Force stealth hotkey triggered (click-through for live overlays).")
        for o in _overlays():
            o.set_interaction(False)

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

        # Screenshot hotkeys only make sense when the Vision overlay exists.
        if not enable_vision:
            logger.info("Screenshot hotkeys skipped (no Vision overlay in this launch).")

        # Area screenshot hotkey
        if enable_vision:
            hk_screenshot = HotkeyManager(config.SCREENSHOT_HOTKEY, on_screenshot, loop, suppress=False)
            hk_screenshot.start()
            _ss = (config.SCREENSHOT_HOTKEY or "").strip().lower()
            if _ss:
                registered_hotkeys.add(_ss)

        # Full-screen screenshot hotkey (separate binding)
        full_combo = (getattr(config, "SCREENSHOT_FULL_HOTKEY", "") or "").strip() if enable_vision else ""
        if full_combo:
            key = full_combo.lower()
            if key in registered_hotkeys:
                logger.warning("Skipping duplicate full-screen hotkey %r — already bound", full_combo)
            else:
                registered_hotkeys.add(key)
                hk_full = HotkeyManager(full_combo, on_screenshot_full, loop, suppress=False)
                hk_full.start()
                # Keep a reference so it can be stopped on shutdown
                hk_interactions.append(hk_full)

        # ASR_INTERACTION_* defaults (alt+a): toggles both overlays together (draggable ↔ click-through).
        hk_interactions += _start_hotkey_pair(
            getattr(config, "ASR_INTERACTION_HOTKEY", ""),
            getattr(config, "ASR_INTERACTION_HOTKEY_BACKUP", ""),
            on_toggle_both_overlays_interaction,
            label="Both overlays (ASR_INTERACTION_HOTKEY)",
            registered=registered_hotkeys,
        )
        # Bound to the Vision overlay only — nothing to toggle without it.
        if enable_vision:
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

    if enable_asr:
        router_task = loop.create_task(asr_router())

    lines = [f"GhostPilot is running (overlay mode={overlay_mode})."]
    if enable_vision:
        lines += [
            f"  {config.SCREENSHOT_HOTKEY}  → Area screenshot → GLM-OCR → text LLM",
            f"  {getattr(config, 'SCREENSHOT_FULL_HOTKEY', '') or '(unset)'}"
            "  → Full-screen screenshot → GLM-OCR → text LLM",
        ]
    if enable_asr:
        lines.append("  WASAPI loopback → ASR → text LLM")
    lines += [
        "  Right-click tray icon → Settings / Quit",
    ]
    logger.info("\n".join(lines))

    try:
        # Suspend here; the qasync loop keeps Qt + asyncio alive
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down pipelines...")
        if audio_capture is not None:
            audio_capture.stop()
        if asr_client is not None:
            asr_client.stop()
        if hk_screenshot:
            hk_screenshot.stop()
        for hk in hk_interactions:
            hk.stop()
        for task in (asr_task, ui_task_asr, ui_task_vision, router_task, audio_watchdog_task):
            if task is not None:
                task.cancel()


def main():
    # Tray-only app: drop the inherited console by re-spawning under pythonw.exe
    # (source runs only; frozen builds are already windowed).
    if _relaunch_console_free():
        logger.info("Relaunched detached under pythonw.exe; exiting the console process.")
        return

    set_dpi_awareness()

    log_path = crash_logger.install(
        level=logging.INFO if sys.stderr is None else logging.WARNING
    )
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
    # Dialog / window icon (Settings, area-capture overlay); the tray icon is
    # set separately in OverlayUI._init_tray.
    from src import icons as _icons
    _app_icon = _icons.app_icon()
    if not _app_icon.isNull():
        app.setWindowIcon(_app_icon)

    # config.json (written by Settings, and by the chooser's "remember" box)
    # must be applied *before* the launch mode is resolved — otherwise
    # STARTUP_OVERLAY_MODE saved there would be invisible and the chooser would
    # reappear on every launch.
    from src.settings_ui import apply_saved_config
    apply_saved_config()

    # Which overlay(s) to run this launch: --overlay > config > ask (chooser).
    overlay_mode = startup_mode.resolve_mode()
    if overlay_mode == startup_mode.MODE_ASK:
        overlay_mode = startup_mode.choose_overlay_mode()
        if overlay_mode is None:
            logger.info("Overlay chooser dismissed — exiting.")
            return
    logger.info("Overlay mode: %s", overlay_mode)

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

        pipelines_task = loop.create_task(run_pipelines(app, loop, overlay_mode))

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
