import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, *env_names: str, default: str = "") -> str:
    """Resolve a secret: keyring → env vars (in order) → default. Import is local
    to avoid a circular import at module load time."""
    try:
        from src import secret_store
        v = secret_store.get(name)
        if v:
            return v
    except Exception:
        pass
    for env in env_names or (name,):
        v = os.getenv(env, "")
        if v:
            return v
    return default


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name, str(default)) or str(default)
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name, str(default)) or str(default)
        return float(raw.strip())
    except ValueError:
        return default


@dataclass
class Config:
    # API Keys — resolved via keyring → env (in that order). Plain-text
    # config.json values are still picked up via the SettingsUI loader because
    # it writes os.environ on load.
    OPENAI_API_KEY = _secret("OPENAI_API_KEY")
    GEMINI_API_KEY = _secret("GEMINI_API_KEY")
    DEEPSEEK_API_KEY = _secret("DEEPSEEK_API_KEY")

    # Azure Speech Service
    AZURE_SPEECH_KEY = _secret("AZURE_SPEECH_KEY", "SPEECH_KEY", "AZURE_SPEECH_KEY")
    AZURE_SPEECH_REGION = os.getenv("SPEECH_REGION", os.getenv("AZURE_SPEECH_REGION", "eastus"))
    AZURE_SPEECH_ENDPOINT = os.getenv("ENDPOINT", os.getenv("AZURE_SPEECH_ENDPOINT", ""))
    # ASR recognition language. zh-CN for Chinese, en-US for English, etc.
    ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en-US")

    # ASR backend: "azure" (cloud, default) or "whisper" (local via faster-whisper).
    ASR_BACKEND = os.getenv("ASR_BACKEND", "azure").strip().lower()
    # faster-whisper knobs (only used when ASR_BACKEND=whisper).
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto | cpu | cuda
    WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    # How many seconds of audio to buffer before each whisper inference pass.
    WHISPER_WINDOW_SEC = _env_float("WHISPER_WINDOW_SEC", 2.5)

    # LLM Settings
    # Answer model. DeepSeek-V4.1-Flash ("deepseek-flash") is the current
    # DeepSeek API model (deepseek-chat was retired in July 2026).
    TEXT_MODEL = os.getenv("TEXT_MODEL", "deepseek-flash")

    # Optional explicit provider override ("openai", "deepseek", "gemini", "ollama").
    # When empty, the engine infers from the model name.
    TEXT_PROVIDER = os.getenv("TEXT_PROVIDER", "")
    # Optional comma-separated fallback chain — used if the primary provider
    # raises before emitting any output. Example: "deepseek,gemini".
    TEXT_PROVIDER_FALLBACK = os.getenv("TEXT_PROVIDER_FALLBACK", "")

    # Screenshot OCR runs locally through Ollama (native /api/generate), then
    # the recognized text is answered by the text model above.
    #   ollama pull glm-ocr && python setup_glm_ocr.py
    OCR_MODEL = os.getenv("OCR_MODEL", "glm-ocr-optimized")
    # The stock "Text recognition:" prompt makes GLM-OCR continue past the page
    # and repeat itself (measured: one line emitted 164×, 1984 chars, 59-94 s of
    # decode for text it had already transcribed). Asking for the text alone and
    # an explicit stop terminates cleanly on the same image in ~12 s with
    # identical accuracy. Kept configurable because a provider-prefix style
    # prompt is still needed if you switch to a model without a GLM renderer.
    OCR_PROMPT = os.getenv("OCR_PROMPT", "Transcribe all text in this image. Output only the text.")
    # Longest edge (px) the screenshot is scaled to before OCR: downscaled above
    # it (the vision prefill dominates the latency), upscaled below it (the model
    # reads small text poorly and falls into a repetition loop).
    OCR_MAX_DIMENSION = _env_int("OCR_MAX_DIMENSION", 1024)
    # A small drag-selected region sent at native size (or scaled only to 768) is
    # unreadable for GLM-OCR, which then loops until the token cap. Normalising
    # the long edge to the same 1024 keeps it legible; lower this to skip the
    # upscale and accept that small crops are slower.
    OCR_MIN_DIMENSION = _env_int("OCR_MIN_DIMENSION", 1024)
    # CPU OCR takes seconds to minutes; this bounds one recognition request.
    OCR_TIMEOUT_SEC = _env_float("OCR_TIMEOUT_SEC", 180.0)
    # Wall-clock budget for one recognition. OCR_TIMEOUT_SEC is the httpx *read*
    # timeout and never fires while tokens keep arriving, so a looping generation
    # would otherwise run to the token cap uninterrupted.
    OCR_TOTAL_TIMEOUT_SEC = _env_float("OCR_TOTAL_TIMEOUT_SEC", 120.0)
    # Ollama evicts an idle model after 5 minutes by default, and reloading the
    # 2.2 GB F16 GLM-OCR costs ~30 s on CPU — paid on the next screenshot.
    # keep_alive holds it resident across a normal interview's gap between shots.
    OCR_KEEP_ALIVE = os.getenv("OCR_KEEP_ALIVE", "30m")
    # Hard cap on tokens generated per screenshot. GLM-OCR is greedy
    # (temperature 0 / top_k 1): on an image it cannot read it latches onto the
    # last token group and repeats it until the cap. The model's own num_predict
    # is 8192 — ~13 measured minutes of CPU for one degenerate screenshot.
    OCR_NUM_PREDICT = _env_int("OCR_NUM_PREDICT", 1024)
    # Stop consuming a degenerate stream after this many identical trailing lines.
    OCR_REPEAT_GUARD_LINES = _env_int("OCR_REPEAT_GUARD_LINES", 12)
    # Append-only JSONL usage log. Empty path → ~/.ghostpilot/usage.jsonl.
    USAGE_LOG_ENABLED = (os.getenv("USAGE_LOG_ENABLED", "1").strip().lower()
                         not in {"0", "false", "no", ""})
    USAGE_LOG_PATH = os.getenv("USAGE_LOG_PATH", "")
    # Ollama (local OpenAI-compatible endpoint).
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    # Start a *local* Ollama at launch when it is not already answering. Ollama's
    # installer registers a Startup-folder shortcut, so this is normally just a
    # probe; it covers the cases that shortcut does not (quit, crashed, autostart
    # disabled, or we came up before it finished booting). Never applies to a
    # remote OLLAMA_BASE_URL. Set to 0 to manage the server yourself.
    OLLAMA_AUTOSTART = os.getenv("OLLAMA_AUTOSTART", "1")
    # Multi-turn context: number of prior (Q, A) pairs to feed back to the
    # text model. 0 disables history (default — keeps token usage tight).
    CONTEXT_TURNS = _env_int("CONTEXT_TURNS", 0)

    # Audio Settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = int(SAMPLE_RATE * 0.16)  # 160ms buffer
    # Audio device selection / resilience
    AUDIO_DEVICE_CONTAINS = os.getenv("AUDIO_DEVICE_CONTAINS", "")  # substring match
    AUDIO_RECONNECT_SEC = _env_float("AUDIO_RECONNECT_SEC", 2.0)
    AUDIO_WATCHDOG_SEC = _env_float("AUDIO_WATCHDOG_SEC", 2.0)  # no-callback threshold
    # Backend: "" (auto: win32→pyaudiowpatch, else→sounddevice), "pyaudiowpatch", or "sounddevice".
    # sounddevice captures the *microphone*, not system loopback (use BlackHole on macOS for loopback).
    AUDIO_BACKEND = os.getenv("AUDIO_BACKEND", "")

    # ASR segmentation
    ASR_PARTIAL_SILENCE_MS = _env_int("ASR_PARTIAL_SILENCE_MS", 650)
    ASR_PUNCTUATION_FINALIZE = os.getenv("ASR_PUNCTUATION_FINALIZE", "1") != "0"
    # ASR overlay: keep only the last N Q&A blocks (split by the same separator as append_block). 0 = unlimited.
    ASR_OVERLAY_MAX_CONVERSATIONS = _env_int("ASR_OVERLAY_MAX_CONVERSATIONS", 3)

    # Answer generation output cap. 0 (the default) means UNCAPPED: the provider
    # applies its own output limit.
    #
    # A caller-imposed cap is a trap with reasoning models. `deepseek-flash`
    # bills its hidden chain-of-thought against this budget *before* writing the
    # visible answer, and the trace length varies per run. Measured on one
    # algorithm question: at max_tokens=450 the entire budget went to reasoning
    # and the answer was empty; at 1200 the same thing happened (1200 reasoning
    # tokens, 0 characters of answer); uncapped it took 1376 tokens total
    # (1241 reasoning + the rest answer) and stopped cleanly with the complete
    # code block. Capping this call therefore cannot be made safe by picking a
    # bigger number — it only makes truncation less likely, at the cost of
    # silently losing the answer when the trace runs long.
    ANSWER_MAX_TOKENS = _env_int("ANSWER_MAX_TOKENS", 0)

    # UI Settings
    # 0.0~1.0, higher = more opaque (less transparent)
    OVERLAY_OPACITY = 0.78

    # Which overlay(s) to bring up at launch, and whether to ask first:
    #   "ask"    → show a chooser dialog on every launch (default)
    #   "both"   → ASR overlay + Vision overlay (audio/ASR service runs)
    #   "vision" → Vision overlay only (no audio capture, no ASR service)
    #   "asr"    → ASR overlay only (no screenshot/vision overlay)
    # Overridable per launch with `python main.py --overlay vision`.
    STARTUP_OVERLAY_MODE = os.getenv("STARTUP_OVERLAY_MODE", "ask").strip().lower()

    # Response language preference:
    # - "auto": follow question language (use each prompt's built-in rule)
    # - "zh": always Chinese
    # - "en": always English (default for interview coach output)
    RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", "en")

    # RAG: minimum cosine similarity (0–1) to keep a chunk; below threshold chunks are dropped
    RAG_MIN_SCORE = _env_float("RAG_MIN_SCORE", 0.32)

    # Algorithm turns are the one route that does *not* retrieve: an algorithm
    # question is answered from the patterns cheatsheet, injected whole. Whole
    # rather than retrieved because a cheatsheet's value is its structure — a
    # 900-char chunk of it ranks well but reads as a fragment — and because the
    # route must not depend on retrieval quality (the dense half needs
    # sentence-transformers; BM25 alone scores a resume chunk against a coding
    # question too). Name/path is resolved under KNOWLEDGE_DIR, or absolute.
    ALGORITHM_KNOWLEDGE_FILE = os.getenv("ALGORITHM_KNOWLEDGE_FILE", "algorithm.md")
    # Safety bound on the whole-file injection (≈2k tokens at the default).
    # Generous enough that a real cheatsheet is never cut; a larger file is
    # truncated with a visible marker rather than sent whole or dropped.
    ALGORITHM_KNOWLEDGE_MAX_CHARS = _env_int("ALGORITHM_KNOWLEDGE_MAX_CHARS", 8000)

    # LLM question classifier (text model, non-streaming)
    CLASSIFIER_MAX_TOKENS = _env_int("CLASSIFIER_MAX_TOKENS", 64)
    CLASSIFIER_TEMPERATURE = _env_float("CLASSIFIER_TEMPERATURE", 0.1)

    # Hotkeys
    SCREENSHOT_HOTKEY = os.getenv("SCREENSHOT_HOTKEY", "alt+p")
    # Full-screen screenshot (captures entire primary monitor)
    SCREENSHOT_FULL_HOTKEY = os.getenv("SCREENSHOT_FULL_HOTKEY", "alt+shift+p")
    # Backward compatible (toggles BOTH overlays together if set)
    INTERACTION_HOTKEY = os.getenv("INTERACTION_HOTKEY", "")
    INTERACTION_HOTKEY_BACKUP = os.getenv("INTERACTION_HOTKEY_BACKUP", "")

    # Preferred (separate hotkeys per overlay)
    ASR_INTERACTION_HOTKEY = os.getenv("ASR_INTERACTION_HOTKEY", "alt+a")
    ASR_INTERACTION_HOTKEY_BACKUP = os.getenv("ASR_INTERACTION_HOTKEY_BACKUP", "ctrl+alt+a")
    VISION_INTERACTION_HOTKEY = os.getenv("VISION_INTERACTION_HOTKEY", "alt+i")
    VISION_INTERACTION_HOTKEY_BACKUP = os.getenv("VISION_INTERACTION_HOTKEY_BACKUP", "ctrl+alt+i")

    # Force both overlays to stealth (click-through) regardless of current state
    FORCE_STEALTH_HOTKEY = os.getenv("FORCE_STEALTH_HOTKEY", "alt+s")
    FORCE_STEALTH_HOTKEY_BACKUP = os.getenv("FORCE_STEALTH_HOTKEY_BACKUP", "ctrl+alt+s")

    # Local knowledge base (RAG)
    KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge")
    KNOWLEDGE_PATTERNS = os.getenv("KNOWLEDGE_PATTERNS", "*.md,*.txt")
    KNOWLEDGE_CHUNK_CHARS = _env_int("KNOWLEDGE_CHUNK_CHARS", 900)
    KNOWLEDGE_OVERLAP_CHARS = _env_int("KNOWLEDGE_OVERLAP_CHARS", 120)
    # KB auto-rebuild watcher: poll knowledge files mtimes every N seconds (0 disables).
    KB_WATCH_INTERVAL_SEC = _env_int("KB_WATCH_INTERVAL_SEC", 5)

    # Per-overlay geometry persisted by the UI (list [x, y, w, h]). None = use defaults.
    OVERLAY_ASR_GEOMETRY = None
    OVERLAY_VISION_GEOMETRY = None


config = Config()
