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

    # ASR backend: "azure" (cloud, default), "whisper" (local via faster-whisper)
    # or "sherpa" (local via sherpa-onnx — streaming, model downloads on first use).
    ASR_BACKEND = os.getenv("ASR_BACKEND", "azure").strip().lower()
    # faster-whisper knobs (only used when ASR_BACKEND=whisper).
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto | cpu | cuda
    WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    # How many seconds of audio to buffer before each whisper inference pass.
    WHISPER_WINDOW_SEC = _env_float("WHISPER_WINDOW_SEC", 2.5)

    # sherpa-onnx knobs (only used when ASR_BACKEND=sherpa).
    # Fully local: no key, no per-minute billing, audio never leaves the machine.
    # SHERPA_MODEL is a registry name (src/sherpa_models.py — see
    # `python setup_sherpa_asr.py --list`) or a path to an already-extracted
    # model directory holding tokens.txt + *encoder*.onnx + *decoder*.onnx.
    SHERPA_MODEL = os.getenv("SHERPA_MODEL", "zipformer-bilingual-zh-en")
    # Where registry models are stored. Empty → ~/.ghostpilot/models/sherpa-onnx.
    SHERPA_MODEL_DIR = os.getenv("SHERPA_MODEL_DIR", "")
    # Fetch the model on first use when it is missing (one time: 74 MB – 1 GB).
    SHERPA_AUTODOWNLOAD = os.getenv("SHERPA_AUTODOWNLOAD", "1") != "0"
    # onnxruntime threads. 0 → min(4, cpu_count) — ASR shares the CPU with the
    # OCR model, so it is deliberately not "all cores".
    SHERPA_NUM_THREADS = _env_int("SHERPA_NUM_THREADS", 0)
    SHERPA_PROVIDER = os.getenv("SHERPA_PROVIDER", "cpu")  # cpu | cuda | coreml
    # Endpointing. RULE2 (end of utterance after speech) MUST stay below
    # ASR_PARTIAL_SILENCE_MS/1000, or main.py's partial timer answers the
    # utterance first and the model's final produces a second, duplicate answer.
    SHERPA_RULE1_SILENCE_SEC = _env_float("SHERPA_RULE1_SILENCE_SEC", 2.4)   # before any text
    SHERPA_RULE2_SILENCE_SEC = _env_float("SHERPA_RULE2_SILENCE_SEC", 0.5)   # after speech
    SHERPA_RULE3_UTTERANCE_SEC = _env_float("SHERPA_RULE3_UTTERANCE_SEC", 20.0)  # hard cap
    # "greedy_search" (cheapest) or "modified_beam_search" (better accuracy, and
    # required for hotwords). Empty → modified_beam_search when a hotwords file
    # is set, else greedy_search.
    SHERPA_DECODING_METHOD = os.getenv("SHERPA_DECODING_METHOD", "")
    # Vocabulary biasing (product names, acronyms). File format is BPE/CJKchar
    # tokens, one phrase per line — generate it with the model's own bpe.model:
    #   sherpa-onnx-cli text2token --tokens <dir>/tokens.txt --bpe-model <dir>/bpe.model \
    #       --texts "your phrase" out.txt
    SHERPA_HOTWORDS_FILE = os.getenv("SHERPA_HOTWORDS_FILE", "")
    SHERPA_HOTWORDS_SCORE = _env_float("SHERPA_HOTWORDS_SCORE", 1.5)
    SHERPA_HOTWORDS_MODELING_UNIT = os.getenv("SHERPA_HOTWORDS_MODELING_UNIT", "cjkchar+bpe")

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
    #   ollama pull glm-ocr && pip install gguf && python setup_glm_ocr.py
    OCR_MODEL = os.getenv("OCR_MODEL", "glm-ocr-optimized")
    # Recognition prompt sent with every screenshot. Measured head-to-head
    # against GLM-OCR's native task prompts ("Text Recognition:" and friends),
    # the wording does *not* decide whether generation terminates — that is the
    # end-of-generation defect described on OCR_NUM_PREDICT below, and it
    # reproduces with every prompt tried. The wording does select the output
    # *format*: the native prompts switch the model to table HTML / LaTeX, so
    # this states the text task explicitly.
    OCR_PROMPT = os.getenv("OCR_PROMPT", "Transcribe all text in this image. Output only the text.")
    # Longest edge (px) the screenshot is scaled to before OCR. Downscale above
    # it — the vision prefill dominates the latency and scales with the
    # *downscaled* pixel count (measured: 512px 5.1s, 768px 14.5s, 1024px 23.1s,
    # 1600px 98.6s for one page). Raising it is a bad trade: 1600px cost 3x the
    # prefill of 1024px for 20/27 vs 18/27 lines recovered, and a *region*
    # captured at native density read 24/27 at the 1024px price. Prefer a
    # tighter region over a bigger dimension; see OCR_MIN_DIMENSION.
    OCR_MAX_DIMENSION = _env_int("OCR_MAX_DIMENSION", 1024)
    # Floor for the long edge: 0 disables upscaling. Upscaling a small crop adds
    # no information, and it was measured to add no accuracy either — a 420px
    # crop read 13/14 lines sent as-is, at 512px, and at 1024px, while the
    # prefill went 5.6s -> 7.7s -> 37.7s (tokens scale with the square of the
    # long edge). Same at 560px: 14/14 either way, 7.0s vs 27.4s. The old
    # default of 1024 made every drag-selected region ~5x slower for nothing.
    OCR_MIN_DIMENSION = _env_int("OCR_MIN_DIMENSION", 0)
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
    # Backstop cap on tokens generated per screenshot.
    #
    # The failure this bounds has a root cause, and it is *not* the prompt and
    # not the cap: Ollama's `glm-ocr` GGUF ships without
    # `tokenizer.ggml.eot_token_id`, so `<|user|>` — the token the model emits
    # to end its turn — is never an end-of-generation token and generation
    # cannot stop. GLM-OCR therefore transcribes the page correctly and keeps
    # going, re-emitting it until a cap; on Ollama >= 0.34.1 the byte-identical
    # replay also trips llama.cpp's token-repeat guard, which is where the
    # "repeat token" / limit-exceeded errors come from. Measured on one
    # synthetic banner: 2000 tokens / 141 s unpatched -> 21 tokens / 7.3 s once
    # `<|user|>` is registered; a full code page: 2000 -> 334 tokens.
    #
    # `setup_glm_ocr.py` applies that repair, so with a normal install
    # generation stops on its own and this cap is never reached. It stays as a
    # backstop for a model built without the repair (no `gguf` package, or an
    # older model), where raising it buys more duplication rather than a better
    # transcript. `src/ocr_client.py`'s repetition guard covers the rest.
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
