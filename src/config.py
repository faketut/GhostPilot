import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


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
    # API Keys (Loaded from .env via os.getenv)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

    # Azure Speech Service
    AZURE_SPEECH_KEY = os.getenv("SPEECH_KEY", os.getenv("AZURE_SPEECH_KEY", ""))
    AZURE_SPEECH_REGION = os.getenv("SPEECH_REGION", os.getenv("AZURE_SPEECH_REGION", "eastus"))
    AZURE_SPEECH_ENDPOINT = os.getenv("ENDPOINT", os.getenv("AZURE_SPEECH_ENDPOINT", ""))
    # ASR recognition language. zh-CN for Chinese, en-US for English, etc.
    ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en-US")

    # LLM Settings
    # Text: DeepSeek-V3 (fast, cheap, great for technical Q&A)
    TEXT_MODEL = os.getenv("TEXT_MODEL", "deepseek-chat")  # deepseek-chat = DeepSeek-V3
    # Vision: Gemini 1.5 Flash by default (works with GEMINI_API_KEY out of the box).
    # Use "gpt-4o" if you prefer OpenAI vision (requires OPENAI_API_KEY).
    # NOTE: DeepSeek vision models are NOT compatible with this pipeline.
    VISION_MODEL = os.getenv("VISION_MODEL", "gemini-1.5-flash")

    # Audio Settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = int(SAMPLE_RATE * 0.16)  # 160ms buffer
    # Audio device selection / resilience
    AUDIO_DEVICE_CONTAINS = os.getenv("AUDIO_DEVICE_CONTAINS", "")  # substring match
    AUDIO_RECONNECT_SEC = _env_float("AUDIO_RECONNECT_SEC", 2.0)
    AUDIO_WATCHDOG_SEC = _env_float("AUDIO_WATCHDOG_SEC", 2.0)  # no-callback threshold

    # ASR segmentation
    ASR_PARTIAL_SILENCE_MS = _env_int("ASR_PARTIAL_SILENCE_MS", 650)
    ASR_PUNCTUATION_FINALIZE = os.getenv("ASR_PUNCTUATION_FINALIZE", "1") != "0"
    # ASR overlay: keep only the last N Q&A blocks (split by the same separator as append_block). 0 = unlimited.
    ASR_OVERLAY_MAX_CONVERSATIONS = _env_int("ASR_OVERLAY_MAX_CONVERSATIONS", 3)

    # Vision timeout / safety (applies to the whole two-step vision pipeline)
    VISION_TIMEOUT_SEC = _env_float("VISION_TIMEOUT_SEC", 8.0)

    # UI Settings
    # 0.0~1.0, higher = more opaque (less transparent)
    OVERLAY_OPACITY = 0.78

    # Response language preference:
    # - "auto": follow question language (use each prompt's built-in rule)
    # - "zh": always Chinese
    # - "en": always English (default for interview coach output)
    RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", "en")

    # RAG: minimum cosine similarity (0–1) to keep a chunk; below threshold chunks are dropped
    RAG_MIN_SCORE = _env_float("RAG_MIN_SCORE", 0.32)

    # LLM question classifier (text model, non-streaming)
    CLASSIFIER_MAX_TOKENS = _env_int("CLASSIFIER_MAX_TOKENS", 64)
    CLASSIFIER_TEMPERATURE = _env_float("CLASSIFIER_TEMPERATURE", 0.1)

    # Hotkeys
    SCREENSHOT_HOTKEY = os.getenv("SCREENSHOT_HOTKEY", "alt+p")
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

    # Per-overlay geometry persisted by the UI (list [x, y, w, h]). None = use defaults.
    OVERLAY_ASR_GEOMETRY = None
    OVERLAY_VISION_GEOMETRY = None


config = Config()
