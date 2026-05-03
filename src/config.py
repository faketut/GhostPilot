import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

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
    # Vision: GPT-4o (best multimodal; requires OPENAI_API_KEY)
    VISION_MODEL = os.getenv("VISION_MODEL", "deepseek-vl2")
    
    # Audio Settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = int(SAMPLE_RATE * 0.16) # 160ms buffer
    # Audio device selection / resilience
    AUDIO_DEVICE_CONTAINS = os.getenv("AUDIO_DEVICE_CONTAINS", "")  # substring match
    AUDIO_RECONNECT_SEC = float(os.getenv("AUDIO_RECONNECT_SEC", "2.0"))
    AUDIO_WATCHDOG_SEC = float(os.getenv("AUDIO_WATCHDOG_SEC", "2.0"))  # no-callback threshold

    # ASR segmentation
    ASR_PARTIAL_SILENCE_MS = int(os.getenv("ASR_PARTIAL_SILENCE_MS", "650"))
    ASR_PUNCTUATION_FINALIZE = os.getenv("ASR_PUNCTUATION_FINALIZE", "1") != "0"

    # Vision timeout / safety
    VISION_TIMEOUT_SEC = float(os.getenv("VISION_TIMEOUT_SEC", "8.0"))

    # UI Settings
    # 0.0~1.0, higher = more opaque (less transparent)
    OVERLAY_OPACITY = 0.78
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE = 14

    # Response language preference:
    # - "auto": follow question language (fallback zh)
    # - "zh": always Chinese
    # - "en": always English
    RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", "auto")
    
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
    KNOWLEDGE_CHUNK_CHARS = int(os.getenv("KNOWLEDGE_CHUNK_CHARS", "900"))
    KNOWLEDGE_OVERLAP_CHARS = int(os.getenv("KNOWLEDGE_OVERLAP_CHARS", "120"))

config = Config()
