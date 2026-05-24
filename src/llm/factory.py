"""Build providers from current `config`."""

from __future__ import annotations

import logging

from src.config import config
from src.llm.base import LLMProvider
from src.llm.failover import FailoverProvider
from src.llm.gemini import GeminiProvider
from src.llm.openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)


def _is_deepseek(model: str) -> bool:
    return "deepseek" in (model or "").lower()


def _is_gemini(model: str) -> bool:
    return "gemini" in (model or "").lower()


def _is_ollama_model(model: str) -> bool:
    # Heuristic: anything starting with a recognised local-model family OR
    # explicitly prefixed "ollama/".
    m = (model or "").lower()
    return m.startswith("ollama/") or m.startswith("llama") or m.startswith("qwen") or m.startswith("mistral") or m.startswith("phi")


def _ollama_base() -> str:
    return getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")


def _build_text_by_name(name: str) -> LLMProvider | None:
    name = (name or "").strip().lower()
    if name == "openai":
        return OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai")
    if name == "ollama":
        return OpenAICompatProvider(api_key="ollama", base_url=_ollama_base(), label="ollama")
    if name == "deepseek":
        return OpenAICompatProvider(
            api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", label="deepseek",
        )
    if name == "gemini":
        return GeminiProvider(api_key=config.GEMINI_API_KEY)
    return None


def _build_vision_by_name(name: str) -> LLMProvider | None:
    name = (name or "").strip().lower()
    if name == "openai":
        return OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai-vision")
    if name == "gemini":
        return GeminiProvider(api_key=config.GEMINI_API_KEY)
    return None


def _parse_chain(spec: str) -> list[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def _maybe_wrap(primary: LLMProvider, fallback_names: list[str], builder) -> LLMProvider:
    fallbacks: list[LLMProvider] = []
    for n in fallback_names:
        p = builder(n)
        if p is not None:
            fallbacks.append(p)
        else:
            logger.warning("Unknown fallback provider name: %s", n)
    if not fallbacks:
        return primary
    logger.info("Provider failover enabled: %s -> %s", primary.name, [f.name for f in fallbacks])
    return FailoverProvider(primary, fallbacks)


def make_text_provider() -> LLMProvider:
    provider = (getattr(config, "TEXT_PROVIDER", "") or "").strip().lower()
    model = config.TEXT_MODEL or ""

    # Explicit provider always wins; model-name inference only kicks in when
    # provider is left blank (auto).
    if provider == "openai":
        primary: LLMProvider = OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai")
    elif provider == "ollama" or (not provider and _is_ollama_model(model)):
        primary = OpenAICompatProvider(api_key="ollama", base_url=_ollama_base(), label="ollama")
    elif provider == "deepseek" or (not provider and _is_deepseek(model)):
        primary = OpenAICompatProvider(
            api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", label="deepseek",
        )
    elif provider == "gemini" or (not provider and _is_gemini(model)):
        primary = GeminiProvider(api_key=config.GEMINI_API_KEY)
    else:
        primary = OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai")

    fallback_names = _parse_chain(getattr(config, "TEXT_PROVIDER_FALLBACK", ""))
    return _maybe_wrap(primary, fallback_names, _build_text_by_name)


def make_vision_provider() -> LLMProvider:
    provider = (getattr(config, "VISION_PROVIDER", "") or "").strip().lower()
    model = config.VISION_MODEL or ""

    if provider == "openai":
        primary: LLMProvider = OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai-vision")
    elif provider == "gemini" or (not provider and _is_gemini(model)):
        primary = GeminiProvider(api_key=config.GEMINI_API_KEY)
    elif not provider and _is_deepseek(model):
        primary = OpenAICompatProvider(api_key=config.DEEPSEEK_API_KEY, label="deepseek-vision-unsupported")
    else:
        primary = OpenAICompatProvider(api_key=config.OPENAI_API_KEY, label="openai-vision")

    fallback_names = _parse_chain(getattr(config, "VISION_PROVIDER_FALLBACK", ""))
    return _maybe_wrap(primary, fallback_names, _build_vision_by_name)
