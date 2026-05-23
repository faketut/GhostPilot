"""
LLM provider package.

`base.LLMProvider` is the common ABC. Concrete providers wrap one underlying
SDK each:
- `openai_compat.OpenAICompatProvider`: OpenAI, DeepSeek, Ollama, any
  endpoint speaking the OpenAI chat-completions API.
- `gemini.GeminiProvider`: Google's google-genai SDK.

Engine code picks a provider per pipeline (text / vision) via
`make_text_provider()` and `make_vision_provider()` which read `config`.
"""

from src.llm.base import LLMProvider, Delta, Usage


def make_text_provider():
    from src.llm.factory import make_text_provider as _f
    return _f()


def make_vision_provider():
    from src.llm.factory import make_vision_provider as _f
    return _f()


__all__ = [
    "LLMProvider", "Delta", "Usage",
    "make_text_provider", "make_vision_provider",
]
