"""LLM provider abstraction.

Providers wrap one vendor SDK each:

- `openai_compat.OpenAICompatProvider`: OpenAI, DeepSeek, Ollama, any
  endpoint speaking the OpenAI chat-completions API.
- `gemini.GeminiProvider`: Google's google-genai SDK.

Screenshot OCR does *not* go through this layer — it runs locally via
`src.ocr_client` (Ollama native API); only the recognized text reaches the
text provider. `make_text_provider()` reads `config` to pick the provider.
"""

from src.llm.base import LLMProvider, Delta, Usage


def make_text_provider():
    from src.llm.factory import make_text_provider as _f
    return _f()


__all__ = [
    "LLMProvider", "Delta", "Usage",
    "make_text_provider",
]
