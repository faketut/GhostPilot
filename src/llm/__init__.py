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
from src.llm.factory import make_text_provider, make_vision_provider

__all__ = [
    "LLMProvider", "Delta", "Usage",
    "make_text_provider", "make_vision_provider",
]
