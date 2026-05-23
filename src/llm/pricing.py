"""Per-1k-token pricing (USD). Fall back to None if model unknown."""

from __future__ import annotations

# Source: provider public pricing pages; keep approximate and easy to update.
# Format: model_name_lower -> (input_per_1k, output_per_1k)
_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o":           (0.0025, 0.010),
    "gpt-4o-mini":      (0.00015, 0.0006),
    "gpt-4-turbo":      (0.010, 0.030),
    # DeepSeek
    "deepseek-chat":     (0.00027, 0.0011),
    "deepseek-reasoner": (0.00055, 0.00219),
    # Gemini
    "gemini-1.5-flash":  (0.000075, 0.0003),
    "gemini-1.5-pro":    (0.00125, 0.005),
    "gemini-2.0-flash":  (0.0001, 0.0004),
    # Ollama / local
    "ollama":            (0.0, 0.0),
}


def cost_usd(model: str, in_tokens: int, out_tokens: int) -> float | None:
    if not model:
        return None
    m = model.lower()
    # Try exact, then prefix match.
    rate = _PRICING.get(m)
    if rate is None:
        for k, v in _PRICING.items():
            if m.startswith(k):
                rate = v
                break
    # Local models always free
    if rate is None and (m.startswith("llama") or m.startswith("qwen") or m.startswith("mistral") or m.startswith("phi")):
        rate = (0.0, 0.0)
    if rate is None:
        return None
    return (in_tokens / 1000.0) * rate[0] + (out_tokens / 1000.0) * rate[1]


def format_cost(usd: float | None) -> str:
    if usd is None:
        return ""
    if usd == 0:
        return "$0"
    if usd < 0.01:
        return f"${usd*1000:.2f}m"  # millicents
    return f"${usd:.3f}"
