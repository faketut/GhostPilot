from src.llm.pricing import cost_usd, format_cost


def test_openai_known_model():
    c = cost_usd("gpt-4o", 1000, 1000)
    assert c is not None and c > 0


def test_unknown_model_returns_none():
    assert cost_usd("totally-made-up", 1000, 1000) is None


def test_local_models_zero():
    assert cost_usd("llama3:8b", 5000, 5000) == 0.0
    assert cost_usd("qwen2.5:7b", 5000, 5000) == 0.0


def test_zero_tokens():
    assert cost_usd("gpt-4o", 0, 0) == 0.0


def test_format_cost():
    assert format_cost(None) == ""
    assert format_cost(0) == "$0"
    assert format_cost(0.005).startswith("$") and "m" in format_cost(0.005)
    assert format_cost(0.5).startswith("$0.")
