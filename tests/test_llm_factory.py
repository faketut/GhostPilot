"""Unit tests for src.llm.factory provider selection.

Patches the constructors so we don't actually open SDK clients.
"""
from __future__ import annotations

import pytest

from src.config import config
from src.llm import factory


class _StubOAI:
    def __init__(self, *, api_key, base_url=None, label=None):
        self.api_key = api_key
        self.base_url = base_url
        self.label = label


class _StubGem:
    def __init__(self, *, api_key):
        self.api_key = api_key
        self.label = "gemini"


@pytest.fixture(autouse=True)
def _patch_providers(monkeypatch):
    monkeypatch.setattr(factory, "OpenAICompatProvider", _StubOAI)
    monkeypatch.setattr(factory, "GeminiProvider", _StubGem)


@pytest.fixture
def cfg(monkeypatch):
    """Reset relevant config fields for each test."""
    defaults = {
        "TEXT_PROVIDER": "",
        "TEXT_MODEL": "",
        "VISION_PROVIDER": "",
        "VISION_MODEL": "",
        "OPENAI_API_KEY": "oai-key",
        "DEEPSEEK_API_KEY": "ds-key",
        "GEMINI_API_KEY": "g-key",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
    }
    for k, v in defaults.items():
        monkeypatch.setattr(config, k, v, raising=False)
    return config


# ── Text provider ────────────────────────────────────────────────────────

def test_text_default_is_openai(cfg):
    p = factory.make_text_provider()
    assert isinstance(p, _StubOAI)
    assert p.label == "openai"
    assert p.base_url is None
    assert p.api_key == "oai-key"


def test_text_explicit_ollama_provider(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "TEXT_PROVIDER", "ollama")
    p = factory.make_text_provider()
    assert p.label == "ollama"
    assert p.base_url == "http://localhost:11434/v1"


def test_text_infers_ollama_from_model_prefix(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "TEXT_MODEL", "qwen2.5")
    p = factory.make_text_provider()
    assert p.label == "ollama"


def test_text_infers_deepseek_from_model_name(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "TEXT_MODEL", "deepseek-chat")
    p = factory.make_text_provider()
    assert p.label == "deepseek"
    assert p.base_url == "https://api.deepseek.com/v1"
    assert p.api_key == "ds-key"


def test_text_explicit_gemini_uses_gemini_provider(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "TEXT_PROVIDER", "gemini")
    p = factory.make_text_provider()
    assert isinstance(p, _StubGem)
    assert p.api_key == "g-key"


def test_text_provider_override_wins_over_model_inference(cfg, monkeypatch):
    # Model name suggests deepseek, but explicit provider should win.
    monkeypatch.setattr(cfg, "TEXT_PROVIDER", "openai")
    monkeypatch.setattr(cfg, "TEXT_MODEL", "deepseek-chat")
    p = factory.make_text_provider()
    assert p.label == "openai"


# ── Vision provider ──────────────────────────────────────────────────────

def test_vision_default_is_openai(cfg):
    p = factory.make_vision_provider()
    assert isinstance(p, _StubOAI)
    assert p.label == "openai-vision"


def test_vision_infers_gemini_from_model(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "VISION_MODEL", "gemini-2.5-flash")
    p = factory.make_vision_provider()
    assert isinstance(p, _StubGem)


def test_vision_explicit_gemini_provider(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "VISION_PROVIDER", "gemini")
    monkeypatch.setattr(cfg, "VISION_MODEL", "gpt-4o")  # ignored
    p = factory.make_vision_provider()
    assert isinstance(p, _StubGem)


def test_vision_deepseek_returns_placeholder(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "VISION_MODEL", "deepseek-vision")
    p = factory.make_vision_provider()
    assert isinstance(p, _StubOAI)
    assert "unsupported" in (p.label or "")
