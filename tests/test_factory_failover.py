"""Factory wiring: TEXT_PROVIDER_FALLBACK builds a FailoverProvider."""
from __future__ import annotations

import pytest

from src.config import config
from src.llm.factory import make_text_provider
from src.llm.failover import FailoverProvider


@pytest.fixture(autouse=True)
def _restore_config():
    keep = {
        "TEXT_PROVIDER": config.TEXT_PROVIDER,
        "TEXT_PROVIDER_FALLBACK": config.TEXT_PROVIDER_FALLBACK,
    }
    yield
    for k, v in keep.items():
        setattr(config, k, v)


def test_text_provider_plain_when_no_fallback():
    config.TEXT_PROVIDER = "openai"
    config.TEXT_PROVIDER_FALLBACK = ""
    p = make_text_provider()
    assert not isinstance(p, FailoverProvider)


def test_text_provider_wraps_when_fallback_set():
    config.TEXT_PROVIDER = "openai"
    config.TEXT_PROVIDER_FALLBACK = "deepseek,gemini"
    p = make_text_provider()
    assert isinstance(p, FailoverProvider)
    assert len(p.fallbacks) == 2
    assert [f.name for f in p.fallbacks] == ["deepseek", "gemini"]


def test_text_provider_skips_unknown_fallback_names():
    config.TEXT_PROVIDER = "openai"
    config.TEXT_PROVIDER_FALLBACK = "bogus, openai ,nope"
    p = make_text_provider()
    assert isinstance(p, FailoverProvider)
    assert [f.name for f in p.fallbacks] == ["openai"]
