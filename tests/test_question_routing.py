"""Question routing: which prompt structure an ASR utterance gets.

The router decides the *shape* of the answer the overlay renders — STAR lines for
a behavioral question, a fenced code block for a coding one, a single
tab-separated line for concepts. Getting it wrong is not a cosmetic issue: an
English coding question routed to the technical prompt produces a one-line
answer with no code block at all.

The LLM classifier is primary; these cover the keyword fallback used when that
call fails, and the guarantee that every route resolves to a real prompt whose
structure the overlay can actually render.
"""
from __future__ import annotations

import pytest

from src.config import config
from src.llm_engine import LLMEngine, TurnContext, _lang_suffix, classify_question
from src import prompt_loader

# Realistic spoken English interview questions (what the ASR path actually sees).
ROUTING_CASES = [
    # ── past-story / fit → STAR or WYEC ──
    ("Tell me about a time you had a conflict with a teammate.", "behavioral"),
    ("Describe a situation where you disagreed with your manager.", "behavioral"),
    ("Give me an example of a project you led end to end.", "behavioral"),
    ("Why do you want to work at our company?", "behavioral"),
    ("Tell us about yourself.", "behavioral"),
    ("What is your greatest weakness?", "behavioral"),
    ("What was your biggest challenge at work last year?", "behavioral"),
    ("Walk me through your resume.", "behavioral"),
    # A past story that names a data structure is still a past story.
    ("Tell me about a time you optimized a slow database query.", "behavioral"),
    # ── coding → UMPIR + code block ──
    ("Given an array of intervals, merge the overlapping ones.", "algorithm"),
    ("Write a function that returns the two numbers summing to a target.", "algorithm"),
    ("Reverse a linked list iteratively.", "algorithm"),
    ("Find the longest substring without repeating characters.", "algorithm"),
    ("What is the time complexity of binary search?", "algorithm"),
    ("Implement an LRU cache using a hash map and a doubly linked list.", "algorithm"),
    # ── concepts / systems → single structured line ──
    ("What are the challenges of eventual consistency in distributed systems?", "technical"),
    ("Design a rate limiter for a public API.", "technical"),
    ("What is the difference between a process and a thread?", "technical"),
    ("How would you debug a memory leak in production?", "technical"),
    ("Describe how you would scale a database to handle more writes.", "technical"),
]


@pytest.mark.parametrize("question,expected", ROUTING_CASES)
def test_interview_questions_route_to_the_right_structure(question, expected):
    assert classify_question(question) == expected


def test_every_route_resolves_to_a_prompt_file():
    """A route with no prompt would fall back to inline text and lose structure."""
    for q_type in {expected for _q, expected in ROUTING_CASES}:
        assert q_type in prompt_loader.list_prompts()
        assert prompt_loader.get_prompt(q_type).strip()


# ── the structure each route promises the overlay ────────────────────────


def test_behavioral_prompt_promises_star_or_wyec_lines():
    p = prompt_loader.get_prompt("behavioral")
    for tag in ("[S]", "[T]", "[A]", "[R]"):   # STAR
        assert tag in p
    for tag in ("[W]", "[Y]", "[E]", "[C]"):   # WYEC (fit / motivation)
        assert tag in p


def test_algorithm_prompt_asks_for_a_fenced_code_block():
    """The 'code displays fully' contract on the answer side.

    The prompt asks for the fence in prose rather than showing backticks, so
    assert the instruction — the overlay's fence rendering (indentation under
    token streaming, open-fence handling) is covered by test_overlay_render.py.
    """
    p = prompt_loader.get_prompt("algorithm")
    assert "fenced code block" in p
    for tag in ("[U]", "[M]", "[P]", "[I]", "[R]"):
        assert tag in p


def test_technical_prompt_promises_the_single_tab_separated_line():
    p = prompt_loader.get_prompt("technical")
    assert "TAB" in p.upper() or "\t" in p
    for tag in ("[R]", "[E]", "[A]", "[C]", "[T]"):
        assert tag in p


# ── English interview → Chinese answer ───────────────────────────────────


def test_chinese_is_forced_even_when_the_question_is_english(monkeypatch):
    """The prompts answer in Chinese only when the question is *not* clearly
    English — which an English interview always is. Without the explicit
    directive, an English question comes back in English."""
    monkeypatch.setattr(config, "RESPONSE_LANGUAGE", "zh", raising=False)
    monkeypatch.setattr(config, "ALGORITHM_KNOWLEDGE_FILE", "", raising=False)

    eng = LLMEngine.__new__(LLMEngine)
    eng.rag = None
    eng._algorithm_reference = lambda: None

    messages = eng._build_interview_messages(
        "technical", "What is a deadlock?", TurnContext(), from_screenshot=False
    )
    system = messages[0]["content"]
    assert "Chinese" in system and "Respond in Chinese only" in system
    assert messages[1]["content"].endswith("What is a deadlock?")


def test_auto_language_leaves_the_choice_to_the_prompt(monkeypatch):
    monkeypatch.setattr(config, "RESPONSE_LANGUAGE", "auto", raising=False)
    assert _lang_suffix() == ""
    monkeypatch.setattr(config, "RESPONSE_LANGUAGE", "en", raising=False)
    assert "English" in _lang_suffix()
