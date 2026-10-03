"""Structural assertions for prompt-shaped answers.

Each route's answer shape is a contract in ``prompts/*.md`` — four tagged lines,
a fenced code block, one tab-separated line. Assert the *structure*, and judge
each part by the rule that actually applies to it.

Why not the global "is this Chinese?" ratio this replaces (``han / len(answer)``):
an algorithm answer is prose **plus a Python block**, and Python is ASCII by
construction — the prompt even forbids comments inside the code. Pooling the two
makes the number wrong in both directions:

* a correct Chinese answer over a correct code block scores ~0.25 and reads as
  "not Chinese" (a false alarm);
* an English-commented code block under Chinese prose would still clear the same
  threshold (a missed failure).

So: split into prose and code segments first, then assert per segment — Chinese
prose, and code that is present and *parses*. The parse check is the one that
matters for "the code displays fully": half a function is a `SyntaxError`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Mirrors overlay_ui._RE_FENCE: the closing fence is optional so a still-streaming
# block is still recognised as code.
_FENCE = re.compile(r"```(\w*)\n([\s\S]*?)(```|\Z)")
_TAG = re.compile(r"\[([A-Z])\]")
_HAN = re.compile(r"[\u4e00-\u9fff]")

# Tag sets, in the order each prompt mandates.
STAR = ("S", "T", "A", "R")
WYEC = ("W", "Y", "E", "C")
UMPIR = ("U", "M", "P", "I", "R")
TAB_LINE = ("R", "E", "A", "C", "T")


@dataclass(frozen=True)
class Segment:
    kind: str          # "prose" | "code"
    text: str
    lang: str = ""     # fence info string, lowercased


# ── parsing ──────────────────────────────────────────────────────────────


def segments(answer: str) -> list[Segment]:
    """Split an answer into alternating prose/code segments."""
    out: list[Segment] = []
    pos = 0
    for m in _FENCE.finditer(answer):
        if m.start() > pos:
            out.append(Segment("prose", answer[pos:m.start()]))
        out.append(Segment("code", m.group(2), (m.group(1) or "").lower()))
        pos = m.end()
    if pos < len(answer):
        out.append(Segment("prose", answer[pos:]))
    return out


def prose_text(answer: str) -> str:
    """Everything outside fenced blocks."""
    return "".join(s.text for s in segments(answer) if s.kind == "prose")


def code_blocks(answer: str) -> list[Segment]:
    return [s for s in segments(answer) if s.kind == "code"]


def tagged(answer: str) -> list[tuple[str, str]]:
    """``[(tag, clause), …]`` in written order, tags inside code ignored."""
    text = prose_text(answer)
    hits = list(_TAG.finditer(text))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        out.append((m.group(1), text[m.end():end].strip()))
    return out


def tags_of(answer: str) -> list[str]:
    return [t for t, _ in tagged(answer)]


def lines_of(answer: str) -> list[str]:
    """Non-blank prose lines (code excluded, so a fence cannot fake a line count)."""
    return [ln for ln in prose_text(answer).splitlines() if ln.strip()]


_TAG_OR_TAB = re.compile(r"\[[A-Z]\]|\t")


def content_text(answer: str) -> str:
    """Prose with the format's own scaffolding removed: tags and tab separators.

    ``[R] `` and the field-separating tabs are markup the prompt injects, not
    words. Counting them as prose is the same mistake as counting code as prose —
    it under-measures Chinese, and it lands hardest on the *technical* route,
    whose one-line format is five tags and four tabs of scaffolding around very
    few words.
    """
    return _TAG_OR_TAB.sub(" ", prose_text(answer))


def han_ratio(text: str) -> float:
    """Han characters as a fraction of all characters. Segment-scoped by design."""
    if not text:
        return 0.0
    return len(_HAN.findall(text)) / len(text)


def has_han(text: str) -> bool:
    return bool(_HAN.search(text))


# A fully-English answer measures ~0.00. Chinese technical writing that borrows
# Latin terms — "senior IC", "CP", "O(n log n)", "target-num" — measures 0.40+.
# The gap is wide, so the bar sits in the middle: high enough that an English
# answer cannot clear it, low enough that borrowing terminology cannot trip it.
MIN_CHINESE_RATIO = 0.25


# ── per-route structure ──────────────────────────────────────────────────


def assert_behavioral(answer: str) -> None:
    """``prompts/behavioral.md``: exactly four lines, STAR or WYEC."""
    tags = tags_of(answer)
    assert tags in (list(STAR), list(WYEC)), (
        f"behavioral answer must carry exactly the {list(STAR)} or {list(WYEC)} tags "
        f"in order; got {tags}"
    )
    lines = lines_of(answer)
    assert len(lines) == 4, (
        f"behavioral answer must be exactly 4 lines (one per tag); got {len(lines)}: {lines}"
    )
    for (tag, clause), line in zip(tagged(answer), lines):
        assert line.lstrip().startswith(f"[{tag}]"), (
            f"line for [{tag}] must start with its tag; got {line!r}"
        )
        assert clause, f"[{tag}] carries no content"
    assert not code_blocks(answer), "behavioral answer must not contain a code block"


def assert_technical(answer: str) -> None:
    """``prompts/technical.md``: exactly one line, five tags, real TAB separators."""
    assert tags_of(answer) == list(TAB_LINE), (
        f"technical answer must carry exactly {list(TAB_LINE)} in order; got {tags_of(answer)}"
    )
    lines = lines_of(answer)
    assert len(lines) == 1, (
        f"technical answer must be exactly 1 line; got {len(lines)}: {lines}"
    )
    line = lines[0]
    # Check the *wrong* thing first: a model that writes the two characters
    # backslash + t has also failed the "real TAB" test, and reporting only the
    # generic failure hides the specific mistake the prompt warns about.
    assert "\\t" not in line, (
        "technical answer contains a literal backslash-t instead of a TAB "
        "(the prompt names this exact failure mode)"
    )
    assert "\t" in line, "technical answer must separate its fields with a real TAB (0x09)"
    assert not code_blocks(answer), "technical answer must not contain a code block"


def assert_algorithm(answer: str) -> None:
    """``prompts/algorithm.md``: a [U][M][P] line, then [I] + fence, then [R]."""
    assert tags_of(answer) == list(UMPIR), (
        f"algorithm answer must carry exactly {list(UMPIR)} in order; got {tags_of(answer)}"
    )
    blocks = code_blocks(answer)
    assert len(blocks) == 1, f"algorithm answer must contain exactly one code block; got {len(blocks)}"
    block = blocks[0]
    assert block.lang, "the code fence needs a language tag (```python), not a bare fence"
    assert block.text.strip(), "the code block is empty"
    assert "```" not in prose_text(answer), "stray fence markers left in the prose"
    # The prompt keeps [U]/[M]/[P] on one line and forbids code there; the '[I]'
    # tag is what introduces the fence.
    plan = lines_of(answer)[0]
    assert "[U]" in plan and "[M]" in plan and "[P]" in plan, (
        f"the first line must carry [U] [M] [P] together; got {plan!r}"
    )
    # Code that does not parse is the whole point of this route: the user pastes
    # it. Truncation (a token cap, a deadline, a repetition guard) lands here.
    assert_code_parses(answer)


def assert_code_parses(answer: str) -> list[str]:
    """Every Python block must be syntactically complete.

    This is the assertion for "the code displayed fully": truncation — a token
    cap, a deadline, a repetition guard — lands mid-construct and fails here,
    where a character-count check sees only a slightly shorter string.

    Returns the languages that were parsed, so a caller can assert the check
    actually ran instead of silently skipping a non-Python block.
    """
    parsed: list[str] = []
    for block in code_blocks(answer):
        if block.lang in ("python", "python3", "py"):
            try:
                compile(block.text, "<answer>", "exec")
            except SyntaxError as e:
                head = "\n".join(block.text.splitlines()[:3])
                raise AssertionError(
                    f"code block does not parse ({e.msg} at line {e.lineno}) — a partial "
                    f"listing looks like this. First lines:\n{head}"
                ) from e
            parsed.append(block.lang)
    return parsed


def assert_prose_is_chinese(answer: str, *, minimum: float = MIN_CHINESE_RATIO) -> float:
    """The prose must be Chinese. Code is exempt — it cannot be, by construction.

    Measured on prose *content*, with the format's tags and tabs stripped.
    Returns the ratio so callers can log it.
    """
    content = content_text(answer)
    assert content.strip(), "answer has no prose to judge"
    ratio = han_ratio(content)
    assert ratio >= minimum, (
        f"prose is only {ratio:.2f} Han (need >= {minimum:.2f}) — the answer is not in "
        f"Chinese; prose was: {content.strip()[:200]!r}"
    )
    return ratio


def assert_answer(q_type: str, answer: str, *, response_language: str = "zh") -> dict:
    """Assert the whole contract for one route and return a structural summary."""
    if q_type == "behavioral":
        assert_behavioral(answer)
    elif q_type == "technical":
        assert_technical(answer)
    elif q_type == "algorithm":
        assert_algorithm(answer)
    else:
        raise AssertionError(f"unknown q_type {q_type!r}")

    if response_language == "zh":
        assert_prose_is_chinese(answer)

    summary = describe(answer)
    summary["q_type"] = q_type
    return summary


# ── reporting ────────────────────────────────────────────────────────────


def describe(answer: str) -> dict:
    """A structural summary — the replacement for a single pooled ratio.

    Printed in test output and failure reports so a reader sees *which* part is
    wrong (tags, line count, code, language) rather than one ambiguous number.
    """
    content = content_text(answer)
    blocks = code_blocks(answer)
    parsed: list[str] = []
    for block in blocks:
        if block.lang in ("python", "python3", "py"):
            try:
                compile(block.text, "<answer>", "exec")
                parsed.append(block.lang)
            except SyntaxError:
                pass
    return {
        "tags": tags_of(answer),
        "lines": len(lines_of(answer)),
        "code_blocks": len(blocks),
        "code_langs": [b.lang for b in blocks],
        "code_lines": sum(len(b.text.splitlines()) for b in blocks),
        "code_parses": len(parsed) == len([b for b in blocks if b.lang in ("python", "python3", "py")]),
        "prose_chars": len(content.strip()),
        "prose_han_ratio": round(han_ratio(content), 3),
        "whole_answer_han_ratio": round(han_ratio(answer), 3),
    }


def format_summary(summary: dict) -> str:
    return (
        f"{summary.get('q_type', '?')} | tags={summary['tags']} lines={summary['lines']} "
        f"code={summary['code_blocks']}x{summary['code_langs']}({summary['code_lines']}L, "
        f"parses={summary['code_parses']}) | prose content {summary['prose_chars']}c "
        f"{summary['prose_han_ratio']:.0%} Han (whole answer {summary['whole_answer_han_ratio']:.0%})"
    )
