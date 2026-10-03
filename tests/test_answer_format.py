"""The structural assertions themselves.

These exist because the check they replace was wrong in *both* directions: a
single ``han / len(answer)`` ratio over a whole answer. It called a correct
Chinese answer with a code block "not Chinese", and would have waved through
English buried inside the code. So the interesting tests here are the ones that
pin each violation class to a specific failure — and the one that pins the
false-alarm case (`test_code_does_not_dilute_the_prose_language_check`).
"""
from __future__ import annotations

import pytest

from tests import format_contract as fc

# Captured verbatim from a real deepseek-flash run (RESPONSE_LANGUAGE=zh), for
# the question "Given an array of integers and a target, return the indices of
# the two numbers that add up to the target."
REAL_ALGORITHM_ANSWER = """[U] 在整数数组中找两个不同下标使对应元素之和等于目标值，并返回这两个下标 [M] 哈希表一次遍历 [P] 用字典记录已见数值到下标；对每个数先查 target-num 是否已出现，若出现则立即返回两下标，否则记录当前数并继续

[I]
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        need = target - num
        if need in seen:
            return [seen[need], i]
        seen[num] = i
    return []
```

[R] T(n)=O(n)，S(n)=O(n)。边界：空数组或不足两个元素时返回 []。
"""

REAL_BEHAVIORAL_ANSWER = """[S] 上一份工作中，一个客户项目因需求变更，交付期从六周压缩到三周，上线日不可移动。
[T] 我作为项目负责人，必须在资源不变下，保证核心功能按时上线并控制质量风险。
[A] 我先冻结非核心需求，按用户价值排MVP；每天站会同步阻塞项，并亲自盯接口联调和回归测试。
[R] 项目按期上线，核心流程未出现P1故障，客户同意续签；我学到先砍范围再保节奏。"""

TECHNICAL_ANSWER = (
    "[R] senior IC\t[E] 一致性与可用性的取舍\t[A] 法定人数读\t[C] 选 CP，账本正确优先\t[T] 不丢写"
)


# ── the false alarm the old metric produced ──────────────────────────────


def test_code_does_not_dilute_the_prose_language_check():
    """The regression this module exists for.

    The real answer scores well under 0.3 for the *whole* answer — the number the
    old check printed as "NOT Chinese" — while its prose is plainly Chinese.
    """
    whole = fc.han_ratio(REAL_ALGORITHM_ANSWER)
    prose = fc.han_ratio(fc.prose_text(REAL_ALGORITHM_ANSWER))

    assert whole < 0.3, "precondition: the pooled ratio is what used to look wrong"
    assert prose > 0.5, f"prose should read as Chinese, got {prose:.2f}"

    # And the assertion agrees with the prose, not with the pooled number.
    assert fc.assert_prose_is_chinese(REAL_ALGORITHM_ANSWER) > fc.MIN_CHINESE_RATIO


def test_segments_separate_code_from_prose():
    segs = fc.segments(REAL_ALGORITHM_ANSWER)
    assert [s.kind for s in segs] == ["prose", "code", "prose"]
    assert fc.code_blocks(REAL_ALGORITHM_ANSWER)[0].lang == "python"
    # Tags inside a code block must not be mistaken for answer structure.
    assert "def" not in fc.prose_text(REAL_ALGORITHM_ANSWER)


def test_language_assertion_ignores_tags_and_identifiers_in_prose():
    """"target-num" and the [U]/[M]/[P] tags are Latin but not a language signal."""
    prose = fc.prose_text(REAL_ALGORITHM_ANSWER)
    assert "target-num" in prose
    assert fc.han_ratio(prose) > 0.5


def test_format_scaffolding_is_not_counted_as_prose():
    """The technical route is five tags and four tabs of markup around few words.

    Counting that scaffolding as prose is the same error as counting code as
    prose — it under-measures Chinese, and it falls hardest on this route.
    """
    raw = fc.han_ratio(fc.prose_text(TECHNICAL_ANSWER))
    content = fc.han_ratio(fc.content_text(TECHNICAL_ANSWER))
    assert content > raw, f"stripping markup should raise the ratio: raw={raw:.2f} content={content:.2f}"
    assert content > fc.MIN_CHINESE_RATIO


# ── compliant answers pass, per route ────────────────────────────────────


def test_real_algorithm_answer_satisfies_the_contract():
    fc.assert_algorithm(REAL_ALGORITHM_ANSWER)
    assert fc.assert_code_parses(REAL_ALGORITHM_ANSWER) == ["python"]


def test_real_behavioral_answer_satisfies_the_contract():
    fc.assert_behavioral(REAL_BEHAVIORAL_ANSWER)
    assert fc.tags_of(REAL_BEHAVIORAL_ANSWER) == list(fc.STAR)


def test_fit_questions_use_the_wyec_layout():
    answer = (
        "[W] 贵司要把账本团队扩到十倍交易量。\n"
        "[Y] 我交付过两套支付账本，端到端负责风险面。\n"
        "[E] 把日均四百万笔结算的对账差异降低 92%。\n"
        "[C] 我可以在头两个季度复制同样的事。"
    )
    fc.assert_behavioral(answer)
    assert fc.tags_of(answer) == list(fc.WYEC)


def test_technical_answer_satisfies_the_contract():
    fc.assert_technical(TECHNICAL_ANSWER)
    # Terse Chinese with borrowed Latin terms ("senior IC", "CP") — comfortably
    # Chinese, and nowhere near the 0.5+ a prose-only route reaches.
    ratio = fc.assert_prose_is_chinese(TECHNICAL_ANSWER)
    assert fc.MIN_CHINESE_RATIO <= ratio < 0.6


# ── violation classes must fail, with a message that names the problem ───


def test_truncated_code_is_caught_by_the_parser_not_by_length():
    """The exact failure the vision path can produce: recognition stopped early,
    so the answer's code is a partial listing. A character count sees only a
    slightly shorter string; the parser sees a syntax error."""
    truncated = REAL_ALGORITHM_ANSWER.replace(
        "            return [seen[need], i]", "            return [seen[need]"
    )
    with pytest.raises(AssertionError, match="does not parse"):
        fc.assert_code_parses(truncated)

    # The old-style measurement cannot distinguish it.
    assert abs(fc.han_ratio(truncated) - fc.han_ratio(REAL_ALGORITHM_ANSWER)) < 0.01
    assert fc.describe(truncated)["code_parses"] is False


def test_missing_code_block_fails_the_algorithm_route():
    """What an English coding question got before the routing fix: the technical
    prompt's one-liner, i.e. no code at all."""
    only_prose = (
        "[U] 找两个数之和等于目标值 [M] 哈希表 [P] 一次遍历\n\n"
        "[I]\n[R] T(n)=O(n)"
    )
    with pytest.raises(AssertionError, match="exactly one code block"):
        fc.assert_algorithm(only_prose)


def test_wrong_tag_set_fails():
    with pytest.raises(AssertionError, match="must carry exactly"):
        fc.assert_technical("[R] a\t[E] b\t[C] c")  # [A] and [T] dropped


def test_behavioral_answer_with_five_lines_fails():
    answer = REAL_BEHAVIORAL_ANSWER + "\n[R2] 另一件事。"
    with pytest.raises(AssertionError, match="tags|exactly 4 lines"):
        fc.assert_behavioral(answer)


def test_technical_answer_split_across_lines_fails():
    """The prompt requires one line; a wrapped answer loses the column layout."""
    answer = "[R] senior IC\t[E] 取舍\t[A] 备选\n[C] 选择\t[T] 结果"
    with pytest.raises(AssertionError, match="exactly 1 line"):
        fc.assert_technical(answer)


def test_literal_backslash_t_fails_the_tab_contract():
    """The prompt calls this exact mistake out: two characters, not a TAB."""
    answer = "[R] a\\t[E] b\\t[A] c\\t[C] d\\t[T] e"
    with pytest.raises(AssertionError, match="backslash-t"):
        fc.assert_technical(answer)


def test_english_prose_fails_the_chinese_assertion():
    english = (
        "[U] Merge overlapping intervals [M] sort then sweep [P] track the last end\n\n"
        "[I]\n```python\ndef merge(xs):\n    return xs\n```\n\n[R] O(n log n)"
    )
    with pytest.raises(AssertionError, match="not in Chinese"):
        fc.assert_prose_is_chinese(english)


def test_bare_fence_without_a_language_fails():
    answer = REAL_ALGORITHM_ANSWER.replace("```python", "```")
    with pytest.raises(AssertionError, match="language tag"):
        fc.assert_algorithm(answer)


# ── dispatcher + reporting ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "q_type,answer",
    [
        ("behavioral", REAL_BEHAVIORAL_ANSWER),
        ("algorithm", REAL_ALGORITHM_ANSWER),
        ("technical", TECHNICAL_ANSWER),
    ],
)
def test_dispatcher_reports_a_structural_summary(q_type, answer):
    summary = fc.assert_answer(q_type, answer, response_language="zh")

    assert summary["q_type"] == q_type
    assert summary["lines"] >= 1
    assert summary["prose_han_ratio"] >= fc.MIN_CHINESE_RATIO
    if q_type == "algorithm":
        assert summary["code_parses"] is True
    else:
        assert summary["code_blocks"] == 0
    # The summary keeps both numbers so the distinction stays visible.
    text = fc.format_summary(summary)
    assert "prose" in text and "whole answer" in text


def test_dispatcher_rejects_an_unknown_route():
    with pytest.raises(AssertionError, match="unknown q_type"):
        fc.assert_answer("vision", REAL_BEHAVIORAL_ANSWER)


@pytest.mark.parametrize("q_type,answer", [("algorithm", REAL_ALGORITHM_ANSWER)])
def test_the_full_contract_includes_the_parser_check(q_type, answer):
    """Regression: the parse check existed but was not wired into the route
    contract, so a truncated answer passed `assert_answer` — found by mutating a
    real answer and confirming it was rejected."""
    truncated = answer.replace("            return [seen[need], i]", "            return [seen[need]")
    assert truncated != answer, "the mutation must actually change the answer"
    assert fc.describe(truncated)["code_parses"] is False

    with pytest.raises(AssertionError, match="does not parse"):
        fc.assert_answer(q_type, truncated, response_language="zh")
    # And the intact answer still passes, so the check is not simply always-fail.
    fc.assert_answer(q_type, answer, response_language="zh")


def test_summary_flags_a_non_parsing_block():
    broken = REAL_ALGORITHM_ANSWER.replace("    seen = {}", "    seen = {")
    assert fc.describe(broken)["code_parses"] is False
    assert fc.describe(REAL_ALGORITHM_ANSWER)["code_parses"] is True
