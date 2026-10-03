"""
LLM Engine
----------
Routes classified questions to the appropriate LLM with prompts loaded from
the `prompts/` directory via PromptLoader.

Text route   → the configured text model (DeepSeek-V4 by default)
Vision route → local OCR (Ollama + GLM-OCR) → same text route as above
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.config import config
from src.ocr_client import OCRError, OCRTruncated, OCRClient, trim_repetition
from src import ollama_boot
from src.knowledge_loader import load_knowledge_file
from src.rag_manager import RAGManager
from src.prompt_loader import get_prompt
from src.llm import make_text_provider, Usage
from src.usage_log import log_usage

logger = logging.getLogger(__name__)


@dataclass
class TurnContext:
    """Reference material injected ahead of one question.

    Two kinds, deliberately not interchangeable: ``snippets`` are *retrieved*
    from the knowledge base and ``algorithm_ref`` is the patterns cheatsheet
    injected *whole* for algorithm turns. Only ``snippets`` may be reported as
    RAG hits — a fixed injection is not a retrieval, and counting it as one
    would make the observability footer describe something that did not happen.
    """
    snippets: list[str] = field(default_factory=list)
    algorithm_ref: str | None = None


# ── Question classifier (keyword fallback) ────────────────────────────────

_BEHAVIORAL_KW = [
    "经历", "举例", "讲一个", "描述一次", "给我讲", "冲突",
    "为什么加入", "为什么选择", "为什么想来", "为何应聘", "自我介绍",
    "动机", "还有什么补充", "对我们公司", "对本公司", "了解我们公司",
    "describe a time", "tell me about", "walk me through",
    "give an example", "give me an example", "example of a time",
    "example of a project", "a time when", "tell us about yourself",
    # Motivation / fit / open pitch (not episodic STAR)
    "why do you want", "why us", "why this company", "why our company",
    "why are you interested", "why join", "what attracts you",
    "anything else", "like to share about yourself", "strong candidate",
    "why should we hire", "introduce yourself",
    "what do you know about us", "cultural fit", "values align",
    "why this role", "why the role", "what interests you about",
]

# Words that *hint* at behavioral but describe a subject as often as they ask
# for a story: "describe how a hash table resolves collisions" and "what are the
# challenges of eventual consistency" are technical questions. Checked only
# after the algorithm list, so a coding question that happens to say "describe"
# still routes to the algorithm prompt (the one that emits a code block).
_BEHAVIORAL_WEAK_KW = [
    "描述", "挑战", "困难", "优点", "缺点",
    "describe", "challenge", "difficult", "strength", "weakness",
]

# Coding/DSA vocabulary. The English half was largely missing, so an English
# coding question ("given an array of intervals, merge the overlapping ones")
# matched nothing here and fell through to the technical prompt — which is a
# single tab-separated line with no code block, i.e. the code the user wanted
# never appeared. This app is used for English interviews, so that was the
# common case, not an edge case.
_ALGORITHM_KW = [
    # Chinese
    "算法", "数组", "链表", "树", "图", "动态规划", "排序", "搜索",
    "复杂度", "二叉", "栈", "队列", "堆", "哈希",
    # English — names and techniques
    "leetcode", "two pointer", "sliding window", "binary search", "dfs", "bfs",
    "backtrack", "dynamic programming", "greedy", "big o",
    "time complexity", "space complexity", "recursion", "dp",
    # English — data structures
    "array", "linked list", "listnode", "hashmap", "hash map", "hash table",
    "binary tree", "bst", "tree", "graph", "matrix", "stack", "queue",
    "heap", "priority queue", "trie", "substring", "subarray", "palindrome",
    "permutation", "anagram", "interval", "merge", "sort", "traverse",
    "iterate",
    # English — how these questions are phrased
    "write a function", "implement a function", "given an array",
    "given a string", "given a list", "brute force", "optimal solution",
    "return the",
]

# Concepts and systems vocabulary. Checked before the generic behavioral hints
# so that "what are the challenges of eventual consistency" is answered as the
# technical question it is, while "what was your biggest challenge" (no CS
# subject) still reads as behavioral.
_TECHNICAL_KW = [
    "distributed", "consistency", "availability", "partition tolerance",
    "replication", "sharding", "consensus", "eventual consistency",
    "latency", "throughput", "scalab", "bottleneck", "load balancer",
    "microservice", "message queue", "database", "transaction", "deadlock",
    "isolation level", "index", "query plan", "sql", "nosql",
    "operating system", "kernel", "virtual memory", "garbage collection",
    "thread", "concurrency", "race condition", "mutex", "context switch",
    "tcp", "udp", "http", "dns", "tls", "networking", "routing",
    "caching", "kubernetes", "docker", "container", "ci/cd", "deployment",
    "system design", "architecture", "api design", "observability",
    "debug", "memory leak", "profiling", "unit test", "compiler",
]


def classify_question(question: str) -> str:
    """
    Returns one of: 'behavioral', 'algorithm', 'technical'.

    Order matters and is deliberate:

    1. an unambiguous behavioral marker wins outright — a past-story or fit
       question stays behavioral even when it names a data structure;
    2. coding vocabulary → algorithm (the prompt that emits a code block);
    3. concepts/systems vocabulary → technical;
    4. generic behavioral hints ("describe", "challenge") only when no CS
       subject was named at all;
    5. otherwise technical.
    """
    q = question.lower()
    if any(kw in q for kw in _BEHAVIORAL_KW):
        return "behavioral"
    if any(kw in q for kw in _ALGORITHM_KW):
        return "algorithm"
    if any(kw in q for kw in _TECHNICAL_KW):
        return "technical"
    if any(kw in q for kw in _BEHAVIORAL_WEAK_KW):
        return "behavioral"
    return "technical"


def _basename_kb(rel: str) -> str:
    return rel.replace("\\", "/").split("/")[-1].lower()


# ── Code-shape detection (OCR transcripts) ────────────────────────────────
#
# A screenshot of an editor or a problem statement reaches the router as OCR
# text, and the classifiers read *words*: an OCR'd listing whose prose is thin
# ("Input: nums, k" / "1 <= k <= n") can be filed as `technical`, whose prompt
# answers in one tab-separated line with no code block — the one shape that is
# useless for a coding question, and the failure this module has hit before.
#
# The transcript's *shape* is a stronger and free signal, so it is checked
# before the model: a line that ends in `;` / `{` / `}` or opens with a
# definition keyword is code, and two of them are not something prose produces.
_CODE_LINE_END = re.compile(r"[;{}]\s*$")
_CODE_LINE_START = re.compile(
    r"^\s*(?:def|class|function|func|struct|impl|public|private|protected|static|void|int|"
    r"const|let|var|elif|else|for|while|return|import|from|#include|package|using)\b"
)
_FENCE = "```"


def looks_like_code(text: str) -> bool:
    """True when a transcript is shaped like source code rather than prose.

    Deliberately conservative — a *false positive* routes a concept question to
    the algorithm prompt, which still answers it (with a demonstration
    snippet); a false negative silently answers a coding question with no code.
    So the bar is "a fenced block, or two line-level code markers", which
    ordinary prose does not clear.
    """
    if not text:
        return False
    if _FENCE in text:
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    markers = sum(1 for ln in lines if _CODE_LINE_END.search(ln) or _CODE_LINE_START.match(ln))
    return markers >= 2


def _normalize_q_type(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip().lower()
    if s in ("behavioral", "technical", "algorithm"):
        return s
    return None


def _parse_json_object(text: str) -> dict:
    t = (text or "").strip()
    if "```" in t:
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(t[start : end + 1])


_CLASSIFIER_SYSTEM = """You classify one interview question.
Reply with JSON only, no markdown. Schema: {"type":"<one>"} where <one> is exactly one word: behavioral | technical | algorithm.

Definitions:
- behavioral: soft skills, teamwork, conflict, leadership, STAR-style past stories ("tell me about a time"); ALSO motivation/why-this-company/why-this-role, self-intro, "anything else about yourself", culture/values fit, "why should we hire you" — anything mainly about people, fit, or personal narrative rather than CS theory or coding.
- technical: CS concepts, system design, frameworks, tools, debugging — not mainly "write this algorithm from scratch".
- algorithm: coding / DSA / LeetCode style, implementations, complexity, specific algorithm names."""


def _answer_token_cap() -> int | None:
    """Configured output cap for an answer call, or None for "uncapped".

    ``ANSWER_MAX_TOKENS=0`` (the default) leaves the call uncapped: for a
    reasoning model a cap is shared with the hidden chain-of-thought, so a fixed
    number can be spent entirely on reasoning and leave the visible answer
    empty. Only a positive value imposes one.
    """
    try:
        cap = int(getattr(config, "ANSWER_MAX_TOKENS", 0) or 0)
    except (TypeError, ValueError):
        return None
    return cap if cap > 0 else None


def _is_truncated(finish_reason: str | None) -> bool:
    """True when the provider stopped because it ran out of output budget.

    OpenAI-compatible servers report ``"length"``; google-genai reports
    ``MAX_TOKENS`` (normalised to the bare name when the delta was built).
    """
    if not finish_reason:
        return False
    return finish_reason.strip().lower() in {"length", "max_tokens"}


def _lang_suffix() -> str:
    """Return the language-control suffix appended to every system prompt.

    Driven by `config.RESPONSE_LANGUAGE` so the Settings UI choice takes effect
    on the very next LLM call (no restart needed).
    """
    lang = (getattr(config, "RESPONSE_LANGUAGE", "auto") or "auto").strip().lower()
    if lang == "en":
        return "\n\nOutput language: English. Respond in English only."
    if lang == "zh":
        return "\n\nOutput language: Chinese. Respond in Chinese only."
    return ""  # auto → let each prompt's built-in rule decide


# ── LLM Engine ────────────────────────────────────────────────────────────


class LLMEngine:
    def __init__(self, rag_manager: RAGManager):
        self.rag = rag_manager
        self._init_clients()
        # Track in-flight streaming tasks so a UI Stop button / Esc can cancel them.
        self._active_tasks: dict[str, asyncio.Task] = {}
        # Multi-turn conversation history (Phase 2.2). Each entry is a (q, a) pair
        # of plain strings. Trimmed to CONTEXT_TURNS on insert.
        from collections import deque
        n = int(getattr(config, "CONTEXT_TURNS", 0) or 0)
        self._text_history: deque = deque(maxlen=max(n, 0))
        # One warning (not one per turn) when the algorithm cheatsheet is absent.
        self._algo_ref_warned = False

    def clear_history(self) -> None:
        """Drop multi-turn conversation history (called from UI Clear button)."""
        self._text_history.clear()

    async def preheat(self) -> None:
        """Fire-and-forget tiny request to warm the HTTPS pool. Throttled to once
        per 60s. Errors are swallowed."""
        import time
        now = time.monotonic()
        last = getattr(self, "_last_preheat", 0.0)
        if now - last < 60:
            return
        self._last_preheat = now
        try:
            await self.text_provider.chat_complete(
                [{"role": "user", "content": "ping"}],
                model=config.TEXT_MODEL,
                max_tokens=1,
                temperature=0.0,
            )
            logger.info("Preheat OK (%s).", config.TEXT_MODEL)
        except Exception as e:
            logger.info("Preheat skipped: %s", e)

    def register_task(self, kind: str, task: asyncio.Task) -> None:
        """Register a running stream task under 'text' | 'vision' so it can be cancelled.

        A *new* stream supersedes the previous one for that kind. It must not
        cancel `task` when the caller re-registers itself: `asr_router` is one
        long-lived task that awaits an answer per question, so cancelling the
        previous registration would cancel the router — killing every second
        question (the CancelledError is swallowed by the router's own handler, so
        the only symptom is an answer that never arrives).
        """
        prev = self._active_tasks.get(kind)
        if prev is not None and prev is not task and not prev.done():
            prev.cancel()
        self._active_tasks[kind] = task
        task.add_done_callback(lambda t, k=kind: self._active_tasks.pop(k, None) if self._active_tasks.get(k) is t else None)

    def cancel(self, kind: str = "all") -> int:
        """Cancel current text/vision/all streams. Returns number of tasks cancelled."""
        targets = ("text", "vision") if kind == "all" else (kind,)
        n = 0
        for k in targets:
            t = self._active_tasks.get(k)
            if t is not None and not t.done():
                t.cancel()
                n += 1
        return n

    def _init_clients(self):
        # Provider abstraction (text only — OCR is local, see src/ocr_client.py).
        self.text_provider = make_text_provider()
        # Last observed usage per pipeline (consumed by Phase 2.3 token counter).
        self.last_usage: dict[str, Usage] = {}
        self.ocr = self._make_ocr_client()

    @staticmethod
    def _make_ocr_client() -> OCRClient:
        return OCRClient(
            base_url=getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=getattr(config, "OCR_MODEL", "glm-ocr-optimized"),
            prompt=getattr(config, "OCR_PROMPT", "Transcribe all text in this image. Output only the text."),
            timeout=float(getattr(config, "OCR_TIMEOUT_SEC", 180.0) or 180.0),
        )

    def reload_clients(self):
        """
        Recreate LLM clients to apply runtime config changes (keys/models/providers).
        Safe to call between requests; in-flight requests keep using old clients.
        """
        self._init_clients()
        # Resize history deque if CONTEXT_TURNS changed.
        from collections import deque
        n = int(getattr(config, "CONTEXT_TURNS", 0) or 0)
        if n != (self._text_history.maxlen or 0):
            old = list(self._text_history)
            self._text_history = deque(old[-n:] if n else [], maxlen=max(n, 0))

    def _rag_min_score(self) -> float:
        return float(getattr(config, "RAG_MIN_SCORE", 0.32))

    async def _emit_usage(
        self,
        ui_queue: asyncio.Queue,
        kind: str,
        usage: Usage,
        *,
        total_ms: int,
        ttft_ms: int | None = None,
        ui_usage: Usage | None = None,
        note: str | None = None,
    ) -> None:
        """Publish one call's usage to the footer and the cost ledger.

        ``ui_usage`` is what the footer shows, ``usage`` is what the ledger
        records. They differ only for a *second* call in one turn (the code
        repair): the footer must show the turn's total, while a ledger row
        holding cumulative numbers would double-count the first call's tokens.
        """
        shown = ui_usage or usage
        _used = getattr(self.text_provider, "last_used", self.text_provider)
        payload = {
            "type": "usage",
            "kind": kind,
            "in": shown.in_tokens,
            "out": shown.out_tokens,
            "reasoning": shown.reasoning_tokens,
            "model": config.TEXT_MODEL,
            "total_ms": total_ms,
            "ttft_ms": ttft_ms,
            "provider": getattr(_used, "name", "?"),
        }
        if note:
            payload["note"] = note
        await ui_queue.put(payload)
        try:
            log_usage(
                {**payload, "in": usage.in_tokens, "out": usage.out_tokens,
                 "reasoning": usage.reasoning_tokens},
                path=Path(config.USAGE_LOG_PATH) if config.USAGE_LOG_PATH else None,
                enabled=config.USAGE_LOG_ENABLED,
            )
        except Exception:
            pass

    async def classify_question_llm(self, question: str) -> str:
        """LLM-based type; falls back to keyword classifier on error."""
        q = (question or "").strip()
        if not q:
            return "technical"
        try:
            text = await self.text_provider.chat_complete(
                [
                    {"role": "system", "content": _CLASSIFIER_SYSTEM},
                    {"role": "user", "content": q},
                ],
                model=config.TEXT_MODEL,
                max_tokens=int(getattr(config, "CLASSIFIER_MAX_TOKENS", 64)),
                temperature=float(getattr(config, "CLASSIFIER_TEMPERATURE", 0.1)),
            )
            text = (text or "").strip()
            data = _parse_json_object(text)
            t = _normalize_q_type(data.get("type"))
            if t:
                logger.info(f"LLM classified as {t}")
                return t
        except Exception as e:
            logger.warning(f"LLM classifier failed, using keywords: {e}")
        return classify_question(q)

    def _gather_context(self, q_type: str, question: str) -> TurnContext:
        """Everything injected ahead of the question on this turn.

        Behavioral and technical turns retrieve: ``knowledge/`` holds candidate
        background (resume, JD, notes) and ``RAG_MIN_SCORE`` gates what is
        relevant. Algorithm turns instead take the patterns cheatsheet whole —
        see ``ALGORITHM_KNOWLEDGE_FILE``. The two are kept in named fields
        because they are not the same thing: ``rag_hits`` reports retrieval, and
        a fixed injection counted into it would make the footer lie.
        """
        min_score = self._rag_min_score()
        q = question or ""

        if q_type == "behavioral":

            def _beh_filter(rel: str) -> bool:
                return _basename_kb(rel) in ("resume.md", "jd.md")

            return TurnContext(snippets=self.rag.search(
                q, top_k=3, min_score=min_score, source_filter=_beh_filter
            ))

        if q_type == "technical":

            def _tech_filter(rel: str) -> bool:
                return _basename_kb(rel) != "jd.md"

            return TurnContext(snippets=self.rag.search(
                q, top_k=3, min_score=min_score, source_filter=_tech_filter
            ))

        if q_type == "algorithm":
            return TurnContext(algorithm_ref=self._algorithm_reference())

        return TurnContext()

    def _algorithm_reference(self) -> str | None:
        """The patterns cheatsheet injected into algorithm turns, or None.

        A missing file is logged once per process, not once per turn: this used
        to warn on every algorithm question, which buries the signal it is
        trying to send.
        """
        name = getattr(config, "ALGORITHM_KNOWLEDGE_FILE", "algorithm.md")
        text, path = load_knowledge_file(
            getattr(config, "KNOWLEDGE_DIR", "knowledge"),
            name,
            max_chars=int(getattr(config, "ALGORITHM_KNOWLEDGE_MAX_CHARS", 8000) or 0),
        )
        if text:
            return text
        if not getattr(self, "_algo_ref_warned", False):
            self._algo_ref_warned = True
            logger.warning(
                "No algorithm cheatsheet found for ALGORITHM_KNOWLEDGE_FILE=%r under %s; "
                "algorithm turns go out without pattern reference material.",
                name, getattr(config, "KNOWLEDGE_DIR", "knowledge"),
            )
        return None

    def _build_interview_messages(
        self,
        q_type: str,
        question: str,
        context: TurnContext,
        *,
        from_screenshot: bool = False,
    ) -> list[dict]:
        system_prompt = get_prompt(q_type) + _lang_suffix()

        parts: list[str] = []
        if context.snippets:
            joined = "\n---\n".join(context.snippets[:3])
            parts.append(f"[Candidate background]\n{joined}")
        if context.algorithm_ref:
            parts.append(f"[Algorithm patterns reference]\n{context.algorithm_ref}")

        if from_screenshot:
            parts.append(
                "The question below was transcribed by a local OCR model from a "
                "screenshot of the interviewer's screen. The transcript can be "
                "incomplete, mis-recognized or mixed with UI chrome and toolbar "
                "text: ignore the chrome, repair the obvious OCR damage "
                "(brackets, colons, keywords, identifiers) and answer the "
                "question that was actually asked — a coding question is "
                "answered with code, as this route's format requires.\n"
                f"Transcribed question:\n{question}"
            )
        else:
            parts.append(f"Interviewer question:\n{question}")

        user_text = "\n\n".join(parts)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

    async def generate_answer_stream(
        self,
        question: str,
        ui_queue: asyncio.Queue,
        *,
        q_type: str | None = None,
    ):
        """Classify (optional) → RAG / algorithm prompt → LLM → stream tokens."""
        cur = asyncio.current_task()
        if cur is not None:
            self.register_task("text", cur)
        if q_type is None:
            q_type = await self.classify_question_llm(question)
        await self._answer_stream(question, ui_queue, q_type=q_type, kind="text")

    async def _answer_stream(
        self,
        question: str,
        ui_queue: asyncio.Queue,
        *,
        q_type: str,
        kind: str,
        from_screenshot: bool = False,
    ):
        """RAG / algorithm prompt → text LLM → stream tokens, usage and telemetry."""
        logger.info("[%s] %s", q_type.upper(), question)

        # Session recorder: pair this user turn with the assistant row that
        # the UI updater flushes when streaming finishes (v0.9.0 replay).
        try:
            from src.session_recorder import recorder as _rec
            _rec.log_user_turn(question, q_type=q_type, kind=kind)
        except Exception:
            pass

        context = self._gather_context(q_type, question)
        rag_hits = len(context.snippets)
        algo_chars = len(context.algorithm_ref or "")
        # Observability: notify UI of provider + injected-context counts before
        # the stream starts. `rag_hits` is retrieval only; the algorithm
        # cheatsheet is reported separately so neither number lies.
        try:
            await ui_queue.put({
                "type": "info",
                "provider": getattr(self.text_provider, "name", "?"),
                "rag_hits": rag_hits,
                "algo_chars": algo_chars,
            })
        except Exception:
            pass
        messages = self._build_interview_messages(
            q_type,
            question,
            context,
            from_screenshot=from_screenshot,
        )
        # Insert prior turns between system and the new user message.
        if self._text_history.maxlen and len(self._text_history) > 0:
            history_msgs: list[dict] = []
            for q, a in self._text_history:
                history_msgs.append({"role": "user", "content": q})
                history_msgs.append({"role": "assistant", "content": a})
            messages = [messages[0], *history_msgs, messages[1]]

        full_answer_parts: list[str] = []
        import time
        _t0 = time.monotonic()
        _ttft_ms: int | None = None
        try:
            # Wire failover hook so the UI sees provider switches.
            def _on_failover(prev, nxt, err):
                try:
                    ui_queue.put_nowait({
                        "type": "info",
                        "provider": getattr(nxt, "name", "?"),
                        "note": f"failover from {getattr(prev, 'name', '?')}",
                        "error": str(err)[:120],
                        "rag_hits": rag_hits,
                    })
                except Exception:
                    pass
            if hasattr(self.text_provider, "on_failover"):
                self.text_provider.on_failover = _on_failover
            agen = self.text_provider.chat_stream(
                messages,
                model=config.TEXT_MODEL,
                max_tokens=_answer_token_cap(),
                temperature=0.25,
            )
            _finish_reason: str | None = None
            async for delta in agen:
                fr = getattr(delta, "finish_reason", None)
                if fr:
                    _finish_reason = fr
                if delta.text:
                    if _ttft_ms is None:
                        _ttft_ms = int((time.monotonic() - _t0) * 1000)
                    full_answer_parts.append(delta.text)
                    await ui_queue.put({"type": "token", "text": delta.text})
                if delta.usage:
                    self.last_usage[kind] = delta.usage
                    await self._emit_usage(
                        ui_queue,
                        kind,
                        delta.usage,
                        total_ms=int((time.monotonic() - _t0) * 1000),
                        ttft_ms=_ttft_ms,
                    )
            # A provider that stopped because it ran out of output budget does not
            # signal it any other way than `finish_reason`. Without this the user
            # sees a code block that simply ends mid-line and assumes the model
            # gave up — or, worse, mistakes it for the whole answer. With
            # reasoning models this is the common case rather than a rare one:
            # the budget covers the hidden chain-of-thought *and* the answer, so
            # the trace can consume it before any answer is written.
            if _is_truncated(_finish_reason):
                cap = _answer_token_cap()
                reasoning = int(getattr(self.last_usage.get(kind), "reasoning_tokens", 0) or 0)
                logger.warning(
                    "Answer truncated (%s stop) with cap=%s; %s of the output tokens "
                    "were reasoning.", _finish_reason, cap, reasoning,
                )
                via = (
                    f" ({reasoning} of them spent on hidden reasoning before the answer)"
                    if reasoning else ""
                )
                if cap is None:
                    # Uncapped: the provider's own limit stopped it, so raising
                    # our setting is not the remedy.
                    detail = "the model's own output limit"
                else:
                    detail = f"the {cap}-token ANSWER_MAX_TOKENS cap"
                await ui_queue.put({
                    "type": "token",
                    "text": (f"\n\n[⚠️ truncated — stopped at {detail}{via}]"),
                })

        except asyncio.CancelledError:
            logger.info("Text stream cancelled.")
            await ui_queue.put({"type": "token", "text": "\n[⏹ stopped]"})
            raise
        except Exception as e:
            logger.error("LLM text error: %s", e)
            await ui_queue.put({"type": "token", "text": f"\n[⚠️ {e}]"})
        else:
            answer = "".join(full_answer_parts).strip()
            # The algorithm prompt requires a fenced block on *every* turn, and
            # the routing override sends code-shaped OCR transcripts here. A model
            # can still answer a coding question with prose alone, and the user
            # has no way to tell that from a deliberate answer — so ask once more,
            # explicitly, and stream whatever comes back.
            if q_type == "algorithm" and not self._has_code_block(answer):
                repaired = await self._repair_missing_code(
                    question,
                    answer,
                    ui_queue,
                    kind=kind,
                    t0=_t0,
                    from_screenshot=from_screenshot,
                )
                if repaired:
                    full_answer_parts.append(repaired)
                    answer = "".join(full_answer_parts).strip()
            # Only record successful completions in history.
            if self._text_history.maxlen and answer:
                self._text_history.append((question, answer))

    @staticmethod
    def _has_code_block(answer: str) -> bool:
        """Any fence at all counts.

        An *unterminated* fence is still code on screen — and it is truncation,
        which a re-ask cannot fix (that is the ANSWER_MAX_TOKENS path), so
        requiring a closed block here would only burn a second call per
        truncated answer.
        """
        return _FENCE in answer

    @staticmethod
    def _code_intro(draft: str) -> str:
        """Separator placed before a repaired block.

        ``[I]`` introduces the fence in the answer format, so it is re-emitted
        only when the draft did not already open the code section.
        """
        return "\n" if draft.rstrip().endswith("[I]") else "\n\n[I]\n"

    async def _repair_missing_code(
        self,
        question: str,
        draft: str,
        ui_queue: asyncio.Queue,
        *,
        kind: str,
        t0: float,
        from_screenshot: bool,
    ) -> str:
        """Re-ask for the code block an algorithm turn failed to emit.

        One attempt, no loop: if the second call also refuses to produce code,
        the notice stays in the answer and the user knows the answer is prose
        rather than silently trusting it. Returns the text appended to the
        answer (empty when the repair produced nothing usable).
        """
        import time
        logger.warning(
            "Algorithm answer carried no code block (%d chars); asking once more.", len(draft)
        )
        await ui_queue.put({
            "type": "token",
            "text": "\n\n[⚠️ 答案缺代码块 — 重新生成中…]",
        })

        ocr_note = (
            "The question was transcribed by OCR from a screenshot and may be "
            "truncated or garbled; infer the intended problem and answer that.\n"
            if from_screenshot else ""
        )
        messages = [
            {"role": "system", "content": get_prompt("algorithm") + _lang_suffix()},
            {
                "role": "user",
                "content": (
                    f"{ocr_note}"
                    "Your previous answer to the interview question below contained no code "
                    "block. A fenced code block is mandatory on this route — it is the part "
                    "the candidate needs.\n\n"
                    f"Question:\n{question}\n\n"
                    f"Your previous answer:\n{draft[:1500]}\n\n"
                    "Reply with exactly ONE fenced code block and nothing else: no tags, no "
                    "prose, no explanation. Use ```python unless another language is clearly "
                    "required. The code must be complete and runnable."
                ),
            },
        ]

        parts: list[str] = []
        ttft_ms: int | None = None
        try:
            agen = self.text_provider.chat_stream(
                messages,
                model=config.TEXT_MODEL,
                max_tokens=_answer_token_cap(),
                temperature=float(getattr(config, "REPAIR_TEMPERATURE", 0.0) or 0.0),
            )
            prefix = self._code_intro(draft)
            await ui_queue.put({"type": "token", "text": prefix})
            parts.append(prefix)
            async for delta in agen:
                if delta.text:
                    if ttft_ms is None:
                        ttft_ms = int((time.monotonic() - t0) * 1000)
                    parts.append(delta.text)
                    await ui_queue.put({"type": "token", "text": delta.text})
                if delta.usage:
                    previous = self.last_usage.get(kind)
                    self.last_usage[kind] = self._add_usage(previous, delta.usage)
                    await self._emit_usage(
                        ui_queue,
                        kind,
                        delta.usage,
                        total_ms=int((time.monotonic() - t0) * 1000),
                        ttft_ms=ttft_ms,
                        ui_usage=self.last_usage[kind],
                        note="code repair",
                    )
        except asyncio.CancelledError:
            logger.info("Code repair cancelled.")
            await ui_queue.put({"type": "token", "text": "\n[⏹ stopped]"})
            raise
        except Exception as e:
            logger.error("Code repair failed: %s", e)
            await ui_queue.put({
                "type": "token",
                "text": f"\n[⚠️ 重新生成失败（{e}）— 请再按一次截图键]",
            })
            return ""

        repaired = "".join(parts)
        if not self._has_code_block(repaired):
            logger.warning("Code repair also produced no fenced block.")
            await ui_queue.put({
                "type": "token",
                "text": "\n[⚠️ 仍未生成代码块 — 可再按一次截图键重试]",
            })
            return ""
        return repaired

    @staticmethod
    def _add_usage(a: Usage | None, b: Usage) -> Usage:
        """Sum two calls' usage — the footer reports the turn, not the last call."""
        if a is None:
            return b
        return Usage(
            in_tokens=a.in_tokens + b.in_tokens,
            out_tokens=a.out_tokens + b.out_tokens,
            reasoning_tokens=a.reasoning_tokens + b.reasoning_tokens,
        )

    @staticmethod
    def _prepare_ocr_image(image_bytes: bytes) -> bytes:
        """Fit a screenshot for OCR and re-encode as JPEG.

        GLM-OCR's accuracy and speed are driven by pixel dimensions. Scaling is
        bounded in *both* directions:

        * **down** to ``OCR_MAX_DIMENSION`` — the vision prefill is the dominant
          cost of a full-screen shot (~9 s at 1024 px long edge, ~25 s at
          900 px on a cropped region).
        * **up** to ``OCR_MIN_DIMENSION`` — a small drag-selected region sent at
          its native size (e.g. 500x260) is unreadable for the model, which then
          latches onto a token group and repeats it; upscaling makes the text
          legible again and the run terminates on its own.
        """
        img = Image.open(io.BytesIO(image_bytes))
        max_dim = max(256, int(getattr(config, "OCR_MAX_DIMENSION", 1024) or 1024))
        min_dim = max(0, int(getattr(config, "OCR_MIN_DIMENSION", 0) or 0))
        longest = max(img.size)
        if longest > max_dim:
            ratio = max_dim / longest
        elif min_dim and longest < min_dim:
            ratio = min_dim / longest
        else:
            ratio = 1.0
        if ratio != 1.0:
            img = img.resize(
                (max(1, round(img.width * ratio)), max(1, round(img.height * ratio))),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()

    # A transcript this short that has already been cut back to its first
    # repetition almost certainly lost content — warn rather than answer from it.
    _MIN_TRUSTED_TRANSCRIPT_CHARS = 120

    async def _ocr_screenshot(self, prepared: bytes, ui_queue: asyncio.Queue) -> tuple[str, str]:
        """Stream a screenshot through local OCR; return ``(text, warning)``.

        Progress is surfaced as ``status`` events (model name + recognized tail)
        so the overlay shows something during the CPU-bound recognition pass.

        Two different shortfalls are handled differently, because reporting them
        the same way was a false alarm:

        * **The model began repeating itself.** Measured on three real
          configurations: GLM-OCR transcribes the screenshot *correctly and
          completely* and then re-emits the page inside a code fence, or invents
          a code block and re-emits that. The recognition is intact, so the
          degenerate tail is trimmed (``trim_repetition``) and no warning is
          raised — the transcript is not incomplete.
        * **The token cap or the wall-clock budget cut generation.** This one can
          genuinely be missing its ending, so the warning stands.
        """
        # Single choke point for talking to Ollama, so the "is it up?" wait lives
        # here. A screenshot taken seconds after launch can arrive while a local
        # Ollama is still starting; waiting (a cached no-op once the server
        # answers) turns that into a slightly slower answer instead of a
        # ConnectError in the overlay.
        if not await ollama_boot.ensure_available_for_config():
            logger.warning("Proceeding without a reachable Ollama; OCR will likely fail.")

        parts: list[str] = []
        note = ""
        stop_kind = ""
        try:
            async for chunk in self.ocr.recognize_stream(prepared):
                parts.append(chunk)
                tail = "".join(parts)[-120:].replace("\n", " ").strip()
                try:
                    await ui_queue.put({"type": "status", "text": f"OCR · {tail}"})
                except Exception:
                    pass
        except OCRTruncated as e:
            stop_kind = getattr(e, "kind", "")
            note = str(e)
            logger.info("OCR stopped early (%s): %s", stop_kind or "truncated", e)

        raw = "".join(parts).strip()
        text, removed = trim_repetition(raw)
        if removed:
            logger.info(
                "OCR: trimmed %d chars (%.0f%%) of repeated output — the recognition "
                "before it is kept (stopped by: %s).",
                removed, 100 * removed / max(1, len(raw)), stop_kind or "n/a",
            )

        warning = ""
        if stop_kind == "loop":
            # The tail was repetition, not recognition; after trimming there is
            # nothing left to warn about — unless almost nothing survived.
            if len(text) < self._MIN_TRUSTED_TRANSCRIPT_CHARS:
                warning = (
                    "OCR stopped after only "
                    f"{len(text)} chars — the screenshot may have been unreadable"
                )
        elif note:
            warning = f"{note}. The text may be missing its ending."

        if warning:
            try:
                await ui_queue.put({"type": "status", "text": f"OCR · {warning}"})
            except Exception:
                pass
        return text, warning

    async def generate_vision_answer_stream(
        self, image_bytes: bytes, ui_queue: asyncio.Queue
    ) -> str:
        """Screenshot → local OCR → text LLM answer.

        Returns the recognized text (the question that was answered) so callers
        can log or display it. Emits the same ``token``/``usage``/``info`` events
        as the text pipeline, plus:
          * ``status``       — OCR progress (model + recognized tail)
          * ``answer_start`` — recognized question is final; begin the answer turn
        """
        cur = asyncio.current_task()
        if cur is not None:
            self.register_task("vision", cur)
        logger.info("Screenshot OCR requested (local model=%s).", self.ocr.model)

        try:
            prepared = await asyncio.to_thread(self._prepare_ocr_image, image_bytes)
            logger.info(
                "Screenshot prepared for OCR: %dKB → %dKB (max edge %dpx)",
                len(image_bytes) // 1024,
                len(prepared) // 1024,
                int(getattr(config, "OCR_MAX_DIMENSION", 1024) or 1024),
            )
            try:
                await ui_queue.put({"type": "status", "text": f"OCR · {self.ocr.model}"})
            except Exception:
                pass

            recognized, ocr_note = await self._ocr_screenshot(prepared, ui_queue)
            if not recognized:
                raise OCRError(
                    "GLM-OCR recognized no text in this screenshot — try a tighter crop."
                )
            logger.info("OCR recognized %d chars.", len(recognized))

            # Persist the screenshot *before* the user turn: replay pairs a
            # `user` row with the `screenshot` row that follows it.
            try:
                from src.session_recorder import recorder as _rec
                _rec.log_screenshot(prepared)
            except Exception:
                pass

            q_type = await self.classify_question_llm(recognized)
            if q_type != "algorithm" and looks_like_code(recognized):
                # The screenshot shows source code. Whatever the classifier made
                # of the *words*, the answer has to be code — the technical
                # prompt's one-line format would return no code block at all.
                logger.info(
                    "OCR transcript is code-shaped — overriding %s route with algorithm.", q_type
                )
                q_type = "algorithm"
            try:
                await ui_queue.put({
                    "type": "answer_start",
                    "question": recognized,
                    "q_type": q_type,
                })
                if ocr_note:
                    # Emitted as answer text, not as a status: a status is wiped
                    # by the very next answer_start, so a screenshot whose code
                    # was cut off would be answered as though it were complete.
                    await ui_queue.put({
                        "type": "token",
                        "text": f"[⚠️ OCR incomplete — {ocr_note}]\n",
                    })
            except Exception:
                pass

            await self._answer_stream(
                recognized, ui_queue, q_type=q_type, kind="vision",
                from_screenshot=True,
            )
            return recognized

        except asyncio.CancelledError:
            logger.info("Screenshot pipeline cancelled.")
            await ui_queue.put({"type": "token", "text": "\n[⏹ stopped]"})
            raise
        except OCRError as e:
            logger.error("Screenshot OCR failed: %s", e)
            await ui_queue.put({"type": "token", "text": f"\n[⚠️ OCR Error: {e}]"})
            return ""
        except Exception as e:
            logger.error("Screenshot pipeline error: %s", e)
            await ui_queue.put({"type": "token", "text": f"\n[⚠️ OCR Error: {e}]"})
            return ""
