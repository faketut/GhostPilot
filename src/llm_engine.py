"""
LLM Engine
----------
Routes classified questions to the appropriate LLM with prompts loaded from
the `prompts/` directory via PromptLoader.

Text route  → DeepSeek-V3 (or any OpenAI-compatible model)
Vision route → GPT-4o (or any vision-capable model)
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

from src.config import config
from src.rag_manager import RAGManager
from src.prompt_loader import get_prompt
from src.llm import make_text_provider, make_vision_provider, Usage
from src.usage_log import log_usage

logger = logging.getLogger(__name__)

# ── Question classifier (keyword fallback) ────────────────────────────────

_BEHAVIORAL_KW = [
    "经历", "举例", "讲一个", "描述一次", "给我讲", "冲突", "挑战",
    "为什么加入", "为什么选择", "为什么想来", "为何应聘", "自我介绍",
    "动机", "还有什么补充", "对我们公司", "对本公司", "了解我们公司",
    "describe", "tell me about", "conflict", "challenge", "walk me through",
    "give an example", "a time when",
    # Motivation / fit / open pitch (not episodic STAR)
    "why do you want", "why us", "why this company", "why our company",
    "why are you interested", "why join", "what attracts you",
    "anything else", "like to share about yourself", "strong candidate",
    "why should we hire", "tell us about yourself", "introduce yourself",
    "what do you know about us", "cultural fit", "values align",
    "why this role", "why the role", "what interests you about",
]
_ALGORITHM_KW = [
    "算法", "数组", "链表", "树", "图", "动态规划", "排序", "搜索",
    "复杂度", "leetcode", "二叉", "栈", "队列", "堆", "哈希",
    "two pointer", "sliding window", "binary search", "dfs", "bfs",
    "dp", "recursion", "backtrack",
]


def classify_question(question: str) -> str:
    """
    Returns one of: 'behavioral', 'algorithm', 'technical'.
    Priority: behavioral > algorithm > technical (default).
    """
    q = question.lower()
    if any(kw in q for kw in _BEHAVIORAL_KW):
        return "behavioral"
    if any(kw in q for kw in _ALGORITHM_KW):
        return "algorithm"
    return "technical"


def _basename_kb(rel: str) -> str:
    return rel.replace("\\", "/").split("/")[-1].lower()


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


_VISION_STEP_A_SYSTEM = """You analyze one screenshot from a technical interview.
Reply with JSON only, no markdown. Schema:
{"visible_question":"<plain text of the main question on screen, or empty string>","type":"behavioral|technical|algorithm"}

Use the same definitions as a text classifier: behavioral = soft skills, STAR stories, motivation/why-us/fit/self-intro; technical = concepts/design/tools; algorithm = coding/DSA."""


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
        # Vision history (Phase 3.2): last N (timestamp, compressed_jpeg, question, answer).
        vn = int(getattr(config, "VISION_HISTORY", 5) or 0)
        self._vision_history: deque = deque(maxlen=max(vn, 0))

    def clear_history(self) -> None:
        """Drop multi-turn conversation history (called from UI Clear button)."""
        self._text_history.clear()
        self._vision_history.clear()

    def recent_vision(self) -> list[tuple]:
        """Return a snapshot of recent vision turns: [(ts, image_bytes, q, a), ...]."""
        return list(self._vision_history)

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
        """Register a running stream task under 'text' | 'vision' so it can be cancelled."""
        prev = self._active_tasks.get(kind)
        if prev is not None and not prev.done():
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
        # New provider abstraction (Phase 2.1).
        self.text_provider = make_text_provider()
        self.vision_provider = make_vision_provider()
        # Last observed usage per pipeline (consumed by Phase 2.3 token counter).
        self.last_usage: dict[str, Usage] = {}

        # ── Legacy raw-SDK client kept only for vision step A (OpenAI family) ──
        # The streaming text path and vision step B both go through the provider
        # abstraction now; only `_vision_step_a_openai` still calls the OpenAI
        # SDK directly because it needs a one-shot JSON response. Routing it
        # through the provider abstraction is tracked as future work.
        def _is_deepseek(m: str) -> bool:
            return "deepseek" in (m or "").lower()

        vm = (config.VISION_MODEL or "").lower()
        self.vision_client = None
        if vm.startswith("gemini"):
            self._vision_provider = "gemini"
        elif _is_deepseek(config.VISION_MODEL):
            self._vision_provider = "deepseek"
        else:
            self._vision_provider = "openai_compatible"
            self.vision_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

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

    def _gather_context(self, q_type: str, question: str) -> tuple[list[str], str | None]:
        """
        Returns (rag_snippets, algorithm_markdown_or_none).
        Algorithm mode skips vector RAG and loads knowledge/algorithm.md.
        """
        min_score = self._rag_min_score()
        q = question or ""

        if q_type == "behavioral":

            def _beh_filter(rel: str) -> bool:
                return _basename_kb(rel) in ("resume.md", "jd.md")

            snippets = self.rag.search(
                q, top_k=3, min_score=min_score, source_filter=_beh_filter
            )
            return snippets, None

        if q_type == "technical":

            def _tech_filter(rel: str) -> bool:
                return _basename_kb(rel) != "jd.md"

            snippets = self.rag.search(
                q, top_k=3, min_score=min_score, source_filter=_tech_filter
            )
            return snippets, None

        if q_type == "algorithm":
            path = Path(getattr(config, "KNOWLEDGE_DIR", "knowledge")) / "algorithm.md"
            blob: str | None = None
            if path.exists():
                try:
                    blob = path.read_text(encoding="utf-8", errors="ignore").strip()
                except OSError as e:
                    logger.warning("Could not read algorithm.md: %s", e)
            else:
                logger.warning("algorithm.md not found at %s", path.resolve())
            if not blob:
                blob = None
            return [], blob

        return [], None

    def _build_interview_messages(
        self,
        q_type: str,
        question: str,
        context_snippets: list[str],
        algorithm_blob: str | None,
        *,
        vision_mode: bool,
        visible_question: str = "",
    ) -> list[dict]:
        system_prompt = get_prompt(q_type) + _lang_suffix()

        parts: list[str] = []
        if context_snippets:
            joined = "\n---\n".join(context_snippets[:3])
            parts.append(f"[Candidate background]\n{joined}")
        if algorithm_blob:
            parts.append(f"[Reference: knowledge/algorithm.md]\n{algorithm_blob}")

        if vision_mode:
            vq = (visible_question or "").strip()
            parts.append(
                "A screenshot of the interview is attached. "
                "Base your answer on both the image and any context above.\n"
                f"Transcribed visible question (may be empty): {vq!r}"
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
        """Classify (optional) → RAG / algorithm.md → LLM → stream tokens."""
        cur = asyncio.current_task()
        if cur is not None:
            self.register_task("text", cur)
        if q_type is None:
            q_type = await self.classify_question_llm(question)
        logger.info("[%s] %s", q_type.upper(), question)

        snippets, algo_blob = self._gather_context(q_type, question)
        rag_hits = len(snippets) + (1 if algo_blob else 0)
        # Observability: notify UI of provider + RAG hit count before stream starts.
        try:
            await ui_queue.put({
                "type": "info",
                "provider": getattr(self.text_provider, "name", "?"),
                "rag_hits": rag_hits,
            })
        except Exception:
            pass
        messages = self._build_interview_messages(
            q_type,
            question,
            snippets,
            algo_blob,
            vision_mode=False,
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
                max_tokens=450,
                temperature=0.25,
            )
            async for delta in agen:
                if delta.text:
                    if _ttft_ms is None:
                        _ttft_ms = int((time.monotonic() - _t0) * 1000)
                    full_answer_parts.append(delta.text)
                    await ui_queue.put({"type": "token", "text": delta.text})
                if delta.usage:
                    self.last_usage["text"] = delta.usage
                    _used = getattr(self.text_provider, "last_used", self.text_provider)
                    _payload = {"type": "usage", "kind": "text",
                                "in": delta.usage.in_tokens,
                                "out": delta.usage.out_tokens,
                                "model": config.TEXT_MODEL,
                                "total_ms": int((time.monotonic() - _t0) * 1000),
                                "ttft_ms": _ttft_ms,
                                "provider": getattr(_used, "name", "?")}
                    await ui_queue.put(_payload)
                    try:
                        log_usage(
                            _payload,
                            path=Path(config.USAGE_LOG_PATH) if config.USAGE_LOG_PATH else None,
                            enabled=config.USAGE_LOG_ENABLED,
                        )
                    except Exception:
                        pass

        except asyncio.CancelledError:
            logger.info("Text stream cancelled.")
            await ui_queue.put({"type": "token", "text": "\n[⏹ stopped]"})
            raise
        except Exception as e:
            logger.error("LLM text error: %s", e)
            await ui_queue.put({"type": "token", "text": f"\n[⚠️ {e}]"})
        else:
            # Only record successful completions in history.
            if self._text_history.maxlen:
                answer = "".join(full_answer_parts).strip()
                if answer:
                    self._text_history.append((question, answer))

    @staticmethod
    def _compress_jpeg(image_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes))
        max_w = 1920
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()

    async def _vision_step_a_openai(self, compressed: bytes) -> tuple[str, str]:
        b64 = base64.b64encode(compressed).decode("utf-8")
        messages = [
            {"role": "system", "content": _VISION_STEP_A_SYSTEM + _lang_suffix()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this screenshot and output the JSON described in your instructions.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]
        resp = await self.vision_client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=messages,
            max_tokens=256,
            temperature=0.1,
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            data = _parse_json_object(text)
            vq = str(data.get("visible_question", "") or "")
            t = _normalize_q_type(data.get("type")) or classify_question(vq or "interview")
            return t, vq
        except Exception as e:
            logger.warning("Vision step A (OpenAI) JSON parse failed: %s", e)
            return "technical", ""

    async def _vision_step_a_gemini(self, compressed: bytes) -> tuple[str, str]:
        from google import genai
        from google.genai import types

        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is empty. Please set it in .env / Settings.")

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        gemini_config = types.GenerateContentConfig(
            system_instruction=_VISION_STEP_A_SYSTEM + _lang_suffix(),
            max_output_tokens=256,
            temperature=0.1,
        )
        resp = await client.aio.models.generate_content(
            model=config.VISION_MODEL,
            contents=[
                "Analyze this screenshot and output the JSON described in your instructions.",
                types.Part.from_bytes(data=compressed, mime_type="image/jpeg"),
            ],
            config=gemini_config,
        )
        text = (resp.text or "").strip()
        try:
            data = _parse_json_object(text)
            vq = str(data.get("visible_question", "") or "")
            t = _normalize_q_type(data.get("type")) or classify_question(vq or "interview")
            return t, vq
        except Exception as e:
            logger.warning("Vision step A (Gemini) JSON parse failed: %s", e)
            return "technical", ""

    def _build_vision_payload(self, system_prompt: str, user_text: str, compressed: bytes):
        """Return a payload shaped for the active vision provider family.

        Vision providers don't share a payload format (gemini takes a list of
        parts; openai-compat takes a chat-style messages list). So we build
        the right shape here based on provider name. Failover between
        different families will fail at request time; see README.
        """
        name = (getattr(self.vision_provider, "name", "") or self._vision_provider or "").lower()
        if "gemini" in name:
            from google.genai import types
            return [
                user_text,
                types.Part.from_bytes(data=compressed, mime_type="image/jpeg"),
            ]
        # OpenAI-compatible (incl. fallbacks)
        b64 = base64.b64encode(compressed).decode("utf-8")
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

    async def _vision_step_b(
        self, compressed: bytes, q_type: str, visible_question: str, ui_queue: asyncio.Queue,
    ):
        """Unified vision step B that routes through self.vision_provider."""
        rag_q = visible_question.strip() or "interview screenshot"
        snippets, algo_blob = self._gather_context(q_type, rag_q)
        base = self._build_interview_messages(
            q_type,
            rag_q,
            snippets,
            algo_blob,
            vision_mode=True,
            visible_question=visible_question,
        )
        system_prompt = base[0]["content"]
        user_text = base[1]["content"]
        payload = self._build_vision_payload(system_prompt, user_text, compressed)

        # Wire failover hook so vision switches are visible in the footer.
        def _on_failover(prev, nxt, err):
            try:
                ui_queue.put_nowait({
                    "type": "info",
                    "kind": "vision",
                    "provider": getattr(nxt, "name", "?"),
                    "note": f"failover from {getattr(prev, 'name', '?')}",
                    "error": str(err)[:120],
                })
            except Exception:
                pass
        if hasattr(self.vision_provider, "on_failover"):
            self.vision_provider.on_failover = _on_failover

        agen = self.vision_provider.vision_stream(
            payload,
            model=config.VISION_MODEL,
            system_prompt=system_prompt,
            max_tokens=550,
            temperature=0.25,
        )
        async for delta in agen:
            if delta.text:
                await ui_queue.put({"type": "token", "text": delta.text})

    async def generate_vision_answer_stream(
        self, image_bytes: bytes, ui_queue: asyncio.Queue
    ):
        """
        Compress screenshot → classify + extract question (vision) →
        RAG / algorithm.md → final vision stream.
        """
        cur = asyncio.current_task()
        if cur is not None:
            self.register_task("vision", cur)
        logger.info("Vision answer requested (two-step).")
        import time
        _vt0 = time.monotonic()
        try:
            compressed = self._compress_jpeg(image_bytes)
            logger.info(
                "Screenshot compressed: %dKB → %dKB",
                len(image_bytes) // 1024,
                len(compressed) // 1024,
            )

            if self._vision_provider == "deepseek":
                raise ValueError(
                    "DeepSeek /chat/completions does not support OpenAI-style image inputs "
                    "(message content blocks with type=image_url). "
                    "Use Gemini (VISION_MODEL=gemini-1.5-flash) or OpenAI vision (gpt-4o), "
                    "or configure a vision provider that supports multimodal chat."
                )

            if self._vision_provider == "gemini":
                try:
                    from google import genai  # noqa: F401
                except Exception as e:
                    raise RuntimeError(f"Failed to import google-genai SDK: {e}") from e
                q_type, visible_question = await self._vision_step_a_gemini(compressed)
            else:
                q_type, visible_question = await self._vision_step_a_openai(compressed)

            logger.info("Vision step A: type=%s visible_question_len=%d", q_type, len(visible_question))

            await self._vision_step_b(compressed, q_type, visible_question, ui_queue)

            # Record entry on successful completion (Phase 3.2).
            if self._vision_history.maxlen:
                import time
                self._vision_history.append((time.time(), compressed, visible_question, ""))

            # Latency telemetry for the vision pipeline.
            try:
                _vpayload = {
                    "type": "latency",
                    "kind": "vision",
                    "total_ms": int((time.monotonic() - _vt0) * 1000),
                    "provider": getattr(self.vision_provider, "name", self._vision_provider),
                }
                await ui_queue.put(_vpayload)
                try:
                    log_usage(
                        _vpayload,
                        path=Path(config.USAGE_LOG_PATH) if config.USAGE_LOG_PATH else None,
                        enabled=config.USAGE_LOG_ENABLED,
                    )
                except Exception:
                    pass
            except Exception:
                pass

        except asyncio.CancelledError:
            logger.info("Vision stream cancelled.")
            await ui_queue.put({"type": "token", "text": "\n[⏹ stopped]"})
            raise
        except Exception as e:
            logger.error("Vision LLM error: %s", e)
            await ui_queue.put({"type": "token", "text": f"\n[⚠️ Vision Error: {e}]"})
