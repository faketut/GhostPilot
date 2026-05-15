"""
LLM Engine
----------
Routes classified questions to the appropriate LLM with prompts loaded from
the `prompts/` directory via PromptLoader.

Text route  → DeepSeek-V3 (or any OpenAI-compatible model)
Vision route → GPT-4o (or any vision-capable model)
"""

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

logger = logging.getLogger(__name__)

# ── Question classifier (keyword fallback) ────────────────────────────────

_BEHAVIORAL_KW = [
    "经历", "举例", "讲一个", "描述一次", "给我讲", "冲突", "挑战",
    "describe", "tell me about", "conflict", "challenge", "walk me through",
    "give an example", "a time when",
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
- behavioral: soft skills, teamwork, conflict, leadership, "tell me about a time", STAR-style stories.
- technical: CS concepts, system design, frameworks, tools, debugging — not mainly "write this algorithm from scratch".
- algorithm: coding / DSA / LeetCode style, implementations, complexity, specific algorithm names."""


_VISION_STEP_A_SYSTEM = """You analyze one screenshot from a technical interview.
Reply with JSON only, no markdown. Schema:
{"visible_question":"<plain text of the main question on screen, or empty string>","type":"behavioral|technical|algorithm"}

Use the same definitions as a text classifier: behavioral = soft/STAR; technical = concepts/design/tools; algorithm = coding/DSA."""


_ENGLISH_SUFFIX = "\n\nOutput language: English. Respond in English only."


# ── LLM Engine ────────────────────────────────────────────────────────────


class LLMEngine:
    def __init__(self, rag_manager: RAGManager):
        self.rag = rag_manager
        self._init_clients()

    def _init_clients(self):
        def _is_deepseek_model(model_name: str) -> bool:
            return "deepseek" in (model_name or "").lower()

        # Text LLM — DeepSeek-V3 by default (OpenAI-compatible API)
        if _is_deepseek_model(config.TEXT_MODEL):
            self.text_client = AsyncOpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1",
            )
        else:
            self.text_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

        # Vision LLM — choose provider based on model name
        vm = (config.VISION_MODEL or "").lower()
        self._vision_provider = "openai_compatible"
        self.vision_client = None

        if vm.startswith("gemini"):
            self._vision_provider = "gemini"
        elif _is_deepseek_model(config.VISION_MODEL):
            self._vision_provider = "deepseek"
            self.vision_client = AsyncOpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com/v1",
            )
        else:
            self._vision_provider = "openai_compatible"
            self.vision_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    def reload_clients(self):
        """
        Recreate LLM clients to apply runtime config changes (keys/models/providers).
        Safe to call between requests; in-flight requests keep using old clients.
        """
        self._init_clients()

    def _rag_min_score(self) -> float:
        return float(getattr(config, "RAG_MIN_SCORE", 0.32))

    async def classify_question_llm(self, question: str) -> str:
        """LLM-based type; falls back to keyword classifier on error."""
        q = (question or "").strip()
        if not q:
            return "technical"
        try:
            resp = await self.text_client.chat.completions.create(
                model=config.TEXT_MODEL,
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM},
                    {"role": "user", "content": q},
                ],
                max_tokens=int(getattr(config, "CLASSIFIER_MAX_TOKENS", 64)),
                temperature=float(getattr(config, "CLASSIFIER_TEMPERATURE", 0.1)),
            )
            text = (resp.choices[0].message.content or "").strip()
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
        system_prompt = get_prompt(q_type) + _ENGLISH_SUFFIX

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
        if q_type is None:
            q_type = await self.classify_question_llm(question)
        logger.info("[%s] %s", q_type.upper(), question)

        snippets, algo_blob = self._gather_context(q_type, question)
        messages = self._build_interview_messages(
            q_type,
            question,
            snippets,
            algo_blob,
            vision_mode=False,
        )

        try:
            stream = await self.text_client.chat.completions.create(
                model=config.TEXT_MODEL,
                messages=messages,
                stream=True,
                max_tokens=450,
                temperature=0.25,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta is not None:
                    await ui_queue.put({"type": "token", "text": delta})

        except Exception as e:
            logger.error("LLM text error: %s", e)
            await ui_queue.put({"type": "token", "text": f"\n[⚠️ {e}]"})

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
            {"role": "system", "content": _VISION_STEP_A_SYSTEM + _ENGLISH_SUFFIX},
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
            system_instruction=_VISION_STEP_A_SYSTEM + _ENGLISH_SUFFIX,
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

    async def _vision_step_b_openai(
        self, compressed: bytes, q_type: str, visible_question: str, ui_queue: asyncio.Queue
    ):
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
        b64 = base64.b64encode(compressed).decode("utf-8")
        messages = [
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
        stream = await self.vision_client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=messages,
            stream=True,
            max_tokens=550,
            temperature=0.25,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta is not None:
                await ui_queue.put({"type": "token", "text": delta})

    async def _vision_step_b_gemini(
        self, compressed: bytes, q_type: str, visible_question: str, ui_queue: asyncio.Queue
    ):
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)
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
        gemini_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=550,
            temperature=0.25,
        )
        stream = await client.aio.models.generate_content_stream(
            model=config.VISION_MODEL,
            contents=[
                user_text,
                types.Part.from_bytes(data=compressed, mime_type="image/jpeg"),
            ],
            config=gemini_config,
        )
        async for chunk in stream:
            if chunk.text:
                await ui_queue.put({"type": "token", "text": chunk.text})

    async def generate_vision_answer_stream(
        self, image_bytes: bytes, ui_queue: asyncio.Queue
    ):
        """
        Compress screenshot → classify + extract question (vision) →
        RAG / algorithm.md → final vision stream.
        """
        logger.info("Vision answer requested (two-step).")
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

            if self._vision_provider == "gemini":
                await self._vision_step_b_gemini(compressed, q_type, visible_question, ui_queue)
            else:
                await self._vision_step_b_openai(compressed, q_type, visible_question, ui_queue)

        except Exception as e:
            logger.error("Vision LLM error: %s", e)
            await ui_queue.put({"type": "token", "text": f"\n[⚠️ Vision Error: {e}]"})
