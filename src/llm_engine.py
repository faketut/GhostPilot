"""
LLM Engine
----------
Routes classified questions to the appropriate LLM with prompts loaded from
the `prompts/` directory via PromptLoader.

Text route  → DeepSeek-V3 (or any OpenAI-compatible model)
Vision route → GPT-4o (or any vision-capable model)
"""

import asyncio
import logging
import base64
import io
from openai import AsyncOpenAI
from PIL import Image

from src.config import config
from src.rag_manager import RAGManager
from src.prompt_loader import get_prompt

logger = logging.getLogger(__name__)

# ── Question classifier ───────────────────────────────────────────────────

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


# ── LLM Engine ────────────────────────────────────────────────────────────

class LLMEngine:
    def __init__(self, rag_manager: RAGManager):
        self.rag = rag_manager

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

    # ── Text answer ───────────────────────────────────────────────────────

    def _build_messages(
        self, q_type: str, question: str, context_snippets: list[str]
    ) -> list[dict]:
        system_prompt = get_prompt(q_type)

        # Inject only the top-3 most relevant resume/JD snippets (minimisation principle)
        context_block = ""
        if context_snippets:
            joined = "\n---\n".join(context_snippets[:3])
            context_block = f"\n\n[候选人相关背景]\n{joined}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context_block}\n\n面试官问题：{question}"},
        ]

    async def generate_answer_stream(self, question: str, ui_queue: asyncio.Queue):
        """Classify → RAG → LLM → stream tokens to ui_queue."""
        q_type = classify_question(question)
        logger.info(f"[{q_type.upper()}] {question}")

        context = self.rag.search(question)
        messages = self._build_messages(q_type, question, context)

        try:
            stream = await self.text_client.chat.completions.create(
                model=config.TEXT_MODEL,
                messages=messages,
                stream=True,
                max_tokens=450,
                temperature=0.25,   # Low temp → factual, consistent answers
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta is not None:
                    await ui_queue.put({"type": "token", "text": delta})

        except Exception as e:
            logger.error(f"LLM text error: {e}")
            await ui_queue.put({"type": "token", "text": f"\n[⚠️ {e}]"})

    # ── Vision answer ─────────────────────────────────────────────────────

    async def generate_vision_answer_stream(
        self, image_bytes: bytes, ui_queue: asyncio.Queue
    ):
        """
        Compress screenshot → send to vision model → stream tokens to ui_queue.
        Image bytes never touch disk.
        """
        logger.info("Vision answer requested.")
        try:
            # ── Compress: cap at 1920px wide, JPEG q=85 ────────────────
            img = Image.open(io.BytesIO(image_bytes))
            max_w = 1920
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize(
                    (max_w, int(img.height * ratio)), Image.LANCZOS
                )
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
            compressed = buf.getvalue()
            logger.info(
                f"Screenshot compressed: {len(image_bytes)//1024}KB → "
                f"{len(compressed)//1024}KB"
            )

            # ── Build vision messages ──────────────────────────────────
            system_prompt = get_prompt("vision")
            user_text = "请分析这道面试题并给出解答："

            if self._vision_provider == "gemini":
                if not config.GEMINI_API_KEY:
                    raise ValueError("GEMINI_API_KEY is empty. Please set it in .env / Settings.")

                try:
                    from google import genai
                    from google.genai import types
                except Exception as e:
                    raise RuntimeError(f"Failed to import google-genai SDK: {e}") from e

                client = genai.Client(api_key=config.GEMINI_API_KEY)

                gemini_config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=550,
                    temperature=0.25,
                )

                # True async streaming (no threads; avoids Qt timer/thread warnings)
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
                return

            # DeepSeek note: their /chat/completions currently rejects OpenAI-style image blocks.
            if self._vision_provider == "deepseek":
                raise ValueError(
                    "DeepSeek /chat/completions does not support OpenAI-style image inputs "
                    "(message content blocks with type=image_url). "
                    "Use Gemini (VISION_MODEL=gemini-1.5-flash) or OpenAI vision (gpt-4o), "
                    "or configure a vision provider that supports multimodal chat."
                )

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
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta is not None:
                    await ui_queue.put({"type": "token", "text": delta})

        except Exception as e:
            logger.error(f"Vision LLM error: {e}")
            await ui_queue.put({"type": "token", "text": f"\n[⚠️ Vision Error: {e}]"})
