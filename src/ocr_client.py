"""Local screenshot OCR through Ollama's native ``/api/generate`` endpoint.

Replaces the old cloud vision-LLM step: the screenshot is base64-encoded and
sent to a local vision-language model (GLM-OCR) with a fixed recognition
prompt; the recognized text is then handed to the text LLM as the question.

Ollama is addressed on its *native* API — the configured ``OLLAMA_BASE_URL``
points at the OpenAI-compatible ``/v1`` surface, so we strip that suffix.

Latency is dominated by two things on CPU, both handled here:

* **Model load.** Reloading the 2.2 GB F16 GLM-OCR takes ~30 s, and Ollama
  evicts an idle model after 5 minutes by default. ``keep_alive`` keeps it
  resident between screenshots.
* **A greedy repetition loop.** GLM-OCR runs at temperature 0 / top_k 1. On an
  image it cannot read it latches onto the last token group and repeats it
  until the token cap; the model's own cap is 8192, which measured ~13 minutes
  per screenshot. Bounded three ways: a per-request ``num_predict``, a
  wall-clock deadline, and a repetition guard that abandons a degenerate tail.

Setup (see ``setup_glm_ocr.py``)::

    ollama pull glm-ocr
    python setup_glm_ocr.py        # ollama create glm-ocr-optimized
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import deque
from typing import AsyncIterator

import httpx

from src.config import config

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Text recognition:"


class OCRError(RuntimeError):
    """OCR request failed (server unreachable, model missing, bad response)."""


class OCRTruncated(OCRError):
    """Recognition stopped early (token cap, deadline, or a degenerate loop)."""


def ollama_root(base_url: str) -> str:
    """Return the Ollama root from an OpenAI-compat ``…/v1`` base URL."""
    root = (base_url or "http://localhost:11434/v1").strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    return root or "http://localhost:11434"


def _error_from_body(status: int, body: str) -> str:
    detail = ""
    try:
        detail = str(json.loads(body).get("error") or "")
    except Exception:
        detail = body.strip()[:200]
    if status == 404 or "not found" in detail.lower():
        return (
            f"Ollama model missing ({detail or 'not found'}). "
            "Run: ollama pull glm-ocr && python setup_glm_ocr.py"
        )
    return f"Ollama HTTP {status}: {detail or 'request failed'}"


class _RepeatGuard:
    """Detects GLM-OCR's degeneracy: it stops reading the image and repeats.

    Two signatures, both observed on input the model cannot read:

    * the *same delta* emitted over and over — the model latches onto
      ```` ``` ```` and never stops (this is the one that runs for minutes);
    * the last N complete *lines* identical.

    Cheap and incremental: neither check scans the accumulated text.
    """

    def __init__(self, run: int):
        self.run = max(0, int(run))
        self._last_delta: str | None = None
        self._delta_run = 0
        self._lines: deque[str] = deque(maxlen=max(1, self.run))
        self._current_line = ""

    def feed(self, delta: str) -> bool:
        """Record a delta; True once the output has degenerated."""
        if self.run <= 0:
            return False

        if delta == self._last_delta:
            self._delta_run += 1
        else:
            self._last_delta = delta
            self._delta_run = 1
        if self._delta_run >= self.run and delta.strip():
            return True

        parts = delta.split("\n")
        for i, part in enumerate(parts):
            if i:
                self._lines.append(self._current_line)
                self._current_line = ""
            self._current_line += part
        if len(self._lines) == self._lines.maxlen:
            if self._lines[0].strip() and len(set(self._lines)) == 1:
                return True
        return False


class OCRClient:
    """Screenshot bytes → recognized text via a local Ollama vision model."""

    def __init__(self, *, base_url: str, model: str, prompt: str = DEFAULT_PROMPT,
                 timeout: float = 180.0, total_timeout: float | None = None,
                 keep_alive: str | None = None, num_predict: int | None = None,
                 repeat_guard_lines: int | None = None):
        self.root = ollama_root(base_url)
        self.model = model
        self.prompt = prompt or DEFAULT_PROMPT
        self.timeout = float(timeout or 180.0)
        # Wall-clock budget for one recognition (the read timeout above only
        # fires when the stream *stalls*, not while it keeps producing tokens).
        self.total_timeout = (
            float(total_timeout) if total_timeout is not None
            else float(getattr(config, "OCR_TOTAL_TIMEOUT_SEC", 120.0) or 0)
        )
        self.keep_alive = getattr(config, "OCR_KEEP_ALIVE", "30m") if keep_alive is None else keep_alive
        self.num_predict = (
            int(getattr(config, "OCR_NUM_PREDICT", 1024)) if num_predict is None else int(num_predict)
        )
        self.repeat_guard_lines = (
            int(getattr(config, "OCR_REPEAT_GUARD_LINES", 12))
            if repeat_guard_lines is None else int(repeat_guard_lines)
        )
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=self.timeout, write=30.0, pool=5.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    async def list_models(self) -> list[str]:
        """Names of models the local Ollama server has pulled."""
        client = self._ensure_client()
        try:
            resp = await client.get(f"{self.root}/api/tags")
        except httpx.HTTPError as e:
            raise OCRError(f"Ollama unreachable at {self.root}: {e}") from e
        if resp.status_code != 200:
            raise OCRError(_error_from_body(resp.status_code, resp.text))
        try:
            data = resp.json()
        except ValueError as e:
            raise OCRError(f"Ollama /api/tags returned non-JSON: {resp.text[:200]}") from e
        return [str(m.get("name") or "") for m in data.get("models", [])]

    def _payload(self, image_bytes: bytes) -> dict:
        payload: dict = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": True,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if self.num_predict > 0:
            payload["options"] = {"num_predict": self.num_predict}
        return payload

    async def _stream_raw(self, image_bytes: bytes) -> AsyncIterator[dict]:
        """Yield decoded chunks until the stream ends or the deadline passes."""
        client = self._ensure_client()
        try:
            async with client.stream(
                "POST", f"{self.root}/api/generate", json=self._payload(image_bytes)
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "ignore")
                    raise OCRError(_error_from_body(resp.status_code, body))
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("OCR: skipping non-JSON stream line: %.120s", line)
                        continue
                    if obj.get("error"):
                        raise OCRError(f"Ollama: {obj['error']}")
                    yield obj
        except httpx.HTTPError as e:
            raise OCRError(f"Ollama request to {self.root} failed ({type(e).__name__}): {e}") from e

    async def recognize_stream(self, image_bytes: bytes) -> AsyncIterator[str]:
        """Yield recognized text chunks as the model produces them.

        Raises :class:`OCRTruncated` once the generation was cut short (token
        cap, wall-clock deadline, or a degenerate repetition loop). Everything
        yielded before that point is still valid recognized text, so a caller
        that can live with a partial transcript should keep it.
        """
        guard = _RepeatGuard(self.repeat_guard_lines)
        out_tokens = 0
        done_reason: str | None = None
        reason: str | None = None
        deadline = (
            asyncio.get_running_loop().time() + self.total_timeout
            if self.total_timeout > 0 else None
        )

        agen = self._stream_raw(image_bytes)
        try:
            while True:
                # NOTE: the httpx read timeout only fires when the stream stalls,
                # so the overall budget is enforced here, per chunk.
                timeout = None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())
                try:
                    obj = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    reason = (
                        f"OCR exceeded its {self.total_timeout:.0f}s budget "
                        f"(~{out_tokens} tokens generated)"
                    )
                    break

                if obj.get("done"):
                    done_reason = obj.get("done_reason")
                    out_tokens = int(obj.get("eval_count") or out_tokens)
                chunk = obj.get("response") or ""
                if not chunk:
                    continue
                out_tokens += 1
                if guard.feed(chunk):
                    logger.warning(
                        "OCR: model started repeating itself after %d chunks — stopping "
                        "(the recognized text is kept).",
                        out_tokens,
                    )
                    reason = "OCR stopped early: the model began repeating itself"
                    break
                yield chunk
        finally:
            await agen.aclose()

        if reason:
            raise OCRTruncated(f"{reason}. Text below may be incomplete.")
        if done_reason == "length" or (self.num_predict > 0 and out_tokens >= self.num_predict):
            raise OCRTruncated(
                f"OCR hit the {self.num_predict}-token cap before finishing — "
                "the text may be incomplete (raise OCR_NUM_PREDICT)."
            )

    async def preload(self) -> bool:
        """Load the OCR model into memory without generating, so the first
        screenshot does not pay the model load (~6s for the 2.2GB F16 model on
        CPU). Returns True when the server accepted the request."""
        client = self._ensure_client()
        payload: dict = {"model": self.model}
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        try:
            # An empty prompt is what makes this a load-only call.
            resp = await client.post(f"{self.root}/api/generate", json=payload)
            if resp.status_code != 200:
                logger.debug("OCR preload returned HTTP %s", resp.status_code)
                return False
            logger.info("OCR model preloaded: %s", self.model)
            return True
        except httpx.HTTPError as e:
            logger.debug("OCR preload failed: %s", e)
            return False

    async def recognize(self, image_bytes: bytes) -> str:
        """Return the full recognized text for one screenshot."""
        return "".join([chunk async for chunk in self.recognize_stream(image_bytes)])

