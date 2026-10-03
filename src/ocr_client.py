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
import re
from collections import deque
from typing import AsyncIterator

import httpx

from src.config import config

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Text recognition:"


class OCRError(RuntimeError):
    """OCR request failed (server unreachable, model missing, bad response)."""


class OCRTruncated(OCRError):
    """Recognition stopped early.

    ``kind`` records *why*, because the two cases need opposite handling
    downstream:

    * ``"loop"`` — the model stopped reading and began repeating itself. What it
      produced before that point is a complete recognition pass, so the text only
      needs its degenerate tail trimmed (see :func:`trim_repetition`); it is not
      an incomplete transcript, and reporting it as one is a false alarm.
    * ``"cap"`` / ``"deadline"`` — generation was cut mid-content, so the text
      really can be missing its ending.
    """
    def __init__(self, message: str, *, kind: str = "truncated"):
        super().__init__(message)
        self.kind = kind


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


# --- Repetition detection -------------------------------------------------
#
# Measured on real screenshots (GLM-OCR, 1024 long edge, three different
# resolutions, light/dark/inverted, PNG/JPEG): the model transcribes the image
# *correctly and completely* and then keeps going. It re-emits the page inside a
# code fence, or invents a code block and re-emits that, or collapses into fence
# spam. The recognition before the loop was byte-perfect in every case, and
# raising the resolution made it worse rather than better. So a loop is not a
# recognition failure — it has to be stopped early and cut out of the transcript.

# How many characters the model must replay before we call it a loop. This is
# the discriminator that matters: a re-emission replays a long stretch
# (measured: 411-738 chars), while real documents legitimately repeat *lines* —
# a page of numbered requirements repeats its "Constraints:" line verbatim in
# every section. A shorter bar cut that 1910-char document down to 287 chars,
# throwing away the question.
_MIN_DUPLICATE_CHARS = 200

# The same idea for a transcript that ends with two copies of one block: the
# replay need not reach 200 chars to be a replay, but it must be long enough not
# to be a repeated sentence (measured safe at 80 on real transcripts).
_MIN_DOUBLED_TAIL_CHARS = 80

# A fragment must contain at least this many distinct characters to count as
# duplicate *content*. Without it a run of one character — a separator line, a
# progress bar, a row of dots — looks like a duplicate at every offset, and a
# transcript gets truncated at the second 'x'.
_MIN_DISTINCT_CHARS = 12

# Guard buffer: the whole transcript is kept (a 1024-token cap bounds it anyway)
# up to this size, because the replay is detected by finding the tail *earlier in
# the output* — a window smaller than the replayed block would miss it.
_GUARD_BUFFER_CHARS = 16384

# Longest replay the trim will look for. Both scans below are quadratic in the
# text they consider, and they run on the event loop, so the length they search
# is bounded: measured worst case (`_GUARD_BUFFER_CHARS` of non-repeating text)
# is ~1.1s unbounded and ~0.3s with this cap. Every replay measured was <= 738
# chars, and a longer one is stopped by the guard before it gets here anyway.
_MAX_REPLAY_CHARS = 4096

_FENCE_LINE = re.compile(r"^\s*```")
_FENCE_ONLY = re.compile(r"^\s*```\s*$")


def _is_content(fragment: str) -> bool:
    """True when a fragment carries enough variety to be recognised text."""
    return bool(fragment.strip()) and len(set(fragment)) >= _MIN_DISTINCT_CHARS


class _RepeatGuard:
    """Detects GLM-OCR's degeneracy: it stops reading the image and repeats.

    Three signatures, all observed on real screenshots:

    * the *same delta* emitted over and over — the model latches onto
      ```` ``` ```` and never stops (this is the one that runs for minutes);
    * the last N complete *lines* identical;
    * **the tail of the output already occurring earlier in it** — a replay.
      This is the one that matters in practice: it is what turns a 165s
      run-to-the-cap into a ~20s answer, with the duplicate stopped one block in
      instead of filling the transcript.

    The tail check compares a fixed window rather than searching for the longest
    duplicate, so it costs one substring search per delta. ``trim_repetition``
    does the precise surgery afterwards.
    """

    def __init__(self, run: int, *, min_dup: int = _MIN_DUPLICATE_CHARS):
        self.run = max(0, int(run))
        self.min_dup = max(1, int(min_dup))
        self._last_delta: str | None = None
        self._delta_run = 0
        self._lines: deque[str] = deque(maxlen=max(1, self.run))
        self._current_line = ""
        self._buf = ""

    def _tail_repeats(self) -> bool:
        """True when the last `min_dup` chars already occurred earlier."""
        n = len(self._buf)
        if n < 2 * self.min_dup:
            return False
        tail = self._buf[-self.min_dup :]
        if not _is_content(tail):
            return False
        return self._buf.find(tail) < n - self.min_dup

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
                completed, self._current_line = self._current_line, ""
                self._lines.append(completed)
            self._current_line += part
        if len(self._lines) == self._lines.maxlen:
            if self._lines[0].strip() and len(set(self._lines)) == 1:
                return True

        self._buf += delta
        if len(self._buf) > _GUARD_BUFFER_CHARS:
            self._buf = self._buf[-_GUARD_BUFFER_CHARS:]
        return self._tail_repeats()


def _duplicated_suffix_start(text: str, min_dup: int = _MIN_DUPLICATE_CHARS) -> int | None:
    """Start of the longest replayed tail (>= ``min_dup`` chars), or None.

    The earlier occurrence must *precede* the tail, which is what makes this a
    replay rather than a coincidence: text that merely repeats a line somewhere
    has no duplicated suffix.
    """
    n = len(text)
    longest = min(n - 1, _MAX_REPLAY_CHARS)
    for length in range(longest, min_dup - 1, -1):
        seg = text[n - length :]
        if text.find(seg) < n - length:
            return n - length
    return None


def _doubled_tail_start(text: str, min_unit: int = _MIN_DOUBLED_TAIL_CHARS) -> int | None:
    """Start of a tail that is the same block twice (``X X`` -> start of ``X``)."""
    n = len(text)
    largest = min(n // 2, _MAX_REPLAY_CHARS)
    for unit in range(largest, min_unit - 1, -1):
        if text[n - 2 * unit : n - unit] == text[n - unit :]:
            return n - 2 * unit
    return None


def _trailing_fence_run(text: str, *, min_reps: int = 3) -> tuple[int, int] | None:
    """Start and unit length of a trailing run of identical bare-fence lines."""
    n = len(text)
    if n < 6:
        return None
    for unit_len in range(3, 9):
        unit = text[-unit_len:]
        if not _FENCE_ONLY.match(unit):
            continue
        reps, i = 0, n
        while i - unit_len >= 0 and text[i - unit_len : i] == unit:
            reps += 1
            i -= unit_len
        if reps >= min_reps:
            return i, unit_len
    return None


def _trailing_unit_run(text: str, *, min_unit: int = 3, min_reps: int = 3,
                       min_chars: int = 12) -> int | None:
    """Start of a trailing run of one short unit repeated back-to-back.

    Bare fences are excluded — ``_trailing_fence_run`` owns those, and it knows
    to keep one when the code block still needs closing. Single-character units
    are ignored too: a run of "-" or "." is a separator or a progress bar, not a
    degenerate model.
    """
    n = len(text)
    for u in range(min_unit, min(40, n // min_reps) + 1):
        unit = text[-u:]
        if not _FENCE_ONLY.match(unit) or not unit.strip() or len(set(unit)) < 2:
            continue
        reps, i = 0, n
        while i - u >= 0 and text[i - u : i] == unit:
            reps += 1
            i -= u
        if reps >= min_reps and reps * u >= min_chars:
            return i
    return None


def _strip_dangling_opener(text: str) -> str:
    """Drop a fence the model opened after the content and never filled.

    Only *tagged* openers (````` ```python `````) are dropped. A bare `````` ``` ``````
    is left alone: it is the legitimate closer of a code screenshot's block, and
    removing it would leave the block open.
    """
    lines = text.rstrip().split("\n")
    while lines and lines[-1].strip().startswith("```") and not _FENCE_ONLY.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def _degenerate_cut(text: str) -> int | None:
    """Where the model stopped recognizing and started generating, or None.

    Four signatures, each measured on real transcripts. The cut is the smallest
    they propose, after which :func:`trim_repetition` runs again — the rules
    interact: fence spam has to go before a replayed page underneath it becomes
    visible as a duplicated suffix.
    """
    cuts: list[int] = []

    fence_run = _trailing_fence_run(text)
    if fence_run is not None:
        start, unit_len = fence_run
        # Fences pair, so keep one marker only when a block is actually open
        # there: a code screenshot must keep its closing fence. This also makes
        # the fence rule authoritative over the generic unit rule, which would
        # otherwise cut the same run to zero.
        open_before = sum(1 for ln in text[:start].split("\n") if _FENCE_LINE.match(ln))
        cuts.append(start + unit_len if open_before % 2 else start)
    elif (unit_start := _trailing_unit_run(text)) is not None:
        cuts.append(unit_start)

    for candidate in (
        _duplicated_suffix_start(text),
        _doubled_tail_start(text),
    ):
        if candidate is not None:
            cuts.append(candidate)

    return min(cuts) if cuts else None


def trim_repetition(text: str, *, rounds: int = 4) -> tuple[str, int]:
    """Cut the degenerate tail off a transcript, returning ``(text, removed)``.

    GLM-OCR reads a screenshot correctly and then keeps going: it re-emits the
    page inside a code fence, or invents a code block and re-emits that, or
    collapses into fence spam — measured on three real configurations, where the
    *entire* recognition was correct and everything after it was repetition.

    Content the model emitted twice is generation, not recognition, so the cut
    lands where the first replay begins. Two things make this fiddly, and both
    are why the rules are what they are:

    * the transcripts replay at different offsets (whole page, invented block),
      so the search cannot assume a fixed period;
    * documents legitimately repeat *lines*, so a single repeated line is not
      evidence of anything — the replay has to be a long stretch, or the whole
      tail, or a run of structural markers.

    A no-op on anything the model produced once; that is the property the tests
    pin hardest, because a false positive silently deletes the question.
    """
    removed = 0
    for _ in range(max(1, rounds)):
        cut = _degenerate_cut(text)
        if cut is None:
            break
        trimmed = _strip_dangling_opener(text[:cut])
        if len(trimmed) >= len(text):
            break
        removed += len(text) - len(trimmed)
        text = trimmed
    return text, removed


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
        kind = "truncated"
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
                    kind = "deadline"
                    break

                if obj.get("done"):
                    done_reason = obj.get("done_reason")
                    out_tokens = int(obj.get("eval_count") or out_tokens)
                chunk = obj.get("response") or ""
                if not chunk:
                    continue
                out_tokens += 1
                if guard.feed(chunk):
                    logger.info(
                        "OCR: the model stopped reading and began repeating itself after "
                        "%d chunks — stopping (the recognition before it is kept).",
                        out_tokens,
                    )
                    # Yield it before stopping: the chunk that completes a
                    # duplicate window can also carry the end of the real
                    # recognition, and `trim_repetition` removes the duplicate
                    # part without touching what came before it.
                    yield chunk
                    reason = "the model began repeating itself"
                    kind = "loop"
                    break
                yield chunk
        finally:
            await agen.aclose()

        if reason:
            raise OCRTruncated(reason, kind=kind)
        if done_reason == "length" or (self.num_predict > 0 and out_tokens >= self.num_predict):
            raise OCRTruncated(
                f"generation hit the {self.num_predict}-token cap before stopping",
                kind="cap",
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

