"""Screenshot pipeline tests: local OCR → text-LLM answer streaming.

The Google/OpenAI vision-provider path is gone; OCR runs locally through
Ollama and only the recognized text reaches the text provider. These tests
pin the two seams that matter: what the OCR client sends/returns, and the
event sequence the screenshot pipeline emits to the overlay.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from collections import deque

import httpx
import pytest
from PIL import Image

from src.config import config
from src.llm.base import Delta
from src.llm_engine import LLMEngine, TurnContext
from src.ocr_client import (
    OCRError, OCRTruncated, OCRClient, ollama_root, trim_repetition,
)


class _FakeOCR:
    """Duck-typed OCRClient: streams `chunks`, or raises `error`."""

    model = "glm-ocr-optimized"

    def __init__(self, chunks: list[str] | None = None, error: Exception | None = None):
        self._chunks = chunks or []
        self._error = error
        self.calls: list[int] = []

    async def recognize_stream(self, image_bytes: bytes):
        self.calls.append(len(image_bytes))
        if self._error:
            raise self._error
        for c in self._chunks:
            yield c


class _FakeTextProvider:
    name = "deepseek"

    def __init__(self, deltas: list[Delta] | None = None, seen: list | None = None):
        self._deltas = deltas or [Delta(text="answer")]
        self.seen = seen if seen is not None else []

    async def chat_stream(self, messages, *, model, max_tokens=512, temperature=0.25):
        self.seen.append(messages)
        for d in self._deltas:
            yield d


# A compliant algorithm answer. The route's contract demands the fenced block,
# so a fake that answers in prose alone would (correctly) trigger the code
# repair — these pipeline tests are about the OCR seam, not about that.
FENCED_ALGORITHM_ANSWER = (
    "[U] 反转链表 [M] 三指针迭代 [P] 逐个改向\n\n"
    "[I]\n```python\ndef reverse(head):\n    prev = None\n    while head:\n"
    "        head.next, prev, head = prev, head, head.next\n    return prev\n```\n\n"
    "[R] T(n)=O(n)，S(n)=O(1)。"
)


class _ScriptedProvider:
    """One scripted answer per call — the code repair is a second call."""

    name = "deepseek"

    def __init__(self, answers: list[str]):
        self._answers = answers
        self.seen: list[list[dict]] = []

    async def chat_stream(self, messages, *, model, max_tokens=512, temperature=0.25):
        self.seen.append(messages)
        text = self._answers[min(len(self.seen) - 1, len(self._answers) - 1)]
        yield Delta(text=text)


def _jpeg_bytes(w: int = 40, h: int = 20) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _make_engine(ocr, text_provider, *, q_type: str = "technical") -> LLMEngine:
    eng = LLMEngine.__new__(LLMEngine)
    eng._active_tasks = {}
    eng._text_history = deque(maxlen=0)
    eng.last_usage = {}
    eng.text_provider = text_provider
    eng.ocr = ocr
    eng._gather_context = lambda *_a, **_k: TurnContext()
    eng.classify_question_llm = lambda *_a, **_k: _async_value(q_type)
    return eng


async def _async_value(v):
    return v


async def _drain(q: asyncio.Queue) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ── Ollama URL handling ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "base,expected",
    [
        ("http://localhost:11434/v1", "http://localhost:11434"),
        ("http://localhost:11434/v1/", "http://localhost:11434"),
        ("http://box:11434", "http://box:11434"),
        ("", "http://localhost:11434"),
    ],
)
def test_ollama_root_strips_openai_v1_suffix(base, expected):
    """Ollama's native /api/generate lives on the root, not under /v1."""
    assert ollama_root(base) == expected


# ── OCR client against a mocked transport ────────────────────────────────


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def _client_with(handler, **kwargs) -> OCRClient:
    c = OCRClient(base_url="http://localhost:11434/v1", model="glm-ocr-optimized", **kwargs)
    c._client = httpx.AsyncClient(transport=_mock_transport(handler), timeout=5.0)
    return c


@pytest.mark.asyncio
async def test_recognize_stream_emits_chunks_and_sends_image():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        lines = [
            json.dumps({"response": "Hello ", "done": False}),
            json.dumps({"response": "world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]
        return httpx.Response(200, content="\n".join(lines).encode())

    client = _client_with(handler)
    out = [c async for c in client.recognize_stream(b"\xff\xd8img")]
    await client.aclose()

    assert "".join(out) == "Hello world"
    assert seen["url"].endswith("/api/generate")
    assert seen["body"]["model"] == "glm-ocr-optimized"
    assert seen["body"]["images"] and seen["body"]["images"][0]  # base64 payload present


@pytest.mark.asyncio
async def test_recognize_surfaces_ollama_error_object():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"error": "model not found"}).encode())

    client = _client_with(handler)
    with pytest.raises(OCRError, match="model not found"):
        await client.recognize(b"\xff\xd8")
    await client.aclose()


@pytest.mark.asyncio
async def test_recognize_maps_404_to_pull_instructions():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=json.dumps({"error": "model 'x' not found"}).encode())

    client = _client_with(handler)
    with pytest.raises(OCRError, match="setup_glm_ocr.py"):
        await client.recognize(b"\xff\xd8")
    await client.aclose()


@pytest.mark.asyncio
async def test_recognize_maps_connection_failure_to_OCRError():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client_with(handler)
    with pytest.raises(OCRError, match="Ollama"):
        await client.recognize(b"\xff\xd8")
    await client.aclose()


# ── OCR resilience: resident model, bounded generation ───────────────────


def _stream_of(objs) -> httpx.Response:
    lines = [json.dumps(o) for o in objs]
    return httpx.Response(200, content="\n".join(lines).encode())


def test_payload_requests_resident_model_and_bounded_generation():
    """Without keep_alive Ollama evicts the 2.2 GB model after 5 min and the
    next screenshot pays ~30 s of load; without num_predict the model's own
    cap (8192) turns a repetition loop into ~13 min of CPU."""
    client = OCRClient(base_url="http://localhost:11434/v1", model="glm-ocr-optimized")
    payload = client._payload(b"\xff\xd8")

    assert payload["keep_alive"] == config.OCR_KEEP_ALIVE
    assert payload["options"]["num_predict"] == config.OCR_NUM_PREDICT


@pytest.mark.asyncio
async def test_recognize_stream_abandons_a_repetition_loop():
    """GLM-OCR is greedy: past the end of the page it latches onto one token
    group and repeats it until the token cap (measured: one line emitted 164×,
    59-94 s of decode for text it had already transcribed). The guard must stop
    the stream early, keeping the text that was already recognized."""
    repeated = json.dumps({"response": "```\n", "done": False})
    body = "\n".join([repeated] * 400) + "\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode())

    client = _client_with(handler, repeat_guard_lines=12)
    chunks: list[str] = []
    with pytest.raises(OCRTruncated, match="repeating itself"):
        async for chunk in client.recognize_stream(b"\xff\xd8"):
            chunks.append(chunk)
    await client.aclose()

    assert len(chunks) < 50, f"guard did not stop the loop: {len(chunks)} chunks consumed"


@pytest.mark.asyncio
async def test_recognize_stream_reports_the_token_cap_as_truncation():
    """A `length` stop means the recognized text is incomplete — that must not
    be indistinguishable from a complete transcript."""
    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_of([
            {"response": "half a scre", "done": False},
            {"response": "", "done": True, "done_reason": "length", "eval_count": 1024},
        ])

    client = _client_with(handler)
    chunks: list[str] = []
    with pytest.raises(OCRTruncated, match="token cap"):
        async for chunk in client.recognize_stream(b"\xff\xd8"):
            chunks.append(chunk)
    await client.aclose()

    assert "".join(chunks) == "half a scre"


@pytest.mark.asyncio
async def test_recognize_stream_enforces_a_total_deadline():
    """The httpx read timeout only fires when the stream stalls, so a slow but
    steady generation would otherwise run to the token cap."""
    class _Drip(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (json.dumps({"response": "tick", "done": False}) + "\n").encode()
            for _ in range(100):
                await asyncio.sleep(0.2)
                yield (json.dumps({"response": "tick", "done": False}) + "\n").encode()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Drip())

    client = _client_with(handler, total_timeout=0.5)
    started = time.monotonic()
    with pytest.raises(OCRTruncated, match="budget"):
        async for _ in client.recognize_stream(b"\xff\xd8"):
            pass
    elapsed = time.monotonic() - started
    await client.aclose()

    assert elapsed < 5.0, f"deadline not enforced ({elapsed:.1f}s)"


@pytest.mark.asyncio
async def test_engine_keeps_partial_text_when_recognition_is_truncated():
    """Truncated recognition still carries the question; discarding it would
    throw away a usable transcript. The shortfall is returned alongside it."""
    class _TruncatingOCR:
        model = "glm-ocr-optimized"

        async def recognize_stream(self, _image_bytes):
            yield "Q: reverse "
            raise OCRTruncated("OCR hit the 1024-token cap before finishing")

    eng = LLMEngine.__new__(LLMEngine)
    eng.ocr = _TruncatingOCR()
    ui: asyncio.Queue = asyncio.Queue()

    text, warning = await eng._ocr_screenshot(b"\xff\xd8", ui)
    statuses = [m["text"] for m in await _drain(ui) if m["type"] == "status"]

    assert text == "Q: reverse"
    assert "cap" in warning
    assert any("cap" in s for s in statuses), statuses


@pytest.mark.asyncio
async def test_truncated_recognition_is_visible_after_the_answer_starts(monkeypatch):
    """A status line is transient — `answer_start` clears it — so a screenshot
    whose code was cut off must carry its warning into the answer text, or the
    user reads a half-transcribed listing as if it were complete."""
    class _TruncatingOCR:
        model = "glm-ocr-optimized"

        async def recognize_stream(self, _image_bytes):
            yield "def merge(a, b):"
            raise OCRTruncated("OCR hit the 1024-token cap before finishing")

    class _StreamingProvider:
        name = "deepseek"

        async def chat_stream(self, *_a, **_k):
            yield Delta(text="[U] merge", usage=None)

    eng = LLMEngine.__new__(LLMEngine)
    eng.ocr = _TruncatingOCR()
    eng.text_provider = _StreamingProvider()
    eng.rag = None
    eng.last_usage = {}
    eng.register_task = lambda *_a, **_k: None
    eng._gather_context = lambda *_a, **_k: TurnContext()
    eng.classify_question_llm = lambda *_a, **_k: _async_value("algorithm")
    eng._record_usage = lambda *_a, **_k: None
    monkeypatch.setattr(config, "OCR_MAX_DIMENSION", 256, raising=False)

    ui: asyncio.Queue = asyncio.Queue()
    await eng.generate_vision_answer_stream(_jpeg_bytes(), ui)
    msgs = await _drain(ui)

    order = [m["type"] for m in msgs]
    assert "answer_start" in order
    body = "".join(m["text"] for m in msgs if m["type"] == "token")
    assert "OCR incomplete" in body, body
    assert "cap" in body, body
    # It must be part of the answer body, not merely a status the UI replaces.
    assert order.index("answer_start") < next(
        i for i, m in enumerate(msgs) if m["type"] == "token" and "OCR incomplete" in m["text"]
    )


# ── Image preparation ────────────────────────────────────────────────────


def test_prepare_ocr_image_downscales_longest_edge(monkeypatch):
    monkeypatch.setattr(config, "OCR_MAX_DIMENSION", 512, raising=False)
    eng = LLMEngine.__new__(LLMEngine)
    src = io.BytesIO()
    Image.new("RGB", (1024, 256), (200, 100, 50)).save(src, format="PNG")

    out = eng._prepare_ocr_image(src.getvalue())
    img = Image.open(io.BytesIO(out))
    assert max(img.size) == 512
    assert img.size == (512, 128)  # aspect ratio preserved


def test_prepare_ocr_image_leaves_images_inside_bounds_alone(monkeypatch):
    monkeypatch.setattr(config, "OCR_MAX_DIMENSION", 1024, raising=False)
    monkeypatch.setattr(config, "OCR_MIN_DIMENSION", 256, raising=False)
    eng = LLMEngine.__new__(LLMEngine)
    src = io.BytesIO()
    Image.new("RGB", (300, 100), (0, 0, 0)).save(src, format="PNG")

    out = eng._prepare_ocr_image(src.getvalue())
    assert Image.open(io.BytesIO(out)).size == (300, 100)


def test_prepare_ocr_image_upscales_images_below_the_floor(monkeypatch):
    """A small crop sent at native size is unreadable for GLM-OCR: it latches
    onto a token group and repeats it until the token cap (~13 min of CPU)."""
    monkeypatch.setattr(config, "OCR_MAX_DIMENSION", 1024, raising=False)
    monkeypatch.setattr(config, "OCR_MIN_DIMENSION", 512, raising=False)
    eng = LLMEngine.__new__(LLMEngine)
    src = io.BytesIO()
    Image.new("RGB", (300, 100), (0, 0, 0)).save(src, format="PNG")

    out = eng._prepare_ocr_image(src.getvalue())
    img = Image.open(io.BytesIO(out))
    assert img.size == (512, 171)  # scaled up to the floor, aspect preserved


# ── End-to-end screenshot pipeline ───────────────────────────────────────


@pytest.mark.asyncio
async def test_screenshot_streams_ocr_then_answer_with_progress_events(monkeypatch):
    monkeypatch.setattr(config, "OCR_MAX_DIMENSION", 256, raising=False)
    ocr = _FakeOCR(["Q: reverse ", "a linked list"])
    text = _FakeTextProvider([Delta(text=FENCED_ALGORITHM_ANSWER)])
    eng = _make_engine(ocr, text, q_type="algorithm")
    ui: asyncio.Queue = asyncio.Queue()

    question = await eng.generate_vision_answer_stream(_jpeg_bytes(), ui)
    msgs = await _drain(ui)

    assert question == "Q: reverse a linked list"
    # Progress is visible while OCR runs.
    statuses = [m["text"] for m in msgs if m["type"] == "status"]
    assert any("glm-ocr-optimized" in s for s in statuses)
    assert any("Q: reverse" in s for s in statuses)
    # The recognized question is announced before the answer streams.
    start = next(m for m in msgs if m["type"] == "answer_start")
    assert start["question"] == "Q: reverse a linked list"
    assert start["q_type"] == "algorithm"
    assert [m["text"] for m in msgs if m["type"] == "token"] == [FENCED_ALGORITHM_ANSWER]
    # One call only: the answer carried its code block, so nothing to repair.
    assert len(text.seen) == 1
    # OCR saw the *downscaled* image, not the raw capture.
    assert ocr.calls and ocr.calls[0] < len(_jpeg_bytes(1024, 1024))
    # And the text model was told the question came from a screenshot.
    prompt = text.seen[0][1]["content"]
    assert "Q: reverse a linked list" in prompt
    assert "OCR" in prompt


@pytest.mark.asyncio
async def test_screenshot_with_no_recognized_text_reports_error():
    ocr = _FakeOCR([])  # model returned nothing
    text = _FakeTextProvider()
    eng = _make_engine(ocr, text)
    ui: asyncio.Queue = asyncio.Queue()

    question = await eng.generate_vision_answer_stream(_jpeg_bytes(), ui)
    msgs = await _drain(ui)

    assert question == ""
    assert not text.seen  # no wasted text-model call
    err = "".join(m["text"] for m in msgs if m["type"] == "token")
    assert "OCR Error" in err


@pytest.mark.asyncio
async def test_screenshot_reports_ollama_outage_in_overlay():
    ocr = _FakeOCR(error=OCRError("Ollama unreachable at http://localhost:11434"))
    text = _FakeTextProvider()
    eng = _make_engine(ocr, text)
    ui: asyncio.Queue = asyncio.Queue()

    question = await eng.generate_vision_answer_stream(_jpeg_bytes(), ui)
    msgs = await _drain(ui)

    assert question == ""
    assert not text.seen
    err = "".join(m["text"] for m in msgs if m["type"] == "token")
    assert "Ollama unreachable" in err


# ── Route override for code-shaped transcripts ───────────────────────────


OCR_EDITOR_BUFFER = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, num in enumerate(nums):\n"
    "        if target - num in seen:\n"
    "            return [seen[target - num], i]\n"
    "        seen[num] = i\n"
    "    return []\n"
)

OCR_PROBLEM_STATEMENT = (
    "Given an array of integers nums and an integer target, return indices of\n"
    "the two numbers such that they add up to target.\n"
    "Example 1: Input: nums = [2,7,11,15], target = 9  Output: [0,1]\n"
)


@pytest.mark.parametrize("transcript", [OCR_EDITOR_BUFFER])
def test_code_shaped_transcript_forces_the_algorithm_route(transcript):
    """A screenshot of an editor must not be answered by the technical prompt.

    The classifiers read words: an OCR'd listing with little prose in it routes
    as `technical`, whose one-line format contains no code block at all — the
    exact answer shape that is useless for a coding question.
    """
    from src.llm_engine import looks_like_code

    assert looks_like_code(transcript)
    # The problem-statement shape is *not* what this check is for: it is prose,
    # and the classifier's vocabulary handles it.
    assert not looks_like_code(OCR_PROBLEM_STATEMENT)


@pytest.mark.asyncio
async def test_ocr_editor_buffer_is_answered_with_code_despite_technical_routing():
    """End to end: classifier says `technical`, the transcript is code, the
    answer is a code answer."""
    ocr = _FakeOCR([OCR_EDITOR_BUFFER])
    text = _FakeTextProvider([Delta(text=FENCED_ALGORITHM_ANSWER)])
    eng = _make_engine(ocr, text, q_type="technical")  # classifier misses
    ui: asyncio.Queue = asyncio.Queue()

    await eng.generate_vision_answer_stream(_jpeg_bytes(), ui)
    msgs = await _drain(ui)

    assert next(m for m in msgs if m["type"] == "answer_start")["q_type"] == "algorithm"
    # The algorithm prompt is what reaches the model.
    assert "[I]" in text.seen[0][0]["content"]
    assert "code block" in text.seen[0][0]["content"].lower()


# ── Code repair when an algorithm answer arrives without code ────────────


@pytest.mark.asyncio
async def test_prose_only_algorithm_answer_is_repaired_once():
    """The route's contract is a fenced block; a prose-only answer is re-asked."""
    ocr = _FakeOCR(["Q: reverse a linked list"])
    text = _ScriptedProvider([
        "[U] 反转链表 [M] 三指针 [P] 逐个改向 [R] O(n)",   # no code block
        "```python\ndef reverse(head):\n    return head\n```",
    ])
    eng = _make_engine(ocr, text, q_type="algorithm")
    ui: asyncio.Queue = asyncio.Queue()

    await eng.generate_vision_answer_stream(_jpeg_bytes(), ui)
    msgs = await _drain(ui)

    assert len(text.seen) == 2, "the repair must be a second call"
    body = "".join(m["text"] for m in msgs if m["type"] == "token")
    assert "```python" in body and "def reverse" in body
    # The repair must not repeat the draft back to the model as an answer.
    repair_prompt = text.seen[1][1]["content"]
    assert "no code block" in repair_prompt
    assert "ONE fenced code block" in repair_prompt


@pytest.mark.asyncio
async def test_repair_is_not_attempted_again_when_it_also_fails():
    """One bounded attempt: a model that refuses twice must not loop, and the
    user is told the answer is prose rather than silently trusting it."""
    ocr = _FakeOCR(["Q: reverse a linked list"])
    text = _ScriptedProvider(["没有代码", "还是没有代码"])
    eng = _make_engine(ocr, text, q_type="algorithm")
    ui: asyncio.Queue = asyncio.Queue()

    await eng.generate_vision_answer_stream(_jpeg_bytes(), ui)
    msgs = await _drain(ui)

    assert len(text.seen) == 2
    body = "".join(m["text"] for m in msgs if m["type"] == "token")
    assert "仍未生成代码块" in body


@pytest.mark.asyncio
async def test_an_unterminated_fence_is_not_repaired():
    """Truncation is a different failure: the code is on screen, and a re-ask
    cannot un-cut it. Repairing here would burn a call per truncated answer."""
    from src.llm_engine import LLMEngine

    ocr = _FakeOCR(["Q: reverse a linked list"])
    text = _ScriptedProvider(["[I]\n```python\ndef reverse(head):\n"])
    eng = _make_engine(ocr, text, q_type="algorithm")
    ui: asyncio.Queue = asyncio.Queue()

    await eng.generate_vision_answer_stream(_jpeg_bytes(), ui)

    assert len(text.seen) == 1
    assert LLMEngine._has_code_block("[I]\n```python\ndef reverse(head):\n")


@pytest.mark.asyncio
async def test_non_algorithm_routes_are_never_repaired():
    """The repair belongs to the one route whose contract is a code block."""
    ocr = _FakeOCR(["Q: what is a deadlock"])
    text = _ScriptedProvider(["[R] senior IC\t[E] 取舍\t[A] 备选\t[C] 选择\t[T] 结果"])
    eng = _make_engine(ocr, text, q_type="technical")
    ui: asyncio.Queue = asyncio.Queue()

    await eng.generate_vision_answer_stream(_jpeg_bytes(), ui)

    assert len(text.seen) == 1


# ── Repetition: stop early, trim the transcript, report honestly ──────────
#
# Measured on real screenshots (GLM-OCR, 1024 long edge): the model transcribes
# the image *correctly and completely* and then keeps going — re-emitting the
# page inside a code fence, inventing a code block and re-emitting that, or
# collapsing into fence spam. Every configuration tested (768/1024/1280/1600 px,
# PNG/JPEG, dark/inverted/light) did this, and the recognition before the loop
# was byte-perfect.
#
# The cause is now known and fixed in the model, not here: Ollama's glm-ocr GGUF
# ships without `tokenizer.ggml.eot_token_id`, so the model has no way to end
# its turn (see setup_glm_ocr.py). What the tests below cover is this module's
# job for a model built without that repair: stop the stream early, trim the
# replay off, and only *then* decide what the user should be told.

PAGE = (
    "LeetCode 121. Best Time to Buy and Sell Stock\n"
    "\n"
    "You are given an array prices where prices[i] is the price of a given stock\n"
    "on the ith day. Return the maximum profit you can achieve.\n"
    "\n"
    "Constraints:\n"
    "1 <= prices.length <= 10^5\n"
    "0 <= prices[i] <= 10^4"
)
FENCE_SPAM = "\n" + "```\n" * 10


def test_repeated_page_is_cut_back_to_the_recognition():
    """The live shape: page, then the same page again inside a fence, then spam."""
    raw = PAGE + "\n```python\n" + PAGE + "\n```" + FENCE_SPAM
    text, removed = trim_repetition(raw)

    assert text == PAGE
    assert removed == len(raw) - len(PAGE)


def test_invented_code_block_repeating_is_cut_at_its_first_occurrence():
    """/1280-1600px did this: past the page, the model invents code and repeats it.

    The invented content was never in the image, so what must survive is the
    recognition; what must not survive is the second copy.
    """
    invented = (
        "\n```python\n"
        "prices = [7,1,5,3,6,4]\n"
        "buy_on_day_2(price=1)\n"
        "sell_on_day_5(price=6)\n"
        "profit = 6-1 = 5\n"
        "print(profit)\n"
    )
    raw = PAGE + invented + invented
    text, removed = trim_repetition(raw)

    assert text == PAGE
    assert text.count("buy_on_day_2") == 0
    assert removed > len(invented)


def test_fence_spam_is_dropped_but_a_real_code_block_keeps_its_closer():
    """A code screenshot ends with a legitimate ``` — that one is not spam.

    Cutting the run at its first line would leave the block unterminated, so the
    run collapses to a *balanced* number of markers instead.
    """
    code = (
        "```python\n"
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[target - n], i]\n"
        "        seen[n] = i\n"
        "    return []\n"
        "```"
    )
    text, removed = trim_repetition(code + FENCE_SPAM)

    assert text == code, "the code and its closing fence must survive"
    assert text.count("```") == 2
    assert removed > 0


def test_fence_spam_without_an_open_block_disappears_entirely():
    """No opener means the markers pair with nothing — keeping one would open a
    block that the recognition never had."""
    text, removed = trim_repetition(PAGE + FENCE_SPAM)

    assert text == PAGE
    assert "```" not in text
    assert removed > 0


def test_dangling_opener_left_by_the_cut_is_dropped():
    """The cut lands after the model's own ```python line; leaving it would show
    an empty code block or swallow whatever follows."""
    raw = PAGE + "\n```python\n" + PAGE
    text, _removed = trim_repetition(raw)

    assert text == PAGE
    assert "```" not in text


@pytest.mark.parametrize(
    "name,transcript",
    [
        ("clean page", PAGE),
        ("table with aligned columns",
         "col_a  col_b  col_c\n" + "\n".join(f"{i:5d}  {i*i:5d}  {i**3:6d}" for i in range(60))),
        ("repeated-looking rows",
         "\n".join(f"row {i}: id=1 status=ok" for i in range(60))),
        ("code with repeated bodies",
         "```python\ndef a():\n    return 1\ndef b():\n    return 1\ndef c():\n    return 1\n```"),
        ("single-line question",
         "算法题：给定整数数组与目标值，返回和为目标值的两个下标，哈希表一次遍历即可。"),
    ],
)
def test_legitimate_transcripts_are_never_rewritten(name, transcript):
    """Trimming must be a no-op on content the model produced once. A false
    positive here silently deletes the question."""
    text, removed = trim_repetition(transcript)

    assert removed == 0, f"{name} was rewritten"
    assert text == transcript


def test_guard_stops_a_stream_that_replays_the_output():
    """The streaming guard is what saves the time: without it the model ran on to
    the 1024-token cap (measured 165s for one screenshot).

    Detection is by tail replay, so it fires once enough of the replay has
    accumulated — a window of `_MIN_DUPLICATE_CHARS` past the replay's start.
    """
    from src.ocr_client import _RepeatGuard

    block = (
        "def max_profit(prices):\n"
        "    best = 0\n"
        "    low = float('inf')\n"
        "    for p in prices:\n"
        "        low = min(low, p)\n"
        "        best = max(best, p - low)\n"
        "    return best\n"
    )
    assert len(block) >= 120, "the fixture must be big enough to be a real replay"

    guard = _RepeatGuard(12)
    stream = block * 3
    fired = next((i for i, ch in enumerate(stream) if guard.feed(ch)), None)

    assert fired is not None, "a replay must be detected"


def test_guard_leaves_a_normal_transcript_alone():
    """Ordinary output must not trip it — every line here is new."""
    from src.ocr_client import _RepeatGuard

    guard = _RepeatGuard(12)
    for i in range(40):
        assert guard.feed(f"line {i}: some ordinary transcribed content here\n") is False


def test_guard_ignores_runs_of_one_character():
    """A separator line, a progress bar or a row of dots duplicates at every
    offset; treating that as 'the model is looping' would truncate real content."""
    from src.ocr_client import _RepeatGuard, trim_repetition

    guard = _RepeatGuard(12)
    assert not any(guard.feed("x" * 30) for _ in range(10))

    text = "Section 1\n" + "-" * 80 + "\nEnd"
    assert trim_repetition(text) == (text, 0)


def test_guard_does_not_fire_on_a_line_that_is_still_being_written():
    """Regression: the window at a line boundary was re-tested on every
    subsequent delta while that line was still arriving, so it matched its own
    anchor and the guard fired on good text — measured live, it cut a real
    problem statement at 108 chars, mid-sentence, and the overlay reported the
    screenshot as unreadable."""
    from src.ocr_client import _RepeatGuard

    line = "You are given an array prices where prices[i] is the price of a given stock"
    guard = _RepeatGuard(12)
    assert not any(guard.feed(ch) for ch in line)


def test_guard_detects_a_replayed_page_char_by_char():
    """The realistic shape: the model re-emits the page it just read, delivered
    as a stream of small deltas (measured transcripts run 588-738 chars)."""
    from src.ocr_client import _RepeatGuard

    page = (
        "LeetCode 121. Best Time to Buy and Sell Stock\n"
        "\n"
        "You are given an array prices where prices[i] is the price of a given stock\n"
        "on the ith day. You want to maximize your profit by choosing a single day\n"
        "to buy one stock and choosing a different day in the future to sell that\n"
        "stock. Return the maximum profit you can achieve from this transaction.\n"
    )
    assert len(page) >= 260, "the fixture must be a realistic page"

    guard = _RepeatGuard(12)
    fired = next((i for i, ch in enumerate(page + page) if guard.feed(ch)), None)

    assert fired is not None, "a replayed page must be detected"
    assert fired >= len(page), "it must not fire while the first pass is still arriving"


@pytest.mark.asyncio
async def test_a_repetition_is_trimmed_and_not_reported_as_incomplete():
    """The false alarm: the whole page was read, the model then repeated itself,
    and the overlay said "OCR incomplete" — which is why users were told to raise
    OCR_NUM_PREDICT, a change that makes it worse rather than better."""
    class _LoopingOCR:
        model = "glm-ocr-optimized"

        async def recognize_stream(self, _image_bytes):
            yield PAGE
            yield "\n```python\n"
            yield PAGE
            raise OCRTruncated("the model began repeating itself", kind="loop")

    eng = LLMEngine.__new__(LLMEngine)
    eng.ocr = _LoopingOCR()
    ui: asyncio.Queue = asyncio.Queue()

    text, warning = await eng._ocr_screenshot(b"\xff\xd8", ui)

    assert text == PAGE
    assert warning == "", "a trimmed repetition is not an incomplete transcript"


@pytest.mark.asyncio
async def test_a_cap_stopped_transcript_still_warns():
    """The other case is real content loss: generation was cut mid-content."""
    partial = "def merge(intervals):\n    intervals.sort(key=lambda x: x[0])\n    out = ["

    class _CappedOCR:
        model = "glm-ocr-optimized"

        async def recognize_stream(self, _image_bytes):
            yield partial
            raise OCRTruncated("generation hit the 1024-token cap", kind="cap")

    eng = LLMEngine.__new__(LLMEngine)
    eng.ocr = _CappedOCR()
    ui: asyncio.Queue = asyncio.Queue()

    text, warning = await eng._ocr_screenshot(b"\xff\xd8", ui)

    assert text == partial
    assert "cap" in warning and "missing its ending" in warning


@pytest.mark.asyncio
async def test_a_cap_stop_on_a_full_transcript_is_not_reported():
    """The false alarm that survives the repetition guard: the model reads the
    page, *generates* past it, and the runaway — not the recognition — is what
    reaches the token cap. The transcript is complete, so warning that it is
    "missing its ending" tells the user to distrust a whole page.

    This is the shape the missing end-of-generation token produces, and the
    cap is not the fix for it (see setup_glm_ocr.py), so the judgement has to
    hold even when the cap is genuinely what stopped generation."""
    complete = PAGE  # a whole page, well past _MIN_TRUSTED_TRANSCRIPT_CHARS

    class _CappedOCR:
        model = "glm-ocr-optimized"

        async def recognize_stream(self, _image_bytes):
            yield complete
            raise OCRTruncated("generation hit the 1024-token cap", kind="cap")

    eng = LLMEngine.__new__(LLMEngine)
    eng.ocr = _CappedOCR()
    ui: asyncio.Queue = asyncio.Queue()

    text, warning = await eng._ocr_screenshot(b"\xff\xd8", ui)

    assert text == complete
    assert warning == "", "a complete transcript must not be reported as truncated"


@pytest.mark.asyncio
async def test_a_loop_that_left_almost_nothing_still_warns():
    """If the model looped after reading almost nothing, the screenshot was
    unreadable and the answer would be nonsense — say so."""
    class _BarelyReadOCR:
        model = "glm-ocr-optimized"

        async def recognize_stream(self, _image_bytes):
            yield "a = 1\n"
            raise OCRTruncated("the model began repeating itself", kind="loop")

    eng = LLMEngine.__new__(LLMEngine)
    eng.ocr = _BarelyReadOCR()
    ui: asyncio.Queue = asyncio.Queue()

    text, warning = await eng._ocr_screenshot(b"\xff\xd8", ui)

    assert text == "a = 1"
    assert "unreadable" in warning


@pytest.mark.asyncio
async def test_cap_message_no_longer_tells_users_to_raise_the_cap(monkeypatch):
    """Raising the cap does not stop the loop — it buys more duplication — so the
    1024-token cap must not be blamed for it."""
    def handler(_request):
        return _stream_of([
            {"response": "half a scre", "done": False},
            {"response": "", "done": True, "done_reason": "length", "eval_count": 1024},
        ])

    client = _client_with(handler)
    with pytest.raises(OCRTruncated) as excinfo:
        async for _ in client.recognize_stream(b"\xff\xd8"):
            pass
    await client.aclose()

    assert excinfo.value.kind == "cap"
    assert "OCR_NUM_PREDICT" not in str(excinfo.value)
