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
from src.ocr_client import OCRError, OCRTruncated, OCRClient, ollama_root


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
    throw away a usable transcript."""
    class _TruncatingOCR:
        model = "glm-ocr-optimized"

        async def recognize_stream(self, _image_bytes):
            yield "Q: reverse "
            raise OCRTruncated("OCR hit the 1024-token cap before finishing")

    eng = LLMEngine.__new__(LLMEngine)
    eng.ocr = _TruncatingOCR()
    ui: asyncio.Queue = asyncio.Queue()

    text = await eng._ocr_screenshot(b"\xff\xd8", ui)
    statuses = [m["text"] for m in await _drain(ui) if m["type"] == "status"]

    assert text == "Q: reverse"
    assert any("cap" in s for s in statuses), statuses


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
    text = _FakeTextProvider([Delta(text="Use three pointers.")])
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
    assert [m["text"] for m in msgs if m["type"] == "token"] == ["Use three pointers."]
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
