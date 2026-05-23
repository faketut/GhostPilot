"""
Async connection tests for Settings UI.

Each `test_<provider>(params)` returns (ok: bool, message: str). Network calls
run in a background thread (QThreadPool) so the dialog never blocks. The
result is delivered to the UI on the Qt main thread via the callback.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

logger = logging.getLogger(__name__)

ResultCb = Callable[[bool, str], None]


class _Signals(QObject):
    done = pyqtSignal(bool, str)


class _TestJob(QRunnable):
    def __init__(self, fn, params, signals: _Signals):
        super().__init__()
        self.fn = fn
        self.params = params
        self.signals = signals
        self.setAutoDelete(True)

    def run(self):
        try:
            ok, msg = self.fn(self.params)
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"
        self.signals.done.emit(ok, msg)


def run_test(provider: str, params: dict, callback: ResultCb) -> None:
    fn = _PROVIDERS.get(provider)
    if fn is None:
        callback(False, f"unknown provider: {provider}")
        return
    sig = _Signals()
    sig.done.connect(lambda ok, msg: callback(ok, msg))
    QThreadPool.globalInstance().start(_TestJob(fn, params, sig))


# ── Provider tests (sync, run on worker thread) ──────────────────────────


def _timed(fn):
    def wrap(params):
        t0 = time.monotonic()
        ok, msg = fn(params)
        dt = int((time.monotonic() - t0) * 1000)
        return ok, f"{msg} · {dt}ms" if ok else msg
    return wrap


@_timed
def _test_openai_compat(params: dict, *, base_url: str | None, key_field: str, model_field: str, default_model: str):
    key = params.get(key_field, "").strip()
    if not key:
        return False, "no key"
    model = (params.get(model_field, "") or default_model).strip() or default_model
    try:
        from openai import OpenAI
    except Exception as e:
        return False, f"openai SDK missing: {e}"
    try:
        client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        _ = resp.choices[0].message.content if resp.choices else ""
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def test_openai(params):
    return _test_openai_compat(
        params, base_url=None,
        key_field="OPENAI_API_KEY", model_field="TEXT_MODEL", default_model="gpt-4o-mini",
    )


def test_deepseek(params):
    return _test_openai_compat(
        params, base_url="https://api.deepseek.com/v1",
        key_field="DEEPSEEK_API_KEY", model_field="TEXT_MODEL", default_model="deepseek-chat",
    )


@_timed
def test_gemini(params):
    key = params.get("GEMINI_API_KEY", "").strip()
    if not key:
        return False, "no key"
    model = (params.get("VISION_MODEL", "") or "gemini-1.5-flash").strip() or "gemini-1.5-flash"
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        return False, f"google-genai missing: {e}"
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=model,
            contents="ping",
            config=types.GenerateContentConfig(max_output_tokens=1, temperature=0),
        )
        _ = getattr(resp, "text", "")
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


@_timed
def test_azure(params):
    key = params.get("AZURE_SPEECH_KEY", "").strip()
    region = params.get("AZURE_SPEECH_REGION", "").strip()
    endpoint = params.get("AZURE_SPEECH_ENDPOINT", "").strip()
    if not key:
        return False, "no key"
    if not region and not endpoint:
        return False, "need region or endpoint"
    # Use Azure's token-issuing endpoint — a successful HTTP 200 with a JWT
    # body confirms key + region without needing the Speech SDK on macOS.
    import urllib.request
    import urllib.error
    if endpoint:
        # Endpoint URLs like https://<region>.api.cognitive.microsoft.com/sts/v1.0/issueToken
        url = endpoint.rstrip("/")
        if not url.endswith("/issueToken"):
            url = url + "/sts/v1.0/issueToken"
    else:
        url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    req = urllib.request.Request(url, method="POST")
    req.add_header("Ocp-Apim-Subscription-Key", key)
    req.add_header("Content-Length", "0")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            if r.status == 200:
                return True, "ok"
            return False, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")[:80]
        except Exception:
            pass
        return False, f"HTTP {e.code}: {body or e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


_PROVIDERS = {
    "openai": test_openai,
    "deepseek": test_deepseek,
    "gemini": test_gemini,
    "azure": test_azure,
}
