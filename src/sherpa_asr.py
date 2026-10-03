"""Local ASR backend using sherpa-onnx (k2-fsa) — streaming, offline, no key.

Slots in behind the same duck-typed surface as ``ASRClient``
(``start_streaming(audio_queue, text_queue, loop)`` + ``stop()``) so ``main.py``
can swap implementations via ``config.ASR_BACKEND``.

Unlike ``FasterWhisperASRClient``, the zipformer/paraformer models used here are
genuinely streaming: partials arrive as words are spoken and the model's own
endpoint detector decides where an utterance ends. That replaces both the cloud
service *and*, for this backend, the app-level "finalize once no new partial has
arrived for N ms" heuristic (``ASR_PARTIAL_SILENCE_MS``) — though that heuristic
still runs in ``main.py``'s router for every backend, see the timing note below.

Two details are load-bearing:

* ``rule2_min_trailing_silence`` (end of utterance after speech) MUST stay well
  below ``ASR_PARTIAL_SILENCE_MS``: the router also finalizes from the last
  partial on that timer, and whichever fires first wins. If the timer won, the
  utterance would be answered twice — once from the partial, once from our
  final. Default: 0.5 s model-side vs the router's 0.65 s.
* Decoding runs in a worker thread. The bindings release the GIL but still
  occupy the calling thread, and the event loop here is the Qt loop — blocking
  it would freeze both overlays and the hotkey watchers.
"""
from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

from src.config import config
from src import sherpa_models

logger = logging.getLogger(__name__)

# 16 kHz mono int16 — the format AudioCapture produces.
SAMPLE_RATE = 16000

# Endpoint rule defaults (seconds). See the class docstring for rule2's constraint.
RULE1_SILENCE_SEC = 2.4    # no text decoded yet → cut the silence, not a sentence
RULE2_SILENCE_SEC = 0.5    # after speech → end of utterance
RULE3_UTTERANCE_SEC = 20.0  # hard cap on one utterance


def pcm16_to_float(chunk: bytes) -> np.ndarray:
    """int16 little-endian PCM → float32 in [-1, 1], the shape sherpa expects."""
    if len(chunk) & 1:
        # A half sample cannot exist; dropping the trailing byte keeps
        # frombuffer from raising on every subsequent chunk.
        chunk = chunk[:-1]
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    samples *= 1.0 / 32768.0
    return samples


class SherpaOnnxASRClient:
    def __init__(
        self,
        *,
        model: str = sherpa_models.DEFAULT_MODEL,
        model_dir: str = "",
        num_threads: int = 0,
        provider: str = "cpu",
        decoding_method: str = "",
        hotwords_file: str = "",
        hotwords_score: float = 1.5,
        modeling_unit: str = "cjkchar+bpe",
        rule1_min_trailing_silence: float = RULE1_SILENCE_SEC,
        rule2_min_trailing_silence: float = RULE2_SILENCE_SEC,
        rule3_min_utterance_length: float = RULE3_UTTERANCE_SEC,
        autodownload: bool = True,
    ) -> None:
        self.model = model or sherpa_models.DEFAULT_MODEL
        self.model_dir = model_dir or ""
        self.provider = provider or "cpu"
        self.hotwords_file = (hotwords_file or "").strip()
        self.hotwords_score = float(hotwords_score)
        self.modeling_unit = modeling_unit or "cjkchar+bpe"
        self.rule1_min_trailing_silence = float(rule1_min_trailing_silence)
        self.rule2_min_trailing_silence = float(rule2_min_trailing_silence)
        self.rule3_min_utterance_length = float(rule3_min_utterance_length)
        self.autodownload = bool(autodownload)
        self.num_threads = int(num_threads) if int(num_threads) > 0 else min(4, os.cpu_count() or 2)
        # Hotwords only bias modified_beam_search; asking for them under greedy
        # would silently do nothing.
        self.decoding_method = (decoding_method or "").strip() or (
            "modified_beam_search" if self.hotwords_file else "greedy_search"
        )
        self._recognizer = None  # lazy: the model is a large file read
        self._is_running = False

        router_finalize_sec = getattr(config, "ASR_PARTIAL_SILENCE_MS", 650) / 1000.0
        if self.rule2_min_trailing_silence >= router_finalize_sec:
            logger.warning(
                "SHERPA_RULE2_SILENCE_SEC (%.2fs) is not below ASR_PARTIAL_SILENCE_MS "
                "(%.2fs): the router's partial timer will answer an utterance just "
                "before this backend's final does, producing a duplicate answer. "
                "Keep the sherpa value lower.",
                self.rule2_min_trailing_silence, router_finalize_sec,
            )
        self._warn_if_language_unsupported()

    def _model_languages(self) -> tuple[str, ...]:
        """Languages the configured model was trained on; () when unknowable.

        ``SHERPA_MODEL`` is a registry name or a path to an extracted model
        directory, so a path is matched by its folder name — a directory
        downloaded by name (or pointed at its parent) is still recognisable.
        """
        candidates = [self.model, os.path.basename(self.model.rstrip("/\\"))]
        if self.model_dir:
            candidates.append(os.path.basename(self.model_dir.rstrip("/\\")))
        for name in candidates:
            entry = sherpa_models.find(name)
            if entry is not None:
                return entry.languages
        return ()

    def _warn_if_language_unsupported(self) -> None:
        """``ASR_LANGUAGE`` is an Azure setting; here the *model* fixes the language.

        Leaving the mismatch silent is the expensive failure: English audio
        through a zh-only model transcribes to nothing (or to nonsense) while
        every status line still looks healthy, and the user debugs the LLM.
        """
        wanted = (getattr(config, "ASR_LANGUAGE", "") or "").split("-")[0].strip().lower()
        covered = self._model_languages()
        if not wanted or not covered:
            return
        if wanted in covered:
            logger.info(
                "sherpa-onnx model '%s' covers %s; ASR_LANGUAGE=%s is satisfied "
                "(ASR_LANGUAGE itself is only sent to Azure).",
                self.model, "/".join(covered), wanted,
            )
            return
        logger.warning(
            "sherpa-onnx model '%s' is trained on %s, but ASR_LANGUAGE=%s. This "
            "backend cannot switch language at runtime — audio in %s will not "
            "transcribe. Set SHERPA_MODEL to a %s model (see "
            "`python setup_sherpa_asr.py --list`).",
            self.model, "/".join(covered), wanted, wanted, wanted,
        )

    # ── lazy model + recognizer load (both run in a worker thread) ────────
    def _get_recognizer(self):
        if self._recognizer is not None:
            return self._recognizer
        try:
            import sherpa_onnx
        except Exception as e:
            raise RuntimeError(
                "sherpa-onnx is not installed. Install with `pip install sherpa-onnx`."
            ) from e

        files = sherpa_models.resolve(
            self.model, root=self.model_dir or None, allow_download=self.autodownload
        )
        logger.info(
            "Loading sherpa-onnx model '%s' (%s, %d threads) …",
            self.model, self.provider, self.num_threads,
        )
        common = dict(
            tokens=files["tokens"],
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=self.rule1_min_trailing_silence,
            rule2_min_trailing_silence=self.rule2_min_trailing_silence,
            rule3_min_utterance_length=self.rule3_min_utterance_length,
            decoding_method=self.decoding_method,
            provider=self.provider,
        )
        if files.get("joiner"):
            kwargs = dict(
                common,
                encoder=files["encoder"],
                decoder=files["decoder"],
                joiner=files["joiner"],
                max_active_paths=4,
            )
            # `reset_encoder=True` is deliberately NOT set. It clears the encoder
            # state when endpoint rule1 fires (trailing silence, even with no
            # speech decoded), and measurement showed that discards the words
            # already decoded for the current utterance: with zipformer-en-20M,
            # "tell me about a time when you had to deliver…" came back as
            # "T UNDER A VERY TIGHT DEAD LINE". It bought nothing measurable, so
            # the default (False) keeps whatever audio has already been decoded.
            if self.hotwords_file:
                kwargs["hotwords_file"] = self.hotwords_file
                kwargs["hotwords_score"] = self.hotwords_score
                kwargs["modeling_unit"] = self.modeling_unit
                bpe_vocab = os.path.join(os.path.dirname(files["tokens"]), "bpe.vocab")
                if os.path.isfile(bpe_vocab):
                    kwargs["bpe_vocab"] = bpe_vocab
                logger.info(
                    "sherpa-onnx hotwords: %s (score %.2f, unit %s)",
                    self.hotwords_file, self.hotwords_score, self.modeling_unit,
                )
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        else:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                **common, encoder=files["encoder"], decoder=files["decoder"],
            )
        return self._recognizer

    def _feed(self, stream, samples: np.ndarray) -> tuple[str, bool]:
        """Push audio through the recognizer; return (partial text, endpoint).

        Runs on a worker thread: every binding it calls releases the GIL but
        still occupies the caller, so this must not be the Qt event loop thread.

        ``get_result_all`` (not ``get_result``, which returns a bare stripped
        string and drops ``is_final``) is the API that carries the result object.
        """
        if samples.size == 0:
            return "", False
        stream.accept_waveform(SAMPLE_RATE, samples)
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        text = self._recognizer.get_result_all(stream).text.strip()
        return text, self._recognizer.is_endpoint(stream)

    # ── public interface (duck-types ASRClient) ──────────────────────────
    async def start_streaming(
        self,
        audio_queue: asyncio.Queue,
        text_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        try:
            recognizer = await loop.run_in_executor(None, self._get_recognizer)
        except Exception as e:
            # One actionable line instead of a per-chunk error flood: without a
            # recognizer there is nothing this backend can do.
            logger.error(f"sherpa-onnx ASR unavailable, transcription disabled: {e}")
            return

        self._is_running = True
        stream = recognizer.create_stream()
        logger.info(
            "sherpa-onnx ASR streaming starting (model=%s, endpoint rule2=%.2fs)…",
            self.model, self.rule2_min_trailing_silence,
        )
        # Last text pushed for the *current* utterance; the recognizer repeats
        # the same partial for every decoded frame, and the overlay must not be
        # repainted (or the router's timer restarted) for an unchanged string.
        last_partial = ""
        try:
            while self._is_running:
                try:
                    chunk = await audio_queue.get()
                except asyncio.CancelledError:
                    break
                if not chunk:
                    continue
                samples = pcm16_to_float(chunk)
                try:
                    text, endpoint = await loop.run_in_executor(None, self._feed, stream, samples)
                except Exception as e:
                    logger.error(f"sherpa-onnx decode failed: {e}")
                    continue

                if text != last_partial:
                    if text:
                        text_queue.put_nowait({"type": "partial", "text": text, "speaker": "Unknown"})
                    last_partial = text
                if endpoint:
                    if text:
                        text_queue.put_nowait({"type": "final", "text": text, "speaker": "Unknown"})
                    # The next utterance decodes against a fresh stream — the
                    # router's per-question state is keyed off this final.
                    recognizer.reset(stream)
                    last_partial = ""
        finally:
            logger.info("sherpa-onnx ASR streaming stopped.")

    def stop(self) -> None:
        self._is_running = False
