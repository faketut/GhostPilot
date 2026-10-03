"""Model registry / downloader tests (no network — ``_fetch`` is stubbed)."""
from __future__ import annotations

import io
import tarfile

import pytest

from src import sherpa_models as sm


# ── path mode: a user-supplied, already-extracted directory ──────────────


def _write_model_dir(d: sm.Path, *, encoder: str = "encoder-epoch-99-avg-1.int8.onnx",
                     decoder: str = "decoder-epoch-99-avg-1.int8.onnx",
                     joiner: str = "joiner-epoch-99-avg-1.int8.onnx") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokens.txt").write_text("a 1\n", encoding="utf-8")
    for name in (encoder, decoder, joiner):
        if name:
            (d / name).write_bytes(b"onnx")


def test_path_mode_resolves_int8_files(tmp_path):
    model = tmp_path / "my-model"
    _write_model_dir(model)
    files = sm.resolve(str(model), allow_download=False)
    assert sm.Path(files["tokens"]).parent == model
    assert files["encoder"].endswith("encoder-epoch-99-avg-1.int8.onnx")
    assert files["joiner"].endswith("joiner-epoch-99-avg-1.int8.onnx")


def test_path_mode_prefers_int8_over_fp32(tmp_path):
    """fp32 decodes 2-4x slower and is 3-5x larger; picking it silently is a real bug."""
    model = tmp_path / "both"
    _write_model_dir(
        model,
        encoder="encoder-epoch-99-avg-1.onnx",
        decoder="decoder-epoch-99-avg-1.onnx",
        joiner="joiner-epoch-99-avg-1.onnx",
    )
    _write_model_dir(model)  # adds the int8 siblings
    files = sm.resolve(str(model), allow_download=False)
    assert ".int8." in files["encoder"]
    assert ".int8." in files["decoder"]
    assert ".int8." in files["joiner"]


def test_path_mode_rejects_incomplete_directory(tmp_path):
    model = tmp_path / "incomplete"
    model.mkdir()
    (model / "tokens.txt").write_text("a 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no sherpa-onnx model found"):
        sm.resolve(str(model), allow_download=False)


def test_path_mode_rejects_directory_without_encoder(tmp_path):
    """A no-state/with-state variant carries no 'encoder' in its name → unusable."""
    model = tmp_path / "variant"
    _write_model_dir(model, encoder="no-state-epoch-99-avg-1.int8.onnx")
    with pytest.raises(RuntimeError, match="no sherpa-onnx model found"):
        sm.resolve(str(model), allow_download=False)


# ── registry ─────────────────────────────────────────────────────────────


def test_unknown_registry_name_lists_alternatives():
    with pytest.raises(ValueError, match="zipformer-bilingual-zh-en"):
        sm.get("not-a-model")


def test_registry_entry_not_downloaded_reports_missing_files(tmp_path):
    entry = sm.get("zipformer-en-20M")
    assert not sm.is_ready(entry, tmp_path)
    assert set(sm.missing_files(entry, tmp_path)) == set(entry.files().values())
    with pytest.raises(RuntimeError, match="setup_sherpa_asr.py"):
        sm.resolve(entry.name, root=tmp_path, allow_download=False)


def test_every_registry_entry_declares_the_files_it_needs():
    for entry in sm.MODELS.values():
        files = entry.files()
        assert files["tokens"] == "tokens.txt"
        assert {"encoder", "decoder"} <= set(files)
        assert ("joiner" in files) == (entry.arch in ("transducer",))


def test_entries_may_override_the_epoch_filename_convention():
    """Kroko ships plain encoder/decoder/joiner.onnx; assuming the epoch-99
    naming would have failed at load with 'file does not exist'."""
    kroko = sm.get("zipformer-en-kroko")
    assert kroko.files() == {
        "tokens": "tokens.txt",
        "encoder": "encoder.onnx",
        "decoder": "decoder.onnx",
        "joiner": "joiner.onnx",
    }
    # An entry without an override still follows the convention.
    assert sm.get("zipformer-en-20M").files()["encoder"] == "encoder-epoch-99-avg-1.int8.onnx"


def test_an_override_is_found_by_folder_name_too(tmp_path):
    entry = sm.get("zipformer-en-kroko")
    assert sm.find(entry.folder) is entry
    assert sm.find("zipformer-en-kroko") is entry
    # And the override survives a download/resolve round trip.
    members = {f"{entry.folder}/{name}": b"onnx" for name in entry.files().values()}
    d = tmp_path / entry.folder
    d.mkdir()
    for name in entry.files().values():
        (d / name).write_bytes(b"onnx")
    assert sm.is_ready(entry, tmp_path)
    files = sm.resolve(entry.name, root=tmp_path, allow_download=False)
    assert files["encoder"].endswith("encoder.onnx")
    assert members  # documents the archive shape used above


# ── download ─────────────────────────────────────────────────────────────


def _fake_fetch_from(members: dict[str, bytes]):
    """Stand-in for ``_fetch``: writes a tar.bz2 built from ``members``."""

    def _fetch(url: str, dest: sm.Path, *, on_progress=None) -> int:  # noqa: ARG001
        with tarfile.open(dest, "w:bz2") as tf:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
        return dest.stat().st_size

    return _fetch


def test_download_extracts_and_leaves_a_usable_model(tmp_path, monkeypatch):
    entry = sm.get("zipformer-en-20M")
    members = {f"{entry.folder}/{name}": b"onnx" for name in entry.files().values()}
    monkeypatch.setattr(sm, "_fetch", _fake_fetch_from(members))

    target = sm.download(entry, root=tmp_path)

    assert target == tmp_path / entry.folder
    assert sm.is_ready(entry, tmp_path)
    files = sm.resolve(entry.name, root=tmp_path, allow_download=False)
    assert sm.Path(files["tokens"]).is_file()
    # The archive itself is not kept — only the extracted model.
    assert not (tmp_path / f"{entry.folder}.tar.bz2").exists()


def test_download_leaves_nothing_behind_when_layout_is_wrong(tmp_path, monkeypatch):
    """An unexpected archive must not produce a directory that resolves as ready."""
    entry = sm.get("zipformer-en-20M")
    members = {f"some-other-folder/{name}": b"onnx" for name in entry.files().values()}
    monkeypatch.setattr(sm, "_fetch", _fake_fetch_from(members))

    with pytest.raises(RuntimeError, match="unexpected archive layout"):
        sm.download(entry, root=tmp_path)

    assert not (tmp_path / entry.folder).exists()
    assert not sm.is_ready(entry, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_escaping_archive_members(tmp_path, monkeypatch):
    entry = sm.get("zipformer-en-20M")
    members = {f"{entry.folder}/tokens.txt": b"a 1\n", "../escaped.txt": b"pwned"}
    monkeypatch.setattr(sm, "_fetch", _fake_fetch_from(members))

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        sm.download(entry, root=tmp_path)

    assert not (tmp_path.parent / "escaped.txt").exists()
    assert not (tmp_path / entry.folder).exists()
