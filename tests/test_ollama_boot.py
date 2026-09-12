"""Ollama bootstrap contract.

Ollama's installer registers a Startup-folder shortcut, so on a normal login it
is already running and this is just a probe. These tests pin the cases that
shortcut does not cover, and the boundaries: never manage a *remote* server, and
never spawn a second process while one is starting.
"""
from __future__ import annotations

import asyncio

import pytest

from src import ollama_boot


@pytest.fixture(autouse=True)
def _reset_boot_state():
    """Module-level state (cache, lock, in-flight task) must not leak between tests."""
    ollama_boot._reachable_until = 0.0
    ollama_boot._start_lock = None
    ollama_boot._inflight = None
    yield
    ollama_boot._reachable_until = 0.0
    ollama_boot._start_lock = None
    ollama_boot._inflight = None


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:11434/v1", True),
        ("http://localhost:11434", True),
        ("http://10.0.0.5:11434/v1", False),
        ("http://ollama.internal:11434/v1", False),
        ("https://ollama.example.com/v1", False),
    ],
)
def test_only_local_endpoints_are_managed(url, expected):
    """A configured remote OLLAMA_BASE_URL means someone else's server — starting
    a local process for that address would be useless and surprising."""
    assert ollama_boot.is_local_endpoint(url) is expected


@pytest.mark.asyncio
async def test_remote_endpoint_is_never_started(monkeypatch):
    spawned: list = []
    monkeypatch.setattr(ollama_boot, "probe", lambda *a, **k: False)
    monkeypatch.setattr(ollama_boot, "_spawn_detached", lambda cmd: spawned.append(cmd) or True)

    assert await ollama_boot.ensure_available("http://10.0.0.5:11434") is True
    assert spawned == []


@pytest.mark.asyncio
async def test_reachable_server_is_not_restarted(monkeypatch):
    spawned: list = []
    monkeypatch.setattr(ollama_boot, "probe", lambda *a, **k: True)
    monkeypatch.setattr(ollama_boot, "_spawn_detached", lambda cmd: spawned.append(cmd) or True)

    assert await ollama_boot.ensure_available("http://localhost:11434") is True
    assert spawned == [], "an already-running Ollama must not be spawned again"


@pytest.mark.asyncio
async def test_down_server_is_started_and_awaited(monkeypatch):
    """The whole point: a screenshot works after launching GhostPilot alone."""
    state = {"up": False, "spawned": []}

    def fake_probe(_root, timeout=2.0):
        return state["up"]

    def fake_spawn(cmd):
        state["spawned"].append(cmd)
        state["up"] = True  # the server comes up shortly after
        return True

    monkeypatch.setattr(ollama_boot, "probe", fake_probe)
    monkeypatch.setattr(ollama_boot, "_spawn_detached", fake_spawn)
    monkeypatch.setattr(ollama_boot, "find_launcher", lambda: ["ollama", "serve"])
    monkeypatch.setattr(ollama_boot, "_POLL_INTERVAL_SEC", 0.01)

    assert await ollama_boot.ensure_available("http://localhost:11434") is True
    assert state["spawned"] == [["ollama", "serve"]]


@pytest.mark.asyncio
async def test_missing_executable_reports_failure_without_raising(monkeypatch):
    """A failure here must not take down the pipeline."""
    monkeypatch.setattr(ollama_boot, "probe", lambda *a, **k: False)
    monkeypatch.setattr(ollama_boot, "find_launcher", lambda: None)

    assert await ollama_boot.ensure_available("http://localhost:11434") is False


@pytest.mark.asyncio
async def test_startup_timeout_is_not_fatal(monkeypatch):
    monkeypatch.setattr(ollama_boot, "probe", lambda *a, **k: False)
    monkeypatch.setattr(ollama_boot, "_spawn_detached", lambda cmd: True)
    monkeypatch.setattr(ollama_boot, "find_launcher", lambda: ["ollama", "serve"])
    monkeypatch.setattr(ollama_boot, "_POLL_INTERVAL_SEC", 0.01)

    assert await ollama_boot.ensure_available("http://localhost:11434", timeout=0.05) is False


@pytest.mark.asyncio
async def test_concurrent_callers_spawn_only_one_server(monkeypatch):
    """Two screenshots at once must not start two servers on the same port."""
    spawned: list = []
    monkeypatch.setattr(ollama_boot, "probe", lambda *a, **k: False)
    monkeypatch.setattr(ollama_boot, "_POLL_INTERVAL_SEC", 0.01)

    def fake_spawn(cmd):
        spawned.append(cmd)
        return True

    monkeypatch.setattr(ollama_boot, "_spawn_detached", fake_spawn)
    monkeypatch.setattr(ollama_boot, "find_launcher", lambda: ["ollama", "serve"])

    results = await asyncio.gather(
        ollama_boot.ensure_available("http://localhost:11434", timeout=0.05),
        ollama_boot.ensure_available("http://localhost:11434", timeout=0.05),
        ollama_boot.ensure_available("http://localhost:11434", timeout=0.05),
    )
    assert spawned == [["ollama", "serve"]], f"expected one spawn, got {spawned}"
    assert results == [False, False, False]


def test_autostart_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OLLAMA_AUTOSTART", "0")
    assert ollama_boot._autostart_enabled() is False
    monkeypatch.setenv("OLLAMA_AUTOSTART", "1")
    assert ollama_boot._autostart_enabled() is True
    monkeypatch.delenv("OLLAMA_AUTOSTART", raising=False)
    assert ollama_boot._autostart_enabled() is True  # default on


@pytest.mark.asyncio
async def test_disabled_toggle_skips_startup(monkeypatch):
    monkeypatch.setenv("OLLAMA_AUTOSTART", "0")
    spawned: list = []
    monkeypatch.setattr(ollama_boot, "probe", lambda *a, **k: False)
    monkeypatch.setattr(ollama_boot, "_spawn_detached", lambda cmd: spawned.append(cmd) or True)

    assert await ollama_boot.ensure_available_for_config() is True
    assert spawned == []


def test_windows_prefers_the_tray_app(monkeypatch, tmp_path):
    """The tray application is what the installer's Startup entry runs, so
    restarting it restores the familiar icon rather than a bare server."""
    fake_dir = tmp_path
    cli = fake_dir / "ollama.exe"
    tray = fake_dir / "ollama app.exe"
    cli.write_bytes(b"")
    tray.write_bytes(b"")
    monkeypatch.setattr(ollama_boot.shutil, "which", lambda _n: str(cli))
    monkeypatch.setattr(ollama_boot.sys, "platform", "win32")

    assert ollama_boot.find_launcher() == [str(tray)]


def test_launcher_falls_back_to_serve(monkeypatch, tmp_path):
    cli = tmp_path / "ollama"
    cli.write_bytes(b"")
    monkeypatch.setattr(ollama_boot.shutil, "which", lambda _n: str(cli))
    monkeypatch.setattr(ollama_boot.sys, "platform", "linux")

    assert ollama_boot.find_launcher() == [str(cli), "serve"]


def test_no_launcher_when_ollama_is_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_boot.shutil, "which", lambda _n: None)
    monkeypatch.setattr(ollama_boot, "_windows_candidates", lambda: [tmp_path / "nope.exe"])
    monkeypatch.setattr(ollama_boot, "_other_candidates", lambda: [tmp_path / "nope"])

    assert ollama_boot.find_launcher() is None
