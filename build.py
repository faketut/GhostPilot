#!/usr/bin/env python3
"""
build.py — PyInstaller deployment script for GhostPilot
=========================================================
Produces a single-file Windows executable with process disguise.

Usage:
    python build.py [--name PROCESS_NAME] [--onefile | --onedir] [--upx]

Examples:
    python build.py                          # default: ApplicationFrameHost.exe
    python build.py --name "SystemSettings"  # disguise as Settings
    python build.py --onedir                 # faster start, folder bundle

Process Disguise Note:
    The exe is named after a common, innocuous Windows process to reduce
    curiosity from basic process-list checks.  Windows Defender will NOT
    flag a legitimately signed binary; unsigned builds may show SmartScreen
    prompts on first run — that is expected.
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
DEFAULT_NAME = "ApplicationFrameHost"   # disguise name (no .exe — PyInstaller adds it)
ENTRY_POINT  = "main.py"
ICON_PATH    = "assets/icon.ico"        # optional; skipped if missing
DIST_DIR     = "dist"
BUILD_DIR    = "build"

# Data files to bundle (src → dest_folder_in_bundle)
EXTRA_DATA = [
    ("prompts", "prompts"),
    ("knowledge", "knowledge"),
]

# Hidden imports that PyInstaller sometimes misses
HIDDEN_IMPORTS = [
    "azure.cognitiveservices.speech",
    "pyaudiowpatch",
    "qasync",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "mss",
    "PIL",
    "PIL.Image",
    "numpy",
    "sentence_transformers",
    "onnxruntime",
]

# ── Helpers ────────────────────────────────────────────────────────────────

def ensure_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build(name: str, onefile: bool, use_upx: bool):
    print(f"\n{'='*60}")
    print(f"  Building GhostPilot  →  {name}.exe")
    print(f"  Mode   : {'--onefile' if onefile else '--onedir'}")
    print(f"  UPX    : {'yes' if use_upx else 'no'}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", name,
        "--noconfirm",
        "--clean",
        "--distpath", DIST_DIR,
        "--workpath", BUILD_DIR,
        # Window mode: no console (stealth)
        "--noconsole",
        "--log-level", "WARN",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Icon
    icon = Path(ICON_PATH)
    if icon.exists():
        cmd += ["--icon", str(icon)]
    else:
        print(f"[WARN] Icon not found at {ICON_PATH} — building without custom icon.")

    # Extra data
    for src, dst in EXTRA_DATA:
        if Path(src).exists():
            cmd += ["--add-data", f"{src}{';' if sys.platform=='win32' else ':'}{dst}"]

    # Hidden imports
    for hi in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", hi]

    # UPX compression (reduces exe size by ~40%)
    if not use_upx:
        cmd.append("--noupx")

    # Exclude heavy unused packages to keep size down
    for excl in ["tkinter", "matplotlib", "scipy", "IPython", "notebook"]:
        cmd += ["--exclude-module", excl]

    cmd.append(ENTRY_POINT)

    print("Running:", " ".join(cmd[:8]), "...\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[ERROR] Build failed — check output above.")
        sys.exit(1)

    exe_path = (
        Path(DIST_DIR) / f"{name}.exe"
        if onefile
        else Path(DIST_DIR) / name / f"{name}.exe"
    )
    print(f"\n✅  Build complete!  →  {exe_path.resolve()}")
    print(f"    Size: {exe_path.stat().st_size / 1_048_576:.1f} MB")

    # Copy .env.example next to the exe for easy setup
    env_example = Path(".env.example")
    if env_example.exists():
        dest_dir = exe_path.parent
        shutil.copy(env_example, dest_dir / ".env.example")
        print(f"    Copied .env.example → {dest_dir / '.env.example'}")

    print("\nQuick-start:")
    print(f"  1. Copy  {exe_path.parent / '.env.example'}  →  {exe_path.parent / '.env'}")
    print(f"  2. Fill in your Azure Speech Key and LLM API keys in .env")
    print(f"  3. Run   {exe_path.name}")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build GhostPilot with PyInstaller")
    parser.add_argument(
        "--name", default=DEFAULT_NAME,
        help=f"Output executable name (default: {DEFAULT_NAME})"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--onefile", action="store_true", default=True,
                      help="Bundle into a single .exe (default)")
    mode.add_argument("--onedir", action="store_false", dest="onefile",
                      help="Bundle into a folder (faster startup)")
    parser.add_argument("--upx", action="store_true",
                        help="Enable UPX compression (requires UPX in PATH)")
    parser.add_argument("--size-report", action="store_true",
                        help="After building, print the top 10 largest bundled packages.")
    args = parser.parse_args()

    ensure_pyinstaller()
    build(name=args.name, onefile=args.onefile, use_upx=args.upx)
    if args.size_report:
        _size_report(args.name)


def _size_report(name: str) -> None:
    """Walk dist/<name>/ and print the top-10 largest directories."""
    from collections import defaultdict
    root = Path(DIST_DIR) / name
    if not root.exists():
        # onefile mode → no folder. Just print the .exe size.
        exe = Path(DIST_DIR) / f"{name}.exe"
        if exe.exists():
            print(f"\n{exe.name}: {exe.stat().st_size / 1_048_576:.1f} MB")
        return
    sizes: dict[str, int] = defaultdict(int)
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root)
            top = rel.parts[0] if rel.parts else "(root)"
            sizes[top] += p.stat().st_size
    print("\nTop 10 largest bundled packages:")
    for top, sz in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {sz / 1_048_576:6.1f} MB  {top}")


if __name__ == "__main__":
    main()
