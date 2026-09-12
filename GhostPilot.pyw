"""GhostPilot — console-free launcher.

Double-click entry point for source checkouts (Windows associates ``.pyw`` with
``pythonw.exe``, which allocates no console window). Equivalent to
``python main.py``, but the app is a tray-resident background process: it must
have no console in the taskbar and no taskbar button for its overlays.

Usage:
    pythonw GhostPilot.pyw
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main  # noqa: E402  (needs sys.path above)

if __name__ == "__main__":
    try:
        main.main()
    except KeyboardInterrupt:
        pass
