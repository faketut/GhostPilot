import ctypes
import sys
import logging

logger = logging.getLogger(__name__)

# Windows API Constants
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000

def enable_window_stealth(hwnd: int) -> bool:
    """
    Applies WDA_EXCLUDEFROMCAPTURE to the given window handle, making it invisible to screen capture.
    Requires Windows 10 Build 19041 (2004) or later.
    """
    if sys.platform != "win32":
        logger.warning("Stealth mode is only supported on Windows.")
        return False
        
    try:
        user32 = ctypes.windll.user32
        result = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        if result:
            logger.info("Window stealth mode enabled successfully.")
            return True
        else:
            logger.error(f"Failed to set window display affinity. Error code: {ctypes.GetLastError()}")
            return False
    except Exception as e:
        logger.error(f"Exception while setting window stealth: {e}")
        return False

def set_window_interaction_mode(hwnd: int, interactive: bool):
    """
    Toggles the WS_EX_TRANSPARENT extended window style.
    If interactive is True, the style is removed, allowing mouse clicks.
    If interactive is False, the style is added, making the window click-through.
    """
    if sys.platform != "win32":
        return
    try:
        user32 = ctypes.windll.user32
        ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        
        if interactive:
            # Remove WS_EX_TRANSPARENT
            new_style = ex_style & ~WS_EX_TRANSPARENT
            logger.info("Window set to INTERACTIVE mode (draggable).")
        else:
            # Add WS_EX_TRANSPARENT (and WS_EX_LAYERED which is required for transparent)
            new_style = ex_style | WS_EX_TRANSPARENT | WS_EX_LAYERED
            logger.info("Window set to CLICK-THROUGH mode (stealth).")
            
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
    except Exception as e:
        logger.error(f"Failed to set interaction mode: {e}")

def set_dpi_awareness():
    """
    Sets the process DPI awareness to Per-Monitor V2 to ensure correct screenshot coordinates 
    and UI rendering on high-DPI displays.
    """
    if sys.platform != "win32":
        return
        
    try:
        # SetProcessDpiAwarenessContext (Windows 10 1607+)
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        logger.info("DPI Awareness set to Per-Monitor V2")
    except AttributeError:
        try:
            # Fallback for older Windows 8.1
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            logger.info("DPI Awareness set to Per-Monitor (fallback)")
        except Exception as e:
            logger.warning(f"Could not set DPI awareness: {e}")
