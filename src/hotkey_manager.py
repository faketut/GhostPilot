import keyboard
import logging
import asyncio
from typing import Callable

logger = logging.getLogger(__name__)

class HotkeyManager:
    def __init__(
        self,
        hotkey: str,
        callback: Callable,
        loop=None,
        *,
        suppress: bool = False,
        trigger_on_release: bool = False,
    ):
        self.hotkey = hotkey
        self.callback = callback
        self.loop = loop
        self.suppress = suppress
        self.trigger_on_release = trigger_on_release
        self._is_running = False
        self._hotkey_handle = None

    def start(self):
        """Starts listening for the hotkey in the background."""
        if self._is_running:
            return
            
        try:
            # suppress=True prevents the key from being sent to other apps
            self._hotkey_handle = keyboard.add_hotkey(
                self.hotkey,
                self._on_hotkey_pressed,
                suppress=self.suppress,
                trigger_on_release=self.trigger_on_release,
            )
            self._is_running = True
            logger.info(f"Started listening for hotkey: {self.hotkey}")
        except Exception as e:
            logger.error(f"Failed to register hotkey {self.hotkey}: {e}")

    def _on_hotkey_pressed(self):
        logger.info(f"Hotkey {self.hotkey} pressed.")
        if asyncio.iscoroutinefunction(self.callback):
            if self.loop:
                try:
                    asyncio.run_coroutine_threadsafe(self.callback(), self.loop)
                except Exception as e:
                    logger.error(f"Hotkey coroutine schedule failed ({self.hotkey}): {e}")
            else:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.callback())
                except RuntimeError:
                    asyncio.run(self.callback())
        else:
            if self.loop:
                def _safe_call():
                    try:
                        self.callback()
                    except Exception as e:
                        logger.error(f"Hotkey callback failed ({self.hotkey}): {e}")
                self.loop.call_soon_threadsafe(_safe_call)
            else:
                try:
                    self.callback()
                except Exception as e:
                    logger.error(f"Hotkey callback failed ({self.hotkey}): {e}")

    def stop(self):
        """Stops listening for hotkeys."""
        if self._is_running:
            try:
                if self._hotkey_handle is not None:
                    keyboard.remove_hotkey(self._hotkey_handle)
                else:
                    keyboard.remove_hotkey(self.hotkey)
            finally:
                self._hotkey_handle = None
            self._is_running = False
            logger.info("Stopped listening for hotkeys.")
