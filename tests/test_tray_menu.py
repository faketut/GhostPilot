"""Tray menu action contract.

A tray item must run its handler whatever the handler's arity, and a checkable
item must hand its slot the new state. Both broke with
`QMenu.addAction(text, callable)`, which invokes the callable with no arguments:
a slot declared `def handler(self, checked)` then raised, and a checkable item's
slot silently saw nothing. These tests pin the contract so the convenience
overload cannot come back unnoticed.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMenu  # noqa: E402

from src.overlay_ui import _add_menu_action  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def _sole_action(menu: QMenu):
    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert len(actions) == 1
    return actions[0]


def test_zero_arg_handler_runs(qt_app):
    """A tray handler that takes no arguments still runs when its item is used."""
    menu = QMenu()
    hits: list[bool] = []
    action = _add_menu_action(menu, "Settings", lambda: hits.append(True))

    _sole_action(menu).trigger()

    assert action.text() == "Settings"
    assert hits == [True]


def test_checkable_action_delivers_new_state(qt_app):
    """A checkable item tells its handler whether it was turned on or off."""
    menu = QMenu()
    seen: list[bool] = []

    def handler(checked: bool = False) -> None:
        seen.append(checked)

    action = _add_menu_action(menu, "Start with Windows", handler, checkable=True)

    _sole_action(menu).trigger()
    assert action.isChecked() is True
    assert seen == [True], "handler must receive the new checked state"

    action.trigger()
    assert action.isChecked() is False
    assert seen == [True, False]
