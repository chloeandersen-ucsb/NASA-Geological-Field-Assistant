#!/usr/bin/env python3
"""
joystick_navigator.py  –  SparkFun Qwiic Joystick → Qt button focus navigator
"""

import time
import sys
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QPushButton, QWidget, QAbstractButton,
    QStackedWidget, QTextEdit, QListWidget,
)
from PySide6.QtGui import QKeyEvent

# ── I2C / joystick constants ────────────────────────────────────────────────
JOYSTICK_ADDR  = 0x20
REG_X_MSB      = 0x03
REG_X_LSB      = 0x04
REG_Y_MSB      = 0x05
REG_Y_LSB      = 0x06
REG_BUTTON     = 0x07

CENTER         = 512
DEADZONE       = 60
POLL_HZ        = 30
INITIAL_DELAY  = 0.40
REPEAT_DELAY   = 0.18

# HIGHLIGHT_BG   = "#f0c040"
# HIGHLIGHT_FG   = "#1a1a1a"
# HIGHLIGHT_BOR  = "#c8960a"

HIGHLIGHT_BG   = "#2f473b"
HIGHLIGHT_FG   = "#d6dcd9"
HIGHLIGHT_BOR  = "#15201a"


def _apply_highlight(widget: QWidget, original_style: str) -> str:
    style = original_style.strip()
    if "{" in style:
        override = (
            f" QPushButton {{ background-color: {HIGHLIGHT_BG} !important; "
            f"color: {HIGHLIGHT_FG} !important; "
            f"border: 3px solid {HIGHLIGHT_BOR} !important; }}"
        )
        return style + override
    else:
        return style + (
            f" background-color: {HIGHLIGHT_BG};"
            f" color: {HIGHLIGHT_FG};"
            f" border: 3px solid {HIGHLIGHT_BOR};"
        )


class _JoystickWorker(QObject):
    move_up    = Signal()
    move_down  = Signal()
    move_left  = Signal()
    move_right = Signal()
    clicked    = Signal()
    error      = Signal(str)

    def __init__(self, bus_number: int = 8, addr: int = JOYSTICK_ADDR):
        super().__init__()
        self.bus_number = bus_number
        self.addr = addr
        self._running = False
        self._last_dir: Optional[str] = None
        self._dir_start: float = 0.0
        self._last_repeat: float = 0.0
        self._btn_was_pressed = False

    def start(self):
        self._running = True
        self._run()

    def stop(self):
        self._running = False

    def _read(self, bus):
        x_msb = bus.read_byte_data(self.addr, REG_X_MSB)
        x_lsb = bus.read_byte_data(self.addr, REG_X_LSB)
        y_msb = bus.read_byte_data(self.addr, REG_Y_MSB)
        y_lsb = bus.read_byte_data(self.addr, REG_Y_LSB)
        btn   = bus.read_byte_data(self.addr, REG_BUTTON)
        x = (x_msb << 2) | (x_lsb >> 6)
        y = (y_msb << 2) | (y_lsb >> 6)
        return x, y, (btn == 0)

    def _direction(self, x: int, y: int) -> Optional[str]:
        dx = x - CENTER
        dy = y - CENTER
        if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
            return None
        if abs(dx) >= abs(dy):
            return "left" if dx > 0 else "right"
        else:
            return "up" if dy > 0 else "down"

    def _run(self):
        try:
            import smbus2
        except ImportError:
            self.error.emit("smbus2 not installed: pip3 install smbus2")
            return
        try:
            bus = smbus2.SMBus(self.bus_number)
        except Exception as e:
            self.error.emit(f"Cannot open I2C bus {self.bus_number}: {e}")
            return
        try:
            bus.read_byte(self.addr)
        except OSError:
            self.error.emit(
                f"No joystick found at {self.addr:#x} on bus {self.bus_number}. "
                "Check wiring or run: i2cdetect -y -r <bus>"
            )
            bus.close()
            return

        interval = 1.0 / POLL_HZ
        while self._running:
            t0 = time.monotonic()
            try:
                x, y, btn_pressed = self._read(bus)
            except OSError as e:
                self.error.emit(f"I2C read error: {e} – retrying in 1 s")
                time.sleep(1.0)
                continue

            direction = self._direction(x, y)
            now = time.monotonic()

            if direction is None:
                self._last_dir = None
            else:
                if direction != self._last_dir:
                    self._fire(direction)
                    self._last_dir = direction
                    self._dir_start = now
                    self._last_repeat = now
                else:
                    held = now - self._dir_start
                    since_last = now - self._last_repeat
                    if held >= INITIAL_DELAY and since_last >= REPEAT_DELAY:
                        self._fire(direction)
                        self._last_repeat = now

            if btn_pressed and not self._btn_was_pressed:
                self.clicked.emit()
            self._btn_was_pressed = btn_pressed

            elapsed = time.monotonic() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        bus.close()

    def _fire(self, direction: str):
        sig = {
            "up":    self.move_up,
            "down":  self.move_down,
            "left":  self.move_left,
            "right": self.move_right,
        }.get(direction)
        if sig:
            sig.emit()


class JoystickNavigator(QObject):
    """
    Focus model:
    - Cursor is either on a QPushButton OR inside a QListWidget (on a specific row).
    - Up/Down on a button: move between button rows, entering lists when encountered.
    - Up/Down inside a list: scroll rows; exit top/bottom to return to buttons.
    - Left/Right inside a list: exit the list to nearest button above/below.
    - Click: fires the focused button or emits itemClicked on the focused list row.
    """

    def __init__(self, window, bus: int = 8, addr: int = JOYSTICK_ADDR):
        super().__init__(window)
        self.window = window

        self._focused_btn: Optional[QPushButton] = None
        self._focused_btn_original_style: str = ""

        self._focused_list: Optional[QListWidget] = None
        self._focused_list_row: int = -1
        self._focused_list_original_style: str = ""

        self._thread = QThread(self)
        self._worker = _JoystickWorker(bus, addr)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.start)
        self._worker.move_up.connect(self._on_up)
        self._worker.move_down.connect(self._on_down)
        self._worker.move_left.connect(self._on_left)
        self._worker.move_right.connect(self._on_right)
        self._worker.clicked.connect(self._on_click)
        self._worker.error.connect(self._on_error)

        if hasattr(window, "stack"):
            window.stack.currentChanged.connect(self._on_page_changed)

    def start(self):
        self._thread.start()

    def stop(self):
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)

    # ── State helpers ────────────────────────────────────────────────────────

    def _in_list(self) -> bool:
        return self._focused_list is not None

    def _clear_btn_highlight(self):
        if self._focused_btn is not None:
            try:
                self._focused_btn.setStyleSheet(self._focused_btn_original_style)
            except RuntimeError:
                pass
            self._focused_btn = None
            self._focused_btn_original_style = ""

    def _highlight_btn(self, btn: QPushButton):
        self._clear_btn_highlight()
        self._clear_list_highlight()
        original = btn.styleSheet()
        self._focused_btn = btn
        self._focused_btn_original_style = original
        btn.setStyleSheet(_apply_highlight(btn, original))
        btn.setFocus(Qt.OtherFocusReason)

    def _clear_list_highlight(self):
        if self._focused_list is not None:
            try:
                self._focused_list.clearSelection()
                self._focused_list.setStyleSheet(self._focused_list_original_style)
            except RuntimeError:
                pass
            self._focused_list = None
            self._focused_list_row = -1
            self._focused_list_original_style = ""

    def _highlight_list_row(self, lst: QListWidget, row: int):
        self._clear_btn_highlight()
        if lst is not self._focused_list:
            self._clear_list_highlight()
            self._focused_list_original_style = lst.styleSheet()
            self._focused_list = lst
        self._focused_list_row = row
        lst.setCurrentRow(row)
        lst.scrollToItem(lst.item(row))
        lst.setFocus(Qt.OtherFocusReason)

    # ── Widget discovery ─────────────────────────────────────────────────────

    def _get_page(self) -> Optional[QWidget]:
        if not hasattr(self.window, "stack"):
            return None
        return self.window.stack.currentWidget()

    def _buttons_on_page(self) -> list:
        page = self._get_page()
        if page is None:
            return []
        results = []
        for w in page.findChildren(QPushButton):
            if w.isVisible() and w.isEnabled():
                results.append(w)

        def sort_key(w):
            pos = w.mapTo(page, w.rect().center())
            return (pos.y() // 50, pos.x())

        results.sort(key=sort_key)
        return results

    def _lists_on_page(self) -> list:
        page = self._get_page()
        if page is None:
            return []
        results = []
        for w in page.findChildren(QListWidget):
            if w.isVisible() and w.count() > 0:
                results.append(w)
        results.sort(key=lambda w: w.mapTo(page, w.rect().center()).y())
        return results

    def _focused_btn_index(self, buttons: list) -> int:
        if self._focused_btn is None:
            return -1
        try:
            return buttons.index(self._focused_btn)
        except ValueError:
            return -1

    def _btn_y(self, btn: QPushButton) -> int:
        page = self._get_page()
        if page is None:
            return 0
        return btn.mapTo(page, btn.rect().center()).y()

    def _btn_x(self, btn: QPushButton) -> int:
        page = self._get_page()
        if page is None:
            return 0
        return btn.mapTo(page, btn.rect().center()).x()

    def _btn_row(self, btn: QPushButton) -> int:
        return self._btn_y(btn) // 50

    # ── Navigation ───────────────────────────────────────────────────────────

    def _exit_list_to_buttons(self, direction: int):
        """Exit list, go to nearest button above (direction=-1) or below (direction=1)."""
        lst = self._focused_list
        self._clear_list_highlight()

        buttons = self._buttons_on_page()
        if not buttons:
            return

        page = self._get_page()
        if lst is None or page is None:
            self._highlight_btn(buttons[0])
            return

        list_y = lst.mapTo(page, lst.rect().center()).y()

        if direction == -1:
            candidates = [b for b in buttons if self._btn_y(b) < list_y]
            target = candidates[-1] if candidates else buttons[0]
        else:
            candidates = [b for b in buttons if self._btn_y(b) > list_y]
            target = candidates[0] if candidates else buttons[-1]

        self._highlight_btn(target)

    def _move_up(self):
        if self._in_list():
            row = self._focused_list_row - 1
            if row >= 0:
                self._highlight_list_row(self._focused_list, row)
            else:
                self._exit_list_to_buttons(-1)
            return

        buttons = self._buttons_on_page()
        if not buttons:
            return

        idx = self._focused_btn_index(buttons)
        if idx == -1:
            self._highlight_btn(buttons[0])
            return

        cur_btn = buttons[idx]
        cur_row = self._btn_row(cur_btn)
        cur_x = self._btn_x(cur_btn)
        cur_y = self._btn_y(cur_btn)

        # Check for a list sitting between the current button row and the next button row above
        prev_row_y = max(
            (self._btn_y(b) for b in buttons if self._btn_row(b) < cur_row),
            default=None
        )
        for lst in self._lists_on_page():
            page = self._get_page()
            list_y = lst.mapTo(page, lst.rect().center()).y()
            if list_y < cur_y:
                if prev_row_y is None or list_y > prev_row_y:
                    self._enter_list(lst, from_top=False)
                    return

        # Move to nearest button above
        candidates = [
            (self._btn_row(b), abs(self._btn_x(b) - cur_x), i)
            for i, b in enumerate(buttons)
            if self._btn_row(b) < cur_row
        ]
        if not candidates:
            # Wrap to bottom — check for list at bottom first
            lists = self._lists_on_page()
            if lists:
                self._enter_list(lists[-1], from_top=False)
                return
            self._highlight_btn(buttons[-1])
            return

        candidates.sort(key=lambda c: (-c[0], c[1]))
        self._highlight_btn(buttons[candidates[0][2]])

    def _move_down(self):
        if self._in_list():
            row = self._focused_list_row + 1
            if row < self._focused_list.count():
                self._highlight_list_row(self._focused_list, row)
            else:
                self._exit_list_to_buttons(1)
            return

        buttons = self._buttons_on_page()
        if not buttons:
            lists = self._lists_on_page()
            if lists:
                self._enter_list(lists[0])
            return

        idx = self._focused_btn_index(buttons)
        if idx == -1:
            self._highlight_btn(buttons[0])
            return

        cur_btn = buttons[idx]
        cur_row = self._btn_row(cur_btn)
        cur_x = self._btn_x(cur_btn)
        cur_y = self._btn_y(cur_btn)

        # Check for a list between this row and the next button row below
        next_row_y = min(
            (self._btn_y(b) for b in buttons if self._btn_row(b) > cur_row),
            default=None
        )
        for lst in self._lists_on_page():
            page = self._get_page()
            list_y = lst.mapTo(page, lst.rect().center()).y()
            if list_y > cur_y:
                if next_row_y is None or list_y < next_row_y:
                    self._enter_list(lst, from_top=True)
                    return

        # Move to nearest button below
        candidates = [
            (self._btn_row(b), abs(self._btn_x(b) - cur_x), i)
            for i, b in enumerate(buttons)
            if self._btn_row(b) > cur_row
        ]
        if not candidates:
            # Wrap to top
            self._highlight_btn(buttons[0])
            return

        candidates.sort(key=lambda c: (c[0], c[1]))
        self._highlight_btn(buttons[candidates[0][2]])

    def _move_left(self):
        if self._in_list():
            self._exit_list_to_buttons(-1)
            return

        buttons = self._buttons_on_page()
        if not buttons:
            return
        idx = self._focused_btn_index(buttons)
        if idx == -1:
            self._highlight_btn(buttons[0])
            return

        cur_row = self._btn_row(buttons[idx])
        cur_x = self._btn_x(buttons[idx])

        same_row_left = [
            i for i, b in enumerate(buttons)
            if self._btn_row(b) == cur_row and self._btn_x(b) < cur_x
        ]
        if same_row_left:
            self._highlight_btn(buttons[same_row_left[-1]])
        else:
            # Wrap to rightmost in row
            same_row = [i for i, b in enumerate(buttons) if self._btn_row(b) == cur_row]
            self._highlight_btn(buttons[same_row[-1]])

    def _move_right(self):
        if self._in_list():
            self._exit_list_to_buttons(1)
            return

        buttons = self._buttons_on_page()
        if not buttons:
            return
        idx = self._focused_btn_index(buttons)
        if idx == -1:
            self._highlight_btn(buttons[0])
            return

        cur_row = self._btn_row(buttons[idx])
        cur_x = self._btn_x(buttons[idx])

        same_row_right = [
            i for i, b in enumerate(buttons)
            if self._btn_row(b) == cur_row and self._btn_x(b) > cur_x
        ]
        if same_row_right:
            self._highlight_btn(buttons[same_row_right[0]])
        else:
            same_row = [i for i, b in enumerate(buttons) if self._btn_row(b) == cur_row]
            self._highlight_btn(buttons[same_row[0]])

    def _enter_list(self, lst: QListWidget, from_top: bool = True):
        row = 0 if from_top else lst.count() - 1
        self._highlight_list_row(lst, row)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_up(self):
        self._move_up()

    def _on_down(self):
        self._move_down()

    def _on_left(self):
        self._move_left()

    def _on_right(self):
        self._move_right()

    def _on_click(self):
        if self._in_list():
            lst = self._focused_list
            row = self._focused_list_row
            item = lst.item(row)
            if item:
                lst.itemClicked.emit(item)
            return

        if self._focused_btn is not None:
            btn = self._focused_btn
            if btn.isVisible() and btn.isEnabled():
                btn.click()
            return

        buttons = self._buttons_on_page()
        if buttons:
            self._highlight_btn(buttons[0])

    def _on_page_changed(self, _index: int):
        QTimer.singleShot(100, self._focus_first)

    def _focus_first(self):
        self._clear_btn_highlight()
        self._clear_list_highlight()
        buttons = self._buttons_on_page()
        if buttons:
            self._highlight_btn(buttons[0])

    def _on_error(self, message: str):
        print(f"[JoystickNavigator] {message}", file=sys.stderr)