#!/usr/bin/env python3
"""
joystick_navigator.py  –  SparkFun Qwiic Joystick → Qt button focus navigator
"""

import re as _re
import time
import sys
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer, QPoint
from PySide6.QtWidgets import (
    QApplication, QPushButton, QWidget, QAbstractButton,
    QStackedWidget, QTextEdit, QListWidget,
)
from PySide6.QtGui import QKeyEvent, QColor

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

HIGHLIGHT_BORDER = "#697d6a"


def _darken_hex(hex_color: str, factor: float = 0.82) -> str:
    c = QColor(hex_color)
    return QColor(
        int(c.red()   * factor),
        int(c.green() * factor),
        int(c.blue()  * factor),
    ).name()


def _apply_highlight(widget: QWidget, original_style: str) -> str:
    style = original_style.strip()
    match = _re.search(r'background(?:-color)?\s*:\s*(#[0-9a-fA-F]{6})', style)
    darkened_bg = _darken_hex(match.group(1)) if match else "rgba(52, 79, 65, 0.18)"
    bg_rule     = f"background-color: {darkened_bg};"
    border_rule = f"border: 2px solid {HIGHLIGHT_BORDER};"
    if "{" in style:
        return style + f" QPushButton {{ {bg_rule} {border_rule} }}"
    else:
        return style + f" {bg_rule} {border_rule}"


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

    relaunch_requested = Signal()

    def __init__(self, window, bus: int = 8, addr: int = JOYSTICK_ADDR):
        super().__init__(window)
        self.window = window

        self._focused_btn: Optional[QPushButton] = None
        self._focused_btn_original_style: str = ""

        self._focused_list: Optional[QListWidget] = None
        self._focused_list_row: int = -1
        self._focused_list_original_style: str = ""
        self._page_memory: dict = {}
        self._list_origin: Optional[tuple] = None  # (QListWidget, row) when focused on a row's dot button

        self._active_menu = None
        self._menu_btns: list = []
        self._menu_btn_idx: int = 0
        self._menu_btn_original_styles: list = []

        self._sleeping = False
        self._relaunch_clicks = 0
        self._relaunch_timer = QTimer(self)
        self._relaunch_timer.setSingleShot(True)
        self._relaunch_timer.setInterval(3000)
        self._relaunch_timer.timeout.connect(self._reset_relaunch_count)

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

    def sleep_mode(self):
        """Suspend normal navigation; count clicks for relaunch sequence."""
        self._sleeping = True
        self._relaunch_clicks = 0

    def wake_mode(self):
        """Resume normal navigation."""
        self._sleeping = False
        self._relaunch_clicks = 0
        self._relaunch_timer.stop()

    def _reset_relaunch_count(self):
        self._relaunch_clicks = 0

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

    def _highlight_list_row(self, lst: QListWidget, row: int):
        self._clear_btn_highlight()
        if lst is not self._focused_list:
            self._clear_list_highlight()
            self._focused_list_original_style = lst.styleSheet()
            self._focused_list = lst
        if self._focused_list_row >= 0 and self._focused_list_row != row:
            prev_item = lst.item(self._focused_list_row)
            if prev_item:
                prev_widget = lst.itemWidget(prev_item)
                if prev_widget:
                    prev_widget.setStyleSheet("")
        self._focused_list_row = row
        lst.setCurrentRow(row)
        lst.scrollToItem(lst.item(row))
        lst.setFocus(Qt.OtherFocusReason)
        item = lst.item(row)
        if item:
            widget = lst.itemWidget(item)
            if widget:
                widget.setStyleSheet(
                    f"background-color: rgba(52, 79, 65, 0.18);"
                    f"border: 2px solid {HIGHLIGHT_BORDER};"
                    f"border-radius: 4px;"
                )
    
    def _clear_list_highlight(self):
        if self._focused_list is not None:
            try:
                # Clear the highlighted item widget
                if self._focused_list_row >= 0:
                    item = self._focused_list.item(self._focused_list_row)
                    if item:
                        widget = self._focused_list.itemWidget(item)
                        if widget:
                            widget.setStyleSheet("")
                self._focused_list.clearSelection()
                self._focused_list.setStyleSheet(self._focused_list_original_style)
            except RuntimeError:
                pass
            self._focused_list = None
            self._focused_list_row = -1
            self._focused_list_original_style = ""

    # ── Menu navigation ──────────────────────────────────────────────────────

    def open_menu(self, menu, buttons: list):
        """Called when a popup menu opens; routes joystick input to menu items."""
        self._active_menu = menu
        self._menu_btns = list(buttons)
        self._menu_btn_original_styles = [btn.styleSheet() for btn in buttons]
        self._menu_btn_idx = 0
        menu.aboutToHide.connect(self._on_menu_closed)
        if buttons:
            self._set_menu_highlight(0)

    def _set_menu_highlight(self, idx: int):
        for i, btn in enumerate(self._menu_btns):
            try:
                original = self._menu_btn_original_styles[i]
                if i == idx:
                    btn.setStyleSheet(_apply_highlight(btn, original))
                else:
                    btn.setStyleSheet(original)
            except RuntimeError:
                pass
        self._menu_btn_idx = idx

    def _on_menu_closed(self):
        if self._active_menu:
            try:
                self._active_menu.aboutToHide.disconnect(self._on_menu_closed)
            except RuntimeError:
                pass
        for i, btn in enumerate(self._menu_btns):
            try:
                btn.setStyleSheet(self._menu_btn_original_styles[i])
            except RuntimeError:
                pass
        self._active_menu = None
        self._menu_btns = []
        self._menu_btn_idx = 0
        self._menu_btn_original_styles = []
        if self._list_origin:
            lst, row = self._list_origin
            try:
                if lst.isVisible() and row < lst.count():
                    self._highlight_list_row(lst, row)
                    return
            except RuntimeError:
                pass
        self._focus_first()

    # ── Widget discovery ─────────────────────────────────────────────────────

    def _get_page(self) -> Optional[QWidget]:
        if not hasattr(self.window, "stack"):
            return self.window
        return self.window.stack.currentWidget()

    def _buttons_on_page(self) -> list:
        # from PySide6.QtWidgets import QListWidget
        page = self._get_page()
        if page is None:
            return []

        # list_viewports = set()
        # for lst in page.findChildren(QListWidget):
        #     vp = lst.viewport()
        #     if vp:
        #         list_viewports.add(vp)

        # def is_inside_list(widget):
        #     p = widget.parent()
        #     while p is not None:
        #         if p in list_viewports:
        #             return True
        #         p = p.parent()
        #     return False

        results = []

        # for w in page.findChildren(QPushButton):
        #     if not w.isVisible():
        #         continue
        #     if not w.isEnabled():
        #         continue
        #     if is_inside_list(w):
        #         continue
        #     results.append(w)

        for w in page.findChildren(QPushButton):
            if w.isVisible() and w.isEnabled() and w.objectName() != "joystick_skip":
                results.append(w)

        #HERE
        # from PySide6.QtWidgets import QListWidget
        # for lst in page.findChildren(QListWidget):
        #     viewport = lst.viewport()
        #     if viewport:
        #         for w in viewport.findChildren(QPushButton):
        #             if w.isVisible() and w.isEnabled() and w not in results:
        #                 results.append(w)

        def sort_key(w):
            pos = w.mapTo(page, w.rect().center())
            return (pos.y() // 50, pos.x())

        results.sort(key=sort_key)
        # DEBUG
        # for i, b in enumerate(results):
        #         pos = b.mapTo(page, b.rect().center())
        #         print(f"  [{i}] '{b.text()[:20]}' objName='{b.objectName()}' y={pos.y()}", file=sys.stderr)
        return results

    def _lists_on_page(self) -> list:

        page = self._get_page()
        if page is None:
            return []
        results = []
        for w in page.findChildren(QListWidget):
            if w.isVisible() and w.count() > 0:
                results.append(w)
        # results.sort(key=lambda w: w.mapTo(page, w.rect().center()).y())
        results.sort(key=lambda w: w.mapTo(page, QPoint(0, 0)).y())
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

        # list_y = lst.mapTo(page, lst.rect().center()).y()

        list_y = lst.mapTo(page, QPoint(0, 0)).y() + lst.height() // 2

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
            # list_y = lst.mapTo(page, lst.rect().center()).y()
    
            list_y = lst.mapTo(page, QPoint(0, 0)).y() + lst.height() // 2  
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
            # list_y = lst.mapTo(page, lst.rect().center()).y()
    
            list_y = lst.mapTo(page, QPoint(0, 0)).y() + lst.height() // 2

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

        # Return to list row if we navigated right from a list to a dot button
        if self._list_origin is not None:
            lst, row = self._list_origin
            self._list_origin = None
            self._clear_btn_highlight()
            try:
                if lst.isVisible() and row < lst.count():
                    self._highlight_list_row(lst, row)
                    return
            except RuntimeError:
                pass

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
            # Try to focus the dot button (⋮) for the currently highlighted row
            lst = self._focused_list
            row = self._focused_list_row
            item = lst.item(row)
            if item:
                widget = lst.itemWidget(item)
                if widget:
                    for btn in widget.findChildren(QPushButton):
                        if btn.objectName() == "joystick_skip" and btn.text() == "⋮":
                            self._list_origin = (lst, row)
                            self._highlight_btn(btn)
                            return
            # No dot button found; exit list normally
            self._exit_list_to_buttons(1)
            return

        self._list_origin = None
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
        if self._active_menu and self._menu_btns:
            idx = (self._menu_btn_idx - 1) % len(self._menu_btns)
            self._set_menu_highlight(idx)
            return
        self._move_up()

    def _on_down(self):
        if self._active_menu and self._menu_btns:
            idx = (self._menu_btn_idx + 1) % len(self._menu_btns)
            self._set_menu_highlight(idx)
            return
        self._move_down()

    def _on_left(self):
        if self._active_menu:
            self._active_menu.close()
            return
        self._move_left()

    def _on_right(self):
        if self._active_menu:
            self._active_menu.close()
            return
        self._move_right()

    def _on_click(self):
        if self._sleeping:
            self._relaunch_clicks += 1
            self._relaunch_timer.start()
            if self._relaunch_clicks >= 5:
                self._relaunch_clicks = 0
                self._relaunch_timer.stop()
                self.relaunch_requested.emit()
            return

        if self._active_menu and self._menu_btns:
            idx = self._menu_btn_idx
            if 0 <= idx < len(self._menu_btns):
                btn = self._menu_btns[idx]
                try:
                    if btn.isVisible() and btn.isEnabled():
                        btn.click()
                except RuntimeError:
                    pass
            return

        if self._in_list():
            lst = self._focused_list
            row = self._focused_list_row
            item = lst.item(row)
            if item:
                lst.itemClicked.emit(item)
                # Fire the row_btn (the main action button) for this list item
                # widget = lst.itemWidget(item)
                # if widget:
                #     for btn in widget.findChildren(QPushButton):
                #         if btn.objectName() != "joystick_skip":
                #             btn.click()
                #             return
                # # Fallback to itemClicked signal
                # lst.itemClicked.emit(item)
            return

        if self._focused_btn is not None:
            btn = self._focused_btn
            if btn.isVisible() and btn.isEnabled():
                btn.click()
            return

        buttons = self._buttons_on_page()
        if buttons:
            self._highlight_btn(buttons[0])

    def _save_position(self):
        """Save current focus position for the current page."""
        page = self._get_page()
        if page is None:
            return
        if self._focused_list is not None:
            self._page_memory[id(page)] = ("list", self._focused_list, self._focused_list_row)
        elif self._focused_btn is not None:
            self._page_memory[id(page)] = ("btn", self._focused_btn, None)
    
    def _on_page_changed(self, _index: int):
        self._save_position()
        self._list_origin = None
        QTimer.singleShot(100, self._focus_first)

    def _focus_first(self):
        page = self._get_page()
        if page is None:
            return

        # Try to restore saved position for this page
        saved = self._page_memory.get(id(page))
        if saved:
            kind = saved[0]
            self._clear_btn_highlight()
            self._clear_list_highlight()
            if kind == "list":
                _, lst, row = saved
                # Verify the list still exists and has that row
                try:
                    if lst.isVisible() and row < lst.count():
                        self._highlight_list_row(lst, row)
                        return
                except RuntimeError:
                    pass
            elif kind == "btn":
                _, btn, _ = saved
                try:
                    if btn.isVisible() and btn.isEnabled():
                        self._highlight_btn(btn)
                        return
                except RuntimeError:
                    pass
        self._clear_btn_highlight()
        self._clear_list_highlight()
        buttons = self._buttons_on_page()
        if buttons:
            primary = [b for b in buttons if b.objectName() != "joystick_secondary"]
            self._highlight_btn((primary or buttons)[0])

    def _on_error(self, message: str):
        print(f"[JoystickNavigator] {message}", file=sys.stderr)