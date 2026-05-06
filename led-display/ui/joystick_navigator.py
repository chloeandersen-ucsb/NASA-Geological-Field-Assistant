#!/usr/bin/env python3
"""
joystick_navigator.py — SparkFun Qwiic Joystick → Qt button/list focus navigator

Navigation model:
  - Focus is on a QPushButton or on a row inside a QListWidget.
  - Up/Down on buttons: move between rows; lists are entered when passed over.
  - Up/Down inside a list: scroll rows; exit at top/bottom returns to buttons.
  - Left/Right on buttons: move within the same visual row (wraps).
  - Left/Right inside a list: no-op (list items are single-column).
  - Click: fires the focused button, or emits itemClicked on the focused row.
  - Page change: focus snaps to the topmost list or button.
"""

import time
import sys
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer
from PySide6.QtWidgets import QPushButton, QWidget, QListWidget, QScrollArea

# ── I2C / Joystick constants ─────────────────────────────────────────────────
JOYSTICK_ADDR = 0x20
REG_X_MSB     = 0x03
REG_X_LSB     = 0x04
REG_Y_MSB     = 0x05
REG_Y_LSB     = 0x06
REG_BUTTON    = 0x07

CENTER       = 512
DEADZONE     = 60
POLL_HZ      = 30
INITIAL_DELAY = 0.40   # seconds before auto-repeat starts
REPEAT_DELAY  = 0.18   # seconds between auto-repeats

# ── Highlight style ───────────────────────────────────────────────────────────
# Keep the button's semantic color and add only a dark focus border. This avoids
# turning every focused control white and keeps primary/delete meaning visible.
_HIGHLIGHT_BLOCK = (
    " QPushButton {"
    " border: 2px solid #111827;"
    " border-radius: 10px; }"
)
_HIGHLIGHT_PLAIN = (
    " border: 2px solid #111827;"
    " border-radius: 10px;"
)

# Row-grouping tolerance: buttons whose centers are within this many px
# vertically are considered to be on the same row.
ROW_PX = 40


def _apply_highlight(btn: QPushButton) -> str:
    """Return the highlight addition to append to btn's current stylesheet."""
    return _HIGHLIGHT_BLOCK if "{" in btn.styleSheet() else _HIGHLIGHT_PLAIN


# ── I2C worker (runs on its own thread) ──────────────────────────────────────

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
            self.error.emit("smbus2 not installed — run: pip3 install smbus2")
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
                f"No joystick at {self.addr:#x} on bus {self.bus_number}. "
                "Check wiring or: i2cdetect -y -r <bus>"
            )
            bus.close()
            return

        interval = 1.0 / POLL_HZ
        while self._running:
            t0 = time.monotonic()
            try:
                x, y, btn_pressed = self._read(bus)
            except OSError as e:
                self.error.emit(f"I2C read error: {e} — retrying in 1 s")
                time.sleep(1.0)
                continue

            direction = self._direction(x, y)
            now = time.monotonic()

            if direction is None:
                self._last_dir = None
            elif direction != self._last_dir:
                self._emit(direction)
                self._last_dir = direction
                self._dir_start = now
                self._last_repeat = now
            else:
                held = now - self._dir_start
                if held >= INITIAL_DELAY and (now - self._last_repeat) >= REPEAT_DELAY:
                    self._emit(direction)
                    self._last_repeat = now

            if btn_pressed and not self._btn_was_pressed:
                self.clicked.emit()
            self._btn_was_pressed = btn_pressed

            elapsed = time.monotonic() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        bus.close()

    def _emit(self, direction: str):
        {"up": self.move_up, "down": self.move_down,
         "left": self.move_left, "right": self.move_right}[direction].emit()


# ── Navigator ─────────────────────────────────────────────────────────────────

class JoystickNavigator(QObject):

    def __init__(self, window, bus: int = 8, addr: int = JOYSTICK_ADDR):
        super().__init__(window)
        self.window = window

        self._focused_btn: Optional[QPushButton] = None
        self._highlight_addition: str = ""       # what we appended to btn's stylesheet

        self._focused_list: Optional[QListWidget] = None
        self._focused_list_row: int = -1
        self._highlighted_item_widget: Optional[QWidget] = None
        self._highlighted_item_widget_style: str = ""

        self._thread = QThread(self)
        self._worker = _JoystickWorker(bus, addr)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.start)
        self._worker.move_up.connect(self._move_up)
        self._worker.move_down.connect(self._move_down)
        self._worker.move_left.connect(self._move_left)
        self._worker.move_right.connect(self._move_right)
        self._worker.clicked.connect(self._on_click)
        self._worker.error.connect(lambda msg: print(f"[Joystick] {msg}", file=sys.stderr))

        if hasattr(window, "stack"):
            window.stack.currentChanged.connect(
                lambda _: QTimer.singleShot(150, self._focus_first)
            )

    def start(self):
        self._thread.start()

    def stop(self):
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)

    # ── Highlight management ─────────────────────────────────────────────────

    def _highlight_btn(self, btn: QPushButton):
        self._clear_btn_highlight()
        self._clear_list_highlight()
        addition = _apply_highlight(btn)
        self._focused_btn = btn
        self._highlight_addition = addition
        btn.setStyleSheet(btn.styleSheet() + addition)
        btn.setFocus(Qt.OtherFocusReason)
        self._scroll_into_view(btn)

    def _clear_btn_highlight(self):
        if self._focused_btn is None:
            return
        try:
            # Strip exactly what we appended from the button's CURRENT style so
            # any intermediate style changes (e.g. recording-state color) survive.
            current = self._focused_btn.styleSheet()
            if self._highlight_addition and current.endswith(self._highlight_addition):
                current = current[: -len(self._highlight_addition)]
            self._focused_btn.setStyleSheet(current.rstrip())
        except RuntimeError:
            pass
        self._focused_btn = None
        self._highlight_addition = ""

    def _highlight_list_row(self, lst: QListWidget, row: int):
        self._clear_btn_highlight()
        # Restore previous item widget style before moving to a new row
        if self._highlighted_item_widget is not None:
            try:
                self._highlighted_item_widget.setStyleSheet(self._highlighted_item_widget_style)
            except RuntimeError:
                pass
            self._highlighted_item_widget = None
            self._highlighted_item_widget_style = ""
        self._focused_list = lst
        self._focused_list_row = row
        lst.setCurrentRow(row)
        lst.scrollToItem(lst.item(row))
        lst.setFocus(Qt.OtherFocusReason)
        # Apply visible highlight to the custom item widget (covers Qt's native selection)
        item = lst.item(row)
        if item:
            widget = lst.itemWidget(item)
            if widget:
                self._highlighted_item_widget = widget
                self._highlighted_item_widget_style = widget.styleSheet()
                widget.setStyleSheet(
                    "background-color: #FFFFFF; border: 2px solid #111827; "
                    "border-radius: 10px;"
                )

    def _clear_list_highlight(self):
        if self._focused_list is not None:
            try:
                self._focused_list.clearSelection()
            except RuntimeError:
                pass
        if self._highlighted_item_widget is not None:
            try:
                self._highlighted_item_widget.setStyleSheet(self._highlighted_item_widget_style)
            except RuntimeError:
                pass
            self._highlighted_item_widget = None
            self._highlighted_item_widget_style = ""
        self._focused_list = None
        self._focused_list_row = -1

    def _scroll_into_view(self, btn: QPushButton):
        """Scroll the nearest ancestor QScrollArea so btn is visible."""
        p = btn.parent()
        while p is not None:
            if isinstance(p, QScrollArea):
                p.ensureWidgetVisible(btn, 20, 20)
                return
            p = p.parent()

    # ── Validation ───────────────────────────────────────────────────────────

    def _btn_still_valid(self) -> bool:
        if self._focused_btn is None:
            return False
        try:
            return self._focused_btn.isVisible() and self._focused_btn.isEnabled()
        except RuntimeError:
            return False

    def _list_still_valid(self) -> bool:
        if self._focused_list is None:
            return False
        try:
            return (self._focused_list.isVisible()
                    and self._focused_list_row < self._focused_list.count())
        except RuntimeError:
            return False

    def _in_list(self) -> bool:
        return self._focused_list is not None

    # ── Widget discovery ─────────────────────────────────────────────────────

    def _page(self) -> Optional[QWidget]:
        stack = getattr(self.window, "stack", None)
        return stack.currentWidget() if stack else None

    def _buttons(self) -> list[QPushButton]:
        page = self._page()
        if page is None:
            return []
        list_widgets = set(page.findChildren(QListWidget))
        result = []
        for btn in page.findChildren(QPushButton):
            if not (btn.isVisible() and btn.isEnabled()):
                continue
            # Exclude buttons embedded inside list-item widgets
            ancestor = btn.parent()
            in_list = False
            while ancestor is not None and ancestor is not page:
                if ancestor in list_widgets:
                    in_list = True
                    break
                ancestor = ancestor.parent()
            if not in_list:
                result.append(btn)
        result.sort(key=lambda b: (self._btn_row(b), self._btn_x(b)))
        return result

    def _lists(self) -> list[QListWidget]:
        page = self._page()
        if page is None:
            return []
        result = [w for w in page.findChildren(QListWidget)
                  if w.isVisible() and w.count() > 0]
        result.sort(key=lambda w: self._widget_y(w))
        return result

    def _btn_y(self, btn: QPushButton) -> int:
        page = self._page()
        return btn.mapTo(page, btn.rect().center()).y() if page else 0

    def _btn_x(self, btn: QPushButton) -> int:
        page = self._page()
        return btn.mapTo(page, btn.rect().center()).x() if page else 0

    def _btn_row(self, btn: QPushButton) -> int:
        return self._btn_y(btn) // ROW_PX

    def _widget_y(self, w: QWidget) -> int:
        page = self._page()
        return w.mapTo(page, w.rect().center()).y() if page else 0

    # ── Navigation ───────────────────────────────────────────────────────────

    def _move_up(self):
        if self._in_list():
            if not self._list_still_valid():
                self._focus_first()
                return
            row = self._focused_list_row - 1
            if row >= 0:
                self._highlight_list_row(self._focused_list, row)
            else:
                self._exit_list(direction=-1)
            return

        buttons = self._buttons()
        if not buttons:
            return

        if not self._btn_still_valid():
            self._highlight_btn(buttons[0])
            return

        cur_row = self._btn_row(self._focused_btn)
        cur_x   = self._btn_x(self._focused_btn)
        cur_y   = self._btn_y(self._focused_btn)

        # Enter a list that sits between the current row and the one above
        prev_row_y = max(
            (self._btn_y(b) for b in buttons if self._btn_row(b) < cur_row),
            default=None,
        )
        for lst in self._lists():
            ly = self._widget_y(lst)
            if ly < cur_y and (prev_row_y is None or ly > prev_row_y):
                self._highlight_list_row(lst, lst.count() - 1)
                return

        # Move to nearest button row above, preferring same horizontal position
        above = [(self._btn_row(b), abs(self._btn_x(b) - cur_x), i)
                 for i, b in enumerate(buttons) if self._btn_row(b) < cur_row]
        if not above:
            # Wrap: go to bottom — enter last list or last button
            lists = self._lists()
            if lists:
                self._highlight_list_row(lists[-1], lists[-1].count() - 1)
            else:
                self._highlight_btn(buttons[-1])
            return
        above.sort(key=lambda t: (-t[0], t[1]))
        self._highlight_btn(buttons[above[0][2]])

    def _move_down(self):
        if self._in_list():
            if not self._list_still_valid():
                self._focus_first()
                return
            row = self._focused_list_row + 1
            if row < self._focused_list.count():
                self._highlight_list_row(self._focused_list, row)
            else:
                self._exit_list(direction=1)
            return

        buttons = self._buttons()
        if not buttons:
            lists = self._lists()
            if lists:
                self._highlight_list_row(lists[0], 0)
            return

        if not self._btn_still_valid():
            self._highlight_btn(buttons[0])
            return

        cur_row = self._btn_row(self._focused_btn)
        cur_x   = self._btn_x(self._focused_btn)
        cur_y   = self._btn_y(self._focused_btn)

        # Enter a list that sits between the current row and the one below
        next_row_y = min(
            (self._btn_y(b) for b in buttons if self._btn_row(b) > cur_row),
            default=None,
        )
        for lst in self._lists():
            ly = self._widget_y(lst)
            if ly > cur_y and (next_row_y is None or ly < next_row_y):
                self._highlight_list_row(lst, 0)
                return

        # Move to nearest button row below, preferring same horizontal position
        below = [(self._btn_row(b), abs(self._btn_x(b) - cur_x), i)
                 for i, b in enumerate(buttons) if self._btn_row(b) > cur_row]
        if not below:
            # Wrap: go to top — enter first list or first button
            lists = self._lists()
            if lists and self._widget_y(lists[0]) < self._btn_y(buttons[0]):
                self._highlight_list_row(lists[0], 0)
            else:
                self._highlight_btn(buttons[0])
            return
        below.sort(key=lambda t: (t[0], t[1]))
        self._highlight_btn(buttons[below[0][2]])

    def _move_left(self):
        if self._in_list():
            # Left skips out of the list to the button above it
            self._exit_list(direction=-1)
            return

        buttons = self._buttons()
        if not buttons:
            return
        if not self._btn_still_valid():
            self._highlight_btn(buttons[0])
            return

        cur_row = self._btn_row(self._focused_btn)
        cur_x   = self._btn_x(self._focused_btn)
        row_btns = [i for i, b in enumerate(buttons) if self._btn_row(b) == cur_row]
        left = [i for i in row_btns if self._btn_x(buttons[i]) < cur_x]
        self._highlight_btn(buttons[left[-1] if left else row_btns[-1]])

    def _move_right(self):
        if self._in_list():
            # Right skips out of the list to the button below it
            self._exit_list(direction=1)
            return

        buttons = self._buttons()
        if not buttons:
            return
        if not self._btn_still_valid():
            self._highlight_btn(buttons[0])
            return

        cur_row = self._btn_row(self._focused_btn)
        cur_x   = self._btn_x(self._focused_btn)
        row_btns = [i for i, b in enumerate(buttons) if self._btn_row(b) == cur_row]
        right = [i for i in row_btns if self._btn_x(buttons[i]) > cur_x]
        self._highlight_btn(buttons[right[0] if right else row_btns[0]])

    def _exit_list(self, direction: int):
        """Leave the list; direction=-1 → button above, +1 → button below."""
        lst = self._focused_list
        self._clear_list_highlight()
        page = self._page()
        buttons = self._buttons()
        if not buttons:
            return
        if lst is None or page is None:
            self._highlight_btn(buttons[0])
            return
        ly = self._widget_y(lst)
        if direction == -1:
            cands = [b for b in buttons if self._btn_y(b) < ly]
            self._highlight_btn(cands[-1] if cands else buttons[0])
        else:
            cands = [b for b in buttons if self._btn_y(b) > ly]
            self._highlight_btn(cands[0] if cands else buttons[-1])

    def _on_click(self):
        if self._in_list():
            if self._list_still_valid():
                item = self._focused_list.item(self._focused_list_row)
                if item:
                    self._focused_list.itemClicked.emit(item)
            return

        if self._btn_still_valid():
            btn = self._focused_btn
            btn.click()
            # After clicking, re-scan after a short delay so newly visible
            # buttons (e.g. sub-buttons in ExpandingVoiceWidget) are discoverable.
            QTimer.singleShot(100, self._validate_focus)
            return

        # Nothing focused — snap to first available element
        self._focus_first()

    def _validate_focus(self):
        """If the focused button was hidden by a click action, re-focus."""
        if not self._btn_still_valid() and not self._in_list():
            self._focus_first()

    def _focus_first(self):
        """Snap to the first navigable element: topmost list or topmost button."""
        self._clear_btn_highlight()
        self._clear_list_highlight()

        buttons = self._buttons()
        lists   = self._lists()

        # Honour an explicitly designated primary button before falling back to
        # positional sorting — lets each page declare its most logical first focus.
        for btn in buttons:
            if btn.property("joystick_primary"):
                self._highlight_btn(btn)
                return

        if lists and buttons:
            # Start in the list if it appears before the first button
            if self._widget_y(lists[0]) < self._btn_y(buttons[0]):
                self._highlight_list_row(lists[0], 0)
                return
        elif lists:
            self._highlight_list_row(lists[0], 0)
            return

        if buttons:
            self._highlight_btn(buttons[0])
