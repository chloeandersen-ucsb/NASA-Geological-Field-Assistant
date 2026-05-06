from __future__ import annotations
import html
import os
import sys
import time
import datetime
from pathlib import Path
from PySide6.QtGui import QPixmap, QImage

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
img_path = project_root / "led-display" / "ui" / "sage-logo-wcbg.png"


from PySide6.QtCore import Qt, QTimer, Signal, QSize
from PySide6.QtGui import QTextCursor, QKeyEvent, QShortcut, QKeySequence, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QMessageBox,
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QHBoxLayout, QSizePolicy, QGridLayout, QDialog, QFrame,
)

import importlib.util, pathlib
_nav_path = pathlib.Path(__file__).parent / "joystick_navigator.py"
_spec = importlib.util.spec_from_file_location("joystick_navigator", _nav_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
JoystickNavigator = _mod.JoystickNavigator

import connector
from core.viewmodel import AppStateType, ClassificationResult, MissionSummary, TripSummary

class SummaryPopupOverlay(QWidget):
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

        self._outer = QVBoxLayout(self)
        self._outer.setAlignment(Qt.AlignCenter)

        self.box = QWidget()
        self.box.setFixedSize(420, 340)
        self.box.setStyleSheet(
            "background-color: #F5F6F1; border: 3px solid #5E6D62; border-radius: 12px;"
        )
        box_layout = QVBoxLayout(self.box)
        box_layout.setContentsMargins(16, 14, 16, 12)
        box_layout.setSpacing(8)

        self.lbl_title = QLabel("Category")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #233127; border: none; padding-bottom: 4px;"
        )
        box_layout.addWidget(self.lbl_title)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background-color: #5E6D62; border: none;")
        divider.setFixedHeight(2)
        box_layout.addWidget(divider)

        self.lbl_content = QLabel("")
        self.lbl_content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_content.setWordWrap(True)
        self.lbl_content.setStyleSheet(
            "font-size: 17px; color: #233127; border: none; margin-top: 4px;"
        )
        box_layout.addWidget(self.lbl_content, stretch=1)

        self.btn_close = QPushButton("Close")
        self.btn_close.setMinimumHeight(44)
        self.btn_close.setProperty("joystick_primary", True)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: #5E6D62; color: white;
                font-size: 16px; font-weight: bold;
                border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #617c32; }
        """)
        self.btn_close.clicked.connect(self._close)
        box_layout.addWidget(self.btn_close)

        self._outer.addWidget(self.box)

    def _close(self):
        self.hide()
        self.closed.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 160))

    def mousePressEvent(self, event):
        if not self.box.geometry().contains(event.pos()):
            self._close()

    def show_popup(self, title: str, content: str):
        self.lbl_title.setText(title)
        self.lbl_content.setText(content)
        self.raise_()
        self.show()

class SpinnerWidget(QWidget):
    """A custom widget that draws a smooth, rotating loading circle."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._rotate)
        
    def _rotate(self):
        self.angle = (self.angle + 15) % 360  # Spin speed
        self.update()
        
    def start(self):
        self.timer.start(30) # 30ms for smooth 60fps rotation
        self.show()
        
    def stop(self):
        self.timer.stop()
        self.hide()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Draw the dim background track
        pen = QPen(QColor(255, 255, 255, 50)) 
        pen.setWidth(6)
        painter.setPen(pen)
        painter.drawArc(self.rect().adjusted(6, 6, -6, -6), 0, 360 * 16)
        
        # 2. Draw the bright spinning part
        pen = QPen(QColor(_COLOR_TIER_MEDIUM))
        pen.setWidth(6)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(self.rect().adjusted(6, 6, -6, -6), -self.angle * 16, 120 * 16)


class LoadingOverlay(QWidget):
    """A semi-transparent overlay that blocks clicks and animates text."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False) # Blocks clicks to the page underneath!
        
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)
        
        # Add the spinning circle
        self.spinner = SpinnerWidget()
        self.spinner.setFixedSize(60, 60)
        
        # Add the animated text label
        self.lbl_text = QLabel("Finalizing Transcript")
        self.lbl_text.setStyleSheet("color: white; font-size: 22px; font-weight: bold; background: transparent;")
        self.lbl_text.setAlignment(Qt.AlignCenter)
        
        self.layout.addWidget(self.spinner, 0, Qt.AlignCenter)
        self.layout.addSpacing(15)
        self.layout.addWidget(self.lbl_text, 0, Qt.AlignCenter)
        
        # Ellipsis Animation Timer
        self.dot_count = 0
        self.text_timer = QTimer(self)
        self.text_timer.timeout.connect(self._animate_text)
        
    def _animate_text(self):
        self.dot_count = (self.dot_count + 1) % 4
        dots = "." * self.dot_count
        self.lbl_text.setText(f"Finalizing Transcript{dots}")
        
    def paintEvent(self, event):
        # Fill the entire screen with 60% opacity black
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 160)) 
        
    def start(self):
        self.show()
        self.spinner.start()
        self.text_timer.start(350) # Ticks exactly every 0.25 seconds!
        
    def stop(self):
        self.hide()
        self.spinner.stop()
        self.text_timer.stop()

# Centralized UI theme. Keep the app minimal, high-contrast, and consistent
# for a 480x800 astronaut-facing display controlled by joystick.
_COLOR_BG          = "#D6DDD2"
_COLOR_PAGE        = "#F5F6F1"
_COLOR_SURFACE     = "#FFFFFF"
_COLOR_SURFACE_ALT = "#EEF2EB"
_COLOR_TEXT        = "#233127"
_COLOR_MUTED       = "#5E6D62"
_COLOR_BORDER      = "#7F8B81"
_COLOR_PRIMARY     = "#566F31"
_COLOR_PRIMARY_H   = "#485D29"
_COLOR_SECONDARY   = "#FFFFFF"
_COLOR_SECONDARY_H = "#EEF2EB"
_COLOR_DANGER      = "#9E2B2B"
_COLOR_DANGER_H    = "#842121"
_COLOR_STOP        = "#7E1F23"
_COLOR_STOP_H      = "#6D191D"
_COLOR_FOCUS       = "#111827"
_BTN_RADIUS        = 4
_BTN_H             = 46
_FONT_UI           = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
_FONT_DATA         = "'DM Mono', 'Courier New', Courier, monospace"

# Extended palette — named so they're grep-able and not scattered as raw hex
_COLOR_HOME_BG         = "#586e5d"   # HomePage background
_COLOR_HOME_TEXT       = "#cad2c5"   # Light text on dark bg
_COLOR_HOME_MUTED      = "#8aab90"   # Subdued accents on dark bg
_COLOR_HOME_BTN        = "#3d5245"   # Dark green border / separator
_COLOR_HOME_BTN_H      = "#2B4035"   # Hover on dark buttons
_COLOR_IMG_BG          = "#0a0e0c"   # Camera / image placeholder bg
_COLOR_HOVER_LIGHT     = "#d4ddd0"   # Hover on light-surface elements
_COLOR_FEAT_SEP        = "#c0c8bb"   # Feature row divider
_COLOR_FEATURES_BORDER = "#b5c9b7"   # Features box border
_COLOR_STATUS_BAR      = "#4d6655"   # Status bar label text
_COLOR_TIER_HIGH       = "#617c32"
_COLOR_TIER_MEDIUM     = "#a88b5c"
_COLOR_TIER_LOW        = "#c46200"
_COLOR_TIER_UNK        = "#888888"
_COLOR_ALT_CHIP_BG     = "#e0e8d8"   # Alternative classification chip bg
_COLOR_HDR_TEXT        = "#cad2c5"   # Header bar label text


def _button_style(
    bg: str,
    fg: str = "white",
    hover: str | None = None,
    border: str = "transparent",
    focus_border: str | None = None,
) -> str:
    hover = hover or bg
    fb = focus_border if focus_border is not None else _COLOR_FOCUS
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {fg};
            font-size: 15px;
            font-weight: 700;
            font-family: {_FONT_UI};
            border: 1px solid {border};
            border-radius: {_BTN_RADIUS}px;
            padding: 6px 10px;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:pressed {{ background-color: {hover}; }}
        QPushButton:focus {{ border: 2px solid {fb}; }}
        QPushButton:disabled {{ background-color: #AAB5AC; color: #E8ECE8; }}
    """

# Dark-bg buttons use a white focus ring; light-bg buttons use the dark focus ring.
_SS_PRIMARY      = _button_style(_COLOR_PRIMARY, "white", _COLOR_PRIMARY_H, focus_border="white")
_SS_SECONDARY    = _button_style(_COLOR_SECONDARY, _COLOR_TEXT, _COLOR_SECONDARY_H, border=_COLOR_BORDER)
_SS_BACK         = _button_style(_COLOR_SECONDARY, _COLOR_TEXT, _COLOR_SECONDARY_H, border=_COLOR_BORDER)
_SS_DELETE       = _button_style(_COLOR_DANGER, "white", _COLOR_DANGER_H, focus_border="white")
_SS_STOP         = _button_style(_COLOR_STOP, "white", _COLOR_STOP_H, focus_border="white")
_SS_ARMED_DELETE = _button_style(_COLOR_STOP, "white", _COLOR_STOP_H, border=_COLOR_FOCUS, focus_border="white")


def _list_style() -> str:
    return f"""
        QListWidget {{
            background-color: transparent;
            color: {_COLOR_TEXT};
            border: none;
            padding: 2px;
            font-size: 15px;
            font-family: {_FONT_UI};
            outline: 0;
        }}
        QListWidget::item {{
            padding: 0px;
            margin: 3px 0px;
            border: none;
        }}
        QListWidget::item:selected {{
            background-color: transparent;
            color: {_COLOR_TEXT};
        }}
    """


def big_button(text: str) -> QPushButton:
    b = QPushButton(text.upper())
    b.setMinimumHeight(62)
    b.setStyleSheet(
        f"QPushButton {{ background-color: {_COLOR_TEXT}; color: {_COLOR_HOME_TEXT};"
        f" font-size: 14px; font-weight: 700; font-family: {_FONT_UI};"
        f" border-radius: {_BTN_RADIUS}px; border: 1px solid {_COLOR_HOME_BTN};"
        " padding: 6px 12px; }}"
        f" QPushButton:hover {{ background-color: {_COLOR_HOME_BTN_H}; }}"
        f" QPushButton:pressed {{ background-color: {_COLOR_HOME_BTN_H}; }}"
        f" QPushButton:focus {{ border: 2px solid {_COLOR_FOCUS}; }}"
    )
    return b

def _btn(style: str, text: str, height: int = _BTN_H) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(height)
    b.setStyleSheet(style)
    return b

def _back_btn(text: str = "Back") -> QPushButton:
    return _btn(_SS_BACK, text)

def _delete_btn(text: str) -> QPushButton:
    return _btn(_SS_DELETE, text)


class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(0)

        self.setStyleSheet(f"""
            background-color: {_COLOR_HOME_BG};
            color: {_COLOR_HOME_TEXT};
            font-family: {_FONT_UI};
        """)

        layout.addStretch(1)

        logo = QLabel()
        pixmap = QPixmap(img_path)
        logo.setStyleSheet("background: transparent; border: none; padding-bottom: 30px;")
        logo.setPixmap(pixmap.scaled(360, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background-color: {_COLOR_HOME_BTN}; border: none; max-height: 1px;")
        layout.addWidget(sep)
        layout.addSpacing(16)

        self.btn_classify = big_button("Classify Rock")
        self.btn_voice    = big_button("Voice to Text")
        self.btn_trip     = big_button("View Trip Notes")
        self.btn_quit     = _btn(
            _button_style(_COLOR_HOME_BTN, _COLOR_HOME_MUTED, _COLOR_HOME_BTN_H, border=_COLOR_HOME_BTN),
            "Quit", height=38
        )

        layout.addWidget(self.btn_classify)
        layout.addSpacing(6)
        layout.addWidget(self.btn_voice)
        layout.addSpacing(6)
        layout.addWidget(self.btn_trip)
        layout.addStretch(1)
        layout.addWidget(self.btn_quit)


class LoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        label = QLabel("ANALYZING")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            f"font-size: 16px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_MUTED}; letter-spacing: 2px;"
        )
        sublabel = QLabel("Processing classification…")
        sublabel.setAlignment(Qt.AlignCenter)
        sublabel.setStyleSheet(
            f"font-size: 13px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};"
        )
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addSpacing(6)
        layout.addWidget(sublabel)
        layout.addStretch(1)
        self.btn_cancel = _back_btn("Cancel")
        layout.addWidget(self.btn_cancel)

    def set_message(self, message: str) -> None:
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setText(message.upper())
                break


class VoiceLoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        label = QLabel("INITIALIZING")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            f"font-size: 16px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_MUTED};"
        )
        sublabel = QLabel("Voice transcription loading…")
        sublabel.setAlignment(Qt.AlignCenter)
        sublabel.setStyleSheet(
            f"font-size: 13px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};"
        )
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addSpacing(6)
        layout.addWidget(sublabel)
        layout.addStretch(1)


class CameraPreviewPage(QWidget):
    """Shown when camera preview is active; user clicks Capture to take the photo."""
    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        layout = QVBoxLayout(self)

        self.lbl_step = QLabel("CAPTURE · VIEW 1")
        self.lbl_step.setAlignment(Qt.AlignCenter)
        self.lbl_step.setStyleSheet(
            f"font-size: 12px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_MUTED}; padding: 6px 0px;"
        )
        layout.addWidget(self.lbl_step)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            f"background-color: {_COLOR_IMG_BG}; border: 1px solid {_COLOR_HOME_BTN}; border-radius: {_BTN_RADIUS}px;"
        )
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout.addWidget(self.video_label, stretch=1)

        self.mic_ctrl = ExpandingVoiceWidget(self.vm, self)
        layout.addWidget(self.mic_ctrl, 0, Qt.AlignCenter)
        
        self.btn_capture = _btn(_SS_PRIMARY, "Capture")
        self.btn_capture.setProperty("joystick_primary", True)
        self.btn_cancel = _back_btn("Cancel")
        row = QHBoxLayout()
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_capture)
        layout.addSpacing(15)
        layout.addLayout(row)


class CaptureReviewPage(QWidget):
    """After both captures: preview top and side images with Classify or Retake."""
    _IMG_W, _IMG_H = 440, 330

    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        layout = QVBoxLayout(self)

        title = QLabel("REVIEW CAPTURES")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 12px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_MUTED}; padding: 6px 0px;"
        )
        layout.addWidget(title, 0, Qt.AlignHCenter)
        layout.addStretch(1)

        images_row = QVBoxLayout()
        images_row.setAlignment(Qt.AlignCenter)
        img_style = f"border: 1px solid {_COLOR_HOME_BTN}; border-radius: {_BTN_RADIUS}px; padding: 2px"
        self.lbl_top = QLabel()
        self.lbl_top.setStyleSheet(img_style)
        self.lbl_side = QLabel()
        self.lbl_side.setStyleSheet(img_style)
        images_row.addWidget(self.lbl_top)
        images_row.addSpacing(50)
        images_row.addWidget(self.lbl_side)
        layout.addLayout(images_row)
        layout.addStretch(1)

        self.mic_ctrl = ExpandingVoiceWidget(self.vm, self)
        layout.addWidget(self.mic_ctrl, 0, Qt.AlignCenter)

        btns = QHBoxLayout()
        self.btn_retake = _btn(_SS_SECONDARY, "Retake")
        self.btn_classify = _btn(_SS_PRIMARY, "Classify")
        self.btn_classify.setProperty("joystick_primary", True)
        btns.addWidget(self.btn_retake)
        btns.addWidget(self.btn_classify)
        layout.addLayout(btns)

    def set_images(self, top_path: str | None, side_path: str | None) -> None:
        w, h = self._IMG_W, self._IMG_H
        if top_path and os.path.exists(top_path):
            pix = QPixmap(top_path)
            scaled_pix = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_top.setPixmap(scaled_pix)
            self.lbl_top.setFixedSize(scaled_pix.size())
        else:
            self.lbl_top.setText("Top")
        if side_path and os.path.exists(side_path):
            pix = QPixmap(side_path)
            scaled_pix = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_side.setPixmap(scaled_pix)
            self.lbl_side.setFixedSize(scaled_pix.size())
        else:
            self.lbl_side.setText("Side")


def _make_divider():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"background-color: {_COLOR_BORDER}; border: none; max-height: 1px;")
    line.setFixedHeight(1)
    return line


def _make_list_row_widget(display_text: str, min_height: int = 60) -> QWidget:
    """Consistent padded row for mission, rock, and note lists."""
    widget = QWidget()
    widget.setObjectName("listRow")
    widget.setMinimumHeight(min_height)
    widget.setStyleSheet(f"""
        QWidget#listRow {{
            background-color: {_COLOR_SURFACE_ALT};
            border: 1px solid transparent;
            border-radius: 4px;
        }}
    """)
    row_layout = QHBoxLayout(widget)
    row_layout.setContentsMargins(12, 8, 12, 8)
    row_layout.setSpacing(8)

    lbl = QLabel(display_text)
    lbl.setTextFormat(Qt.RichText)
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.NoTextInteraction)
    lbl.setStyleSheet(f"font-size: 15px; color: {_COLOR_TEXT}; font-family: {_FONT_UI};")
    lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
    widget.setToolTip(display_text)
    row_layout.addWidget(lbl, stretch=1)
    return widget

class ClassifiedPage(QWidget):
    def __init__(self, vm):
        super().__init__()
        self.vm = vm

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 10, 16, 10)
        self.main_layout.setSpacing(4)

        # ── Images (compact thumbnails) ──────────────────────────────────────
        self.image_container = QWidget()
        self.image_container.setStyleSheet("background-color: transparent;")
        img_layout = QHBoxLayout(self.image_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(10)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: transparent; border: none;")
        self.lbl_image.setFixedSize(105, 79)
        self.lbl_image.setScaledContents(True)
        self.lbl_side_image = QLabel()
        self.lbl_side_image.setAlignment(Qt.AlignCenter)
        self.lbl_side_image.setStyleSheet("background-color: transparent; border: none;")
        self.lbl_side_image.setFixedSize(105, 79)
        self.lbl_side_image.setScaledContents(True)
        img_layout.addStretch()
        img_layout.addWidget(self.lbl_image)
        img_layout.addWidget(self.lbl_side_image)
        img_layout.addStretch()
        self.main_layout.addWidget(self.image_container)

        # ── Line 1: Rock name (large, left) + Volume (right) ─────────────────
        name_vol_row = QHBoxLayout()
        name_vol_row.setContentsMargins(0, 4, 0, 0)
        name_vol_row.setSpacing(8)
        self.lbl_label = QLabel("—")
        self.lbl_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; font-family: {_FONT_UI}; color: {_COLOR_TEXT};"
        )
        self.lbl_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.lbl_volume = QLabel("")
        self.lbl_volume.setStyleSheet(
            f"font-size: 13px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};"
        )
        self.lbl_volume.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        name_vol_row.addWidget(self.lbl_label)
        name_vol_row.addWidget(self.lbl_volume)
        self.main_layout.addLayout(name_vol_row)

        # ── Line 2: Confidence + tier badge + weight ──────────────────────────
        conf_wt_row = QHBoxLayout()
        conf_wt_row.setContentsMargins(0, 0, 0, 2)
        conf_wt_row.setSpacing(8)
        self.lbl_conf = QLabel("—")
        self.lbl_conf.setStyleSheet(
            f"font-size: 13px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};"
        )
        self.lbl_tier = QLabel("")
        self.lbl_tier.setStyleSheet(
            f"font-size: 12px; font-weight: 700; font-family: {_FONT_UI};"
            f" border-radius: {_BTN_RADIUS}px; padding: 2px 7px; color: {_COLOR_PAGE};"
        )
        self.lbl_tier.setVisible(False)
        self.lbl_extra = QLabel("")
        self.lbl_extra.setStyleSheet(
            f"font-size: 12px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};"
        )
        conf_wt_row.addWidget(self.lbl_conf)
        conf_wt_row.addWidget(self.lbl_tier)
        conf_wt_row.addStretch()
        conf_wt_row.addWidget(self.lbl_extra)
        self.main_layout.addLayout(conf_wt_row)

        # ── Divider ───────────────────────────────────────────────────────────
        self.main_layout.addSpacing(4)
        self.main_layout.addWidget(_make_divider())

        # ── Features + geology notes (rebuilt on each classification) ─────────
        self.features_section = QWidget()
        self.features_layout = QVBoxLayout(self.features_section)
        self.features_layout.setContentsMargins(0, 2, 0, 0)
        self.features_layout.setSpacing(2)
        self.main_layout.addWidget(self.features_section, stretch=1)

        # ── Alternative classifications (selectable override) ─────────────────
        self.div_alts = _make_divider()
        self.div_alts.setVisible(False)
        self.main_layout.addWidget(self.div_alts)

        self.alt_row = QWidget()
        alt_outer = QHBoxLayout(self.alt_row)
        alt_outer.setContentsMargins(0, 4, 0, 2)
        alt_outer.setSpacing(8)
        lbl_also = QLabel("ALT:")
        lbl_also.setStyleSheet(
            f"font-size: 11px; color: {_COLOR_MUTED}; font-weight: 700; font-family: {_FONT_UI};"
        )
        self.alt_buttons_layout = QHBoxLayout()
        self.alt_buttons_layout.setSpacing(6)
        alt_outer.addWidget(lbl_also)
        alt_outer.addLayout(self.alt_buttons_layout)
        alt_outer.addStretch()
        self.alt_row.setVisible(False)
        self.main_layout.addWidget(self.alt_row)

        # ── Buttons ───────────────────────────────────────────────────────────
        self.main_layout.addWidget(_make_divider())

        self.mic_ctrl = ExpandingVoiceWidget(self.vm, self)

        self.btn_reclassify = _btn(_SS_SECONDARY, "Reclassify")
        self.btn_save = _btn(_SS_PRIMARY, "Save")
        self.btn_save.setProperty("joystick_primary", True)
        self.btn_delete = _delete_btn("Delete")

        top_btns = QHBoxLayout()
        top_btns.setSpacing(6)
        top_btns.addWidget(self.btn_reclassify)
        top_btns.addWidget(self.btn_save)

        self.main_layout.addWidget(self.mic_ctrl, 0, Qt.AlignCenter)
        self.main_layout.addLayout(top_btns)
        self.main_layout.addWidget(self.btn_delete)
        


class VoicePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.setStyleSheet(f"""
            background-color: {_COLOR_PAGE};
            color: {_COLOR_TEXT};
            font-family: {_FONT_UI};
        """)

        hdr = QWidget()
        hdr.setStyleSheet(
            f"background-color: {_COLOR_TEXT}; border-radius: {_BTN_RADIUS}px;"
        )
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(12, 8, 12, 8)
        title = QLabel("VOICE TO TEXT")
        title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_HDR_TEXT}; background: transparent;"
        )
        hdr_layout.addWidget(title)
        hdr_layout.addStretch()
        layout.addWidget(hdr)

        self.lbl_context = QLabel("Context: None")
        self.lbl_context.setAlignment(Qt.AlignCenter)
        self.lbl_context.setStyleSheet(
            f"font-size: 12px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};"
            " padding: 3px 0px;"
        )
        layout.addWidget(self.lbl_context)

        self.text = QTextEdit()
        self.text.setStyleSheet(f"""
            font-size: 18px;
            font-family: {_FONT_UI};
            border: 1px solid {_COLOR_BORDER};
            border-radius: {_BTN_RADIUS}px;
            padding: 8px;
            background-color: {_COLOR_SURFACE};
            color: {_COLOR_TEXT};
        """)
        self.text.setReadOnly(True)
        layout.addWidget(self.text, stretch=1)

        row = QHBoxLayout()
        self.btn_start  = _btn(_SS_PRIMARY,   "Start")
        self.btn_stop   = _btn(_SS_STOP,      "Stop")
        self.btn_redo   = _btn(_SS_SECONDARY, "Redo")
        self.btn_reset  = _btn(_SS_SECONDARY, "Reset Context")
        self.btn_save   = _btn(_SS_PRIMARY,   "Save")
        self.btn_delete = _btn(_SS_DELETE,    "Delete")
        self.btn_cancel = _back_btn("Cancel")

        for b in [self.btn_start, self.btn_stop, self.btn_redo, self.btn_save, self.btn_delete, self.btn_reset, self.btn_cancel]:
            row.addWidget(b)

        layout.addLayout(row)

        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()

    def resizeEvent(self, event):
        # Force the overlay to always match the exact size of the VoicePage
        self.loading_overlay.setGeometry(self.rect())
        super().resizeEvent(event)


class TripLoadPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background-color: {_COLOR_PAGE}; color: {_COLOR_TEXT};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        hdr = QWidget()
        hdr.setStyleSheet(f"background-color: {_COLOR_TEXT}; border-radius: {_BTN_RADIUS}px;")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(12, 8, 12, 8)
        title = QLabel("TRIP NOTES")
        title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_HDR_TEXT}; background: transparent;"
        )
        hdr_layout.addWidget(title)
        layout.addWidget(hdr)

        info_card = QWidget()
        info_card.setStyleSheet(
            f"background-color: {_COLOR_SURFACE_ALT}; border-radius: {_BTN_RADIUS}px;"
        )
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(4)

        self.lbl_current_mission = QLabel("Current mission: --")
        self.lbl_current_mission.setAlignment(Qt.AlignCenter)
        self.lbl_current_mission.setWordWrap(True)
        self.lbl_current_mission.setStyleSheet(
            f"font-size: 13px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};"
            " background: transparent;"
        )
        info_layout.addWidget(self.lbl_current_mission)

        sep = _make_divider()
        info_layout.addWidget(sep)

        missions_label = QLabel("MISSIONS")
        missions_label.setStyleSheet(
            f"font-size: 11px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_MUTED}; background: transparent;"
        )
        info_layout.addWidget(missions_label)

        layout.addWidget(info_card)

        self.btn_create_new_mission = _btn(_SS_PRIMARY, "Create Mission")
        layout.addWidget(self.btn_create_new_mission)
        
        self.list = QListWidget()
        self.list.setStyleSheet(_list_style())
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setWordWrap(True)
        layout.addWidget(self.list, stretch=1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        self.btn_back = _back_btn()
        self.btn_delete_all = _delete_btn("Delete All")
        bottom_row.addWidget(self.btn_back)
        bottom_row.addWidget(self.btn_delete_all)
        layout.addLayout(bottom_row)

        self._missions_data = []
        self._summary = None


class MissionDetailPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background-color: {_COLOR_PAGE}; color: {_COLOR_TEXT};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(9)

        hdr = QWidget()
        hdr.setStyleSheet(f"background-color: {_COLOR_TEXT}; border-radius: {_BTN_RADIUS}px;")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(12, 8, 12, 8)
        self.lbl_title = QLabel("MISSION")
        self.lbl_title.setWordWrap(True)
        self.lbl_title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_HDR_TEXT}; background: transparent;"
        )
        hdr_layout.addWidget(self.lbl_title)
        layout.addWidget(hdr)

        info_card = QWidget()
        info_card.setStyleSheet(
            f"background-color: {_COLOR_SURFACE_ALT}; border-radius: {_BTN_RADIUS}px;"
        )
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(4)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet(
            f"font-size: 12px; color: {_COLOR_MUTED}; font-family: {_FONT_UI};"
            " background: transparent;"
        )
        info_layout.addWidget(self.lbl_status)

        self.lbl_totals = QLabel("Total volume: --   Total weight: --")
        self.lbl_totals.setAlignment(Qt.AlignCenter)
        self.lbl_totals.setStyleSheet(
            f"font-size: 12px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};"
            " background: transparent;"
        )
        info_layout.addWidget(self.lbl_totals)

        sep = _make_divider()
        info_layout.addWidget(sep)

        items_label = QLabel("MISSION ITEMS")
        items_label.setStyleSheet(
            f"font-size: 11px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_MUTED}; background: transparent;"
        )
        info_layout.addWidget(items_label)

        layout.addWidget(info_card)

        self.list = QListWidget()
        self.list.setStyleSheet(_list_style())
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setWordWrap(True)
        layout.addWidget(self.list, stretch=1)

        self.btn_set_current = _btn(_SS_PRIMARY, "Set as Current Mission")
        layout.addWidget(self.btn_set_current)

        self.btn_cancel_assign = _back_btn("Cancel Assignment")
        layout.addWidget(self.btn_cancel_assign)
        self.btn_cancel_assign.hide()

        action_row = QHBoxLayout()
        self.btn_delete_mission = _delete_btn("Delete Mission")
        self.btn_back = _back_btn()
        action_row.addWidget(self.btn_back)
        action_row.addWidget(self.btn_delete_mission)
        layout.addLayout(action_row)

        self._timeline_data = []
        self._summary = None


class MissionCreatePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        hdr = QWidget()
        hdr.setStyleSheet(f"background-color: {_COLOR_TEXT}; border-radius: {_BTN_RADIUS}px;")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(12, 8, 12, 8)
        title = QLabel("CREATE MISSION")
        title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_HDR_TEXT}; background: transparent;"
        )
        hdr_layout.addWidget(title)
        layout.addWidget(hdr)

        prompt = QLabel("Mission name:")
        prompt.setStyleSheet(
            f"font-size: 13px; font-weight: 600; font-family: {_FONT_UI}; color: {_COLOR_MUTED};"
        )
        layout.addWidget(prompt)

        self.lbl_recording_status = QLabel("Recording… speak the mission name")
        self.lbl_recording_status.setAlignment(Qt.AlignCenter)
        self.lbl_recording_status.setStyleSheet(
            f"font-size: 13px; color: {_COLOR_STOP}; font-weight: 700; font-family: {_FONT_UI};"
        )
        layout.addWidget(self.lbl_recording_status)

        self.text = QTextEdit()
        self.text.setFixedHeight(76)
        self.text.setStyleSheet(f"""
            font-size: 18px;
            font-family: {_FONT_UI};
            border: 1px solid {_COLOR_BORDER};
            border-radius: {_BTN_RADIUS}px;
            padding: 8px;
            background-color: {_COLOR_SURFACE};
            color: {_COLOR_TEXT};
        """)
        layout.addWidget(self.text)

        self.btn_create = _btn(_SS_PRIMARY, "Create Mission")
        layout.addWidget(self.btn_create)

        self.btn_cancel = _back_btn("Cancel")
        layout.addWidget(self.btn_cancel)
        
class RockDetailPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"""
            background-color: {_COLOR_PAGE};
            color: {_COLOR_TEXT};
            font-family: {_FONT_UI};
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        # --- Fixed header ---
        hdr = QWidget()
        hdr.setStyleSheet(f"background-color: {_COLOR_TEXT}; border-radius: {_BTN_RADIUS}px;")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(12, 8, 12, 8)
        self.lbl_title = QLabel("ROCK DETAIL")
        self.lbl_title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_HDR_TEXT}; background: transparent;"
        )
        self.lbl_time = QLabel("")
        self.lbl_time.setStyleSheet(
            f"font-size: 11px; color: {_COLOR_HOME_MUTED}; font-family: {_FONT_DATA}; background: transparent;"
        )
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr_layout.addWidget(self.lbl_title)
        hdr_layout.addStretch()
        hdr_layout.addWidget(self.lbl_time)
        layout.addWidget(hdr)

        # --- Tab bar ---
        tab_bar = QHBoxLayout()
        tab_bar.setSpacing(4)
        self.btn_tab_classification = QPushButton("CLASSIFICATION")
        self.btn_tab_classification.setMinimumHeight(36)
        self.btn_tab_field_notes = QPushButton("FIELD NOTES")
        self.btn_tab_field_notes.setMinimumHeight(36)
        tab_bar.addWidget(self.btn_tab_classification)
        tab_bar.addWidget(self.btn_tab_field_notes)
        layout.addLayout(tab_bar)

        # --- Stacked panels ---
        self.panels = QStackedWidget()
        layout.addWidget(self.panels, stretch=1)

        # Panel 0: Classification
        panel_class = QWidget()
        pc_layout = QVBoxLayout(panel_class)
        pc_layout.setContentsMargins(4, 8, 4, 4)
        pc_layout.setSpacing(10)

        images_row = QHBoxLayout()
        images_row.setSpacing(12)
        self.lbl_top = QLabel()
        self.lbl_top.setAlignment(Qt.AlignCenter)
        self.lbl_top.setMinimumHeight(110)
        self.lbl_top.setStyleSheet(
            f"background-color: {_COLOR_IMG_BG}; border: 1px solid {_COLOR_HOME_BTN}; border-radius: {_BTN_RADIUS}px;"
        )
        self.lbl_side = QLabel()
        self.lbl_side.setAlignment(Qt.AlignCenter)
        self.lbl_side.setMinimumHeight(110)
        self.lbl_side.setStyleSheet(
            f"background-color: {_COLOR_IMG_BG}; border: 1px solid {_COLOR_HOME_BTN}; border-radius: {_BTN_RADIUS}px;"
        )
        images_row.addWidget(self.lbl_top, stretch=1)
        images_row.addWidget(self.lbl_side, stretch=1)
        pc_layout.addLayout(images_row)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet(
            f"font-size: 13px; font-family: {_FONT_DATA}; color: {_COLOR_MUTED};"
            f" background-color: {_COLOR_SURFACE_ALT}; border-radius: {_BTN_RADIUS}px; padding: 6px 10px;"
        )
        pc_layout.addWidget(self.lbl_info)

        lbl_features_title = QLabel("OBSERVED FEATURES")
        lbl_features_title.setStyleSheet(
            f"font-size: 11px; font-weight: 700; font-family: {_FONT_UI}; color: {_COLOR_MUTED};"
        )
        pc_layout.addWidget(lbl_features_title)

        self.features_section = QWidget()
        self.features_section.setObjectName("featuresBox")
        self.features_section.setStyleSheet(
            f"QWidget#featuresBox {{ background-color: {_COLOR_SURFACE_ALT};"
            f" border: 1px solid {_COLOR_FEATURES_BORDER}; border-radius: {_BTN_RADIUS}px; }}"
        )
        self.features_layout = QVBoxLayout(self.features_section)
        self.features_layout.setContentsMargins(10, 8, 10, 8)
        self.features_layout.setSpacing(4)
        pc_layout.addWidget(self.features_section)
        pc_layout.addStretch(1)
        self.panels.addWidget(panel_class)

        # Panel 1: Field Notes
        panel_notes = QWidget()
        pn_layout = QVBoxLayout(panel_notes)
        pn_layout.setContentsMargins(4, 8, 4, 4)
        pn_layout.setSpacing(10)

        summary_header_layout = QHBoxLayout()
        lbl_summary_title = QLabel("AI SUMMARY")
        lbl_summary_title.setStyleSheet(
            f"font-size: 11px; font-weight: 700; font-family: {_FONT_UI}; color: {_COLOR_MUTED};"
        )
        self.btn_force_summary = _btn(_SS_SECONDARY, "Re-Summarize", height=36)
        summary_header_layout.addWidget(lbl_summary_title)
        summary_header_layout.addStretch(1)
        summary_header_layout.addWidget(self.btn_force_summary)
        pn_layout.addLayout(summary_header_layout)

        # Accordion for AI summary categories
        self.summary_buttons_widget = QWidget()
        accordion_layout = QVBoxLayout(self.summary_buttons_widget)
        accordion_layout.setSpacing(3)
        accordion_layout.setContentsMargins(0, 0, 0, 0)

        self.summary_data = {}
        self.summary_buttons = {}
        self._accordion_labels = {}

        self.categories = [
            ("Color & Appearance",          "Color & Appearance"),
            ("Mineralogy & Composition",     "Mineralogy & Composition"),
            ("Texture & Structure",          "Texture & Structure"),
            ("Weathering & Alteration",      "Weathering & Alteration"),
            ("Dimensions & Weight",          "Dimensions & Weight"),
            ("Field Context & Notes",        "Field Context & Sampling Notes"),
        ]

        _btn_ss = (
            f"QPushButton {{ background-color: {_COLOR_SURFACE_ALT}; color: {_COLOR_TEXT};"
            f" font-weight: 700; font-size: 13px; font-family: {_FONT_UI};"
            f" border-radius: {_BTN_RADIUS}px; border: none;"
            " text-align: left; padding: 0 10px; }"
            f" QPushButton:hover {{ background-color: {_COLOR_HOVER_LIGHT}; }}"
            f" QPushButton:focus {{ border: 2px solid {_COLOR_FOCUS}; }}"
        )
        _content_ss = (
            f"background-color: {_COLOR_PAGE}; color: {_COLOR_TEXT};"
            f" font-size: 13px; font-family: {_FONT_UI};"
            " border: none; padding: 6px 12px 10px 14px;"
        )

        for short_name, full_name in self.categories:
            btn = QPushButton(f"▶  {short_name}")
            btn.setMinimumHeight(44)
            btn.setStyleSheet(_btn_ss)
            btn.clicked.connect(lambda checked=False, fn=full_name: self._toggle_accordion(fn))
            self.summary_buttons[full_name] = btn
            self.summary_data[full_name] = "Not specified."

            content_lbl = QLabel("Not specified.")
            content_lbl.setWordWrap(True)
            content_lbl.setStyleSheet(_content_ss)
            content_lbl.setVisible(False)
            self._accordion_labels[full_name] = content_lbl

            accordion_layout.addWidget(btn)
            accordion_layout.addWidget(content_lbl)
            sep = _make_divider()
            accordion_layout.addWidget(sep)

        pn_layout.addWidget(self.summary_buttons_widget)

        self.lbl_summary_loading = QLabel("Generating AI summary…")
        self.lbl_summary_loading.setAlignment(Qt.AlignCenter)
        self.lbl_summary_loading.setStyleSheet(
            f"font-size: 13px; font-style: italic; color: {_COLOR_MUTED};"
            f" font-family: {_FONT_DATA}; background-color: transparent;"
        )
        self.lbl_summary_loading.setMinimumHeight(120)
        pn_layout.addWidget(self.lbl_summary_loading)
        self.lbl_summary_loading.hide()

        lbl_notes_title = QLabel("VOICE NOTES")
        lbl_notes_title.setStyleSheet(
            f"font-size: 11px; font-weight: 700; font-family: {_FONT_UI}; color: {_COLOR_MUTED};"
        )
        pn_layout.addWidget(lbl_notes_title)

        self.notes_text = QTextEdit()
        self.notes_text.setReadOnly(True)
        self.notes_text.setStyleSheet(f"""
            QTextEdit {{
                font-size: 14px;
                font-family: {_FONT_UI};
                border: 1px solid {_COLOR_BORDER};
                border-radius: {_BTN_RADIUS}px;
                padding: 8px;
                background-color: {_COLOR_SURFACE};
                color: {_COLOR_TEXT};
            }}
        """)
        pn_layout.addWidget(self.notes_text, stretch=1)
        self.panels.addWidget(panel_notes)

        # --- Fixed bottom actions ---
        layout.addSpacing(4)
        rock_action_row = QHBoxLayout()
        rock_action_row.setSpacing(8)
        self.btn_make_current = _btn(_SS_PRIMARY, "Make Current Rock")
        self.btn_delete_rock  = _delete_btn("Delete Rock")
        rock_action_row.addWidget(self.btn_make_current)
        rock_action_row.addWidget(self.btn_delete_rock)
        layout.addLayout(rock_action_row)

        self.btn_back = _back_btn()
        self.btn_back.setProperty("joystick_primary", True)
        layout.addWidget(self.btn_back)
        self._current_rock_id = None
        self._apply_tab_styles(0)

    def _apply_tab_styles(self, active_idx: int) -> None:
        active_ss = (
            f"QPushButton {{ background-color: {_COLOR_PRIMARY}; color: white;"
            f" font-size: 12px; font-weight: 700; font-family: {_FONT_UI};"
            f" border-radius: {_BTN_RADIUS}px; border: none; }}"
        )
        inactive_ss = (
            f"QPushButton {{ background-color: {_COLOR_SURFACE_ALT}; color: {_COLOR_MUTED};"
            f" font-size: 12px; font-weight: 700; font-family: {_FONT_UI};"
            f" border-radius: {_BTN_RADIUS}px; border: 1px solid {_COLOR_BORDER}; }}"
            f" QPushButton:hover {{ background-color: {_COLOR_HOVER_LIGHT}; }}"
            f" QPushButton:focus {{ border: 2px solid {_COLOR_FOCUS}; }}"
        )
        self.btn_tab_classification.setStyleSheet(active_ss if active_idx == 0 else inactive_ss)
        self.btn_tab_field_notes.setStyleSheet(active_ss if active_idx == 1 else inactive_ss)

    def _parse_summary_to_table_data(self, summary: str) -> list[tuple[str, str]]:
        """Parses the bulleted AI string into Key/Value pairs."""
        if not summary or "Generating AI summary" in summary or "unavailable" in summary:
            return [("Status", summary or "No data")]

        parsed_data = []
        lines = [line.strip() for line in str(summary).splitlines() if line.strip()]
        
        for line in lines:
            # Strip the leading bullet point if the AI included it
            if line.startswith("- "):
                line = line[2:]
                
            # Split exactly at the first colon
            if ":" in line:
                key, value = line.split(":", 1)
                parsed_data.append((key.strip(), value.strip()))
            else:
                # Catch-all just in case the AI writes a weird sentence
                parsed_data.append(("Note", line.strip()))
                
        return parsed_data

    def _toggle_accordion(self, full_name: str) -> None:
        currently_open = self._accordion_labels[full_name].isVisible()
        # Collapse all
        for fn, lbl in self._accordion_labels.items():
            lbl.setVisible(False)
            short = next(s for s, f in self.categories if f == fn)
            self.summary_buttons[fn].setText(f"▶  {short}")
        # If it was closed, open it
        if not currently_open:
            self._accordion_labels[full_name].setVisible(True)
            short = next(s for s, f in self.categories if f == full_name)
            self.summary_buttons[full_name].setText(f"▼  {short}")

    def _populate_buttons(self, data: list[tuple[str, str]]) -> None:
        # Collapse all and reset content
        for fn, lbl in self._accordion_labels.items():
            lbl.setVisible(False)
            short = next(s for s, f in self.categories if f == fn)
            self.summary_buttons[fn].setText(f"▶  {short}")
            self.summary_data[fn] = "Not specified."
            lbl.setText("Not specified.")

        is_generating = any("Generating" in v for k, v in data)
        if is_generating:
            self.summary_buttons_widget.hide()
            self.lbl_summary_loading.show()
            self.btn_force_summary.setEnabled(False)
            return

        self.lbl_summary_loading.hide()
        self.summary_buttons_widget.show()
        self.btn_force_summary.setEnabled(True)

        for key, val in data:
            for full_name in self.summary_data.keys():
                if key.split()[0].lower() in full_name.lower():
                    self.summary_data[full_name] = val
                    self._accordion_labels[full_name].setText(val)


    def set_entry(self, entry, associated_notes=None, ai_summary: str = "Generating AI summary...") -> None:
        self._current_rock_id = entry.rock_id
        self._current_rock_entry = entry                  # Save for the button
        self._current_associated_notes = associated_notes # Save for the button

        self.panels.setCurrentIndex(0)
        self._apply_tab_styles(0)

        dt = datetime.datetime.fromtimestamp(entry.ts) if entry.ts else None
        self.lbl_time.setText(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}" if dt else "Unknown time")

        res = entry.result
        self.lbl_title.setText(f"{res.label.upper()} ({int(res.confidence * 100)}%)")

        w, h = 130, 100

        if res.image_path and os.path.exists(res.image_path):
            self.lbl_top.setPixmap(QPixmap(res.image_path).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_top.setPixmap(QPixmap())
            self.lbl_top.setText("Top")

        has_side = bool(res.side_image_path and os.path.exists(res.side_image_path))
        self.lbl_side.setVisible(has_side)
        if has_side:
            self.lbl_side.setPixmap(QPixmap(res.side_image_path).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_side.setPixmap(QPixmap())
            self.lbl_side.setText("Side")

        vol_txt = f"{res.estimated_volume} cm³" if res.estimated_volume is not None else "N/A"
        weight_txt = res.estimated_weight if res.estimated_weight is not None else "N/A"
        self.lbl_info.setText(f"Vol: {vol_txt}   Wt: {weight_txt}")

        # Rebuild ML features section
        feat_layout = self.features_layout
        while feat_layout.count():
            child = feat_layout.takeAt(0)
            w = child.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        geo_lookup = {}
        for nd in (res.geology_notes or []):
            if isinstance(nd, dict):
                geo_lookup[nd.get("feature", "")] = nd.get("note", "")
            else:
                geo_lookup[getattr(nd, "feature", "")] = getattr(nd, "note", "")
        ui_meta = (res.raw or {}).get("ui", {}) if isinstance(res.raw, dict) else {}
        show_features = (
            res.features is not None
            and res.tier not in (None, "uncertain")
            and ui_meta.get("show_features", True)
        )
        first_feat = True
        if show_features:
            for feat_name, feat_data in res.features.items():
                if not feat_data.get("display", True):
                    continue
                if feat_data.get("tier") not in ("high", "medium"):
                    continue
                if not first_feat:
                    sep = _make_divider()
                    sep.setStyleSheet(f"background-color: {_COLOR_FEAT_SEP}; max-height: 1px;")
                    feat_layout.addWidget(sep)
                first_feat = False
                conf_pct = int(feat_data.get("confidence", 0.0) * 100)
                display_name = feat_name.replace("_", " ").title()
                feat_row_w = QWidget()
                feat_row = QHBoxLayout(feat_row_w)
                feat_row.setContentsMargins(0, 5, 0, 3)
                feat_row.setSpacing(8)
                lbl_n = QLabel(display_name)
                lbl_n.setStyleSheet(
                    f"font-size: 14px; color: {_COLOR_TEXT}; font-weight: 700; font-family: {_FONT_UI};"
                )
                lbl_v = QLabel(str(feat_data.get("value", "")))
                lbl_v.setStyleSheet(
                    f"font-size: 14px; color: {_COLOR_TEXT}; font-family: {_FONT_DATA};"
                )
                lbl_c = QLabel(f"{conf_pct}%")
                lbl_c.setStyleSheet(
                    f"font-size: 12px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};"
                )
                feat_row.addWidget(lbl_n)
                feat_row.addStretch()
                feat_row.addWidget(lbl_v)
                feat_row.addWidget(lbl_c)
                feat_layout.addWidget(feat_row_w)
                note_text = geo_lookup.get(feat_name, "")
                if note_text:
                    lbl_note = QLabel(note_text)
                    lbl_note.setWordWrap(True)
                    lbl_note.setStyleSheet(
                        f"font-size: 12px; color: {_COLOR_MUTED}; font-style: italic;"
                        f" font-family: {_FONT_UI}; padding: 1px 0px 4px 0px;"
                    )
                    lbl_note.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                    feat_layout.addWidget(lbl_note)

        self._populate_buttons(self._parse_summary_to_table_data(ai_summary))

        # --- Populate Voice Notes ---
        self.notes_text.setPlaceholderText("No recordings linked to this rock.")
        if associated_notes:
            notes_str = ""
            for n in associated_notes:
                ts = n.get("ts", 0)
                note_dt = datetime.datetime.fromtimestamp(ts) if ts else None
                time_str = note_dt.strftime("%H:%M:%S") if note_dt else "Unknown"
                cleaned = n.get("cleaned", n.get("transcript", ""))
                notes_str += f"[{time_str}] {cleaned}\n\n"
            self.notes_text.setPlainText(notes_str.strip())
        else:
            self.notes_text.setPlainText("")

    def set_ai_summary(self, rock_id: str, summary: str) -> None:
        if self._current_rock_id == rock_id:
            self._populate_buttons(self._parse_summary_to_table_data(summary))


class VoiceNoteDetailPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"""
            background-color: {_COLOR_PAGE};
            color: {_COLOR_TEXT};
            font-family: {_FONT_UI};
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        hdr = QWidget()
        hdr.setStyleSheet(f"background-color: {_COLOR_TEXT}; border-radius: {_BTN_RADIUS}px;")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(12, 8, 12, 8)
        self.lbl_title = QLabel("VOICE NOTE")
        self.lbl_title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; font-family: {_FONT_UI};"
            f" color: {_COLOR_HDR_TEXT}; background: transparent;"
        )
        self.lbl_time = QLabel("")
        self.lbl_time.setStyleSheet(
            f"font-size: 11px; color: {_COLOR_HOME_MUTED}; font-family: {_FONT_DATA}; background: transparent;"
        )
        self.lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr_layout.addWidget(self.lbl_title)
        hdr_layout.addStretch()
        hdr_layout.addWidget(self.lbl_time)
        layout.addWidget(hdr)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet(f"""
            font-size: 16px;
            font-family: {_FONT_UI};
            border: 1px solid {_COLOR_BORDER};
            border-radius: {_BTN_RADIUS}px;
            padding: 8px;
            background-color: {_COLOR_SURFACE};
            color: {_COLOR_TEXT};
        """)
        layout.addWidget(self.text, stretch=1)

        note_action_row = QHBoxLayout()
        self.btn_add_to_classification = _btn(_SS_SECONDARY, "Add to Classification")
        self.btn_delete_note = _delete_btn("Delete Note")
        note_action_row.addWidget(self.btn_add_to_classification)
        note_action_row.addWidget(self.btn_delete_note)
        layout.addLayout(note_action_row)

        self.btn_back = _back_btn()
        layout.addWidget(self.btn_back)
        self._current_note = None

    def set_note(self, note: dict) -> None:
        self._current_note = note
        ts = note.get("ts", 0)
        if ts:
            dt = datetime.datetime.fromtimestamp(ts)
            self.lbl_time.setText(dt.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            self.lbl_time.setText("Unknown time")
        cleaned = note.get("cleaned", note.get("transcript", ""))
        transcript = note.get("transcript", "")
        self.text.setPlainText(cleaned if cleaned else transcript)

class ExpandingVoiceWidget(QWidget):
    def __init__(self, vm, parent=None):
        super().__init__(parent)
        self.vm = vm
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(10)

        self.trigger_btn = QPushButton("🎤")
        self.trigger_btn.setFixedSize(44, 44)
        self.trigger_btn.setStyleSheet(
            f"background-color: {_COLOR_TEXT}; color: white;"
            f" border-radius: {_BTN_RADIUS}px; border: 1px solid {_COLOR_HOME_BTN};"
            f" font-size: 18px; font-family: {_FONT_UI};"
        )
        self.main_layout.addWidget(self.trigger_btn)

        self.button_container = QWidget()
        self.container_layout = QHBoxLayout(self.button_container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_start = self._make_sub_btn("Start", _COLOR_PRIMARY)
        self.btn_stop  = self._make_sub_btn("Stop",  _COLOR_STOP)
        self.btn_redo  = self._make_sub_btn("Redo",  _COLOR_SECONDARY)

        self.container_layout.addWidget(self.btn_start)
        self.container_layout.addWidget(self.btn_stop)
        self.container_layout.addWidget(self.btn_redo)
        
        self.main_layout.addWidget(self.button_container)
        self.button_container.hide()

        self.vm.recording_status_changed.connect(self._update_ui_state)
        self.trigger_btn.clicked.connect(self._toggle_sub_buttons)
        self.btn_start.clicked.connect(lambda: self.vm.start_voice_to_text(silent=True))
        self.btn_stop.clicked.connect(self.vm.stop_voice_to_text)
        self.btn_redo.clicked.connect(self.vm.redo_voice_to_text)

        if hasattr(self.vm, 'vtt_active'):
            self._update_ui_state(self.vm.vtt_active)
        else:
            self._update_ui_state(False)

    def _update_ui_state(self, is_recording: bool):
        if is_recording:
            self.trigger_btn.setText("●")
            self.trigger_btn.setStyleSheet(
                f"background-color: {_COLOR_STOP}; color: white;"
                f" border-radius: {_BTN_RADIUS}px; border: 1px solid {_COLOR_STOP_H};"
                f" font-size: 16px; font-weight: 700; font-family: {_FONT_UI};"
            )
        else:
            self.trigger_btn.setText("🎤")
            self.trigger_btn.setStyleSheet(
                f"background-color: {_COLOR_TEXT}; color: white;"
                f" border-radius: {_BTN_RADIUS}px; border: 1px solid {_COLOR_HOME_BTN};"
                f" font-size: 18px; font-family: {_FONT_UI};"
            )

    def _make_sub_btn(self, text, color):
        btn = QPushButton(text)
        btn.setFixedSize(58, 38)
        if text.lower() == "stop":
            btn.setStyleSheet(_button_style(_COLOR_STOP, "white", _COLOR_STOP_H))
        elif text.lower() == "start":
            btn.setStyleSheet(_button_style(_COLOR_PRIMARY, "white", _COLOR_PRIMARY_H))
        else:
            btn.setStyleSheet(_button_style(_COLOR_SECONDARY, _COLOR_TEXT, _COLOR_SECONDARY_H, border=_COLOR_BORDER))
        return btn

    def _toggle_sub_buttons(self):
        self.button_container.setVisible(not self.button_container.isVisible())

class AppWindow(QMainWindow):
    def __init__(self, vm):
        super().__init__()

        self.session_start_time = time.time()
        
        self.vm = vm
        self.setWindowTitle("SAGE Jetson UI")
        self.setStyleSheet(f"""
            background-color: {_COLOR_BG};
            color: {_COLOR_TEXT};
            font-family: {_FONT_UI};
            font-size: 15px;
            font-weight: 400;
            QToolTip {{
                background-color: {_COLOR_SURFACE};
                color: {_COLOR_TEXT};
                border: 1px solid {_COLOR_BORDER};
                border-radius: 4px;
                padding: 5px;
                font-size: 13px;
            }}
        """)

        # --- NEW: Master Layout for the whole app ---
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- GLOBAL STATUS BAR ---
        self.status_widget = QWidget()
        self.status_widget.setStyleSheet("background: transparent;")
        status_layout = QHBoxLayout(self.status_widget)
        status_layout.setContentsMargins(5, 5, 5, 0)
        
        _sb_style = (
            f"font-size: 12px; font-weight: 700; font-family: {_FONT_DATA};"
            f" color: {_COLOR_STATUS_BAR}; background: transparent;"
        )
        self.lbl_time = QLabel("00:00")
        self.lbl_time.setStyleSheet(_sb_style)

        self.lbl_date = QLabel("Mon, Jan 1")
        self.lbl_date.setStyleSheet(_sb_style)
        self.lbl_date.setAlignment(Qt.AlignCenter)

        self.lbl_battery = QLabel("100%")
        self.lbl_battery.setStyleSheet(_sb_style)
        self.lbl_battery.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        status_layout.addWidget(self.lbl_time)
        status_layout.addStretch(1)
        status_layout.addWidget(self.lbl_date)
        status_layout.addStretch(1)
        status_layout.addWidget(self.lbl_battery)
        
        self.main_layout.addWidget(self.status_widget, 0, Qt.AlignTop)

        # Start global timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)
        self._update_status() 
        # ---------------------------

        # The deck of cards (StackedWidget) now goes BELOW the status bar
        self.stack = QStackedWidget()
        self.stack.setMinimumSize(0, 0)
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.stack, stretch=1)

        self.home = HomePage()
        self.loading = LoadingPage()
        self.camera_preview = CameraPreviewPage(self.vm)
        self.capture_review = CaptureReviewPage(self.vm)
        self.classified = ClassifiedPage(self.vm)
        self.voice_loading = VoiceLoadingPage()
        self.voice = VoicePage()
        self.trip = TripLoadPage()
        self.mission_detail = MissionDetailPage()
        self.mission_create = MissionCreatePage()
        self.rock_detail = RockDetailPage()
        self.voice_note_detail = VoiceNoteDetailPage()
        self._selected_mission_id = None
        self._pending_delete_mission_id = None
        self._pending_delete_mission_armed_at = 0.0
        self._delete_mission_timer = QTimer(self)
        self._delete_mission_timer.setSingleShot(True)
        self._delete_mission_timer.timeout.connect(self._reset_delete_mission_arm)
        self._mission_name_typing_mode = False

        self.stack.addWidget(self.home)
        self.stack.addWidget(self.loading)
        self.stack.addWidget(self.camera_preview)
        self.stack.addWidget(self.capture_review)
        self.stack.addWidget(self.classified)
        self.stack.addWidget(self.voice_loading)
        self.stack.addWidget(self.voice)
        self.stack.addWidget(self.trip)
        self.stack.addWidget(self.mission_detail)
        self.stack.addWidget(self.mission_create)
        self.stack.addWidget(self.rock_detail)
        self.stack.addWidget(self.voice_note_detail)

        for i in range(self.stack.count()):
            w = self.stack.widget(i)
            w.setMinimumSize(0, 0)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._wire_ui()
        self._wire_vm()

        self.joystick = JoystickNavigator(self, bus=1)
        self.joystick.start()

        self._show_state(AppStateType.HOME)

        is_jetson = connector.is_jetson()
        
        if is_jetson:
            try:
                self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
                self._apply_fullscreen_geometry()
                self.showFullScreen()
            except Exception as e:
                import sys
                print(f"ERROR: Failed to show fullscreen: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                self.resize(480, 800 )
                self.show()
        else:
            self.resize(480, 800 )
            self.setMinimumSize(480, 800 )
            self.setMaximumSize(480, 800 )
            self.show()
        
        self._setup_shortcuts()

    def _update_vtt_context_label(self) -> None:
        project_root = Path(__file__).resolve().parent.parent.parent
        context_file = project_root / "ML-classifications" / "visual_context.txt"
        
        context_words = ""
        if context_file.exists():
            try:
                with open(context_file, "r") as f:
                    context_words = f.read().strip()
            except Exception:
                pass
                
        if context_words:
            self.voice.lbl_context.setText(f"Context: {context_words}")
        else:
            self.voice.lbl_context.setText("Context: None (Default)")
    
    def _on_reset_context_clicked(self) -> None:
        self.vm.reset_voice_context()
        self.voice.btn_reset.hide()
        self._update_vtt_context_label()

    def _on_force_summary_clicked(self) -> None:
        entry = getattr(self.rock_detail, "_current_rock_entry", None)
        notes = getattr(self.rock_detail, "_current_associated_notes", None)
        if entry and notes:
            self.rock_detail.set_ai_summary(entry.rock_id, "Generating AI summary...")
            self.vm.request_rock_summary(entry, notes, force=True)
   
    def _update_voice_buttons(self, mode: str) -> None:
        """Dynamically hides/shows Voice to Text buttons based on the current phase."""
        v = self.voice
        
        # 1. Hide everything first
        for b in [v.btn_start, v.btn_stop, v.btn_redo, v.btn_save, v.btn_delete, v.btn_reset, v.btn_cancel]:
            b.hide()
            
        # Ensure overlay is hidden by default
        if mode != "formatting":
            v.loading_overlay.stop()
            
        # 2. Show only what belongs in the current phase
        if mode == "initial":
            v.btn_start.show()
            v.btn_cancel.show()
            active_rock = getattr(self.vm, "active_rock_id", None)
            if active_rock and active_rock != "ORPHAN":
                v.btn_reset.show()
                
        elif mode == "recording":
            v.btn_stop.setText("Stop")
            v.btn_stop.show()
            
        elif mode == "formatting":
            # Fire up the gray screen, spinner, and animated text!
            v.loading_overlay.start()
            
        elif mode == "review":
            v.btn_stop.setText("Edit")
            v.btn_stop.show()
            v.btn_redo.show()
            v.btn_save.show()
            v.btn_delete.show()

    def _start_rock_assignment(self, note_ts):
        """Kicks off the jiggle animation and waits for a rock click."""
        self._assigning_note_ts = note_ts
        self._shake_offset = 0
        self._shake_direction = 1

        self._shake_timer = QTimer(self)
        self._shake_timer.timeout.connect(self._do_shake)
        self._shake_timer.start(40)

        if hasattr(self, "mission_detail"):
            self.mission_detail.btn_cancel_assign.show()
            self.mission_detail.lbl_status.setText("▼ Navigate list — select a rock to link this note")

    def _do_shake(self):
        """Bounces the margins of the Rock widgets left and right."""
        self._shake_offset += self._shake_direction * 2
        if abs(self._shake_offset) >= 4:
            self._shake_direction *= -1
            
        for i in range(self.mission_detail.list.count()):
            item_dict = self.mission_detail._timeline_data[i]
            if item_dict["type"] == "rock":
                list_item = self.mission_detail.list.item(i)
                widget = self.mission_detail.list.itemWidget(list_item)
                if widget:
                    # Alternating padding creates the "shake"
                    widget.setContentsMargins(6 + self._shake_offset, 2, 6 - self._shake_offset, 2)

    def _stop_rock_assignment(self):
        """Stops the timer and resets everything back to normal."""
        self._assigning_note_ts = None
        if hasattr(self, "_shake_timer") and self._shake_timer is not None:
            self._shake_timer.stop()
            self._shake_timer = None

        if hasattr(self, "mission_detail"):
            self.mission_detail.btn_cancel_assign.hide()
            summary = getattr(self.mission_detail, "_summary", None)
            if summary:
                is_current = summary.mission.mission_id == getattr(self.vm, "active_mission_id", None)
                self.mission_detail.lbl_status.setText("Current mission" if is_current else "")
            if hasattr(self.mission_detail, "list"):
                for i in range(self.mission_detail.list.count()):
                    list_item = self.mission_detail.list.item(i)
                    widget = self.mission_detail.list.itemWidget(list_item)
                    if widget:
                        widget.setContentsMargins(0, 0, 0, 0)
    
    def _on_delete_all_clicked(self) -> None:
        # Create the custom popup
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Delete All Data")
        msg_box.setText("Are you sure? Clicking CONFIRM will delete data from all your past missions. This data will be irretrievable.")
        msg_box.setStyleSheet(f"QLabel {{ color: {_COLOR_TEXT}; font-size: 15px; font-weight: normal; font-family: {_FONT_UI}; }} QMessageBox {{ background-color: {_COLOR_BG}; }}")

        btn_confirm = msg_box.addButton("CONFIRM", QMessageBox.AcceptRole)
        btn_confirm.setStyleSheet(
            f"background-color: {_COLOR_DANGER}; color: white; font-weight: 700;"
            f" font-family: {_FONT_UI}; padding: 8px; border-radius: {_BTN_RADIUS}px; border: none;"
        )

        btn_cancel = msg_box.addButton("CANCEL", QMessageBox.RejectRole)
        btn_cancel.setStyleSheet(
            f"background-color: {_COLOR_SECONDARY}; color: {_COLOR_TEXT}; font-weight: 700;"
            f" font-family: {_FONT_UI}; padding: 8px; border-radius: {_BTN_RADIUS}px;"
            f" border: 1px solid {_COLOR_BORDER};"
        )
        
        msg_box.exec()
        
        # Check which button they clicked
        if msg_box.clickedButton() == btn_confirm:
            self.vm.clear_all_trip_data()

    def _setup_shortcuts(self) -> None:
        shortcut_f11 = QShortcut(QKeySequence(Qt.Key_F11), self)
        shortcut_f11.activated.connect(self._toggle_fullscreen)
        
        shortcut_escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        shortcut_escape.activated.connect(self._exit_fullscreen)
        
        shortcut_quit = QShortcut(QKeySequence("Ctrl+C"), self)
        shortcut_quit.activated.connect(self._quit_application)
        
    def _apply_virtual_key_to_text_edit(self, editor: QTextEdit, key: str) -> None:
        cursor = editor.textCursor()

        if key == "⌫":
            cursor.deletePreviousChar()
        elif key == "Space":
            cursor.insertText(" ")
        elif key == "←":
            cursor.movePosition(QTextCursor.Left)
        elif key == "→":
            cursor.movePosition(QTextCursor.Right)
        elif key == "↑":
            cursor.movePosition(QTextCursor.Up)
        elif key == "↓":
            cursor.movePosition(QTextCursor.Down)
        else:
            cursor.insertText(key)
        editor.setTextCursor(cursor)
        editor.setFocus()

    def _sync_keyboard_shift(self, editor: QTextEdit, keyboard) -> None:
        cursor = editor.textCursor()
        text_so_far = editor.toPlainText()[:cursor.position()]

        if len(text_so_far) == 0:
            keyboard.set_shift_state(1)
        elif len(text_so_far) >= 2 and text_so_far[-2:] in [". ", "? ", "! "]:
            keyboard.set_shift_state(1)
        elif len(text_so_far) >= 1 and text_so_far[-1] == "\n":
            keyboard.set_shift_state(1)
        else:
            keyboard.set_shift_state(0)

    def _on_virtual_key_pressed(self, key: str) -> None:
        self._apply_virtual_key_to_text_edit(self.voice.text, key)
        self.vm.transcription_text = self.voice.text.toPlainText()
        self._sync_keyboard_shift(self.voice.text, self.inline_keyboard)

    def _on_mission_key_pressed(self, key: str) -> None:
        if not self._mission_name_typing_mode:
            self._mission_name_typing_mode = True
            self.vm.stop_mission_name_recording(abort=True)
        self._apply_virtual_key_to_text_edit(self.mission_create.text, key)
        self._sync_keyboard_shift(self.mission_create.text, self.mission_keyboard)
        self._update_mission_create_button()

    def _update_mission_create_button(self) -> None:
        self.mission_create.btn_create.setEnabled(bool(self.mission_create.text.toPlainText().strip()))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_F11:
            self._toggle_fullscreen()
            event.accept()
            return
        
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self._exit_fullscreen()
                event.accept()
                return
        
        if event.key() == Qt.Key_C and event.modifiers() == Qt.ControlModifier:
            self._quit_application()
            event.accept()
            return
        
        super().keyPressEvent(event)

    def _apply_fullscreen_geometry(self) -> None:
        """Set window to exact screen size (edge-to-edge). Use geometry(), not availableGeometry()."""
        screen = self.screen() if self.windowHandle() else QApplication.primaryScreen()
        if not screen:
            return
        rect = screen.geometry()
        self.setMinimumSize(rect.size())
        self.setMaximumSize(rect.size())
        self.setGeometry(rect)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self._exit_fullscreen()
        else:
            self._apply_fullscreen_geometry()
            self.showFullScreen()

    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            sim_w, sim_h = 480, 800 
            self.setFixedSize(sim_w, sim_h)

    def _quit_application(self) -> None:
        self.joystick.stop()
        self.close()

    def _wire_ui(self) -> None:
        self.home.btn_classify.clicked.connect(self.vm.open_camera_preview)
        self.home.btn_voice.clicked.connect(lambda: self.vm.state_changed.emit(AppStateType.VOICE_TO_TEXT))
        self.home.btn_trip.clicked.connect(self.vm.open_trip_load)
        self.home.btn_quit.clicked.connect(self._quit_application)

        self.camera_preview.btn_capture.clicked.connect(self.vm.trigger_capture)
        self.camera_preview.btn_cancel.clicked.connect(self.vm.cancel_camera)
        self.capture_review.btn_classify.clicked.connect(self.vm.confirm_captures_and_classify)
        self.capture_review.btn_retake.clicked.connect(self.vm.retake_captures)
        self.loading.btn_cancel.clicked.connect(self.vm.cancel_classification)

        self.classified.btn_reclassify.clicked.connect(self.vm.reclassify)
        self.classified.btn_save.clicked.connect(self.vm.save_classification)
        self.classified.btn_delete.clicked.connect(self.vm.delete_classification)

        self.voice.btn_start.clicked.connect(self.vm.start_voice_to_text)
        self.voice.btn_stop.clicked.connect(self._on_stop_or_edit_clicked)
        self.voice.btn_redo.clicked.connect(self.vm.redo_voice_to_text)
        self.voice.btn_save.clicked.connect(self.vm.save_transcription)
        self.voice.btn_delete.clicked.connect(self.vm.delete_transcription)
        self.voice.btn_reset.clicked.connect(self._on_reset_context_clicked)
        self.voice.btn_cancel.clicked.connect(self.vm.go_home)
        

        self.trip.btn_back.clicked.connect(self.vm.go_home)
        self.trip.btn_create_new_mission.clicked.connect(self._open_create_mission_page)
        self.trip.btn_delete_all.clicked.connect(self._on_delete_all_clicked)
        self.trip.list.itemClicked.connect(self._on_mission_clicked)

        self.mission_detail.btn_back.clicked.connect(self._show_trip_home)
        self.mission_detail.list.itemClicked.connect(self._on_timeline_clicked)
        self.mission_detail.btn_set_current.clicked.connect(self._on_set_current_mission_clicked)
        self.mission_detail.btn_cancel_assign.clicked.connect(self._stop_rock_assignment)
        self.mission_detail.btn_delete_mission.clicked.connect(self._on_delete_current_mission_clicked)

        self.mission_create.btn_create.clicked.connect(self._on_create_mission_clicked)
        self.mission_create.btn_cancel.clicked.connect(self._show_trip_home)

        self.rock_detail.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.mission_detail))
        self.rock_detail.btn_force_summary.clicked.connect(self._on_force_summary_clicked)
        self.rock_detail.btn_make_current.clicked.connect(self._on_make_current_rock_clicked)
        self.rock_detail.btn_delete_rock.clicked.connect(self._on_delete_current_rock_clicked)
        self.rock_detail.btn_tab_classification.clicked.connect(
            lambda: (self.rock_detail.panels.setCurrentIndex(0),
                     self.rock_detail._apply_tab_styles(0)))
        self.rock_detail.btn_tab_field_notes.clicked.connect(
            lambda: (self.rock_detail.panels.setCurrentIndex(1),
                     self.rock_detail._apply_tab_styles(1)))

        self.voice_note_detail.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.mission_detail))
        self.voice_note_detail.btn_add_to_classification.clicked.connect(self._on_add_to_classification_clicked)
        self.voice_note_detail.btn_delete_note.clicked.connect(self._on_delete_current_note_clicked)
        
    def _on_stop_or_edit_clicked(self) -> None:
        if self.voice.btn_stop.text() == "Stop":
            self.vm.stop_voice_to_text()
        else:
            self.inline_keyboard.show()
            self.voice.text.setReadOnly(False)
            self.voice.text.setFocus()
            self.voice.text.moveCursor(QTextCursor.End)
            
            # --- STARTUP CHECK ---
            text = self.voice.text.toPlainText()
            # If the text is empty, OR ends in punctuation, start with Shift ON!
            if not text or (len(text) >= 2 and text[-2:] in [". ", "? ", "! "]):
                self.inline_keyboard.set_shift_state(1)
            else:
                self.inline_keyboard.set_shift_state(0)

    def _wire_vm(self) -> None:
        self.vm.state_changed.connect(self._show_state)
        self.vm.classification_changed.connect(self._on_classification)
        self.vm.volume_display_changed.connect(self._on_volume_display)
        self.vm.transcription_changed.connect(self._on_transcription)
        self.vm.transcription_formatted.connect(self._on_transcription_formatted)
        self.vm.mission_name_transcription_changed.connect(self._on_mission_name_transcription)
        self.vm.rock_summary_changed.connect(self._on_rock_summary_changed)
        self.vm.trip_changed.connect(self._on_trip)
        self.vm.error.connect(self._on_error)
        self.vm.recording_status_changed.connect(self._on_recording_status_changed)
        self.vm.mission_name_recording_status_changed.connect(self._on_mission_name_recording_status_changed)
        self.vm.two_step_capture_message.connect(self.camera_preview.lbl_step.setText)
        self.vm.camera.frame_ready.connect(self._on_camera_frame)
        
        self.inline_keyboard = Keyboard()
        self.inline_keyboard.hide()
        
        self.inline_keyboard.btn_close.clicked.connect(self.inline_keyboard.hide)
        
        voice_layout = self.voice.layout()
        if voice_layout:
            voice_layout.insertWidget(voice_layout.count() - 1, self.inline_keyboard)
            
        self.inline_keyboard.hide()
        
        self.inline_keyboard.key_pressed.connect(self._on_virtual_key_pressed)

        self.mission_keyboard = Keyboard()
        self.mission_keyboard.hide()
        self.mission_keyboard.btn_close.clicked.connect(self._show_trip_home)
        mission_layout = self.mission_create.layout()
        if mission_layout:
            mission_layout.addWidget(self.mission_keyboard)
        self.mission_keyboard.key_pressed.connect(self._on_mission_key_pressed)
        self.mission_create.text.textChanged.connect(self._update_mission_create_button)
        self._update_mission_create_button()

    def _on_transcription_formatted(self):
        """Called when the LLM finishes formatting the text."""
        if self.stack.currentWidget() == self.voice:
            # Turn off the overlay and show the buttons!
            if not self.vm.transcription_text.strip():
                self._update_voice_buttons("initial")
            else:
                self._update_voice_buttons("review")
        
    def _on_camera_frame(self, image: QImage) -> None:
        if self.vm.state == AppStateType.CAMERA_PREVIEW:
            img = image.copy()
            w, h = img.width(), img.height()
            painter = QPainter(img)
            painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine))
            cx, cy = w // 2, h // 2
            size = min(w, h) // 15
            painter.drawLine(cx - size, cy, cx + size, cy)
            painter.drawLine(cx, cy - size, cx, cy + size)
            painter.end()
            pixmap = QPixmap.fromImage(img)
            scaled_pixmap = pixmap.scaled(
                self.camera_preview.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.camera_preview.video_label.setPixmap(scaled_pixmap)

    def _show_state(self, state: AppStateType) -> None:
        # 1. Switch the widget on the screen
        mapping = {
            AppStateType.HOME:                  self.home,
            AppStateType.CAMERA_PREVIEW:        self.camera_preview,
            AppStateType.CONFIRM_CAPTURES:      self.capture_review,
            AppStateType.CLASSIFYING:           self.loading,
            AppStateType.CLASSIFIED:            self.classified,
            AppStateType.VOICE_TO_TEXT_LOADING: self.voice_loading,
            AppStateType.VOICE_TO_TEXT:         self.voice,
            AppStateType.TRIP_LOAD:             self.trip,
        }
        if state in mapping:
            # Edge case: If we are in camera preview, don't switch screens just to show VTT
            if state == AppStateType.VOICE_TO_TEXT and self.stack.currentWidget() == self.camera_preview:
                pass
            else:
                self.stack.setCurrentWidget(mapping[state])

        if state == AppStateType.HOME:
            self.vm.stop_mission_name_recording(abort=True)
            self.mission_keyboard.hide()
            # FIX: Check vtt_active instead of the background process state!
            if not getattr(self.vm, 'vtt_active', False):
                self.voice.text.clear()
                self.vm.transcription_text = ""  # Wipes ghost text from memory!
                
                # Reset the little camera-preview voice button
                self.camera_preview.mic_ctrl.trigger_btn.setText("🎤")
                self.camera_preview.mic_ctrl.trigger_btn.setStyleSheet(
                    f"background-color: {_COLOR_TEXT}; color: white;"
                    f" border-radius: {_BTN_RADIUS}px; border: 1px solid {_COLOR_HOME_BTN};"
                    f" font-size: 18px; font-family: {_FONT_UI};"
                )

        elif state == AppStateType.CAMERA_PREVIEW:
            self.vm.stop_mission_name_recording(abort=True)
            self.mission_keyboard.hide()
            # From HEAD: Updated label text
            self.camera_preview.lbl_step.setText("Capture First View")
            self.vm.start_camera_stream(0, 0, 0, 0)

        elif state == AppStateType.CONFIRM_CAPTURES:
            self.vm.stop_mission_name_recording(abort=True)
            self.mission_keyboard.hide()
            top_path, side_path = self.vm.get_review_image_paths()
            self.capture_review.set_images(top_path, side_path)

        elif state == AppStateType.VOICE_TO_TEXT:
            self.vm.stop_mission_name_recording(abort=True)
            self.mission_keyboard.hide()
            self.inline_keyboard.hide()       
            self.inline_keyboard.set_shift_state(0)
            self.voice.text.setReadOnly(True)
            
            self._update_vtt_context_label()
            
            current_text = self.vm.transcription_text
            self.voice.text.setPlainText(current_text)
            self.voice.btn_save.setEnabled(bool(current_text.strip()))
            
            # --- NEW: Trigger Dynamic Layout ---
            if getattr(self.vm, 'vtt_active', False):
                self._update_voice_buttons("recording")
            elif current_text.strip():
                self._update_voice_buttons("review")
            else:
                self._update_voice_buttons("initial")

        elif state == AppStateType.TRIP_LOAD:
            self._show_trip_home()
            
    def _on_classification(self, result: ClassificationResult) -> None:
        has_side = result.side_image_path and os.path.exists(result.side_image_path)
        if has_side:
            self.classified.lbl_side_image.show()
            w, h = 280, 210
            if result.image_path and os.path.exists(result.image_path):
                pixmap = QPixmap(result.image_path)
                self.classified.lbl_image.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.classified.lbl_image.setText("No image")
            pixmap_side = QPixmap(result.side_image_path)
            self.classified.lbl_side_image.setPixmap(pixmap_side.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.classified.lbl_side_image.hide()
            self.classified.lbl_side_image.setPixmap(QPixmap())
            if result.image_path and os.path.exists(result.image_path):
                pixmap = QPixmap(result.image_path)
                self.classified.lbl_image.setPixmap(pixmap.scaled(560, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.classified.lbl_image.setText("No Image Available")

        self._original_classification = result
        self.classified.lbl_label.setText(result.label.upper())
        self.classified.lbl_conf.setText(f"Confidence: {int(result.confidence * 100)}%")

        # Tier badge
        tier_colors = {
            "high":      _COLOR_TIER_HIGH,
            "medium":    _COLOR_TIER_MEDIUM,
            "low":       _COLOR_TIER_LOW,
            "uncertain": _COLOR_TIER_UNK,
        }
        if result.tier and result.tier in tier_colors:
            bg = tier_colors[result.tier]
            self.classified.lbl_tier.setText(result.tier.upper())
            self.classified.lbl_tier.setStyleSheet(
                f"font-size: 13px; font-weight: 700; font-family: {_FONT_UI};"
                f" border-radius: {_BTN_RADIUS}px;"
                f" padding: 2px 8px; color: {_COLOR_PAGE}; background-color: {bg};"
            )
            self.classified.lbl_tier.setVisible(True)
        else:
            self.classified.lbl_tier.setVisible(False)

        # Weight label (volume updated separately via _on_volume_display)
        if result.estimated_weight is not None:
            self.classified.lbl_extra.setText(f"Wt: {result.estimated_weight}")
        else:
            self.classified.lbl_extra.setText("")

        # Build geology lookup keyed by feature name
        geology_notes = result.geology_notes or []
        geo_lookup = {}
        for nd in geology_notes:
            if isinstance(nd, dict):
                geo_lookup[nd.get("feature", "")] = nd.get("note", "")
            else:
                geo_lookup[getattr(nd, "feature", "")] = getattr(nd, "note", "")

        # Features + inline geology notes (rebuilt each classification)
        feat_layout = self.classified.features_layout
        while feat_layout.count():
            item = feat_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        ui_meta = (result.raw or {}).get("ui", {}) if isinstance(result.raw, dict) else {}
        show_features = (
            result.features is not None
            and result.tier not in (None, "uncertain")
            and ui_meta.get("show_features", True)
        )
        first_feat = True
        if show_features:
            for feat_name, feat_data in result.features.items():
                if not feat_data.get("display", True):
                    continue
                if feat_data.get("tier") not in ("high", "medium"):
                    continue
                if not first_feat:
                    sep = _make_divider()
                    sep.setStyleSheet(f"background-color: {_COLOR_FEAT_SEP}; max-height: 1px;")
                    feat_layout.addWidget(sep)
                first_feat = False
                conf_pct = int(feat_data.get("confidence", 0.0) * 100)
                display_name = feat_name.replace("_", " ").title()
                feat_row_w = QWidget()
                feat_row = QHBoxLayout(feat_row_w)
                feat_row.setContentsMargins(0, 8, 0, 4)
                feat_row.setSpacing(6)
                lbl_feat_name = QLabel(display_name)
                lbl_feat_name.setStyleSheet(f"font-size: 16px; color: {_COLOR_TEXT}; font-weight: 700; font-family: {_FONT_UI};")
                lbl_feat_val = QLabel(str(feat_data.get("value", "")))
                lbl_feat_val.setStyleSheet(f"font-size: 16px; color: {_COLOR_TEXT}; font-family: {_FONT_DATA};")
                lbl_feat_conf = QLabel(f"conf. {conf_pct}%")
                lbl_feat_conf.setStyleSheet(f"font-size: 13px; color: {_COLOR_MUTED}; font-family: {_FONT_DATA};")
                feat_row.addWidget(lbl_feat_name)
                feat_row.addStretch()
                feat_row.addWidget(lbl_feat_val)
                feat_row.addWidget(lbl_feat_conf)
                feat_layout.addWidget(feat_row_w)
                note_text = geo_lookup.get(feat_name, "")
                if note_text:
                    lbl_note = QLabel(note_text)
                    lbl_note.setWordWrap(True)
                    lbl_note.setStyleSheet(
                        f"font-size: 13px; color: {_COLOR_MUTED}; font-family: {_FONT_UI}; padding: 0px 0px 4px 0px;"
                    )
                    lbl_note.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                    feat_layout.addWidget(lbl_note)

        # Alternative classification chips (selectable override)
        top3 = None
        if result.raw and isinstance(result.raw, dict):
            top3 = result.raw.get("top3", [])
        while self.classified.alt_buttons_layout.count():
            item = self.classified.alt_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        alt_chips = []
        alts = []
        if top3:
            for entry in top3[1:3]:
                conf = float(entry.get("confidence", 0.0))
                if conf > 0:
                    alts.append((entry.get("label", ""), conf))
        for lbl_text, conf in alts:
            chip = QPushButton(f"{lbl_text.upper()}  {int(conf * 100)}%")
            chip.setCheckable(True)
            chip.setFocusPolicy(Qt.StrongFocus)
            chip.setStyleSheet(f"""
                QPushButton {{
                    background-color: {_COLOR_ALT_CHIP_BG}; color: {_COLOR_TEXT};
                    border: 1px solid {_COLOR_MUTED}; border-radius: {_BTN_RADIUS}px;
                    font-size: 13px; font-family: {_FONT_UI}; font-weight: 700; padding: 3px 10px;
                }}
                QPushButton:checked {{
                    background-color: {_COLOR_TIER_HIGH}; color: {_COLOR_PAGE};
                    border-color: {_COLOR_TIER_HIGH};
                }}
                QPushButton:focus {{ border: 2px solid {_COLOR_FOCUS}; }}
            """)
            alt_chips.append(chip)
            self.classified.alt_buttons_layout.addWidget(chip)
        for chip, (lbl_text, conf) in zip(alt_chips, alts):
            chip.clicked.connect(
                lambda checked, l=lbl_text, c=conf, b=chip, all_b=list(alt_chips):
                    self._on_alt_override(l, c, b, all_b, checked)
            )
        has_alts = len(alt_chips) > 0
        self.classified.div_alts.setVisible(has_alts)
        self.classified.alt_row.setVisible(has_alts)

    def _on_alt_override(self, label: str, confidence: float, clicked_btn, all_buttons, checked: bool) -> None:
        from dataclasses import replace as _dc_replace
        if not checked:
            for b in all_buttons:
                b.setChecked(False)
            orig = getattr(self, "_original_classification", None)
            if orig is not None:
                if self.vm.current_classification is not None:
                    self.vm.current_classification = _dc_replace(
                        self.vm.current_classification, label=orig.label, confidence=orig.confidence
                    )
                self.classified.lbl_label.setText(orig.label.upper())
                self.classified.lbl_conf.setText(f"Confidence: {int(orig.confidence * 100)}%")
        else:
            for b in all_buttons:
                b.setChecked(b is clicked_btn)
            if self.vm.current_classification is not None:
                self.vm.current_classification = _dc_replace(
                    self.vm.current_classification, label=label, confidence=confidence
                )
            self.classified.lbl_label.setText(label.upper())
            self.classified.lbl_conf.setText(f"Confidence: {int(confidence * 100)}%")

    def _on_volume_display(self, text: str) -> None:
        # Reformat "Volume = X cm³" / "Volume = N/A" → "Vol: X cm³" / "Vol: N/A"
        display = text.replace("Volume = ", "Vol: ").replace("Volume=", "Vol:")
        self.classified.lbl_volume.setText(display)

    def _on_transcription(self, text: str) -> None:
        if "No audio recorded" in text or "STREAMING COMPLETE" in text:
            return
        if not text:
            self.voice.text.clear()
            self.voice.btn_save.setEnabled(False)
            return
        self.voice.text.setPlainText(text)
        self.voice.btn_save.setEnabled(True)
        if self.stack.currentWidget() == self.voice:
            cursor = self.voice.text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.voice.text.setTextCursor(cursor)
        cursor = self.voice.text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.voice.text.setTextCursor(cursor)

        self.voice.btn_save.setEnabled(bool(text.strip()))

    def _on_mission_name_transcription(self, text: str) -> None:
        if self._mission_name_typing_mode:
            return
        if "No audio recorded" in text or "STREAMING COMPLETE" in text:
            return
        self.mission_create.text.setPlainText(text)
        cursor = self.mission_create.text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.mission_create.text.setTextCursor(cursor)
        self._update_mission_create_button()

    def _on_rock_summary_changed(self, rock_id: str, summary: str) -> None:
        self.rock_detail.set_ai_summary(rock_id, summary)

    def _format_trip_totals(self, total_volume: float, total_weight: float) -> str:
        total_vol = f"{total_volume:.2f}" if total_volume else "0.00"
        total_wt = f"{total_weight:.2f}" if total_weight else "0.00"
        return f"Total volume: {total_vol} cm³   Total weight: {total_wt} kg"

    def _find_mission_summary(self, mission_id: str) -> MissionSummary | None:
        summary = getattr(self.trip, "_summary", None)
        if not summary:
            return None
        for mission_summary in summary.missions:
            if mission_summary.mission.mission_id == mission_id:
                return mission_summary
        return None

    def _show_trip_home(self) -> None:
        self._reset_delete_mission_arm()
        self.vm.stop_mission_name_recording(abort=True)
        self.mission_keyboard.hide()
        self.stack.setCurrentWidget(self.trip)

    def _open_create_mission_page(self) -> None:
        self._stop_rock_assignment()
        self.vm.stop_mission_name_recording(abort=True)
        self._mission_name_typing_mode = False
        self.mission_create.text.clear()
        self.mission_create.lbl_recording_status.setText("Recording... speak the mission name")
        self.mission_create.text.setFocus()
        self.mission_keyboard.show()
        self.mission_keyboard.set_shift_state(1)
        self._update_mission_create_button()
        self.stack.setCurrentWidget(self.mission_create)
        self.vm.start_mission_name_recording()

    def _open_mission_detail(self, mission_summary: MissionSummary) -> None:
        self._reset_delete_mission_arm()
        self._selected_mission_id = mission_summary.mission.mission_id
        self._render_mission_detail(mission_summary)
        self.stack.setCurrentWidget(self.mission_detail)

    def _build_mission_timeline_items(self, mission_summary: MissionSummary) -> list[dict]:
        rocks_sorted = sorted(mission_summary.rocks, key=lambda r: r.ts)
        combined = [{"type": "rock", "ts": r.ts, "data": r} for r in mission_summary.rocks]

        for note in mission_summary.voice_notes:
            note_ts = note.get("ts", 0)
            explicit_rock_id = note.get("rock_id")

            owning_rock = None
            if explicit_rock_id:
                for rock in mission_summary.rocks:
                    if rock.rock_id == explicit_rock_id:
                        owning_rock = rock
                        break
            else:
                for rock in reversed(rocks_sorted):
                    if rock.ts <= note_ts:
                        owning_rock = rock
                        break

            if not owning_rock:
                combined.append({"type": "voice", "ts": note_ts, "data": note})

        combined.sort(key=lambda item: item["ts"], reverse=True)
        return combined

    def _render_mission_detail(self, mission_summary: MissionSummary) -> None:
        from PySide6.QtWidgets import QListWidgetItem

        self.mission_detail._summary = mission_summary
        self.mission_detail._timeline_data = []
        self.mission_detail.list.clear()
        self.mission_detail.lbl_title.setText(mission_summary.mission.name)
        if mission_summary.mission.mission_id == getattr(self.vm, "active_mission_id", None):
            self.mission_detail.lbl_status.setText("Current mission")
        else:
            self.mission_detail.lbl_status.setText("")
        self.mission_detail.lbl_totals.setText(
            self._format_trip_totals(mission_summary.total_volume, mission_summary.total_weight)
        )

        combined = self._build_mission_timeline_items(mission_summary)

        for item in combined:
            ts = item["ts"]
            dt = datetime.datetime.fromtimestamp(ts) if ts else None
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "Unknown"

            if item["type"] == "rock":
                rock = item["data"]
                label = rock.result.label
                conf = int(rock.result.confidence * 100)
                display_text = f"🪨 <span style='font-size: 13px; color: #555;'>[{time_str}]</span>&nbsp;&nbsp;<b>{label.upper()}</b> ({conf}%)"
            else:
                note = item["data"]
                cleaned = note.get("cleaned", note.get("transcript", ""))
                short_text = cleaned[:60] + "..." if len(cleaned) > 60 else cleaned
                display_text = f"🎤 <span style='font-size: 13px; color: #555;'>[{time_str}]</span>&nbsp;&nbsp;{short_text}"

            list_item = QListWidgetItem()
            self.mission_detail.list.addItem(list_item)

            widget = _make_list_row_widget(display_text)
            list_item.setSizeHint(QSize(0, max(68, widget.sizeHint().height())))
            self.mission_detail.list.setItemWidget(list_item, widget)
            self.mission_detail._timeline_data.append(item)

    def _on_trip(self, summary: TripSummary) -> None:
        self._stop_rock_assignment()
        self.trip.list.clear()
        self.trip._summary = summary
        self.trip._missions_data = summary.missions

        from PySide6.QtWidgets import QListWidgetItem

        current_mission_name = "--"
        for mission_summary in summary.missions:
            if mission_summary.mission.mission_id == summary.current_mission_id:
                current_mission_name = mission_summary.mission.name
                break
        self.trip.lbl_current_mission.setText(f"Current mission: {current_mission_name}")

        for mission_summary in summary.missions:
            dt = datetime.datetime.fromtimestamp(mission_summary.mission.updated_ts)
            time_str = dt.strftime("%Y-%m-%d %H:%M")
            item_count = len(self._build_mission_timeline_items(mission_summary))
            badge = "  • CURRENT" if mission_summary.mission.mission_id == summary.current_mission_id else ""
            display_text = (
                f"<b>{mission_summary.mission.name}</b>{badge}<br>"
                f"<span style='font-size: 13px; color: {_COLOR_MUTED};'>Updated {time_str}   Items: {item_count}</span>"
            )

            list_item = QListWidgetItem()
            self.trip.list.addItem(list_item)

            widget = _make_list_row_widget(display_text, min_height=78)
            list_item.setSizeHint(QSize(0, max(76, widget.sizeHint().height())))
            self.trip.list.setItemWidget(list_item, widget)
        
        if self._selected_mission_id:
            selected_summary = self._find_mission_summary(self._selected_mission_id)
            if selected_summary:
                self._render_mission_detail(selected_summary)
            elif self.stack.currentWidget() in {self.mission_detail, self.rock_detail, self.voice_note_detail}:
                self._selected_mission_id = None
                self._show_trip_home()

    def _on_set_current_mission_clicked(self) -> None:
        summary = self.mission_detail._summary
        if summary:
            self.vm.make_mission_current(summary.mission.mission_id)

    def _reset_delete_mission_arm(self) -> None:
        self._pending_delete_mission_id = None
        self._pending_delete_mission_armed_at = 0.0
        if hasattr(self, "mission_detail"):
            self.mission_detail.btn_delete_mission.setText("Delete Mission")
            self.mission_detail.btn_delete_mission.setStyleSheet(_SS_DELETE)
            summary = getattr(self.mission_detail, "_summary", None)
            if summary:
                is_current = summary.mission.mission_id == getattr(self.vm, "active_mission_id", None)
                self.mission_detail.lbl_status.setText("Current mission" if is_current else "")

    def _on_delete_current_mission_clicked(self) -> None:
        summary = self.mission_detail._summary
        if summary:
            self._delete_mission(summary.mission.mission_id, summary.mission.name)

    def _on_make_current_rock_clicked(self) -> None:
        entry = getattr(self.rock_detail, "_current_rock_entry", None)
        if entry:
            self.vm.make_rock_current(entry)

    def _on_delete_current_rock_clicked(self) -> None:
        entry = getattr(self.rock_detail, "_current_rock_entry", None)
        if entry:
            self._delete_timeline_item({"type": "rock", "data": entry})
            self.stack.setCurrentWidget(self.mission_detail)

    def _on_add_to_classification_clicked(self) -> None:
        note = getattr(self.voice_note_detail, "_current_note", None)
        if note:
            self._start_rock_assignment(note.get("ts"))
            self.stack.setCurrentWidget(self.mission_detail)

    def _on_delete_current_note_clicked(self) -> None:
        note = getattr(self.voice_note_detail, "_current_note", None)
        if note:
            self._delete_timeline_item({"type": "voice", "data": note})
            self.stack.setCurrentWidget(self.mission_detail)

    def _delete_mission(self, mission_id: str, mission_name: str) -> None:
        """Arm-confirm mission deletion without a modal popup.

        First activation arms the button. A second activation on the same
        button confirms. This supports joystick, mouse, Enter, and Space
        without requiring a fast double-click timing gesture.
        """
        now = time.monotonic()
        if self._pending_delete_mission_id == mission_id:
            # Ignore switch bounce or an accidental repeated signal from one press.
            if now - self._pending_delete_mission_armed_at < 0.25:
                return
            self._delete_mission_timer.stop()
            self._pending_delete_mission_id = None
            self._pending_delete_mission_armed_at = 0.0
            self.mission_detail.btn_delete_mission.setText("Delete Mission")
            self.mission_detail.btn_delete_mission.setStyleSheet(_SS_DELETE)
            if mission_id == self._selected_mission_id:
                self._selected_mission_id = None
            self.vm.delete_mission(mission_id)
            return

        self._pending_delete_mission_id = mission_id
        self._pending_delete_mission_armed_at = now
        self.mission_detail.btn_delete_mission.setText("Press Again to Delete")
        self.mission_detail.btn_delete_mission.setStyleSheet(_SS_ARMED_DELETE)
        self.mission_detail.lbl_status.setText(f'Delete armed for "{mission_name}". Press again to confirm.')
        self._delete_mission_timer.start(4500)
    def _delete_timeline_item(self, item_dict):
        if item_dict["type"] == "voice":
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Delete Voice Note")
            msg_box.setText("Are you sure? Clicking CONFIRM will delete this data permanently.")
            msg_box.setStyleSheet(f"QLabel {{ color: {_COLOR_TEXT}; font-size: 15px; font-weight: normal; font-family: {_FONT_UI}; }} QMessageBox {{ background-color: {_COLOR_BG}; }}")

            btn_cancel = msg_box.addButton("CANCEL", QMessageBox.RejectRole)
            btn_cancel.setStyleSheet(
                f"background-color: {_COLOR_SECONDARY}; color: {_COLOR_TEXT}; font-weight: 700;"
                f" font-family: {_FONT_UI}; padding: 8px; border-radius: {_BTN_RADIUS}px;"
                f" border: 1px solid {_COLOR_BORDER};"
            )

            btn_confirm = msg_box.addButton("CONFIRM", QMessageBox.AcceptRole)
            btn_confirm.setStyleSheet(
                f"background-color: {_COLOR_DANGER}; color: white; font-weight: 700;"
                f" font-family: {_FONT_UI}; padding: 8px; border-radius: {_BTN_RADIUS}px; border: none;"
            )
            
            msg_box.exec()
            
            if msg_box.clickedButton() == btn_confirm:
                self.vm.delete_voice_note_by_ts(item_dict["data"].get("ts"))
                
        elif item_dict["type"] == "rock":
            dialog = QDialog(self)
            dialog.setWindowTitle("Delete Classification")
            dialog.setStyleSheet(f"QDialog {{ background-color: {_COLOR_BG}; font-family: {_FONT_UI}; }}")
            layout = QVBoxLayout(dialog)

            msg = QLabel("Are you sure? How would you like to delete this rock?")
            msg.setStyleSheet(f"color: {_COLOR_TEXT}; font-size: 15px; font-weight: normal; font-family: {_FONT_UI};")
            msg.setWordWrap(True)
            layout.addWidget(msg)
            layout.addSpacing(10)

            btn_both = QPushButton("DELETE ROCK && ALL NOTES")
            btn_both.setStyleSheet(
                f"background-color: {_COLOR_STOP}; color: white; font-weight: 700;"
                f" font-family: {_FONT_UI}; padding: 12px; font-size: 14px;"
                f" border-radius: {_BTN_RADIUS}px; border: none;"
            )

            btn_only = QPushButton("DELETE ROCK ONLY")
            btn_only.setStyleSheet(
                f"background-color: {_COLOR_DANGER}; color: white; font-weight: 700;"
                f" font-family: {_FONT_UI}; padding: 12px; font-size: 14px;"
                f" border-radius: {_BTN_RADIUS}px; border: none;"
            )

            btn_cancel = QPushButton("CANCEL")
            btn_cancel.setStyleSheet(
                f"background-color: {_COLOR_SECONDARY}; color: {_COLOR_TEXT}; font-weight: 700;"
                f" font-family: {_FONT_UI}; padding: 12px; font-size: 14px;"
                f" border-radius: {_BTN_RADIUS}px; border: 1px solid {_COLOR_BORDER};"
            )
            
            layout.addWidget(btn_both)
            layout.addWidget(btn_only)
            layout.addWidget(btn_cancel)
            
            # Helper function to capture the choice and close the dialog
            choice = [None]
            def on_choice(c):
                choice[0] = c
                dialog.accept()
                
            btn_both.clicked.connect(lambda: on_choice("both"))
            btn_only.clicked.connect(lambda: on_choice("only"))
            btn_cancel.clicked.connect(dialog.reject)
            
            dialog.exec()
            
            if choice[0] == "only":
                self.vm.delete_rock_by_id(item_dict["data"].rock_id)
                
            elif choice[0] == "both":
                entry = item_dict["data"]
                associated_ts = []
                mission_summary = self.mission_detail._summary
                rocks_sorted = sorted(mission_summary.rocks, key=lambda r: r.ts)
                
                next_rock_ts = float('inf')
                for r in rocks_sorted:
                    if r.ts > entry.ts:
                        next_rock_ts = r.ts
                        break
                        
                for n in mission_summary.voice_notes:
                    note_ts = n.get("ts", 0)
                    explicit_rock_id = n.get("rock_id")
                    
                    if explicit_rock_id == entry.rock_id:
                        associated_ts.append(note_ts)
                    elif explicit_rock_id is None:
                        if entry.ts <= note_ts < next_rock_ts:
                            if n.get("session_id") == entry.session_id:
                                associated_ts.append(note_ts)
                
                for ts in associated_ts:
                    self.vm.store.delete_voice_note(ts)
                    
                self.vm.delete_rock_by_id(entry.rock_id)

    def _on_mission_clicked(self, item) -> None:
        index = self.trip.list.row(item)
        if 0 <= index < len(self.trip._missions_data):
            self._open_mission_detail(self.trip._missions_data[index])

    def _on_timeline_clicked(self, item) -> None:
        index = self.mission_detail.list.row(item)
        if 0 <= index < len(self.mission_detail._timeline_data):
            item_dict = self.mission_detail._timeline_data[index]
            
            if getattr(self, "_assigning_note_ts", None) is not None:
                if item_dict["type"] == "rock":
                    self.vm.assign_note_to_rock(self._assigning_note_ts, item_dict["data"].rock_id)
                self._stop_rock_assignment()
                return
            
            if item_dict["type"] == "rock":
                entry = item_dict["data"]
                associated_notes = []
                mission_summary = self.mission_detail._summary
                rocks_sorted = sorted(mission_summary.rocks, key=lambda r: r.ts)
                
                next_rock_ts = float('inf')
                for r in rocks_sorted:
                    if r.ts > entry.ts:
                        next_rock_ts = r.ts
                        break
                        
                for n in mission_summary.voice_notes:
                    note_ts = n.get("ts", 0)
                    explicit_rock_id = n.get("rock_id")
                    
                    if explicit_rock_id == entry.rock_id:
                        associated_notes.append(n)
                    elif explicit_rock_id is None:
                        if entry.ts <= note_ts < next_rock_ts:
                            if n.get("session_id") == entry.session_id:
                                associated_notes.append(n)
                
                associated_notes.sort(key=lambda x: x.get("ts", 0))

                initial_summary = "Generating AI summary..." if associated_notes else "No associated recordings to summarize yet."
                self.rock_detail.set_entry(entry, associated_notes, ai_summary=initial_summary)
                self.vm.request_rock_summary(entry, associated_notes)
                self.stack.setCurrentWidget(self.rock_detail)
                
            else:
                note = item_dict["data"]
                self.voice_note_detail.set_note(note)
                self.stack.setCurrentWidget(self.voice_note_detail)

    def _on_create_mission_clicked(self) -> None:
        self.vm.stop_mission_name_recording(abort=True)
        name = " ".join(self.mission_create.text.toPlainText().strip().split())
        if not name:
            return
        try:
            mission = self.vm.create_mission(name)
        except ValueError as exc:
            QMessageBox.warning(self, "Name Taken", str(exc))
            return
        self.mission_create.text.clear()
        self.mission_keyboard.hide()
        self._mission_name_typing_mode = False
        mission_summary = self._find_mission_summary(mission.mission_id)
        if mission_summary:
            self._open_mission_detail(mission_summary)
        else:
            self._show_trip_home()

    def _on_error(self, message: str) -> None:
        QMessageBox.warning(self, "Error", "Something went wrong. Please press escape to return to home screen.")
    
    def _on_recording_status_changed(self, is_recording: bool):
        self.camera_preview.mic_ctrl._update_ui_state(is_recording)
        
        if is_recording:
            self.voice.btn_stop.setText("Stop")
            self.inline_keyboard.hide()
            self.voice.text.setReadOnly(True)
            
            # If we are actually looking at the voice page, update the layout
            if self.stack.currentWidget() == self.voice:
                self._update_voice_buttons("recording")
        else:
            self.voice.btn_stop.setText("Edit")
            self.voice.btn_save.setEnabled(False)
            
            # If we are actually looking at the voice page, update the layout
            if self.stack.currentWidget() == self.voice:
                self._update_voice_buttons("formatting")

    def _on_mission_name_recording_status_changed(self, is_recording: bool):
        if is_recording:
            self.mission_create.lbl_recording_status.setText("Recording... speak the mission name")
        elif self._mission_name_typing_mode:
            self.mission_create.lbl_recording_status.setText("Typing mode")
        else:
            self.mission_create.lbl_recording_status.setText("Recording stopped")

    def _update_status(self):
        """Fetches the live system time, locks to West Coast, and gets battery percentage."""
        import datetime
        
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("America/Los_Angeles")
        except Exception:
            tz = datetime.timezone(datetime.timedelta(hours=-7))
            
        now = datetime.datetime.now(tz)
        
        date_str = now.strftime("%A, %b %d, %Y").replace(" 0", " ")
        time_str = now.strftime("%I:%M %p").lstrip("0")
        
        self.lbl_time.setText(time_str)
        self.lbl_date.setText(date_str)
        
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery:
                percent = int(battery.percent)
                is_plugged = battery.power_plugged
                icon = "⚡" if is_plugged else "🔋"
                self.lbl_battery.setText(f"{percent}% {icon}")
            else:
                self.lbl_battery.setText("Power Connected") 
        except Exception as e:
            self.lbl_battery.setText("Battery N/A")
            

class Keyboard(QDialog):
    key_pressed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(4)
        
        # 0 = lower, 1 = Single shift, 2 = CAPS LOCK
        self.shift_state = 0
        self.letter_btns = []
        
        rows = [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "⌫"],
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", "'"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "!", "?"]
        ]
        
        for row in rows:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(4)
            for key in row:
                btn = QPushButton(key)
                btn.setMinimumHeight(45)
                btn.setMinimumWidth(20)
                btn.setStyleSheet(f"font-size: 14px; font-weight: 700; font-family: {_FONT_UI}; background-color: {_COLOR_SURFACE}; color: {_COLOR_TEXT}; border: 1px solid {_COLOR_BORDER}; border-radius: {_BTN_RADIUS}px;")
                
                if key.isalpha() and len(key) == 1:
                    self.letter_btns.append(btn)
                    btn.clicked.connect(lambda checked=False, b=btn: self._on_letter_clicked(b.text()))
                else:
                    btn.clicked.connect(lambda checked=False, char=key: self.key_pressed.emit(char))
                
                row_layout.addWidget(btn)
            main_layout.addLayout(row_layout)
            
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(4)
        
        self.btn_shift = QPushButton("⇧")
        self.btn_shift.setMinimumHeight(45)
        self.btn_shift.setMinimumWidth(40)
        self.btn_shift.setStyleSheet(f"font-size: 16px; font-weight: 700; font-family: {_FONT_UI}; background-color: {_COLOR_SURFACE_ALT}; color: {_COLOR_TEXT}; border: 1px solid {_COLOR_BORDER}; border-radius: {_BTN_RADIUS}px;")
        self.btn_shift.clicked.connect(self._on_shift_clicked)
        bottom_layout.addWidget(self.btn_shift, stretch=1)
        
        self.shift_timer = QTimer(self)
        self.shift_timer.setSingleShot(True)
        
        self.btn_space = QPushButton("Space")
        self.btn_space.setMinimumHeight(45)
        self.btn_space.setStyleSheet(f"font-size: 14px; font-weight: 700; font-family: {_FONT_UI}; background-color: {_COLOR_SURFACE}; color: {_COLOR_TEXT}; border: 1px solid {_COLOR_BORDER}; border-radius: {_BTN_RADIUS}px;")
        self.btn_space.clicked.connect(lambda checked=False: self.key_pressed.emit("Space"))
        bottom_layout.addWidget(self.btn_space, stretch=3)
        
        # Arrow Keys
        for arrow in ["←", "↑", "↓", "→"]:
            btn = QPushButton(arrow)
            btn.setMinimumHeight(45)
            btn.setMinimumWidth(20)
            btn.setStyleSheet(f"font-size: 15px; font-weight: 700; font-family: {_FONT_UI}; background-color: {_COLOR_SURFACE_ALT}; color: {_COLOR_TEXT}; border: 1px solid {_COLOR_BORDER}; border-radius: {_BTN_RADIUS}px;")
            btn.clicked.connect(lambda checked=False, char=arrow: self.key_pressed.emit(char))
            bottom_layout.addWidget(btn)
            
        # The Close Button (keeping the name btn_close so our hiding logic still works!)
        self.btn_close = QPushButton("Close")
        self.btn_close.setMinimumHeight(45)
        self.btn_close.setStyleSheet(f"font-size: 14px; font-weight: 700; font-family: {_FONT_UI}; background-color: {_COLOR_DANGER}; color: white; border-radius: {_BTN_RADIUS}px;")
        bottom_layout.addWidget(self.btn_close, stretch=1)
        
        main_layout.addLayout(bottom_layout)
        
        self._update_keyboard_case()
        
    def _on_letter_clicked(self, char):
        self.key_pressed.emit(char)
        # iPhone logic: After typing a single shifted letter, instantly revert to lowercase
        if self.shift_state == 1:
            self.shift_state = 0
            self._update_keyboard_case()

    def _on_shift_clicked(self):
        if self.shift_timer.isActive():
            # Second click happened within 300ms! Activate Caps Lock!
            self.shift_timer.stop()
            self.shift_state = 2
        else:
            self.shift_timer.start(300) # Start the 300ms window waiting for a double click
            # Toggle normal shift
            if self.shift_state == 0:
                self.shift_state = 1
            elif self.shift_state == 1:
                self.shift_state = 0
            elif self.shift_state == 2:
                self.shift_state = 0
                
        self._update_keyboard_case()

    def set_shift_state(self, state):
        # Don't auto-override the user if they specifically turned on Caps Lock
        if self.shift_state != 2:
            self.shift_state = state
            self._update_keyboard_case()

    def _update_keyboard_case(self):
        is_upper = (self.shift_state > 0)
        
        # Flip all the letters A-Z
        for btn in self.letter_btns:
            text = btn.text()
            btn.setText(text.upper() if is_upper else text.lower())
            
        # Update Shift Button icon & colors
        if self.shift_state == 0:
            self.btn_shift.setText("⇧")
            self.btn_shift.setStyleSheet(f"font-size: 16px; font-weight: 700; font-family: {_FONT_UI}; background-color: {_COLOR_SURFACE_ALT}; color: {_COLOR_TEXT}; border: 1px solid {_COLOR_BORDER}; border-radius: {_BTN_RADIUS}px;")
        elif self.shift_state == 1:
            self.btn_shift.setText("⬆")
            self.btn_shift.setStyleSheet(f"font-size: 16px; font-weight: 700; font-family: {_FONT_UI}; background-color: {_COLOR_SURFACE}; color: {_COLOR_TEXT}; border: 2px solid {_COLOR_FOCUS}; border-radius: {_BTN_RADIUS}px;")
        elif self.shift_state == 2:
            self.btn_shift.setText("⇪")
            self.btn_shift.setStyleSheet(f"font-size: 16px; font-weight: 700; font-family: {_FONT_UI}; background-color: {_COLOR_PRIMARY}; color: white; border-radius: {_BTN_RADIUS}px;")

    
                
