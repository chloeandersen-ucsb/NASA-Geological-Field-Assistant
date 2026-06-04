from __future__ import annotations
import html
import os
import sys
import datetime
from pathlib import Path
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QFont
from PySide6.QtGui import QImage
import math

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# img_path = project_root/ "led-display" / "ui" / "sage-logo-wcbg.png"
img_path = project_root/ "led-display" / "ui" / "newlogo.png"


from PySide6.QtCore import Qt, QTimer, Signal, QPoint
from PySide6.QtGui import QTextCursor, QKeyEvent, QShortcut, QKeySequence, QPainter, QPen, QColor, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMainWindow, QStackedWidget,
    QTextEdit, QListWidget, QHBoxLayout, QSizePolicy, QGridLayout, QDialog,
    QFrame, QProgressBar
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
    popup_shown = Signal()
    popup_hidden = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False) # Blocks clicks to the background!
        
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)
        
        # The white/light popup box
        self.box = QWidget()
        self.box.setFixedSize(400, 300)
        self.box.setStyleSheet("background-color: #cbd2c5; border: 3px solid #344f41; border-radius: 12px;")        
        box_layout = QVBoxLayout(self.box)
        
        self.lbl_title = QLabel("Category")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #344f41; border: none; padding-bottom: 5px;")
        box_layout.addWidget(self.lbl_title)
        
        # A nice divider line
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background-color: #697d6a; border: none;")
        divider.setFixedHeight(2)
        box_layout.addWidget(divider)
        
        self.lbl_content = QLabel("Content goes here...")
        self.lbl_content.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_content.setWordWrap(True)
        self.lbl_content.setStyleSheet("font-size: 18px; color: #344f41; border: none; margin-top: 5px;")
        box_layout.addWidget(self.lbl_content, stretch=1)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(36)
        self.btn_cancel.setStyleSheet("""
            QPushButton { background-color: #697d6a; color: #f5f6f4; font-size: 16px; font-weight: bold; border-radius: 6px; border: none; }
            QPushButton:hover { background-color: #344f41; }
        """)
        self.btn_cancel.clicked.connect(self.hide)
        box_layout.addWidget(self.btn_cancel)
        
        self.layout.addWidget(self.box)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 180)) # 70% opacity black background
        
    def mousePressEvent(self, event):
        self.hide()
        
    def hideEvent(self, event):
        self.popup_hidden.emit()
        super().hideEvent(event)

    def show_popup(self, title: str, content: str):
        self.lbl_title.setText(title)
        self.lbl_content.setText(content)
        self.raise_()
        self.show()
        self.popup_shown.emit()

class BatteryInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Frameless dialog that sits on top
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background-color: #cbd2c5; color: #344f41; border: 3px solid #344f41; border-radius: 12px;")
        self.setFixedSize(320, 480)

        layout = QVBoxLayout(self)
        
        lbl_title = QLabel("Set Battery Percentage")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; border: none; padding-top: 10px;")
        layout.addWidget(lbl_title)
        
        self.lbl_display = QLabel("")
        self.lbl_display.setAlignment(Qt.AlignCenter)
        self.lbl_display.setStyleSheet("font-size: 36px; font-weight: bold; background-color: #f5f6f4; border: 2px solid #697d6a; border-radius: 8px; padding: 10px; margin: 10px;")
        layout.addWidget(self.lbl_display)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setContentsMargins(15, 0, 15, 10)
        
        self.entered_value = ""
        
        # 3x4 Grid for number pad
        buttons = [
            ('1', 0, 0), ('2', 0, 1), ('3', 0, 2),
            ('4', 1, 0), ('5', 1, 1), ('6', 1, 2),
            ('7', 2, 0), ('8', 2, 1), ('9', 2, 2),
            ('⌫', 3, 0), ('0', 3, 1), ('✓', 3, 2)
        ]
        
        for text, row, col in buttons:
            btn = QPushButton(text)
            btn.setFixedSize(75, 65)
            if text == '✓':
                btn.setStyleSheet("font-size: 24px; font-weight: bold; background-color: #617c32; color: white; border-radius: 8px; border: none;")
            elif text == '⌫':
                btn.setStyleSheet("font-size: 24px; font-weight: bold; background-color: #95b7dc; color: #385573; border-radius: 8px; border: none;")
            else:
                btn.setStyleSheet("font-size: 24px; font-weight: bold; background-color: #f5f6f4; border: 2px solid #697d6a; border-radius: 8px;")
            
            btn.clicked.connect(lambda checked=False, t=text: self._on_btn_clicked(t))
            grid.addWidget(btn, row, col)
            
        layout.addLayout(grid)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedHeight(45)
        btn_cancel.setStyleSheet("font-size: 18px; font-weight: bold; background-color: #344f41; color: white; border-radius: 8px; margin: 0px 15px 15px 15px;")
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(btn_cancel)

    def _on_btn_clicked(self, text):
        if text == '⌫':
            self.entered_value = self.entered_value[:-1]
        elif text == '✓':
            if self.entered_value:
                val = int(self.entered_value)
                # Validation rule: 1 to 100
                if 1 <= val <= 100:
                    self.accept()
                else:
                    self.lbl_display.setText("Invalid\n(1-100)")
                    self.lbl_display.setStyleSheet("font-size: 20px; font-weight: bold; color: #cc0000; background-color: #f5f6f4; border: 2px solid #cc0000; border-radius: 8px; padding: 10px; margin: 10px;")
                    self.entered_value = ""
                    return
            else:
                return
        else:
            # Maximum 3 digits
            if len(self.entered_value) < 3:
                self.entered_value += text
                self.lbl_display.setStyleSheet("font-size: 36px; font-weight: bold; background-color: #f5f6f4; border: 2px solid #697d6a; border-radius: 8px; padding: 10px; margin: 10px;")
        
        if text != '✓':
            self.lbl_display.setText(self.entered_value + "%" if self.entered_value else "")

    def get_percentage(self):
        return int(self.entered_value) if self.entered_value else -1

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
        pen = QPen(QColor("#a88b5c")) # Matches your accent color
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

def dark_green_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(50)
    b.setStyleSheet("font-size: 22px; background-color: #344f41; color: #cad2c5;")
    return b

def light_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(50)
    b.setStyleSheet("font-size: 22px; background-color: #cad2c5; color: #344f41;")
    return b

def blue_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(50)
    b.setStyleSheet("font-size: 20px; background-color: #95b7dc; color: #385573;")
    return b

def green_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(50)
    b.setStyleSheet("font-size: 20px; background-color: #617c32; color: #f5f6f4;")
    return b

def homepage_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(70)
    b.setStyleSheet("font-size: 20px; background-color: #344f41; color: #cad2c5;")
    return b

def homepage_quit_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(50)
    b.setStyleSheet("font-size: 20px; background-color: #344f41; color: #cad2c5;")
    return b

def light_green_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(35)
    b.setStyleSheet("font-size: 14px; background-color: #344f41; color: #cad2c5;")
    return b


class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # --- NEW: MISSION BANNER AT THE TOP ---
        self.mission_banner = QWidget()
        self.mission_banner.setMinimumHeight(50) 
        banner_layout = QVBoxLayout(self.mission_banner)
        banner_layout.setContentsMargins(15, 10, 15, 0)
        
        self.lbl_current_mission = QLabel()
        self.lbl_current_mission.setAlignment(Qt.AlignCenter)
        self.lbl_current_mission.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #344f41; /* Matches the dark green buttons below */
            background: transparent; 
            border: none;
        """)
        
        self.btn_create_mission = QPushButton("Create New Mission")
        self.btn_create_mission.setMinimumHeight(45)
        self.btn_create_mission.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            background-color: transparent; 
            color: #344f41; 
            border-radius: 8px;
        """)
        
        banner_layout.addWidget(self.lbl_current_mission)
        banner_layout.addWidget(self.btn_create_mission)
        
        # Insert it at the very top of the main layout, before the stretch
        layout.insertWidget(0, self.mission_banner)
        # --------------------------------------
        
        # Add a small spacer so the logo doesn't crash into the top
        layout.addStretch(1)

        logo = QLabel()
        pixmap = QPixmap(img_path)
        logo.setStyleSheet("background: transparent; border: none;")
        logo.setPixmap(pixmap.scaled(500, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

        layout.addStretch(1)

        self.btn_classify = homepage_button("Classify Rock")
        self.btn_voice = homepage_button("Voice to Text")
        self.btn_trip = homepage_button(" View Trip Notes")
        self.btn_quit = homepage_quit_button("QUIT")

        layout.addWidget(self.btn_classify)
        layout.addWidget(self.btn_voice)
        layout.addWidget(self.btn_trip)
        layout.addSpacing(60)
        layout.addWidget(self.btn_quit)

    def update_mission_display(self, mission_name: str | None) -> None:
        if mission_name:
            self.lbl_current_mission.setText(f"Current Mission:\n{mission_name}")
            self.lbl_current_mission.show()
            self.btn_create_mission.hide()
        else:
            self.lbl_current_mission.hide()
            self.btn_create_mission.show()


class LoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self._label = QLabel("Analyzing…")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("font-size: 22px;")

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #cad2c5;
                border-radius: 5px;
                background-color: #577d6a;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #344f41;
                border-radius: 3px;
            }
        """)

        self._exact_progress = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        layout.addStretch(1)
        layout.addWidget(self._label)
        layout.addSpacing(15)
        layout.addWidget(self._progress)
        layout.addStretch(1)
        self.btn_cancel = dark_green_button("Cancel")
        layout.addWidget(self.btn_cancel)

    def _tick(self):
        if self._exact_progress < 100.0:
            self._exact_progress += (100.0 - self._exact_progress) * 0.01
            self._progress.setValue(int(self._exact_progress))

    def start_progress(self):
        self._exact_progress = 0.0
        self._progress.setValue(0)
        self._timer.start(33)

    def stop_progress(self):
        self._timer.stop()
        self._progress.setValue(0)

    def set_message(self, message: str) -> None:
        self._label.setText(message)


class VoiceLoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel("Initializing voice transcription…")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 22px;")
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)


class CameraPreviewPage(QWidget):
    """Shown when camera preview is active; user clicks Capture to take the photo."""
    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        layout = QVBoxLayout(self)

        self.lbl_step = QLabel("Capture First View")
        self.lbl_step.setAlignment(Qt.AlignCenter)
        self.lbl_step.setStyleSheet("font-size: 18px; color: #344f41;")
        layout.addWidget(self.lbl_step)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: 4px solid #344f41; border-radius: 8px;")
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout.addWidget(self.video_label, stretch=1)

        self.mic_ctrl = ExpandingVoiceWidget(self.vm, self)
        layout.addWidget(self.mic_ctrl, 0, Qt.AlignCenter)
        
        self.btn_capture = light_button("Capture")
        self.btn_cancel = dark_green_button("Back")
        self.btn_cancel.setObjectName("joystick_secondary")

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
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
        """)
        layout.addWidget(title, 0, Qt.AlignHCenter)
        layout.addStretch(1)

        images_row = QVBoxLayout()
        images_row.setAlignment(Qt.AlignCenter)
        img_style = "border: 2px solid #344f41; border-radius: 4px; padding: 2px"
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
        self.btn_retake = blue_button("Retake")
        self.btn_retake.setObjectName("joystick_secondary")
        self.btn_classify = light_button("Classify")
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


class ClassifiedPage(QWidget):
    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        self.setStyleSheet("""
            background-color: #cbd2c5;
            color: #344f41;
            font-family: "Courier New";
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(6)

        # Title
        self.lbl_label = QLabel("LABEL")
        self.lbl_label.setAlignment(Qt.AlignCenter)
        self.lbl_label.setStyleSheet(
            "background-color: #f5f6f4; font-size: 24px; font-weight: 700;"
            " border: 2px solid #697d6a; border-radius: 8px; padding: 8px;"
        )
        layout.addWidget(self.lbl_label)

        # Images (fixed height so they don't compete with the features box)
        images_wrapper = QWidget()
        images_wrapper.setMaximumHeight(120)
        images_wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        images_wrapper.setStyleSheet("background: transparent;")
        images_row = QHBoxLayout(images_wrapper)
        images_row.setSpacing(10)
        images_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_image.setStyleSheet(
            "background-color: #222; border: 3px solid #344f41; border-radius: 6px;"
        )
        self.lbl_side_image = QLabel()
        self.lbl_side_image.setAlignment(Qt.AlignCenter)
        self.lbl_side_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_side_image.setStyleSheet(
            "background-color: #222; border: 3px solid #344f41; border-radius: 6px;"
        )
        images_row.addWidget(self.lbl_image, stretch=1)
        images_row.addWidget(self.lbl_side_image, stretch=1)
        layout.addWidget(images_wrapper)

        # Confidence / volume row
        info_row = QHBoxLayout()
        self.lbl_conf = QLabel("Confidence: --")
        self.lbl_conf.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_conf.setStyleSheet("font-size: 14px; color: #555;")
        self.lbl_volume = QLabel("Volume: --")
        self.lbl_volume.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_volume.setStyleSheet("font-size: 14px;")
        self.lbl_extra = QLabel("")
        self.lbl_extra.setAlignment(Qt.AlignCenter)
        self.lbl_extra.setStyleSheet("font-size: 13px;")
        info_row.addWidget(self.lbl_conf)
        info_row.addStretch()
        info_row.addWidget(self.lbl_volume)
        layout.addLayout(info_row)
        layout.addWidget(self.lbl_extra)

        # Features box
        self.features_container = QWidget()
        self.features_container.setStyleSheet(
            "background-color: #f5f6f4; border: 2px solid #697d6a; border-radius: 8px;"
        )
        self.features_layout = QVBoxLayout(self.features_container)
        self.features_layout.setSpacing(4)
        self.features_layout.setContentsMargins(10, 15, 10, 15)
        layout.addWidget(self.features_container, stretch=1)

        # Alternatives
        self.alternatives_container = QWidget()
        self.alternatives_layout = QHBoxLayout(self.alternatives_container)
        self.alternatives_layout.setContentsMargins(0, 0, 0, 0)
        self.alternatives_layout.setSpacing(8)
        layout.addWidget(self.alternatives_container)

        # Voice widget + buttons
        self.mic_ctrl = ExpandingVoiceWidget(self.vm, self)
        layout.addWidget(self.mic_ctrl, 0, Qt.AlignCenter)
        layout.addSpacing(10)

        self.btn_reclassify = blue_button("Reclassify")
        self.btn_save = green_button("Save Classification")
        self.btn_delete = dark_green_button("Delete")
        layout.addWidget(self.btn_reclassify)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_delete)
        


class VoicePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.setStyleSheet("""
            background-color: #f5f6f4;
            color: #344f41;
            font-family: "Courier New";
            font-size: 18px;
        """)
        
        title = QLabel("Voice to Text")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            border: 2px solid #697d6a;
            border-radius: 8px;
            padding: 8px;
        """)
        layout.addWidget(title)

        # --- NEW: The Context Label ---
        self.lbl_context = QLabel("Context: None")
        self.lbl_context.setAlignment(Qt.AlignCenter)
        self.lbl_context.setStyleSheet("font-size: 14px; color: #697d6a; font-style: italic; margin-top: 2px; margin-bottom: 5px;")
        layout.addWidget(self.lbl_context)
        # ------------------------------

        self.text = QTextEdit()
        self.text.setStyleSheet("""
            font-size: 20px;
            border: 2px solid #697d6a;
            border-radius: 8px;
            padding: 8px;
        """)
        self.text.setReadOnly(True)
        layout.addWidget(self.text, stretch=1)

        row = QHBoxLayout()
        self.btn_start = light_button("Start")
        self.btn_stop = dark_green_button("Stop")
        self.btn_redo = blue_button("Redo")
        self.btn_reset = blue_button("Reset Context")
        self.btn_save = green_button("Save")
        self.btn_delete = dark_green_button("Delete")       

        self.btn_cancel = dark_green_button("Back")
        
        for b in [self.btn_start, self.btn_stop, self.btn_redo, self.btn_save, self.btn_delete, self.btn_reset, self.btn_cancel]:
            row.addWidget(b)

        layout.addLayout(row)

        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()

    def resizeEvent(self, event):
        # Force the overlay to always match the exact size of the VoicePage
        self.loading_overlay.setGeometry(self.rect())
        super().resizeEvent(event)


class ConfirmPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setWordWrap(True)
        self._lbl.setStyleSheet("font-size: 20px; color: #344f41;")
        layout.addWidget(self._lbl, stretch=1)

        self._btn_confirm = green_button("CONFIRM")
        layout.addWidget(self._btn_confirm)

        self._btn_cancel = blue_button("CANCEL")
        layout.addWidget(self._btn_cancel)

        self._on_confirm_cb = None
        self._on_cancel_cb = None
        self._btn_confirm.clicked.connect(lambda: self._on_confirm_cb and self._on_confirm_cb())
        self._btn_cancel.clicked.connect(lambda: self._on_cancel_cb and self._on_cancel_cb())

    def prepare(self, text: str, on_confirm, on_cancel) -> None:
        self._lbl.setText(text)
        self._on_confirm_cb = on_confirm
        self._on_cancel_cb = on_cancel


class RockDeletePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setWordWrap(True)
        self._lbl.setStyleSheet("font-size: 20px; color: #344f41;")
        layout.addWidget(self._lbl, stretch=1)

        self._btn_both = green_button("DELETE ROCK && ALL NOTES")
        layout.addWidget(self._btn_both)

        self._btn_only = green_button("DELETE ROCK ONLY")
        layout.addWidget(self._btn_only)

        self._btn_cancel = blue_button("CANCEL")
        layout.addWidget(self._btn_cancel)

        self._on_both_cb = None
        self._on_only_cb = None
        self._on_cancel_cb = None
        self._btn_both.clicked.connect(lambda: self._on_both_cb and self._on_both_cb())
        self._btn_only.clicked.connect(lambda: self._on_only_cb and self._on_only_cb())
        self._btn_cancel.clicked.connect(lambda: self._on_cancel_cb and self._on_cancel_cb())

    def prepare(self, text: str, on_both, on_only, on_cancel) -> None:
        self._lbl.setText(text)
        self._on_both_cb = on_both
        self._on_only_cb = on_only
        self._on_cancel_cb = on_cancel


class AlertPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setWordWrap(True)
        self._lbl.setStyleSheet("font-size: 20px; color: #344f41;")
        layout.addWidget(self._lbl, stretch=1)

        self._btn_ok = blue_button("OK")
        layout.addWidget(self._btn_ok)

        self._on_dismiss_cb = None
        self._btn_ok.clicked.connect(lambda: self._on_dismiss_cb and self._on_dismiss_cb())

    def prepare(self, text: str, on_dismiss) -> None:
        self._lbl.setText(text)
        self._on_dismiss_cb = on_dismiss


class TripLoadPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Trip & Notes")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)

        self.lbl_current_mission = QLabel("Current mission: --")
        self.lbl_current_mission.setAlignment(Qt.AlignCenter)
        self.lbl_current_mission.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.lbl_current_mission)

        missions_label = QLabel("Missions:")
        missions_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(missions_label)

        self.list = QListWidget()
        self.list.setStyleSheet("font-size: 16px;")
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setSpacing(4)
        layout.addWidget(self.list, stretch=1)

        self.btn_create_new_mission = light_button("Create New Mission")
        layout.addWidget(self.btn_create_new_mission)

        self.btn_delete_all = dark_green_button("Delete All Missions")
        layout.addWidget(self.btn_delete_all)

        self.btn_back = dark_green_button("Back")
        layout.addWidget(self.btn_back)

        self._missions_data = []
        self._summary = None


class MissionDetailPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.lbl_title = QLabel("Mission")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(self.lbl_title)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.lbl_status)

        self.lbl_totals = QLabel("Total volume: --   Total weight: --")
        self.lbl_totals.setAlignment(Qt.AlignCenter)
        self.lbl_totals.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.lbl_totals)

        items_label = QLabel("Mission Items:")
        items_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(items_label)

        self.list = QListWidget()
        self.list.setStyleSheet("font-size: 16px;")
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setSpacing(4)
        layout.addWidget(self.list, stretch=1)

        self.btn_back = dark_green_button("Back")
        layout.addWidget(self.btn_back)

        self._timeline_data = []
        self._summary = None


class MissionCreatePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        title = QLabel("Create New Mission")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        prompt = QLabel("Type a name for the mission:")
        prompt.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(prompt)

        self.lbl_recording_status = QLabel("Recording... speak the mission name")
        self.lbl_recording_status.setAlignment(Qt.AlignCenter)
        self.lbl_recording_status.setStyleSheet("font-size: 16px; color: #7e1f23; font-weight: 700;")
        layout.addWidget(self.lbl_recording_status)

        self.text = QTextEdit()
        self.text.setFixedHeight(80)
        self.text.setStyleSheet("""
            font-size: 20px;
            border: 2px solid #697d6a;
            border-radius: 8px;
            padding: 10px;
            background-color: #f5f6f4;
            color: #344f41;
        """)
        layout.addWidget(self.text)

        self.btn_create = light_button("Create Mission")
        layout.addWidget(self.btn_create)

        self.btn_cancel = dark_green_button("Back")
        layout.addWidget(self.btn_cancel)
        
class RockDetailPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            background-color: #cbd2c5;
            color: #344f41;
            font-family: "Courier New";
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        BOX = "background-color: #f5f6f4; border: 2px solid #697d6a; border-radius: 8px; padding: 8px;"

        self.lbl_title = QLabel("Rock Detail")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet(f"{BOX} font-size: 24px; font-weight: 700;")
        layout.addWidget(self.lbl_title)

        self.lbl_time = QLabel("")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setStyleSheet(f"{BOX} font-size: 16px; font-weight: 600;")
        layout.addWidget(self.lbl_time)

        images_row = QHBoxLayout()
        images_row.setSpacing(10)
        self.lbl_top = QLabel()
        self.lbl_top.setAlignment(Qt.AlignCenter)
        self.lbl_top.setFixedSize(200, 150)
        self.lbl_top.setStyleSheet("background-color: #222; border: 2px solid #697d6a; border-radius: 3px;")

        self.lbl_side = QLabel()
        self.lbl_side.setAlignment(Qt.AlignCenter)
        self.lbl_side.setFixedSize(200, 150)
        self.lbl_side.setStyleSheet("background-color: #222; border: 2px solid #697d6a; border-radius: 3px;")

        images_row.addWidget(self.lbl_top)
        images_row.addWidget(self.lbl_side)
        layout.addLayout(images_row)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setFixedHeight(60)
        self.lbl_info.setStyleSheet(f"{BOX} font-size: 18px;")
        layout.addWidget(self.lbl_info)

        summary_box = QFrame()
        summary_box.setStyleSheet("QFrame { background-color: #f5f6f4; border: 2px solid #697d6a; border-radius: 8px; }")
        summary_box_layout = QVBoxLayout(summary_box)
        summary_box_layout.setContentsMargins(8, 8, 8, 8)
        summary_box_layout.setSpacing(6)

        summary_header_layout = QHBoxLayout()
        self.lbl_summary_title = QLabel("Summary:")
        self.lbl_summary_title.setStyleSheet("background-color: transparent; font-size: 18px; font-weight: 700; border: none;")

        self.btn_force_summary = QPushButton("RE-SUMMARIZE")
        self.btn_force_summary.setFixedSize(140, 30)
        self.btn_force_summary.setStyleSheet("""
            QPushButton { background-color: #cbd2c5; color: #344f41; font-weight: bold; border-radius: 8px; border: 2px solid #697d6a; font-size: 14px; }
            QPushButton:hover { background-color: #617c32; color: white; border-color: #617c32; }
        """)

        summary_header_layout.addWidget(self.lbl_summary_title, stretch=1)
        summary_header_layout.addSpacing(10)
        summary_header_layout.addWidget(self.btn_force_summary)
        summary_box_layout.addLayout(summary_header_layout)

        self.summary_buttons_widget = QWidget()
        self.summary_buttons_widget.setStyleSheet("background-color: transparent; border: none;")
        self.summary_grid = QGridLayout(self.summary_buttons_widget)
        self.summary_grid.setSpacing(5)
        self.summary_grid.setContentsMargins(0, 0, 0, 0)

        summary_box_layout.addWidget(self.summary_buttons_widget)

        self.summary_data = {}
        self.summary_buttons = {}

        self.categories = [
            ("Appearance", "Color & Appearance"),
            ("Mineralogy", "Mineralogy & Composition"),
            ("Texture", "Texture & Structure"),
            ("Weathering", "Weathering & Alteration"),
            ("Dimensions", "Dimensions & Weight"),
            ("Other", "Field Context & Sampling Notes")
        ]

        for i, (short_name, full_name) in enumerate(self.categories):
            btn = QPushButton(short_name)
            btn.setMinimumHeight(50)
            btn.setStyleSheet("""
                QPushButton { background-color: #cbd2c5; color: #344f41; font-weight: bold; font-size: 17px; border-radius: 8px; border: 2px solid #697d6a; }
                QPushButton:hover { background-color: #617c32; color: white; border-color: #617c32; }
            """)
            btn.clicked.connect(lambda checked=False, cat=full_name: self._show_category_popup(cat))
            row, col = divmod(i, 3)
            self.summary_grid.addWidget(btn, row, col)
            self.summary_buttons[full_name] = btn
            self.summary_data[full_name] = "Not specified."

        layout.addWidget(summary_box)

        self.lbl_summary_loading = QLabel("Generating AI summary...")
        self.lbl_summary_loading.setAlignment(Qt.AlignCenter)
        self.lbl_summary_loading.setStyleSheet("font-size: 18px; font-style: italic; color: #344f41; background-color: transparent; border: none;")
        self.lbl_summary_loading.setMinimumHeight(140)
        layout.addWidget(self.lbl_summary_loading)
        self.lbl_summary_loading.hide()

        self.notes_text = QTextEdit()
        self.notes_text.setReadOnly(True)
        self.notes_text.setStyleSheet("""
            background-color: #f5f6f4;
            font-size: 16px;
            border: 2px solid #697d6a;
            border-radius: 8px;
            padding: 8px;
        """)
        layout.addWidget(self.notes_text, stretch=1)

        self.btn_back = dark_green_button("Back")
        layout.addWidget(self.btn_back)
        self._current_rock_id = None

        # Initialize the popup hidden on top of everything
        self.popup_overlay = SummaryPopupOverlay(self)
        self.popup_overlay.hide()

    def resizeEvent(self, event):
        if hasattr(self, 'popup_overlay'):
            self.popup_overlay.setGeometry(self.rect())
        super().resizeEvent(event)

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

    def _show_category_popup(self, category: str):
        # Triggered when a grid button is clicked
        content = self.summary_data.get(category, "No data available.")
        self.popup_overlay.show_popup(category, content)

    def _populate_buttons(self, data: list[tuple[str, str]]) -> None:
        """Loads the parsed AI summary data into the button memory dictionary."""
        # 1. Reset everything to defaults
        for cat in self.summary_data.keys():
            self.summary_data[cat] = "Not specified."
            
        # 2. Check if the AI is currently thinking
        is_generating = any("Generating" in v for k, v in data)
        if is_generating:
            # Hide the buttons, show the label, and disable the RE-SUMMARIZE button!
            self.summary_buttons_widget.hide()
            self.lbl_summary_loading.show()
            self.btn_force_summary.setEnabled(False) 
            return
            
        # 3. AI is done! Restore visibility and re-enable the button
        self.lbl_summary_loading.hide()
        self.summary_buttons_widget.show()
        self.btn_force_summary.setEnabled(True)
            
        # 4. Map the AI output securely to the exact buttons
        for key, val in data:
            for full_name in self.summary_data.keys():
                # We match based on the first word to ensure it syncs perfectly
                if key.split()[0].lower() in full_name.lower():
                    self.summary_data[full_name] = val

    def set_entry(self, entry, associated_notes=None, ai_summary: str = "Generating AI summary...") -> None:
        self._current_rock_id = entry.rock_id
        self._current_rock_entry = entry                  # Save for the button
        self._current_associated_notes = associated_notes # Save for the button

        dt = datetime.datetime.fromtimestamp(entry.ts) if entry.ts else None
        self.lbl_time.setText(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}" if dt else "Unknown time")

        res = entry.result
        self.lbl_title.setText(f"{res.label.upper()} ({int(res.confidence * 100)}%)")

        w, h = 200, 150

        self.lbl_top.setVisible(True)
        if res.image_path and os.path.exists(res.image_path):
            self.lbl_top.setPixmap(QPixmap(res.image_path).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.lbl_top.setText("")
        else:
            self.lbl_top.setPixmap(QPixmap())
            self.lbl_top.setText("Top")

        has_side = bool(res.side_image_path and os.path.exists(res.side_image_path))
        self.lbl_side.setVisible(has_side)
        if has_side:
            self.lbl_side.setPixmap(QPixmap(res.side_image_path).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.lbl_side.setText("")
        else:
            self.lbl_side.setPixmap(QPixmap())
            self.lbl_side.setText("Side")

        vol_txt = f"{res.estimated_volume} cm³" if res.estimated_volume is not None else "N/A"
        weight_txt = res.estimated_weight if res.estimated_weight is not None else "N/A"
        self.lbl_info.setText(f"Volume: {vol_txt}\nWeight: {weight_txt}")
        
        # --- NEW: Parse and populate the table instead of HTML ---
        table_data = self._parse_summary_to_table_data(ai_summary)
        self._populate_buttons(table_data)
        # ---------------------------------------------------------

        # --- Populate Voice Notes ---
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
            self.notes_text.setPlainText("No associated voice notes.")

    def set_ai_summary(self, rock_id: str, summary: str) -> None:
        if self._current_rock_id == rock_id:
            table_data = self._parse_summary_to_table_data(summary)
            self._populate_buttons(table_data)


class VoiceNoteDetailPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            background-color: #f5f6f4;
            color: #344f41;
            font-family: "Courier New";
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        self.lbl_title = QLabel("Voice Note")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("background-color: #f5f6f4; font-size: 24px; font-weight: 700; border: 2px solid #697d6a; border-radius: 8px; padding: 8px;")
        
        layout.addWidget(self.lbl_title)

        self.lbl_time = QLabel("")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(self.lbl_time)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("""
            font-size: 18px;
            border: 2px solid #697d6a;
            border-radius: 8px;
            padding: 10px;
        """)
        layout.addWidget(self.text, stretch=1)

        self.btn_back = dark_green_button("Back")
        layout.addWidget(self.btn_back)

    def set_note(self, note: dict) -> None:
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
        self.setMouseTracking(True)
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(10)

        self.trigger_btn = QPushButton("🎤")
        self.trigger_btn.setFixedSize(50, 50)
        self.trigger_btn.setStyleSheet("background-color: #344f41; color: white; border-radius: 25px; font-size: 20px;")
        self.trigger_btn.setObjectName("joystick_secondary")
        self.main_layout.addWidget(self.trigger_btn)
        self._panel_visible = False
        self.trigger_btn.clicked.connect(self._toggle_panel)

        self.button_container = QWidget()
        self.container_layout = QHBoxLayout(self.button_container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_start = self._make_sub_btn("Start", "#617c32")
        self.btn_stop = self._make_sub_btn("Stop", "#7e1f23")
        self.btn_redo = self._make_sub_btn("Redo", "#95b7dc")

        self.container_layout.addWidget(self.btn_start)
        self.container_layout.addWidget(self.btn_stop)
        self.container_layout.addWidget(self.btn_redo)
        
        self.main_layout.addWidget(self.button_container)
        self.button_container.hide()

        self.vm.recording_status_changed.connect(self._update_ui_state)
        self.btn_start.clicked.connect(lambda: self.vm.start_voice_to_text(silent=True))
        self.btn_stop.clicked.connect(self.vm.stop_voice_to_text)
        self.btn_redo.clicked.connect(self.vm.redo_voice_to_text)

        if hasattr(self.vm, 'vtt_active'):
            self._update_ui_state(self.vm.vtt_active)
        else:
            self._update_ui_state(False)

    def _update_ui_state(self, is_recording: bool):
        """Updates the button appearance based on actual recording state"""
        if is_recording:
            self.trigger_btn.setText("🔴")
            self.trigger_btn.setStyleSheet("""
                background-color: #7e1f23; 
                color: white; 
                border-radius: 25px; 
                font-size: 20px;
            """)
        else:
            self.trigger_btn.setText("🎤")
            self.trigger_btn.setStyleSheet("""
                background-color: #344f41; 
                color: white; 
                border-radius: 25px; 
                font-size: 20px;
            """)

    def _on_start_clicked(self):
        self.trigger_btn.setText("🔴")
        self.trigger_btn.setStyleSheet("background-color: #7e1f23; color: white; border-radius: 25px; font-size: 20px;")
        self.vm.start_voice_to_text()

    def _on_stop_clicked(self):
        self.trigger_btn.setText("🎤")
        self.trigger_btn.setStyleSheet("background-color: #344f41; color: white; border-radius: 25px; font-size: 20px;")
        self.vm.stop_voice_to_text()

    def _make_sub_btn(self, text, color):
        btn = QPushButton(text)
        btn.setFixedSize(60, 40)
        btn.setStyleSheet(f"background-color: {color}; color: white; border-radius: 5px; font-size: 12px;")
        return btn

    def _toggle_panel(self):
        self._panel_visible = not self._panel_visible
        self.button_container.setVisible(self._panel_visible)

    def enterEvent(self, event):
        if not self._panel_visible:
            self.button_container.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self._panel_visible:
            self.button_container.hide()
        super().leaveEvent(event)

class SleepOverlay(QWidget):
    """Full-window black overlay shown while the app is in sleep mode."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self.hide()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))


class ReLaunchSplash(QWidget):
    """Full-window loading overlay shown while models reload after sleep."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #cbd2c5; color: #cad2c5; font-family: 'Courier New';")
        self.hide()

        layout = QVBoxLayout(self)

        import pathlib
        logo = QLabel()
        logo_path = pathlib.Path(__file__).parent / "newlogo.png"
        if logo_path.exists():
            from PySide6.QtGui import QPixmap
            logo.setPixmap(QPixmap(str(logo_path)).scaled(500, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("background: transparent; border: none;")

        self.loading_text = QLabel("Loading SAGE...")
        self.loading_text.setStyleSheet("font-size: 24px; font-weight: bold; background: transparent;")
        self.loading_text.setAlignment(Qt.AlignCenter)
        self.loading_text.setWordWrap(True)
        self.loading_text.setMaximumWidth(460)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 13px; color: #344f41; background: transparent;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setMaximumWidth(460)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #cad2c5; border-radius: 5px;
                background-color: #577d6a; height: 30px;
                text-align: center; font-weight: bold; font-size: 16px;
            }
            QProgressBar::chunk { background-color: #344f41; border-radius: 3px; }
        """)

        quit_btn = QPushButton("QUIT")
        quit_btn.setMinimumHeight(50)
        quit_btn.setStyleSheet("font-size: 20px; background-color: #344f41; color: #cad2c5;")
        quit_btn.clicked.connect(QApplication.quit)

        layout.addStretch(1)
        layout.addWidget(logo)
        layout.addSpacing(40)
        layout.addWidget(self.loading_text)
        layout.addSpacing(8)
        layout.addWidget(self.status_label)
        layout.addSpacing(10)
        layout.addWidget(self.progress)
        layout.addStretch(1)
        layout.addWidget(quit_btn)

        self._exact_progress = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _tick(self):
        if self._exact_progress < 95.0:
            self._exact_progress += (95.0 - self._exact_progress) * 0.01
            self.progress.setValue(int(self._exact_progress))

    def start(self):
        self._exact_progress = 0.0
        self.progress.setValue(0)
        self.loading_text.setText("Loading SAGE...")
        self.status_label.setText("Loading models (this may take a moment)...")
        self._timer.start(33)
        self.show()
        self.raise_()

    def finish(self):
        self._timer.stop()
        self.progress.setValue(100)
        self.loading_text.setText("Ready!")

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0xcb, 0xd2, 0xc5))


class AppWindow(QMainWindow):
    def __init__(self, vm):
        super().__init__()

        import time
        import json
        self.MAX_BATTERY_SECONDS = 378 * 60 
        self.current_battery_percentage = 100.0 
        
        # Load saved battery state if it exists
        state_file = project_root / "battery_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    self.current_battery_percentage = float(data.get("percentage", 100.0))
            except Exception as e:
                print(f"Failed to load battery state: {e}")

        self.last_battery_calc_time = time.time()
        self.last_saved_percentage = int(self.current_battery_percentage)
        
        self.vm = vm
        self.setWindowTitle("SAGE Jetson UI")
        self.setStyleSheet("background-color: #cbd2c5;")
        self.setStyleSheet("""
            background-color: #cbd2c5;
            color: #344f41;
            font-family: "Courier New";
            font-size: 22px;
            font-weight: bold;
        """)

        self.stack = QStackedWidget()
        self.stack.setMinimumSize(0, 0)
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.status_widget = QWidget()
        self.status_widget.setFixedHeight(28)
        self.status_widget.setStyleSheet("background-color: transparent;")
        status_layout = QHBoxLayout(self.status_widget)
        status_layout.setContentsMargins(8, 2, 8, 2)

        self.lbl_time = QLabel("00:00")
        self.lbl_time.setStyleSheet("font-size: 16px; font-weight: bold; color: #344f41; background: transparent;")

        self.lbl_date = QLabel("Mon, Jan 1")
        self.lbl_date.setStyleSheet("font-size: 16px; font-weight: bold; color: #344f41; background: transparent;")
        self.lbl_date.setAlignment(Qt.AlignCenter)

        self.lbl_battery = QLabel("100% 🔋")
        self.lbl_battery.setStyleSheet("font-size: 16px; font-weight: bold; color: #344f41; background: transparent;")
        self.lbl_battery.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_battery.mousePressEvent = self._on_battery_clicked

        status_layout.addWidget(self.lbl_time)
        status_layout.addStretch(1)
        status_layout.addWidget(self.lbl_date)
        status_layout.addStretch(1)
        status_layout.addWidget(self.lbl_battery)

        # Container: navbar on top, stack below
        _container = QWidget()
        _container_layout = QVBoxLayout(_container)
        _container_layout.setContentsMargins(0, 0, 0, 0)
        _container_layout.setSpacing(0)
        _container_layout.addWidget(self.status_widget)
        _container_layout.addWidget(self.stack)
        self.setCentralWidget(_container)

        # Start global timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)
        self._update_status()

        self.home = HomePage()
        self.loading = LoadingPage()
        self.camera_preview = CameraPreviewPage(self.vm)
        self.capture_review = CaptureReviewPage(self.vm)
        self.classified = ClassifiedPage(self.vm)
        self.voice_loading = VoiceLoadingPage()
        self.voice = VoicePage()
        self.trip = TripLoadPage()
        self.confirm_page = ConfirmPage()
        self.rock_delete_page = RockDeletePage()
        self.alert_page = AlertPage()
        self.mission_detail = MissionDetailPage()
        self.mission_create = MissionCreatePage()
        self.rock_detail = RockDetailPage()
        self.voice_note_detail = VoiceNoteDetailPage()
        self._selected_mission_id = None
        self._mission_name_typing_mode = False

        self.stack.addWidget(self.home)
        self.stack.addWidget(self.loading)
        self.stack.addWidget(self.camera_preview)
        self.stack.addWidget(self.capture_review)
        self.stack.addWidget(self.classified)
        self.stack.addWidget(self.voice_loading)
        self.stack.addWidget(self.voice)
        self.stack.addWidget(self.trip)
        self.stack.addWidget(self.confirm_page)
        self.stack.addWidget(self.rock_delete_page)
        self.stack.addWidget(self.alert_page)
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
        self.joystick.relaunch_requested.connect(self._relaunch_application)
        self.joystick.start()

        self.sleep_overlay = SleepOverlay(self)
        self.relaunch_splash = ReLaunchSplash(self)

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

    def _save_battery_state(self):
        """Saves the battery state to a JSON file to survive reboots."""
        import json
        try:
            state_file = project_root / "battery_state.json"
            with open(state_file, "w") as f:
                json.dump({"percentage": self.current_battery_percentage}, f)
        except Exception as e:
            print(f"Failed to save battery state: {e}")

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

    def _on_battery_clicked(self, event):
        """Triggered when the user taps the battery percentage in the top right."""
        dialog = BatteryInputDialog(self)
        if dialog.exec():
            new_pct = dialog.get_percentage()
            if new_pct != -1:
                import time
                self.current_battery_percentage = float(new_pct)
                self.last_battery_calc_time = time.time()
                
                # --- NEW: Save the manual override to disk ---
                self.last_saved_percentage = new_pct
                self._save_battery_state()
                
                self._update_status() # Instantly reflect the change
    
    def _on_reset_context_clicked(self) -> None:
        self.vm.reset_voice_context()
        self.voice.btn_reset.hide()
        self._update_vtt_context_label()

    def _on_force_summary_clicked(self) -> None:
        entry = getattr(self.rock_detail, "_current_rock_entry", None)
        notes = getattr(self.rock_detail, "_current_associated_notes", None)
        if entry and notes:
            table_data = self.rock_detail._parse_summary_to_table_data("Generating AI summary...")
            self.rock_detail._populate_buttons(table_data)

            self.vm.request_rock_summary(entry, notes, force=True)

    def _on_summary_popup_shown(self) -> None:
        QTimer.singleShot(50, lambda: self.joystick._highlight_btn(self.rock_detail.popup_overlay.btn_cancel))

    def _on_summary_popup_hidden(self) -> None:
        QTimer.singleShot(50, self.joystick._focus_first)   
   
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
            if self.stack.currentWidget() == self.voice:
                v.loading_overlay.start()
            
        elif mode == "review":
            v.btn_stop.setText("Edit")
            v.btn_stop.show()
            v.btn_redo.show()
            v.btn_save.show()
            v.btn_delete.show()
            
        if hasattr(self, 'joystick'):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(50, self.joystick._focus_first)

    def _start_rock_assignment(self, note_ts):
        """Kicks off the jiggle animation and waits for a rock click."""
        self._assigning_note_ts = note_ts
        self._shake_offset = 0
        self._shake_direction = 1
        
        self._shake_timer = QTimer(self)
        self._shake_timer.timeout.connect(self._do_shake)
        self._shake_timer.start(40) # Update the layout every 40ms

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
                    widget.setContentsMargins(5 + self._shake_offset, 2, 5 - self._shake_offset, 2)

    def _stop_rock_assignment(self):
        """Stops the timer and resets everything back to normal."""
        self._assigning_note_ts = None
        if hasattr(self, "_shake_timer") and self._shake_timer is not None:
            self._shake_timer.stop()
            self._shake_timer = None
            
        if hasattr(self, "mission_detail") and hasattr(self.mission_detail, "list"):
            for i in range(self.mission_detail.list.count()):
                list_item = self.mission_detail.list.item(i)
                widget = self.mission_detail.list.itemWidget(list_item)
                if widget:
                    widget.setContentsMargins(5, 2, 5, 2)
    
    def _on_delete_all_clicked(self) -> None:
        self.confirm_page.prepare(
            "Are you sure? This will delete all past mission data and is irretrievable.",
            on_confirm=self._on_confirm_delete_all,
            on_cancel=self._show_trip_home,
        )
        self.stack.setCurrentWidget(self.confirm_page)

    def _on_confirm_delete_all(self) -> None:
        self.vm.clear_all_trip_data()
        self._show_trip_home()

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
        try:
            self.vm.transcriber.kill()
        except Exception:
            pass

        self.sleep_overlay.setGeometry(self.rect())
        self.sleep_overlay.show()
        self.sleep_overlay.raise_()
        QApplication.processEvents()

        self.joystick.sleep_mode()
        QApplication.instance().installEventFilter(self)

    def _relaunch_application(self) -> None:
        if not self.joystick._sleeping:
            return

        QApplication.instance().removeEventFilter(self)
        self.joystick.wake_mode()

        self.relaunch_splash.setGeometry(self.rect())
        self.relaunch_splash.start()
        self.relaunch_splash.raise_()
        self.sleep_overlay.hide()
        QApplication.processEvents()

        try:
            self.vm.transcriber.boot_model()
        except Exception as e:
            print(f"[SLEEP] Model reboot failed: {e}", file=sys.stderr)

        self.relaunch_splash.finish()
        QTimer.singleShot(400, self.relaunch_splash.hide)

        self._show_state(AppStateType.HOME)
        QTimer.singleShot(100, self.joystick._focus_first)

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.KeyPress and self.joystick._sleeping:
            self._relaunch_application()
            return True
        return super().eventFilter(obj, event)

    def _wire_ui(self) -> None:
        self.home.btn_classify.clicked.connect(self.vm.open_camera_preview)
        self.home.btn_voice.clicked.connect(lambda: self.vm.state_changed.emit(AppStateType.VOICE_TO_TEXT))
        self.home.btn_trip.clicked.connect(self.vm.open_trip_load)
        self.home.btn_quit.clicked.connect(self._quit_application)

        self.home.btn_create_mission.clicked.connect(self._open_create_mission_page)
        self.camera_preview.btn_capture.clicked.connect(self.vm.trigger_capture)
        self.camera_preview.btn_cancel.clicked.connect(self.vm.cancel_camera)
        self.capture_review.btn_classify.clicked.connect(self.vm.confirm_captures_and_classify)
        self.capture_review.btn_retake.clicked.connect(self.vm.retake_captures)
        self.loading.btn_cancel.clicked.connect(self.vm.cancel_classification)

        self.classified.btn_reclassify.clicked.connect(self.vm.reclassify)
        self.classified.btn_save.clicked.connect(self.vm.save_classification)
        self.classified.btn_delete.clicked.connect(self.vm.delete_classification)

        self.voice.btn_start.clicked.connect(self.vm.start_voice_to_text)
        # self.voice.btn_stop.clicked.connect(self.vm.stop_voice_to_text)
        #self.voice.btn_stop.clicked.connect(self.vm.stop_voice_to_text)
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

        self.mission_create.btn_create.clicked.connect(self._on_create_mission_clicked)
        self.mission_create.btn_cancel.clicked.connect(self._show_trip_home)

        self.rock_detail.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.mission_detail))
        self.rock_detail.btn_force_summary.clicked.connect(self._on_force_summary_clicked)
        self.voice_note_detail.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.mission_detail))

        self.rock_detail.popup_overlay.popup_shown.connect(self._on_summary_popup_shown)
        self.rock_detail.popup_overlay.popup_hidden.connect(self._on_summary_popup_hidden)
        
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
        self.vm.vtt_formatting = False
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
                if self.stack.currentWidget() == self.loading and mapping[state] != self.loading:
                    self.loading.stop_progress()
                self.stack.setCurrentWidget(mapping[state])
                if mapping[state] == self.loading:
                    self.loading.start_progress()

        if state == AppStateType.HOME:
            self.vm.stop_mission_name_recording(abort=True)
            self.mission_keyboard.hide()

            current_id = self.vm.store.get_current_mission_id()
            mission_name = None
            if current_id:
                for m in self.vm.store.list_missions():
                    if m.mission_id == current_id:
                        mission_name = m.name
                        break
            self.home.update_mission_display(mission_name)

            # FIX: Check vtt_active instead of the background process state!
            if not getattr(self.vm, 'vtt_active', False):
                self.voice.text.clear()
                self.vm.transcription_text = ""  # Wipes ghost text from memory!

                self.vm.vtt_formatting = False
                
                # Reset the little camera-preview voice button
                self.camera_preview.mic_ctrl.trigger_btn.setText("🎤")
                self.camera_preview.mic_ctrl.trigger_btn.setStyleSheet(
                    "background-color: #344f41; color: white; border-radius: 25px; font-size: 20px;"
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

            # --- THE FIX: Wipe ghost text from late LLM flushes ---
            # If we enter this page and are NOT actively recording, start fresh!
            if not getattr(self.vm, 'vtt_active', False) and not getattr(self.vm, 'transcription_text', '').strip():
                self.vm.transcription_text = ""
                self.voice.text.clear()
            # ------------------------------------------------------

            current_text = self.vm.transcription_text
            self.voice.text.setPlainText(current_text)
            self.voice.btn_save.setEnabled(bool(current_text.strip()))

            # --- NEW: Trigger Dynamic Layout ---
            if getattr(self.vm, 'vtt_active', False):
                self._update_voice_buttons("recording")
            elif getattr(self.vm, 'vtt_formatting', False):
                self._update_voice_buttons("formatting")
            elif current_text.strip():
                self._update_voice_buttons("review")
            else:
                self._update_voice_buttons("initial")

        elif state == AppStateType.TRIP_LOAD:
            self._show_trip_home()
            
    def _on_classification(self, result: ClassificationResult) -> None:
        self.joystick._page_memory.pop(id(self.classified), None)
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

        self.classified.lbl_label.setText(result.label.upper())
        self.classified.lbl_conf.setText(f"Confidence: {int(result.confidence * 100)}%")

        top3 = None
        if result.raw and isinstance(result.raw, dict):
            top3 = result.raw.get("top3") or None
            if not top3:
                scores = result.raw.get("primary", {}).get("scores", {})
                if scores:
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top3 = [{"label": k, "confidence": v} for k, v in sorted_scores[:3]]

        while self.classified.alternatives_layout.count():
            child = self.classified.alternatives_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        alt_entries = []
        if top3 and len(top3) >= 2:
            conf2 = float(top3[1].get("confidence", 0.0))
            if conf2 > 0:
                alt_entries.append((top3[1].get("label", ""), conf2))
        if top3 and len(top3) >= 3:
            conf3 = float(top3[2].get("confidence", 0.0))
            if conf3 > 0:
                alt_entries.append((top3[2].get("label", ""), conf3))

        if alt_entries:
            lbl = QLabel("Alternatively:")
            lbl.setStyleSheet("font-size: 12px; color: #888;")
            self.classified.alternatives_layout.addWidget(lbl)
            alt_buttons = []
            for alt_label, alt_conf in alt_entries:
                btn = blue_button(f"{alt_label.upper()}\nconf: {int(alt_conf * 100)}%")
                btn.setStyleSheet(
                    "font-size: 14px; background-color: #95b7dc; color: #385573;"
                )
                btn.setObjectName("joystick_secondary")
                alt_buttons.append(btn)
                self.classified.alternatives_layout.addWidget(btn)

            original_label = result.label
            original_conf  = result.confidence
            selected = [None]

            def _restore_original_features():
                while self.classified.features_layout.count():
                    child = self.classified.features_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                if show_features and features:
                    note_by_feature = {n["feature"]: n["note"] for n in geo_notes}
                    displayed = [
                        (feat_name, feat_data)
                        for feat_name, feat_data in features.items()
                        if feat_data.get("display")
                    ]
                    for idx, (feat_name, feat_data) in enumerate(displayed):
                        value = feat_data["value"]
                        conf = int(feat_data["confidence"] * 100)
                        feat_lbl = QLabel(f"{feat_name.replace('_', ' ').title()}: {value}  ({conf}%)")
                        feat_lbl.setAlignment(Qt.AlignLeft)
                        feat_lbl.setWordWrap(True)
                        feat_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
                        feat_lbl.setStyleSheet("font-size: 14px; border: none; background: transparent; font-weight: bold;")
                        self.classified.features_layout.addWidget(feat_lbl)
                        note = note_by_feature.get(feat_name, "")
                        if note:
                            note_lbl = QLabel(note)
                            note_lbl.setAlignment(Qt.AlignLeft)
                            note_lbl.setWordWrap(True)
                            note_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
                            note_lbl.setStyleSheet("font-size: 11px; color: #888; border: none; background: transparent; font-weight: normal;")
                            self.classified.features_layout.addWidget(note_lbl)
                        if idx < len(displayed) - 1:
                            sep = QFrame()
                            sep.setFrameShape(QFrame.HLine)
                            sep.setFixedHeight(1)
                            sep.setStyleSheet("background-color: #ccc;")
                            self.classified.features_layout.addWidget(sep)
                            self.classified.features_layout.addSpacing(6)

            def _clear_features():
                while self.classified.features_layout.count():
                    child = self.classified.features_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

            def _on_alt_clicked(_checked, chosen_label, chosen_conf, chosen_btn):
                unselected_style = "font-size: 14px; background-color: #95b7dc; color: #385573;"
                selected_style   = "font-size: 14px; background-color: #385573; color: #f5f6f4;"
                if selected[0] is chosen_btn:
                    chosen_btn.setStyleSheet(unselected_style)
                    selected[0] = None
                    self.vm.override_classification_label(original_label, original_conf)
                    self.classified.lbl_label.setText(original_label.upper())
                    self.classified.lbl_conf.setText(f"Confidence: {int(original_conf * 100)}%")
                    _restore_original_features()
                else:
                    for b in alt_buttons:
                        b.setStyleSheet(unselected_style)
                    chosen_btn.setStyleSheet(selected_style)
                    selected[0] = chosen_btn
                    self.vm.override_classification_label(chosen_label, chosen_conf)
                    self.classified.lbl_label.setText(chosen_label.upper())
                    self.classified.lbl_conf.setText(f"Confidence: {int(chosen_conf * 100)}%")
                    _clear_features()

            for btn, (alt_label, alt_conf) in zip(alt_buttons, alt_entries):
                btn.clicked.connect(
                    lambda _checked, l=alt_label, c=alt_conf, b=btn: _on_alt_clicked(_checked, l, c, b)
                )

            self.classified.alternatives_layout.addStretch()

        raw = result.raw or {}
        features = raw.get("features") or {}
        geo_notes = raw.get("geology_notes") or []
        show_features = raw.get("ui", {}).get("show_features", False)

        while self.classified.features_layout.count():
            child = self.classified.features_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if show_features and features:
            note_by_feature = {n["feature"]: n["note"] for n in geo_notes}
            displayed = [
                (feat_name, feat_data)
                for feat_name, feat_data in features.items()
                if feat_data.get("display")
            ]
            for idx, (feat_name, feat_data) in enumerate(displayed):
                value = feat_data["value"]
                conf = int(feat_data["confidence"] * 100)

                feat_lbl = QLabel(f"{feat_name.replace('_', ' ').title()}: {value}  ({conf}%)")
                feat_lbl.setAlignment(Qt.AlignLeft)
                feat_lbl.setWordWrap(True)
                feat_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
                feat_lbl.setStyleSheet("font-size: 14px; border: none; background: transparent; font-weight: bold;")
                self.classified.features_layout.addWidget(feat_lbl)

                note = note_by_feature.get(feat_name, "")
                if note:
                    note_lbl = QLabel(note)
                    note_lbl.setAlignment(Qt.AlignLeft)
                    note_lbl.setWordWrap(True)
                    note_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
                    note_lbl.setStyleSheet("font-size: 11px; color: #888; border: none; background: transparent; font-weight: normal;")
                    self.classified.features_layout.addWidget(note_lbl)

                if idx < len(displayed) - 1:
                    sep = QFrame()
                    sep.setFrameShape(QFrame.HLine)
                    sep.setFixedHeight(1)
                    sep.setStyleSheet("background-color: #ccc;")
                    self.classified.features_layout.addWidget(sep)
                    self.classified.features_layout.addSpacing(6)

        extras = []
        if result.estimated_weight is not None:
            extras.append(f"Weight: {result.estimated_weight}")
        self.classified.lbl_extra.setText("   ".join(extras) if extras else "")

    def _on_volume_display(self, text: str) -> None:
        self.classified.lbl_volume.setText(text)

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
        self.joystick._page_memory.pop(id(self.mission_detail), None)
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

            widget = QWidget()
            row_layout = QHBoxLayout(widget)
            row_layout.setContentsMargins(5, 2, 5, 2)

            # lbl = QLabel(display_text)
            # lbl.setStyleSheet("font-size: 16px;")
            # lbl.setAttribute(Qt.WA_TransparentForMouseEvents)

            # dot_btn = QPushButton("⋮")
            # dot_btn.setFixedSize(30, 30)
            # dot_btn.setStyleSheet("""
            #     QPushButton { font-size: 24px; font-weight: bold; border: none; background: transparent; color: #344f41; }
            #     QPushButton::menu-indicator { image: none; width: 0px; }
            # """)
            # self._attach_timeline_menu_to_button(dot_btn, item)

            # row_layout.addWidget(lbl, stretch=1)
            # row_layout.addWidget(dot_btn)

            # AFTER:
            row_btn = QPushButton()
            row_btn.setObjectName("joystick_skip")
            row_btn.setText("")  # we use a QLabel inside for rich text
            row_btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    text-align: left;
                    border: none;
                    background: transparent;
                    padding: 4px 0px;
                }
                QPushButton:hover { background-color: rgba(52, 79, 65, 0.08); border-radius: 4px; }
            """)
            row_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            # Use a QLabel inside the button for rich HTML text
            inner_lbl = QLabel(display_text)
            inner_lbl.setStyleSheet("font-size: 16px; background: transparent; border: none;")
            inner_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
            inner_layout = QHBoxLayout(row_btn)
            inner_layout.setContentsMargins(4, 0, 4, 0)
            inner_layout.addWidget(inner_lbl)

            # Wire the row button to the same logic as itemClicked
            captured_item = item  # capture for lambda
            row_btn.clicked.connect(lambda checked=False, i=captured_item: self._on_timeline_item_activated(i))

            dot_btn = QPushButton("⋮")
            dot_btn.setObjectName("joystick_skip")
            dot_btn.setFixedSize(30, 30)
            dot_btn.setStyleSheet("""
                QPushButton { font-size: 24px; font-weight: bold; border: none; background: transparent; color: #344f41; }
                QPushButton::menu-indicator { image: none; width: 0px; }
            """)
            self._attach_timeline_menu_to_button(dot_btn, item)

            row_layout.addWidget(row_btn, stretch=1)
            row_layout.addWidget(dot_btn)
            #AFTER

            list_item.setSizeHint(widget.sizeHint())
            self.mission_detail.list.setItemWidget(list_item, widget)
            self.mission_detail._timeline_data.append(item)

    def _on_trip(self, summary: TripSummary) -> None:
        self._stop_rock_assignment()
        self.trip.list.clear()
        self.joystick._page_memory.pop(id(self.trip), None)
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
                f"<span style='font-size: 13px; color: #555;'>Updated {time_str}   Items: {item_count}</span>"
            )

            list_item = QListWidgetItem()
            self.trip.list.addItem(list_item)

            widget = QWidget()
            row_layout = QHBoxLayout(widget)
            row_layout.setContentsMargins(5, 2, 5, 2)

            # lbl = QLabel(display_text)
            # lbl.setStyleSheet("font-size: 16px;")
            # lbl.setAttribute(Qt.WA_TransparentForMouseEvents)

            # dot_btn = QPushButton("⋮")
            # dot_btn.setFixedSize(30, 30)
            # dot_btn.setStyleSheet("""
            #     QPushButton { font-size: 24px; font-weight: bold; border: none; background: transparent; color: #344f41; }
            #     QPushButton::menu-indicator { image: none; width: 0px; }
            # """)
            # self._attach_mission_menu_to_button(dot_btn, mission_summary)

            # row_layout.addWidget(lbl, stretch=1)
            # AFTER in _on_trip:
            row_btn = QPushButton()
            row_btn.setObjectName("joystick_skip")
            inner_lbl = QLabel(display_text)
            inner_lbl.setStyleSheet("font-size: 16px; background: transparent; border: none;")
            inner_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
            inner_layout = QHBoxLayout(row_btn)
            inner_layout.setContentsMargins(4, 0, 4, 0)
            inner_layout.addWidget(inner_lbl)
            row_btn.setStyleSheet("""
                QPushButton { border: none; background: transparent; padding: 4px 0px; }
                QPushButton:hover { background-color: rgba(52, 79, 65, 0.08); border-radius: 4px; }
            """)
            row_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            captured_summary = mission_summary
            row_btn.clicked.connect(lambda checked=False, ms=captured_summary: self._open_mission_detail(ms))
            
            dot_btn = QPushButton("⋮")
            dot_btn.setFixedSize(30, 30)
            dot_btn.setObjectName("joystick_skip")
            dot_btn.setStyleSheet("""
                QPushButton { font-size: 24px; font-weight: bold; border: none; background: transparent; color: #344f41; }
                QPushButton::menu-indicator { image: none; width: 0px; }
            """)
            self._attach_mission_menu_to_button(dot_btn, mission_summary)

            row_layout.addWidget(row_btn, stretch=1)
            row_layout.addWidget(dot_btn)

            list_item.setSizeHint(widget.sizeHint())
            self.trip.list.setItemWidget(list_item, widget)
        
        if self._selected_mission_id:
            selected_summary = self._find_mission_summary(self._selected_mission_id)
            if selected_summary:
                self._render_mission_detail(selected_summary)
            elif self.stack.currentWidget() in {self.mission_detail, self.rock_detail, self.voice_note_detail}:
                self._selected_mission_id = None
                self._show_trip_home()

    def _attach_mission_menu_to_button(self, button, mission_summary: MissionSummary):
        from PySide6.QtWidgets import QMenu, QWidgetAction

        menu = QMenu(button)
        menu.setStyleSheet("QMenu { background-color: #cbd2c5; border: 2px solid #344f41; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        btn_current = QPushButton("MAKE CURRENT MISSION")
        btn_current.setStyleSheet("""
            QPushButton { font-size: 18px; color: #344f41; padding: 12px 20px; border: none; text-align: left; }
            QPushButton:hover { background-color: #95b7dc; }
        """)
        btn_current.clicked.connect(
            lambda checked=False, mission_id=mission_summary.mission.mission_id, m=menu: [self.vm.make_mission_current(mission_id), m.close()]
        )
        layout.addWidget(btn_current)

        btn_delete = QPushButton("DELETE MISSION")
        btn_delete.setStyleSheet("""
            QPushButton { font-size: 18px; color: #cc0000; padding: 12px 20px; border: none; text-align: left; font-weight: bold; }
            QPushButton:hover { background-color: #95b7dc; }
        """)
        btn_delete.clicked.connect(
            lambda checked=False, mission_id=mission_summary.mission.mission_id, mission_name=mission_summary.mission.name, m=menu:
                [self._delete_mission(mission_id, mission_name), m.close()]
        )
        layout.addWidget(btn_delete)

        action = QWidgetAction(menu)
        action.setDefaultWidget(container)
        menu.addAction(action)
        btns = [btn_current, btn_delete]
        button.clicked.connect(lambda checked=False, b=button, m=menu, bs=btns: self._show_bounded_menu(b, m, bs))

    def _attach_timeline_menu_to_button(self, button, item_dict):
        from PySide6.QtWidgets import QMenu, QWidgetAction

        menu = QMenu(button)
        menu.setStyleSheet("QMenu { background-color: #cbd2c5; border: 2px solid #344f41; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if item_dict["type"] == "rock":
            btn_current = QPushButton("Make current")
            btn_current.setStyleSheet("""
                QPushButton { font-size: 18px; color: #344f41; padding: 12px 20px; border: none; text-align: left; }
                QPushButton:hover { background-color: #95b7dc; }
            """)
            btn_current.clicked.connect(lambda checked=False, d=item_dict["data"], m=menu: [self.vm.make_rock_current(d), m.close()])
            layout.addWidget(btn_current)

        if item_dict["type"] == "voice":
            btn_assign = QPushButton("Add to classification")
            btn_assign.setStyleSheet("""
                QPushButton { font-size: 18px; color: #344f41; padding: 12px 20px; border: none; text-align: left; }
                QPushButton:hover { background-color: #95b7dc; }
            """)
            btn_assign.clicked.connect(lambda checked=False, i=item_dict, m=menu: [self._start_rock_assignment(i["data"].get("ts")), m.close()])
            layout.addWidget(btn_assign)

        btn_delete = QPushButton("Delete")
        btn_delete.setStyleSheet("""
            QPushButton { font-size: 18px; color: #cc0000; padding: 12px 20px; border: none; text-align: left; font-weight: bold; }
            QPushButton:hover { background-color: #95b7dc; }
        """)
        btn_delete.clicked.connect(lambda checked=False, i=item_dict, m=menu: [self._delete_timeline_item(i), m.close()])
        layout.addWidget(btn_delete)

        action = QWidgetAction(menu)
        action.setDefaultWidget(container)
        menu.addAction(action)
        if item_dict["type"] == "rock":
            btns = [btn_current, btn_delete]
        else:
            btns = [btn_assign, btn_delete]
        button.clicked.connect(lambda checked=False, b=button, m=menu, bs=btns: self._show_bounded_menu(b, m, bs))

    def _show_bounded_menu(self, button: QPushButton, menu, menu_buttons: list = None) -> None:
        menu.ensurePolished()
        menu_size = menu.sizeHint()

        preferred_pos = button.mapToGlobal(button.rect().bottomRight())
        screen = QApplication.screenAt(preferred_pos)
        if screen is None:
            screen = button.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen else self.geometry()

        x = preferred_pos.x() - menu_size.width()
        y = preferred_pos.y()

        if x < available.left():
            x = min(
                button.mapToGlobal(button.rect().topLeft()).x(),
                available.right() - menu_size.width()
            )
        if x + menu_size.width() > available.right():
            x = available.right() - menu_size.width()

        if y + menu_size.height() > available.bottom():
            y = button.mapToGlobal(button.rect().topLeft()).y() - menu_size.height()
        if y < available.top():
            y = available.top()

        menu.popup(QPoint(x, y))
        if menu_buttons and hasattr(self, "joystick"):
            self.joystick.open_menu(menu, menu_buttons)

    def _delete_mission(self, mission_id: str, mission_name: str) -> None:
        def on_confirm():
            if mission_id == self._selected_mission_id:
                self._selected_mission_id = None
            self.vm.delete_mission(mission_id)
            self._show_trip_home()
        self.confirm_page.prepare(
            f'Are you sure? Clicking CONFIRM will delete the mission "{mission_name}" and all of its items permanently.',
            on_confirm=on_confirm,
            on_cancel=self._show_trip_home,
        )
        self.stack.setCurrentWidget(self.confirm_page)

    def _delete_timeline_item(self, item_dict):
        if item_dict["type"] == "voice":
            def on_confirm():
                self.vm.delete_voice_note_by_ts(item_dict["data"].get("ts"))
                self.stack.setCurrentWidget(self.mission_detail)
            self.confirm_page.prepare(
                "Are you sure? Clicking CONFIRM will delete this data permanently.",
                on_confirm=on_confirm,
                on_cancel=lambda: self.stack.setCurrentWidget(self.mission_detail),
            )
            self.stack.setCurrentWidget(self.confirm_page)
                
        elif item_dict["type"] == "rock":
            entry = item_dict["data"]

            def on_only():
                self.vm.delete_rock_by_id(entry.rock_id)
                self.stack.setCurrentWidget(self.mission_detail)

            def on_both():
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
                self.stack.setCurrentWidget(self.mission_detail)

            self.rock_delete_page.prepare(
                "Are you sure? Clicking CONFIRM will delete this rock permanently.",
                on_both=on_both,
                on_only=on_only,
                on_cancel=lambda: self.stack.setCurrentWidget(self.mission_detail),
            )
            self.stack.setCurrentWidget(self.rock_delete_page)

    def _on_mission_clicked(self, item) -> None:
        index = self.trip.list.row(item)
        if 0 <= index < len(self.trip._missions_data):
            self._open_mission_detail(self.trip._missions_data[index])

    def _on_timeline_item_activated(self, item_dict: dict) -> None:
        """Shared logic for clicking a timeline row (via list click or joystick button)."""
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
            has_features = bool(entry.result.raw and entry.result.raw.get("features"))
            initial_summary = "Generating AI summary..." if (associated_notes or has_features) else "No associated recordings to summarize yet."
            self.rock_detail.set_entry(entry, associated_notes, ai_summary=initial_summary)
            self.vm.request_rock_summary(entry, associated_notes)
            self.stack.setCurrentWidget(self.rock_detail)

        else:
            note = item_dict["data"]
            self.voice_note_detail.set_note(note)
            self.stack.setCurrentWidget(self.voice_note_detail)


    def _on_timeline_clicked(self, item) -> None:
        index = self.mission_detail.list.row(item)
        if 0 <= index < len(self.mission_detail._timeline_data):
            self._on_timeline_item_activated(self.mission_detail._timeline_data[index])
    
    # def _on_timeline_clicked(self, item) -> None:
    #     index = self.mission_detail.list.row(item)
    #     if 0 <= index < len(self.mission_detail._timeline_data):
    #         item_dict = self.mission_detail._timeline_data[index]
            
    #         if getattr(self, "_assigning_note_ts", None) is not None:
    #             if item_dict["type"] == "rock":
    #                 self.vm.assign_note_to_rock(self._assigning_note_ts, item_dict["data"].rock_id)
    #             self._stop_rock_assignment()
    #             return
            
    #         if item_dict["type"] == "rock":
    #             entry = item_dict["data"]
    #             associated_notes = []
    #             mission_summary = self.mission_detail._summary
    #             rocks_sorted = sorted(mission_summary.rocks, key=lambda r: r.ts)
                
    #             next_rock_ts = float('inf')
    #             for r in rocks_sorted:
    #                 if r.ts > entry.ts:
    #                     next_rock_ts = r.ts
    #                     break
                        
    #             for n in mission_summary.voice_notes:
    #                 note_ts = n.get("ts", 0)
    #                 explicit_rock_id = n.get("rock_id")
                    
    #                 if explicit_rock_id == entry.rock_id:
    #                     associated_notes.append(n)
    #                 elif explicit_rock_id is None:
    #                     if entry.ts <= note_ts < next_rock_ts:
    #                         if n.get("session_id") == entry.session_id:
    #                             associated_notes.append(n)
                
    #             associated_notes.sort(key=lambda x: x.get("ts", 0))

    #             initial_summary = "Generating AI summary..." if associated_notes else "No associated recordings to summarize yet."
    #             self.rock_detail.set_entry(entry, associated_notes, ai_summary=initial_summary)
    #             self.vm.request_rock_summary(entry, associated_notes)
    #             self.stack.setCurrentWidget(self.rock_detail)
                
    #         else:
    #             note = item_dict["data"]
    #             self.voice_note_detail.set_note(note)
    #             self.stack.setCurrentWidget(self.voice_note_detail)

    def _on_create_mission_clicked(self) -> None:
        self.vm.stop_mission_name_recording(abort=True)
        name = " ".join(self.mission_create.text.toPlainText().strip().split())
        if not name:
            return
        try:
            mission = self.vm.create_mission(name)
        except ValueError as exc:
            self.alert_page.prepare(str(exc), on_dismiss=lambda: self.stack.setCurrentWidget(self.mission_create))
            self.stack.setCurrentWidget(self.alert_page)
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
        self.alert_page.prepare(
            "Something went wrong. Please press escape to return to home screen.",
            on_dismiss=self.vm.go_home,
        )
        self.stack.setCurrentWidget(self.alert_page)
    
    def _on_recording_status_changed(self, is_recording: bool):
        self.camera_preview.mic_ctrl._update_ui_state(is_recording)
        
        if is_recording:
            self.vm.vtt_formatting = False
            self.voice.btn_stop.setText("Stop")
            self.inline_keyboard.hide()
            self.voice.text.setReadOnly(True)
            
            # If we are actually looking at the voice page, update the layout
            if self.stack.currentWidget() == self.voice:
                self._update_voice_buttons("recording")
        else:
            if getattr(self.vm, '_is_redoing', False):
                return
                
            self.vm.vtt_formatting = True
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
        """Fetches the live system time, locks to West Coast, and calculates dead-reckoning battery."""
        import datetime
        import time
        
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
        
        # --- NEW SMART DEAD RECKONING BATTERY HACK ---
        current_time = time.time()
        delta_seconds = current_time - self.last_battery_calc_time
        self.last_battery_calc_time = current_time
        
        percentage_drop = (delta_seconds / self.MAX_BATTERY_SECONDS) * 100.0
        self.current_battery_percentage -= percentage_drop
        self.current_battery_percentage = math.ceil(max(0.0, min(100.0, self.current_battery_percentage)))
        
        display_percentage = int(self.current_battery_percentage)
        
        # --- NEW: Save to disk only when the visible number changes ---
        if display_percentage != self.last_saved_percentage:
            self._save_battery_state()
            self.last_saved_percentage = display_percentage
            
        icon = "🪫" if display_percentage < 15 else "🔋"
        self.lbl_battery.setText(f"{display_percentage}% {icon}")
            

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
                btn.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #e0e0e0; color: black;")
                
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
        self.btn_shift.setStyleSheet("font-size: 18px; font-weight: bold; background-color: #c0c0c0; color: black;")
        self.btn_shift.clicked.connect(self._on_shift_clicked)
        bottom_layout.addWidget(self.btn_shift, stretch=1)
        
        self.shift_timer = QTimer(self)
        self.shift_timer.setSingleShot(True)
        
        self.btn_space = QPushButton("Space")
        self.btn_space.setMinimumHeight(45)
        self.btn_space.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #e0e0e0; color: black;")
        self.btn_space.clicked.connect(lambda checked=False: self.key_pressed.emit("Space"))
        bottom_layout.addWidget(self.btn_space, stretch=3)
        
        # Arrow Keys
        for arrow in ["←", "↑", "↓", "→"]:
            btn = QPushButton(arrow)
            btn.setMinimumHeight(45)
            btn.setMinimumWidth(20)
            btn.setStyleSheet("font-size: 18px; font-weight: bold; background-color: #c0c0c0; color: black;")
            btn.clicked.connect(lambda checked=False, char=arrow: self.key_pressed.emit(char))
            bottom_layout.addWidget(btn)
            
        # The Close Button (keeping the name btn_close so our hiding logic still works!)
        self.btn_close = QPushButton("Close")
        self.btn_close.setMinimumHeight(45)
        self.btn_close.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #8c4b4b; color: white;")
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
            self.btn_shift.setStyleSheet("font-size: 18px; font-weight: bold; background-color: #c0c0c0; color: black;")
        elif self.shift_state == 1:
            self.btn_shift.setText("⬆")
            self.btn_shift.setStyleSheet("font-size: 18px; font-weight: bold; background-color: #ffffff; color: black;")
        elif self.shift_state == 2:
            self.btn_shift.setText("⇪") # Caps lock icon
            self.btn_shift.setStyleSheet("font-size: 18px; font-weight: bold; background-color: #4a90e2; color: white;")

    
                
