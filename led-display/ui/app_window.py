from __future__ import annotations
import os
import sys
import time
import datetime
from pathlib import Path
from PySide6.QtGui import QPixmap, QImage

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
img_path = project_root / "led-display" / "ui" / "sage-logo-wcbg.png"


from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor, QKeyEvent, QShortcut, QKeySequence, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QMessageBox,
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QHBoxLayout, QSizePolicy, QDialog, QFrame,
)

import connector
from core.viewmodel import AppStateType, ClassificationResult, TripSummary


def big_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(70)
    b.setStyleSheet("font-size: 20px;")
    return b


class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.setStyleSheet("""
            background-color: #586e5d;
            color: #cad2c5;
        """)

        logo = QLabel()
        pixmap = QPixmap(img_path)
        logo.setStyleSheet("background: transparent; border: none;")
        logo.setPixmap(pixmap.scaled(400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

        self.btn_classify = big_button("Classify Rock")
        self.btn_voice = big_button("Voice to Text")
        self.btn_trip = big_button(" View Trip Notes")
        self.btn_quit = QPushButton("QUIT")
        self.btn_quit.setStyleSheet("""
            background-color: #344f41;
            color: #cad2c5;
        """)

        self.btn_quit.setMinimumHeight(50)

        layout.addWidget(self.btn_classify)
        layout.addWidget(self.btn_voice)
        layout.addWidget(self.btn_trip)
        layout.addSpacing(60)
        layout.addWidget(self.btn_quit)


class LoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel("Analyzing…")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 22px;")
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(50)
        self.btn_cancel.setStyleSheet("""
            background-color: #7e1f23;
            font-size: 22px;
            color: white;
        """)
        layout.addWidget(self.btn_cancel)

    def set_message(self, message: str) -> None:
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setText(message)
                break


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
        
        self.btn_capture = QPushButton("Capture")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(50)
        self.btn_capture.setMinimumHeight(50)
        self.btn_cancel.setStyleSheet("""
            background-color: #7e1f23;
            font-size: 22px;
            color: white;
        """)
        self.btn_capture.setStyleSheet("""
            font-size: 22px;
        """)
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
        self.btn_retake = QPushButton("Retake")
        self.btn_retake.setMinimumHeight(55)
        self.btn_retake.setStyleSheet("background-color: #95b7dc; font-size: 20px; color: #385573;")
        self.btn_classify = QPushButton("Classify")
        self.btn_classify.setMinimumHeight(55)
        self.btn_classify.setStyleSheet("background-color: #617c32; font-size: 20px; color: #f5f6f4;")
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
    line.setStyleSheet("color: #697d6a;")
    line.setFixedHeight(1)
    return line


class ClassifiedPage(QWidget):
    def __init__(self, vm):
        super().__init__()
        self.vm = vm

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Scrollable content ───────────────────────────────────────────────
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFocusPolicy(Qt.NoFocus)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollBar:vertical { width: 6px; }")

        content = QWidget()
        self.main_layout = QVBoxLayout(content)
        self.main_layout.setContentsMargins(16, 14, 16, 8)
        self.main_layout.setSpacing(6)
        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

        # ── Images (compact thumbnails for reference) ────────────────────────
        self.image_container = QWidget()
        self.image_container.setStyleSheet("background-color: transparent;")
        img_layout = QHBoxLayout(self.image_container)
        img_layout.setContentsMargins(0, 0, 0, 4)
        img_layout.setSpacing(10)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: transparent; border: none;")
        self.lbl_image.setFixedSize(130, 98)
        self.lbl_image.setScaledContents(True)
        self.lbl_side_image = QLabel()
        self.lbl_side_image.setAlignment(Qt.AlignCenter)
        self.lbl_side_image.setStyleSheet("background-color: transparent; border: none;")
        self.lbl_side_image.setFixedSize(130, 98)
        self.lbl_side_image.setScaledContents(True)
        img_layout.addStretch()
        img_layout.addWidget(self.lbl_image)
        img_layout.addWidget(self.lbl_side_image)
        img_layout.addStretch()
        self.main_layout.addWidget(self.image_container)

        # ── Line 1: Rock name (large, left) + Volume (right) ─────────────────
        name_vol_row = QHBoxLayout()
        name_vol_row.setContentsMargins(0, 6, 0, 0)
        name_vol_row.setSpacing(8)
        self.lbl_label = QLabel("—")
        self.lbl_label.setStyleSheet("font-size: 32px; font-weight: 700;")
        self.lbl_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.lbl_volume = QLabel("")
        self.lbl_volume.setStyleSheet("font-size: 17px; color: #555;")
        self.lbl_volume.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        name_vol_row.addWidget(self.lbl_label)
        name_vol_row.addWidget(self.lbl_volume)
        self.main_layout.addLayout(name_vol_row)

        # ── Line 2: Confidence + tier badge + weight ──────────────────────────
        conf_wt_row = QHBoxLayout()
        conf_wt_row.setContentsMargins(0, 0, 0, 4)
        conf_wt_row.setSpacing(8)
        self.lbl_conf = QLabel("—")
        self.lbl_conf.setStyleSheet("font-size: 18px; color: #555;")
        self.lbl_tier = QLabel("")
        self.lbl_tier.setStyleSheet(
            "font-size: 17px; font-weight: 700; border-radius: 5px; padding: 2px 10px; color: #f5f6f4;"
        )
        self.lbl_tier.setVisible(False)
        self.lbl_extra = QLabel("")
        self.lbl_extra.setStyleSheet("font-size: 16px; color: #777;")
        conf_wt_row.addWidget(self.lbl_conf)
        conf_wt_row.addWidget(self.lbl_tier)
        conf_wt_row.addStretch()
        conf_wt_row.addWidget(self.lbl_extra)
        self.main_layout.addLayout(conf_wt_row)

        # ── Divider ───────────────────────────────────────────────────────────
        self.main_layout.addSpacing(12)
        self.main_layout.addWidget(_make_divider())

        # ── Features + geology notes (rebuilt on each classification) ─────────
        self.features_section = QWidget()
        self.features_layout = QVBoxLayout(self.features_section)
        self.features_layout.setContentsMargins(0, 4, 0, 0)
        self.features_layout.setSpacing(3)
        self.main_layout.addWidget(self.features_section)

        # ── Alternative classifications (selectable override) ─────────────────
        self.div_alts = _make_divider()
        self.div_alts.setVisible(False)
        self.main_layout.addWidget(self.div_alts)

        self.alt_row = QWidget()
        alt_outer = QHBoxLayout(self.alt_row)
        alt_outer.setContentsMargins(0, 6, 0, 4)
        alt_outer.setSpacing(8)
        lbl_also = QLabel("Alternatively:")
        lbl_also.setStyleSheet("font-size: 16px; color: #697d6a; font-weight: 700;")
        self.alt_buttons_layout = QHBoxLayout()
        self.alt_buttons_layout.setSpacing(6)
        alt_outer.addWidget(lbl_also)
        alt_outer.addLayout(self.alt_buttons_layout)
        alt_outer.addStretch()
        self.alt_row.setVisible(False)
        self.main_layout.addWidget(self.alt_row)
        self.main_layout.addStretch(1)

        # ── Fixed buttons pinned at bottom ────────────────────────────────────
        btn_area = QWidget()
        btn_area.setStyleSheet("border-top: 1px solid #697d6a;")
        btn_vbox = QVBoxLayout(btn_area)
        btn_vbox.setContentsMargins(12, 10, 12, 10)
        btn_vbox.setSpacing(8)

        self.mic_ctrl = ExpandingVoiceWidget(self.vm, self)

        self.btn_reclassify = big_button("Reclassify")
        self.btn_reclassify.setStyleSheet("""
            QPushButton { background-color: #95b7dc; font-size: 20px; color: #385573; }
            QPushButton:hover { background-color: #b8d4ec; }
        """)
        self.btn_save = big_button("Save")
        self.btn_save.setStyleSheet("""
            QPushButton { background-color: #617c32; font-size: 20px; color: #f5f6f4; }
            QPushButton:hover { background-color: #7a9a3e; }
        """)
        self.btn_delete = big_button("Delete")
        self.btn_delete.setStyleSheet("""
            QPushButton { background-color: #313940; font-size: 20px; color: #f5f6f4; }
            QPushButton:hover { background-color: #424d56; }
        """)

        top_btns = QHBoxLayout()
        top_btns.setSpacing(6)
        top_btns.addWidget(self.btn_reclassify)
        top_btns.addWidget(self.btn_save)

        btn_vbox.addWidget(self.mic_ctrl, 0, Qt.AlignCenter)
        btn_vbox.addLayout(top_btns)
        btn_vbox.addWidget(self.btn_delete)

        outer.addWidget(btn_area)
        


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
        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #617c32;
                font-size: 22px;
                color: white;
            }
            QPushButton:hover {
                background-color: #f5f6f4;
            }
        """)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #7e1f23;
                font-size: 22px;
                color: white;
            }
            QPushButton:hover {
                background-color: #f5f6f4;
            }
        """)
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setStyleSheet("""
            QPushButton {
                background-color: #95b7dc;
                font-size: 22px;
                color: #385573;
            }
            QPushButton:hover {
                background-color: #f5f6f4;
            }
        """)
        self.btn_reset = QPushButton("Reset Context")
        self.btn_reset.setStyleSheet("""
            QPushButton { background-color: #a88b5c; font-size: 22px; color: white; }
            QPushButton:hover { background-color: #f5f6f4; color: #344f41; }
        """)
        self.btn_save = QPushButton("Save")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #617c32;
                font-size: 22px;
                color: white;
            }
            QPushButton:hover {
                background-color: #f5f6f4;
            }
        """)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("""
            QPushButton {
                background-color: #313940;
                font-size: 22px;
                color: white;
            }
            QPushButton:hover {
                background-color: #f5f6f4;
            }
        """)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("""
            QPushButton { background-color: #313940; font-size: 22px; color: white; }
            QPushButton:hover { background-color: #f5f6f4; color: #344f41; }
        """)
        
        for b in [self.btn_start, self.btn_stop, self.btn_redo, self.btn_save, self.btn_delete, self.btn_reset, self.btn_cancel]:
            b.setMinimumHeight(55)
            row.addWidget(b)

        layout.addLayout(row)


class TripLoadPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        title = QLabel("Trip & Notes")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)

        # --- NEW: Delete All Button ---
        self.btn_delete_all = QPushButton("DELETE ALL")
        self.btn_delete_all.setMinimumHeight(45)
        self.btn_delete_all.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #ff3333;
            }
        """)
        layout.addWidget(self.btn_delete_all)
        # ------------------------------

        self.lbl_totals = QLabel("Total volume: --   Total weight: --")
        self.lbl_totals.setAlignment(Qt.AlignCenter)
        self.lbl_totals.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.lbl_totals)

        timeline_label = QLabel("Timeline:")
        timeline_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(timeline_label)
        
        self.list = QListWidget()
        self.list.setStyleSheet("font-size: 16px;")
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.list, stretch=1)

        self.btn_back = big_button("Back")
        layout.addWidget(self.btn_back)
        
        self._timeline_data = []
        self._summary = None
        
class RockDetailPage(QWidget):
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

        self.lbl_title = QLabel("Rock Detail")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("font-size: 24px; font-weight: 700; border: 2px solid #697d6a; border-radius: 8px; padding: 8px;")
        layout.addWidget(self.lbl_title)

        self.lbl_time = QLabel("")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(self.lbl_time)

        images_row = QHBoxLayout()
        images_row.setSpacing(10)
        self.lbl_top = QLabel()
        self.lbl_top.setAlignment(Qt.AlignCenter)
        # self.lbl_top.setMinimumSize(260, 200)
        self.lbl_top.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_top.setStyleSheet("background-color: #222; border: 3px solid #344f41; border-radius: 6px;")

        self.lbl_side = QLabel()
        self.lbl_side.setAlignment(Qt.AlignCenter)
        # self.lbl_side.setMinimumSize(260, 200)
        self.lbl_side.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_side.setStyleSheet("background-color: #222; border: 3px solid #344f41; border-radius: 6px;")

        images_row.addWidget(self.lbl_top, stretch=1)
        images_row.addWidget(self.lbl_side, stretch=1)
        layout.addLayout(images_row, stretch=2)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size: 18px; border: 2px solid #697d6a; border-radius: 8px; padding: 8px;")
        layout.addWidget(self.lbl_info, stretch=2)

        # --- NEW: Voice Notes Display ---
        self.lbl_notes_title = QLabel("Associated Voice Notes:")
        self.lbl_notes_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(self.lbl_notes_title)

        self.notes_text = QTextEdit()
        self.notes_text.setReadOnly(True)
        self.notes_text.setStyleSheet("""
            font-size: 16px;
            border: 2px solid #cbd2c5;
            border-radius: 8px;
            padding: 8px;
        """)
        layout.addWidget(self.notes_text, stretch=1)

        self.btn_back = big_button("Back")
        layout.addWidget(self.btn_back)

    def set_entry(self, entry, associated_notes=None) -> None:
        dt = datetime.datetime.fromtimestamp(entry.ts) if entry.ts else None
        self.lbl_time.setText(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}" if dt else "Unknown time")

        res = entry.result
        self.lbl_title.setText(f"{res.label.upper()} ({int(res.confidence * 100)}%)")

        w, h = 300, 225

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
        self.lbl_info.setText(f"Volume: {vol_txt}\nWeight: {weight_txt}")

        # --- NEW: Populate Voice Notes ---
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
        self.lbl_title.setStyleSheet("font-size: 24px; font-weight: 700; border: 2px solid #697d6a; border-radius: 8px; padding: 8px;")
        
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

        self.btn_back = big_button("Back")
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
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(10)

        self.trigger_btn = QPushButton("🎤")
        self.trigger_btn.setFixedSize(50, 50)
        self.trigger_btn.setStyleSheet("background-color: #344f41; color: white; border-radius: 25px; font-size: 20px;")
        self.main_layout.addWidget(self.trigger_btn)

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
        self.trigger_btn.clicked.connect(self._toggle_sub_buttons)
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

    def _make_sub_btn(self, text, color):
        btn = QPushButton(text)
        btn.setFixedSize(60, 40)
        btn.setStyleSheet(f"background-color: {color}; color: white; border-radius: 5px; font-size: 12px;")
        return btn

    def _toggle_sub_buttons(self):
        self.button_container.setVisible(not self.button_container.isVisible())

class AppWindow(QMainWindow):
    def __init__(self, vm):
        super().__init__()

        self.session_start_time = time.time()
        
        self.vm = vm
        self.setWindowTitle("SAGE Jetson UI")
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
        self.setCentralWidget(self.stack)

        self.home = HomePage()
        self.loading = LoadingPage()
        self.camera_preview = CameraPreviewPage(self.vm)
        self.capture_review = CaptureReviewPage(self.vm)
        self.classified = ClassifiedPage(self.vm)
        self.voice_loading = VoiceLoadingPage()
        self.voice = VoicePage()
        self.trip = TripLoadPage()
        self.rock_detail = RockDetailPage()
        self.voice_note_detail = VoiceNoteDetailPage()

        self.stack.addWidget(self.home)
        self.stack.addWidget(self.loading)
        self.stack.addWidget(self.camera_preview)
        self.stack.addWidget(self.capture_review)
        self.stack.addWidget(self.classified)
        self.stack.addWidget(self.voice_loading)
        self.stack.addWidget(self.voice)
        self.stack.addWidget(self.trip)
        self.stack.addWidget(self.rock_detail)
        self.stack.addWidget(self.voice_note_detail)

        for i in range(self.stack.count()):
            w = self.stack.widget(i)
            w.setMinimumSize(0, 0)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._wire_ui()
        self._wire_vm()

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

    def _on_reset_context_clicked(self) -> None:
        self.vm.reset_voice_context()
        self.voice.btn_reset.hide()
        
    def _update_voice_buttons(self, mode: str) -> None:
        """Dynamically hides/shows Voice to Text buttons based on the current phase."""
        v = self.voice
        
        # 1. Hide everything first
        for b in [v.btn_start, v.btn_stop, v.btn_redo, v.btn_save, v.btn_delete, v.btn_reset, v.btn_cancel]:
            b.hide()
            
        # 2. Show only what belongs in the current phase
        if mode == "initial":
            v.btn_start.show()
            v.btn_cancel.show()
            # Only show reset if they are currently latched to a specific rock context
            active_rock = getattr(self.vm, "active_rock_id", None)
            if active_rock and active_rock != "ORPHAN":
                v.btn_reset.show()
                
        elif mode == "recording":
            v.btn_stop.setText("Stop")
            v.btn_stop.show()
            
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
        self._shake_timer.start(40) # Update the layout every 40ms

    def _do_shake(self):
        """Bounces the margins of the Rock widgets left and right."""
        self._shake_offset += self._shake_direction * 2
        if abs(self._shake_offset) >= 4:
            self._shake_direction *= -1
            
        for i in range(self.trip.list.count()):
            item_dict = self.trip._timeline_data[i]
            if item_dict["type"] == "rock":
                list_item = self.trip.list.item(i)
                widget = self.trip.list.itemWidget(list_item)
                if widget:
                    # Alternating padding creates the "shake"
                    widget.setContentsMargins(5 + self._shake_offset, 2, 5 - self._shake_offset, 2)

    def _stop_rock_assignment(self):
        """Stops the timer and resets everything back to normal."""
        self._assigning_note_ts = None
        if hasattr(self, "_shake_timer") and self._shake_timer is not None:
            self._shake_timer.stop()
            self._shake_timer = None
            
        if hasattr(self, "trip") and hasattr(self.trip, "list"):
            for i in range(self.trip.list.count()):
                list_item = self.trip.list.item(i)
                widget = self.trip.list.itemWidget(list_item)
                if widget:
                    widget.setContentsMargins(5, 2, 5, 2)
    
    def _on_delete_all_clicked(self) -> None:
        # Create the custom popup
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Delete All Data")
        msg_box.setText("Are you sure? Clicking CONFIRM will delete data from all your past missions. This data will be irretrievable.")
        msg_box.setStyleSheet("QLabel { color: #344f41; font-size: 18px; font-weight: normal; } QMessageBox { background-color: #cbd2c5; }")
        
        # Add the custom buttons
        btn_confirm = msg_box.addButton("CONFIRM", QMessageBox.AcceptRole)
        btn_confirm.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold; padding: 8px;")
        
        btn_cancel = msg_box.addButton("CANCEL", QMessageBox.RejectRole)
        btn_cancel.setStyleSheet("background-color: #95b7dc; color: #385573; font-weight: bold; padding: 8px;")
        
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
        
    def _on_virtual_key_pressed(self, key: str) -> None:
        cursor = self.voice.text.textCursor()
        
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
            
        self.voice.text.setTextCursor(cursor)
        self.voice.text.setFocus()
        self.vm.transcription_text = self.voice.text.toPlainText()
        
        # --- SMART AUTO-SHIFT LOGIC ---
        # Grab all the text leading up to the cursor's current position
        text_so_far = self.voice.text.toPlainText()[:cursor.position()]
        
        if len(text_so_far) == 0:
            # If the box is completely empty, capitalize the first letter!
            self.inline_keyboard.set_shift_state(1)
        elif len(text_so_far) >= 2 and text_so_far[-2:] in [". ", "? ", "! "]:
            # If the last two characters were punctuation + space, capitalize!
            self.inline_keyboard.set_shift_state(1)
        elif len(text_so_far) >= 1 and text_so_far[-1] == "\n":
            # If they just went to a new line, capitalize!
            self.inline_keyboard.set_shift_state(1)

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
        self.trip.btn_delete_all.clicked.connect(self._on_delete_all_clicked)
        # --- NEW: Unified Timeline Click ---
        self.trip.list.itemClicked.connect(self._on_timeline_clicked)
        self.rock_detail.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.trip))
        self.voice_note_detail.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.trip))
        
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
        self.vm.trip_changed.connect(self._on_trip)
        self.vm.error.connect(self._on_error)
        self.vm.recording_status_changed.connect(self._on_recording_status_changed)
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
            # FIX: Check vtt_active instead of the background process state!
            if not getattr(self.vm, 'vtt_active', False):
                self.voice.text.clear()
                self.vm.transcription_text = ""  # Wipes ghost text from memory!
                
                # Reset the little camera-preview voice button
                self.camera_preview.mic_ctrl.trigger_btn.setText("🎤")
                self.camera_preview.mic_ctrl.trigger_btn.setStyleSheet(
                    "background-color: #344f41; color: white; border-radius: 25px; font-size: 20px;"
                )

        elif state == AppStateType.CAMERA_PREVIEW:
            # From HEAD: Updated label text
            self.camera_preview.lbl_step.setText("Capture First View")
            self.vm.start_camera_stream(0, 0, 0, 0)

        elif state == AppStateType.CONFIRM_CAPTURES:
            top_path, side_path = self.vm.get_review_image_paths()
            self.capture_review.set_images(top_path, side_path)

        elif state == AppStateType.VOICE_TO_TEXT:
            self.inline_keyboard.hide()       
            self.inline_keyboard.set_shift_state(0)
            self.voice.text.setReadOnly(True)
            
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
            "high":      "#617c32",
            "medium":    "#a88b5c",
            "low":       "#c46200",
            "uncertain": "#888888",
        }
        if result.tier and result.tier in tier_colors:
            bg = tier_colors[result.tier]
            self.classified.lbl_tier.setText(result.tier.upper())
            self.classified.lbl_tier.setStyleSheet(
                f"font-size: 15px; font-weight: 700; border-radius: 6px; "
                f"padding: 2px 8px; color: #f5f6f4; background-color: {bg};"
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
                    sep.setStyleSheet("background-color: #c0c8bb; max-height: 1px;")
                    feat_layout.addWidget(sep)
                first_feat = False
                conf_pct = int(feat_data.get("confidence", 0.0) * 100)
                display_name = feat_name.replace("_", " ").title()
                feat_row_w = QWidget()
                feat_row = QHBoxLayout(feat_row_w)
                feat_row.setContentsMargins(0, 8, 0, 4)
                feat_row.setSpacing(6)
                lbl_feat_name = QLabel(display_name)
                lbl_feat_name.setStyleSheet("font-size: 18px; color: #344f41; font-weight: 700;")
                lbl_feat_val = QLabel(str(feat_data.get("value", "")))
                lbl_feat_val.setStyleSheet("font-size: 18px;")
                lbl_feat_conf = QLabel(f"conf. {conf_pct}%")
                lbl_feat_conf.setStyleSheet("font-size: 15px; color: #697d6a;")
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
                        "font-size: 15px; color: #697d6a; padding: 0px 0px 4px 0px;"
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
            chip.setStyleSheet("""
                QPushButton {
                    background-color: #e0e8d8; color: #344f41;
                    border: 1px solid #697d6a; border-radius: 4px;
                    font-size: 15px; padding: 3px 10px;
                }
                QPushButton:checked {
                    background-color: #617c32; color: #f5f6f4;
                    border-color: #617c32;
                }
                QPushButton:focus { border: 2px solid #344f41; }
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

    def _on_trip(self, summary: TripSummary) -> None:
        self._stop_rock_assignment()
        self.trip.list.clear()
        self.trip._timeline_data = []
        self.trip._summary = summary
        
        rocks_sorted = sorted(summary.rocks, key=lambda r: r.ts)
        
        combined = []
        for r in summary.rocks:
            combined.append({"type": "rock", "ts": r.ts, "data": r})
            
        for n in summary.voice_notes:
            note_ts = n.get("ts", 0)
            note_session = n.get("session_id")
            explicit_rock_id = n.get("rock_id")
            
            owning_rock = None
            if explicit_rock_id:
                for r in summary.rocks:
                    if r.rock_id == explicit_rock_id:
                        owning_rock = r
                        break
            else:
                for r in reversed(rocks_sorted):
                    if r.ts <= note_ts:
                        owning_rock = r
                        break
                    
            is_orphan = False
            if not owning_rock:
                is_orphan = True
            elif explicit_rock_id is None and note_session != getattr(self, "session_start_time", 0) and owning_rock.ts < getattr(self, "session_start_time", 0):
                is_orphan = True
                
            if is_orphan:
                combined.append({"type": "voice", "ts": note_ts, "data": n})
            
        combined.sort(key=lambda x: x["ts"], reverse=True)
        
        from PySide6.QtWidgets import QListWidgetItem
        
        for item in combined:
            ts = item["ts"]
            dt = datetime.datetime.fromtimestamp(ts) if ts else None
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "Unknown"
            
            if item["type"] == "rock":
                r = item["data"]
                label = r.result.label
                conf = int(r.result.confidence * 100)
                # Shrunk timestamp, grayed out slightly, removed "ROCK:" and bolded the label
                display_text = f"🪨 <span style='font-size: 13px; color: #555;'>[{time_str}]</span>&nbsp;&nbsp;<b>{label.upper()}</b> ({conf}%)"
            else:
                n = item["data"]
                cleaned = n.get("cleaned", n.get("transcript", ""))
                # Cut the preview length slightly shorter to prevent overflow
                short_text = cleaned[:60] + "..." if len(cleaned) > 60 else cleaned
                # Shrunk timestamp, grayed out slightly, removed "NOTE:"
                display_text = f"🎤 <span style='font-size: 13px; color: #555;'>[{time_str}]</span>&nbsp;&nbsp;{short_text}"
                
            list_item = QListWidgetItem()
            self.trip.list.addItem(list_item)
            
            widget = QWidget()
            row_layout = QHBoxLayout(widget)
            row_layout.setContentsMargins(5, 2, 5, 2)
            
            # --- FIX: Using a Button designed for native QMenus ---
            dot_btn = QPushButton("⋮")
            dot_btn.setFixedSize(30, 30)
            # We hide the default dropdown arrow using `menu-indicator`
            dot_btn.setStyleSheet("""
                QPushButton { font-size: 24px; font-weight: bold; border: none; background: transparent; color: #344f41; }
                QPushButton::menu-indicator { image: none; width: 0px; }
            """)
            
            self._attach_menu_to_button(dot_btn, item)
            
            lbl = QLabel(display_text)
            lbl.setStyleSheet("font-size: 16px;")
            lbl.setAttribute(Qt.WA_TransparentForMouseEvents) 
            
            # Dots on the Left, Text on the Right
            row_layout.addWidget(dot_btn)         
            row_layout.addWidget(lbl, stretch=1)  
            
            list_item.setSizeHint(widget.sizeHint())
            self.trip.list.setItemWidget(list_item, widget)
            self.trip._timeline_data.append(item)

        total_vol = f"{summary.total_volume:.2f}" if summary.total_volume else "0.00"
        total_wt = f"{summary.total_weight:.2f}" if summary.total_weight else "0.00"
        self.trip.lbl_totals.setText(f"Total volume: {total_vol} cm³   Total weight: {total_wt} kg")

    def _attach_menu_to_button(self, button, item_dict):
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
            
        # --- NEW: Assignment button for Voice Notes ---
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
        button.setMenu(menu)
    
    def _delete_timeline_item(self, item_dict):
        if item_dict["type"] == "voice":
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Delete Voice Note")
            msg_box.setText("Are you sure? Clicking CONFIRM will delete this data permanently.")
            msg_box.setStyleSheet("QLabel { color: #344f41; font-size: 18px; font-weight: normal; } QMessageBox { background-color: #cbd2c5; }")
            
            btn_cancel = msg_box.addButton("CANCEL", QMessageBox.RejectRole)
            btn_cancel.setStyleSheet("background-color: #95b7dc; color: #385573; font-weight: bold; padding: 8px;")
            
            btn_confirm = msg_box.addButton("CONFIRM", QMessageBox.AcceptRole)
            btn_confirm.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold; padding: 8px;")
            
            msg_box.exec()
            
            if msg_box.clickedButton() == btn_confirm:
                self.vm.delete_voice_note_by_ts(item_dict["data"].get("ts"))
                
        elif item_dict["type"] == "rock":
            # --- NEW: Custom Dialog for Vertical Stacking ---
            dialog = QDialog(self)
            dialog.setWindowTitle("Delete Classification")
            dialog.setStyleSheet("QDialog { background-color: #cbd2c5; }")
            layout = QVBoxLayout(dialog)
            
            msg = QLabel("Are you sure? How would you like to delete this rock?")
            msg.setStyleSheet("color: #344f41; font-size: 18px; font-weight: normal;")
            msg.setWordWrap(True)
            layout.addWidget(msg)
            layout.addSpacing(10)
            
            btn_both = QPushButton("DELETE ROCK && ALL NOTES")
            btn_both.setStyleSheet("background-color: #7e1f23; color: white; font-weight: bold; padding: 15px; font-size: 16px; border-radius: 6px;")
            
            btn_only = QPushButton("DELETE ROCK ONLY")
            btn_only.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold; padding: 15px; font-size: 16px; border-radius: 6px;")
            
            btn_cancel = QPushButton("CANCEL")
            btn_cancel.setStyleSheet("background-color: #95b7dc; color: #385573; font-weight: bold; padding: 15px; font-size: 16px; border-radius: 6px;")
            
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
                rocks_sorted = sorted(self.trip._summary.rocks, key=lambda r: r.ts)
                
                next_rock_ts = float('inf')
                for r in rocks_sorted:
                    if r.ts > entry.ts:
                        next_rock_ts = r.ts
                        break
                        
                for n in self.trip._summary.voice_notes:
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

    def _on_timeline_clicked(self, item) -> None:
        index = self.trip.list.row(item)
        if 0 <= index < len(self.trip._timeline_data):
            item_dict = self.trip._timeline_data[index]
            
            # --- NEW: Intercept clicks if we are in assignment mode ---
            if getattr(self, "_assigning_note_ts", None) is not None:
                if item_dict["type"] == "rock":
                    # They clicked a rock target! Link the note.
                    self.vm.assign_note_to_rock(self._assigning_note_ts, item_dict["data"].rock_id)
                
                # Turn off the jiggle regardless of what they clicked
                self._stop_rock_assignment()
                return
            # ----------------------------------------------------------
            
            if item_dict["type"] == "rock":
                entry = item_dict["data"]
                associated_notes = []
                rocks_sorted = sorted(self.trip._summary.rocks, key=lambda r: r.ts)
                
                next_rock_ts = float('inf')
                for r in rocks_sorted:
                    if r.ts > entry.ts:
                        next_rock_ts = r.ts
                        break
                        
                for n in self.trip._summary.voice_notes:
                    note_ts = n.get("ts", 0)
                    explicit_rock_id = n.get("rock_id")
                    
                    if explicit_rock_id == entry.rock_id:
                        associated_notes.append(n)
                    elif explicit_rock_id is None:
                        if entry.ts <= note_ts < next_rock_ts:
                            if n.get("session_id") == entry.session_id:
                                associated_notes.append(n)
                                
                associated_notes.sort(key=lambda x: x.get("ts", 0))
                
                self.rock_detail.set_entry(entry, associated_notes)
                self.stack.setCurrentWidget(self.rock_detail)
                
            else:
                note = item_dict["data"]
                self.voice_note_detail.set_note(note)
                self.stack.setCurrentWidget(self.voice_note_detail)

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
                self._update_voice_buttons("review")
            

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
                



