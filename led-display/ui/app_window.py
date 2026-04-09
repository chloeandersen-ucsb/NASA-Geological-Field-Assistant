from __future__ import annotations
import os
import sys
import datetime
from pathlib import Path
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QFont
from PySide6.QtGui import QImage

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
img_path = project_root/ "led-display" / "ui" / "sage-logo-wcbg.png"


from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor, QKeyEvent, QShortcut, QKeySequence, QPainter, QPen, QColor, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QMessageBox,
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QHBoxLayout, QSizePolicy, QDialog
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

        # font = QFont("Arial", 18)
        # font.setBold(True)
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
        # layout.addStretch(1)
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
        #self.btn_cancel.setStyleSheet("font-size: 18px;")
        
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


class ClassifiedPage(QWidget):
    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        # Main layout for the whole page
        self.main_layout = QVBoxLayout(self)
        # self.main_layout.setSpacing(5)
        # self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.addStretch(1)
        
        self.image_container = QWidget()
        # self.image_container.setFixedSize(600, 900)
        self.image_container.setStyleSheet("background-color: transparent;")
        container_layout = QHBoxLayout(self.image_container)
        # container_layout.addSpacing(50)
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: transparent; border: none;")
        self.lbl_image.setFixedSize(200, 150)
        self.lbl_image.setScaledContents(True)
        self.lbl_side_image = QLabel()
        self.lbl_side_image.setAlignment(Qt.AlignCenter)
        self.lbl_side_image.setStyleSheet("background-color: transparent; border: none;")
        self.lbl_side_image.setFixedSize(200, 150)
        self.lbl_side_image.setScaledContents(True)
        container_layout.addWidget(self.lbl_image)
        container_layout.addWidget(self.lbl_side_image)
        self.main_layout.addWidget(self.image_container, 0, Qt.AlignCenter)
        self.main_layout.addStretch(1)

        # Result Labels
        self.lbl_label = QLabel("LABEL")
        self.lbl_label.setAlignment(Qt.AlignCenter)
        self.lbl_label.setStyleSheet("font-size: 28px; font-weight: 700;")
        self.lbl_label.setStyleSheet("font-size: 24px; font-weight: 700; margin-top: 5px;")
        

        self.lbl_conf = QLabel("Confidence: --")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setStyleSheet("font-size: 25px; color: #666;") #color = white

        self.lbl_volume = QLabel("Volume = --")
        self.lbl_volume.setAlignment(Qt.AlignCenter)
        self.lbl_volume.setStyleSheet("font-size: 20px;")

        # self.lbl_top2 = QLabel("")
        # self.lbl_top2.setAlignment(Qt.AlignCenter)
        # self.lbl_top2.setStyleSheet("font-size: 20px;")

        # self.lbl_top3 = QLabel("")
        # self.lbl_top3.setAlignment(Qt.AlignCenter)
        # self.lbl_top3.setStyleSheet("font-size: 20px;")

        self.extra_results_widget = QWidget()
        self.extra_grid = QGridLayout(self.extra_results_widget)
        self.extra_grid.setAlignment(Qt.AlignCenter)
        # self.extra_grid.setContentsMargins(100, 0, 100, 0) # Adjust margins to control width
        
        # Create the sub-labels
        self.lbl_rank2 = QLabel(""); self.lbl_name2 = QLabel(""); self.lbl_perc2 = QLabel("")
        self.lbl_rank3 = QLabel(""); self.lbl_name3 = QLabel(""); self.lbl_perc3 = QLabel("")

        # Style and Add to Grid
        sub_style = "font-size: 20px;"
        for i, lbl in enumerate([self.lbl_rank2, self.lbl_name2, self.lbl_perc2, 
                                 self.lbl_rank3, self.lbl_name3, self.lbl_perc3]):
            lbl.setStyleSheet(sub_style)
            # Row 0 for 2nd, Row 1 for 3rd
            row = i // 3
            col = i % 3
            alignment = [Qt.AlignLeft, Qt.AlignCenter, Qt.AlignRight][col]
            self.extra_grid.addWidget(lbl, row, col, alignment)

        self.lbl_extra = QLabel("")
        self.lbl_extra.setAlignment(Qt.AlignCenter)
        self.lbl_extra.setStyleSheet("font-size: 20px;")

        self.mic_ctrl = ExpandingVoiceWidget(self.vm, self) 

        self.btn_reclassify = big_button("Reclassify")
        self.btn_reclassify.setStyleSheet("""
            QPushButton {
                background-color: #95b7dc;
                font-size: 22px;
                color: #385573;
            }
            QPushButton:hover {
                background-color: #f5f6f4;
            }
        """)
        self.btn_save = big_button("Save Classification")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #617c32;
                font-size: 22px;
                color: #f5f6f4;
            }
            QPushButton:hover {
                background-color: #f5f6f4;
            }
        """)
        self.btn_delete = big_button("Delete")
        self.btn_delete.setStyleSheet("""
            QPushButton {
                background-color: #313940;
                font-size: 22px;
                color: #f5f6f4;
            }
            QPushButton:hover {
                background-color: #f5f6f4;
            }
        """)

        # Assemble main layout
        self.main_layout.addWidget(self.lbl_label)
        self.main_layout.addWidget(self.lbl_conf)
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(self.lbl_volume)
        self.main_layout.addWidget(self.lbl_extra)
        self.main_layout.addSpacing(10)

        self.main_layout.addWidget(self.extra_results_widget)
        self.main_layout.addStretch(1)
        

        self.main_layout.addWidget(self.mic_ctrl, 0, Qt.AlignCenter)
        self.main_layout.addSpacing(15)

        self.main_layout.addWidget(self.btn_reclassify)
        self.main_layout.addWidget(self.btn_save)
        self.main_layout.addWidget(self.btn_delete)
        


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
        # title.setStyleSheet("font-size: 22px; font-weight: 600;")
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
        for b in [self.btn_start, self.btn_stop, self.btn_redo, self.btn_save, self.btn_delete, self.btn_reset]:
            b.setMinimumHeight(55)
            # b.setStyleSheet("font-size: 20px;")
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
        layout.addWidget(self.list, stretch=1)

        
        notes_label = QLabel("Voice Notes:")
        notes_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(notes_label)
        
        self.notes_list = QListWidget()
        self.notes_list.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.notes_list, stretch=1)

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
        self.setMouseTracking(True)
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

    def enterEvent(self, event):
        self.button_container.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.button_container.hide()
        super().leaveEvent(event)

class AppWindow(QMainWindow):
    def __init__(self, vm):
        super().__init__()

        import time
        self.session_start_time = time.time()
        
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
        
    def _on_rock_clicked(self, item) -> None:
        index = self.trip.list.row(item)
        summary = self.vm.store.list_rocks()
        if 0 <= index < len(summary):
            entry = summary[index]
            self.rock_detail.set_entry(entry)
            self.stack.setCurrentWidget(self.rock_detail)

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
        # self.voice.btn_stop.clicked.connect(self.vm.stop_voice_to_text)
        #self.voice.btn_stop.clicked.connect(self.vm.stop_voice_to_text)
        self.voice.btn_stop.clicked.connect(self._on_stop_or_edit_clicked)
        self.voice.btn_redo.clicked.connect(self.vm.redo_voice_to_text)
        self.voice.btn_save.clicked.connect(self.vm.save_transcription)
        self.voice.btn_delete.clicked.connect(self.vm.delete_transcription)
        self.voice.btn_reset.clicked.connect(self._on_reset_context_clicked)
        

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

        # 2. Handle state-specific setup and cleanup
        if state == AppStateType.HOME:
            # If we return Home, and the mic is off, aggressively factory-reset the Voice module
            if not getattr(self.vm.transcriber, 'is_recording', False):
                self.voice.text.clear()
                self.vm.transcription_text = ""  # Wipes ghost text from memory!
                
                # Reset the little camera-preview voice button
                self.camera_preview.voice_ctrl.trigger_btn.setText("🎤")
                self.camera_preview.voice_ctrl.trigger_btn.setStyleSheet(
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
            # From ilai-macbook branch: Reset UI elements for safety
            self.inline_keyboard.hide()       
            self.inline_keyboard.set_shift_state(0)
            self.voice.btn_stop.setText("Stop")      
            self.voice.text.setReadOnly(True)
            
            # From HEAD: Load text directly from VM using a local variable
            current_text = self.vm.transcription_text
            self.voice.text.setPlainText(current_text)
            self.voice.btn_save.setEnabled(bool(current_text.strip()))
            
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

        self.classified.lbl_label.setText(result.label.upper())
        self.classified.lbl_conf.setText(f"Confidence: {int(result.confidence * 100)}%")

        top3 = None
        if result.raw and isinstance(result.raw, dict):
            top3 = result.raw.get("top3", [])
        
        if top3 and len(top3) >= 2:
            label2 = top3[1].get("label", "")
            conf2 = float(top3[1].get("confidence", 0.0))
            if conf2 > 0:
                self.classified.lbl_rank2.setText("2nd:")
                self.classified.lbl_name2.setText(top3[1].get("label", "").upper())
                self.classified.lbl_perc2.setText(f"({int(float(top3[1]['confidence']) * 100)}%)")
            else:
                for lbl in [self.classified.lbl_rank2, self.classified.lbl_name2, self.classified.lbl_perc2]: lbl.setText("")
        else:
            for lbl in [self.classified.lbl_rank2, self.classified.lbl_name2, self.classified.lbl_perc2]: lbl.setText("")
        
        if top3 and len(top3) >= 3:
            label3 = top3[2].get("label", "")
            conf3 = float(top3[2].get("confidence", 0.0))
            if conf3 > 0:
                self.classified.lbl_rank3.setText("3rd:")
                self.classified.lbl_name3.setText(top3[2].get("label", "").upper())
                self.classified.lbl_perc3.setText(f"({int(float(top3[2]['confidence']) * 100)}%)")
            else:
                for lbl in [self.classified.lbl_rank3, self.classified.lbl_name3, self.classified.lbl_perc3]: lbl.setText("")
        else:
            for lbl in [self.classified.lbl_rank3, self.classified.lbl_name3, self.classified.lbl_perc3]: lbl.setText("")

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

    # def _on_transcription_ready(self):
    #     self.state_changed.emit(AppStateType.VOICE_TO_TEXT)

    
        self.trip.list.clear()
        self.trip._timeline_data = []
        self.trip._summary = summary
        
        rocks_sorted = sorted(summary.rocks, key=lambda r: r.ts)
        
        # 1. Combine rocks and ONLY orphan voice notes into one unified list
        combined = []
        for r in summary.rocks:
            combined.append({"type": "rock", "ts": r.ts, "data": r})
            
        for n in summary.voice_notes:
            note_ts = n.get("ts", 0)
            note_session = n.get("session_id")
            
            # Find the most recent rock BEFORE this note
            owning_rock = None
            for r in reversed(rocks_sorted):
                if r.ts <= note_ts:
                    owning_rock = r
                    break
                    
            is_orphan = False
            if not owning_rock:
                # Before any rocks were ever taken
                is_orphan = True
            elif note_session != owning_rock.session_id:
                # STRICT BOUNDARY: If they don't share the exact same launch ID, it's an orphan!
                is_orphan = True
                
            if is_orphan:
                combined.append({"type": "voice", "ts": note_ts, "data": n})
            
        # 2. Sort REVERSE chronologically (Newest at the top)
        combined.sort(key=lambda x: x["ts"], reverse=True)
        
        # 3. Populate the UI List
        for item in combined:
            ts = item["ts"]
            dt = datetime.datetime.fromtimestamp(ts) if ts else None
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "Unknown"
            
            if item["type"] == "rock":
                r = item["data"]
                label = r.result.label
                conf = int(r.result.confidence * 100)
                display_text = f"🪨 [{time_str}] ROCK: {label.upper()} ({conf}%)"
            else:
                n = item["data"]
                cleaned = n.get("cleaned", n.get("transcript", ""))
                short_text = cleaned[:80] + "..." if len(cleaned) > 80 else cleaned
                display_text = f"🎤 [{time_str}] NOTE: {short_text}"
                
            self.trip.list.addItem(display_text)
            self.trip._timeline_data.append(item)

        total_vol = f"{summary.total_volume:.2f}" if summary.total_volume else "0.00"
        total_wt = f"{summary.total_weight:.2f}" if summary.total_weight else "0.00"
        self.trip.lbl_totals.setText(f"Total volume: {total_vol} cm³   Total weight: {total_wt} kg")

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
                display_text = f"🪨 [{time_str}] ROCK: {label.upper()} ({conf}%)"
            else:
                n = item["data"]
                cleaned = n.get("cleaned", n.get("transcript", ""))
                short_text = cleaned[:80] + "..." if len(cleaned) > 80 else cleaned
                display_text = f"🎤 [{time_str}] NOTE: {short_text}"
                
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
        if item_dict["type"] == "rock":
            self.vm.delete_rock_by_id(item_dict["data"].rock_id)
        else:
            self.vm.delete_voice_note_by_ts(item_dict["data"].get("ts"))

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
        # pass
    # Update the hover widget (as we did before)
        self.camera_preview.voice_ctrl._update_ui_state(is_recording)
        
        if is_recording:
            self.voice.btn_stop.setText("Stop")
            self.inline_keyboard.hide()
            self.voice.text.setReadOnly(True)
        # If recording has stopped, explicitly clear the VoicePage text box
        if not is_recording:
            self.voice.btn_stop.setText("Edit")
            self.voice.text.clear() 
            self.voice.btn_save.setEnabled(False)
            

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
                



