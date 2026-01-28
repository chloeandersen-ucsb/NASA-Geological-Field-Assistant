from __future__ import annotations
import os
import sys
from pathlib import Path

# Add project root to path to import connector
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QMessageBox,
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QHBoxLayout
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

        title = QLabel("SAGE")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: 600;")
        layout.addWidget(title)

        self.btn_classify = big_button("Classify Rock")
        self.btn_voice = big_button("Voice to Text")
        self.btn_trip = big_button("Current Trip Load")

        layout.addWidget(self.btn_classify)
        layout.addWidget(self.btn_voice)
        layout.addWidget(self.btn_trip)
        layout.addStretch(1)


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


class ClassifiedPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.lbl_label = QLabel("LABEL")
        self.lbl_label.setAlignment(Qt.AlignCenter)
        self.lbl_label.setStyleSheet("font-size: 24px; font-weight: 700;")

        self.lbl_conf = QLabel("Confidence: --")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setStyleSheet("font-size: 18px;")

        self.lbl_extra = QLabel("")  # volume/weight optional
        self.lbl_extra.setAlignment(Qt.AlignCenter)
        self.lbl_extra.setStyleSheet("font-size: 16px;")

        self.btn_reclassify = big_button("Reclassify")
        self.btn_save = big_button("Save Rock to Trip")
        self.btn_delete = big_button("Delete / Back Home")

        layout.addStretch(1)
        layout.addWidget(self.lbl_label)
        layout.addWidget(self.lbl_conf)
        layout.addWidget(self.lbl_extra)
        layout.addSpacing(10)
        layout.addWidget(self.btn_reclassify)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_delete)
        layout.addStretch(1)


class VoicePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        title = QLabel("Voice to Text")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.text, stretch=1)

        row = QHBoxLayout()
        self.btn_stop = QPushButton("Stop")
        self.btn_redo = QPushButton("Redo")
        self.btn_save = QPushButton("Save")
        self.btn_delete = QPushButton("Delete")
        for b in [self.btn_stop, self.btn_redo, self.btn_save, self.btn_delete]:
            b.setMinimumHeight(55)
            b.setStyleSheet("font-size: 16px;")
            row.addWidget(b)

        layout.addLayout(row)


class TripLoadPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        title = QLabel("Current Trip Load")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)

        self.lbl_totals = QLabel("Total volume: --   Total weight: --")
        self.lbl_totals.setAlignment(Qt.AlignCenter)
        self.lbl_totals.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.lbl_totals)

        self.list = QListWidget()
        self.list.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.list, stretch=1)

        self.btn_back = big_button("Back")
        layout.addWidget(self.btn_back)


class AppWindow(QMainWindow):
    def __init__(self, vm):
        super().__init__()
        import sys
        print("DEBUG: AppWindow.__init__() called", file=sys.stderr)
        
        self.vm = vm
        self.setWindowTitle("SAGE Jetson UI")
        print("DEBUG: Window title set", file=sys.stderr)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        print("DEBUG: Stack widget created and set as central widget", file=sys.stderr)

        print("DEBUG: Creating page widgets...", file=sys.stderr)
        self.home = HomePage()
        self.loading = LoadingPage()
        self.classified = ClassifiedPage()
        self.voice = VoicePage()
        self.trip = TripLoadPage()
        print("DEBUG: All page widgets created", file=sys.stderr)

        self.stack.addWidget(self.home)       # index 0
        self.stack.addWidget(self.loading)    # index 1
        self.stack.addWidget(self.classified) # index 2
        self.stack.addWidget(self.voice)      # index 3
        self.stack.addWidget(self.trip)       # index 4
        print("DEBUG: All widgets added to stack", file=sys.stderr)

        print("DEBUG: Wiring UI and VM...", file=sys.stderr)
        self._wire_ui()
        self._wire_vm()
        print("DEBUG: UI and VM wired", file=sys.stderr)

        print("DEBUG: Showing HOME state...", file=sys.stderr)
        self._show_state(AppStateType.HOME)

        # Always fullscreen on Jetson, windowed mode on other platforms
        is_jetson = connector.is_jetson()
        print(f"DEBUG: is_jetson = {is_jetson}", file=sys.stderr)
        
        if is_jetson:
            print("DEBUG: Jetson detected - setting up fullscreen...", file=sys.stderr)
            try:
                # Set window flags for kiosk mode (no decorations, always on top)
                self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
                print("DEBUG: Window flags set", file=sys.stderr)
                self.showFullScreen()
                print("DEBUG: showFullScreen() called", file=sys.stderr)
                print(f"DEBUG: Window visible after fullscreen: {self.isVisible()}", file=sys.stderr)
                print(f"DEBUG: Window geometry: {self.geometry()}", file=sys.stderr)
            except Exception as e:
                print(f"ERROR: Failed to show fullscreen: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                # Fallback to windowed mode
                self.resize(800, 600)
                self.show()
                print("DEBUG: Fallback to windowed mode", file=sys.stderr)
        else:
            # Windowed mode for development/testing
            print("DEBUG: Non-Jetson platform - using windowed mode", file=sys.stderr)
            self.resize(800, 600)
            self.show()
            print(f"DEBUG: Window shown in windowed mode, visible: {self.isVisible()}", file=sys.stderr)
        
        print("DEBUG: AppWindow initialization complete", file=sys.stderr)

    def _wire_ui(self) -> None:
        # Home
        self.home.btn_classify.clicked.connect(self.vm.start_classification)
        self.home.btn_voice.clicked.connect(self.vm.start_voice_to_text)
        self.home.btn_trip.clicked.connect(self.vm.open_trip_load)

        # Classified
        self.classified.btn_reclassify.clicked.connect(self.vm.reclassify)
        self.classified.btn_save.clicked.connect(self.vm.save_classification)
        self.classified.btn_delete.clicked.connect(self.vm.delete_classification)

        # Voice
        self.voice.btn_stop.clicked.connect(self.vm.stop_voice_to_text)
        self.voice.btn_redo.clicked.connect(self.vm.redo_voice_to_text)
        self.voice.btn_save.clicked.connect(self.vm.save_transcription)
        self.voice.btn_delete.clicked.connect(self.vm.delete_transcription)

        # Trip
        self.trip.btn_back.clicked.connect(self.vm.go_home)

    def _wire_vm(self) -> None:
        self.vm.state_changed.connect(self._show_state)
        self.vm.classification_changed.connect(self._on_classification)
        self.vm.transcription_changed.connect(self._on_transcription)
        self.vm.trip_changed.connect(self._on_trip)
        self.vm.error.connect(self._on_error)

    def _show_state(self, state: AppStateType) -> None:
        if state == AppStateType.HOME:
            self.stack.setCurrentWidget(self.home)
        elif state == AppStateType.CLASSIFYING:
            self.stack.setCurrentWidget(self.loading)
        elif state == AppStateType.CLASSIFIED:
            self.stack.setCurrentWidget(self.classified)
        elif state == AppStateType.VOICE_TO_TEXT:
            self.stack.setCurrentWidget(self.voice)
        elif state == AppStateType.TRIP_LOAD:
            self.stack.setCurrentWidget(self.trip)

    def _on_classification(self, result: ClassificationResult) -> None:
        self.classified.lbl_label.setText(result.label.upper())
        self.classified.lbl_conf.setText(f"Confidence: {int(result.confidence * 100)}%")

        extras = []
        if result.estimated_volume is not None:
            extras.append(f"Volume: {result.estimated_volume}")
        if result.estimated_weight is not None:
            extras.append(f"Weight: {result.estimated_weight}")
        self.classified.lbl_extra.setText("   ".join(extras))

    def _on_transcription(self, text: str) -> None:
        self.voice.text.setPlainText(text)
        cursor = self.voice.text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.voice.text.setTextCursor(cursor)

    def _on_trip(self, summary: TripSummary) -> None:
        self.trip.list.clear()
        for r in summary.rocks:
            label = r.result.label
            conf = int(r.result.confidence * 100)
            item = f"{label} ({conf}%)"
            self.trip.list.addItem(item)
        self.trip.lbl_totals.setText(
            f"Total volume: {summary.total_volume:.2f}   Total weight: {summary.total_weight:.2f}"
        )

    def _on_error(self, message: str) -> None:
        # MVP: pop a modal, then ViewModel returns to HOME
        QMessageBox.warning(self, "Error", message)
