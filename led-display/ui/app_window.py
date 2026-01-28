from __future__ import annotations
import os
import sys
import datetime
from pathlib import Path

# Add project root to path to import connector
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QKeyEvent, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QMessageBox,
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QHBoxLayout, QDialog,
    QDialogButtonBox
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
        self.btn_trip = big_button("Trip & Notes")
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.setMinimumHeight(50)
        self.btn_quit.setStyleSheet("font-size: 16px;")

        layout.addWidget(self.btn_classify)
        layout.addWidget(self.btn_voice)
        layout.addWidget(self.btn_trip)
        layout.addStretch(1)
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
    
    def set_message(self, message: str) -> None:
        """Update the loading message."""
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QLabel):
                item.widget().setText(message)
                break


class VoiceLoadingPage(QWidget):
    """Loading page specifically for voice-to-text initialization."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel("Initializing voice transcription…")
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

        title = QLabel("Trip & Notes")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)

        self.lbl_totals = QLabel("Total volume: --   Total weight: --")
        self.lbl_totals.setAlignment(Qt.AlignCenter)
        self.lbl_totals.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.lbl_totals)

        # Rocks section
        rocks_label = QLabel("Rocks:")
        rocks_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(rocks_label)
        
        self.list = QListWidget()
        self.list.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.list, stretch=1)

        # Voice notes section
        notes_label = QLabel("Voice Notes:")
        notes_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(notes_label)
        
        self.notes_list = QListWidget()
        self.notes_list.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.notes_list, stretch=1)

        self.btn_back = big_button("Back")
        layout.addWidget(self.btn_back)
        
        # Store voice notes data for access when clicked
        self._voice_notes_data = []


class AppWindow(QMainWindow):
    def __init__(self, vm):
        super().__init__()
        
        self.vm = vm
        self.setWindowTitle("SAGE Jetson UI")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home = HomePage()
        self.loading = LoadingPage()
        self.classified = ClassifiedPage()
        self.voice_loading = VoiceLoadingPage()
        self.voice = VoicePage()
        self.trip = TripLoadPage()

        self.stack.addWidget(self.home)          # index 0
        self.stack.addWidget(self.loading)       # index 1
        self.stack.addWidget(self.classified)    # index 2
        self.stack.addWidget(self.voice_loading) # index 3
        self.stack.addWidget(self.voice)         # index 4
        self.stack.addWidget(self.trip)          # index 5

        self._wire_ui()
        self._wire_vm()

        self._show_state(AppStateType.HOME)

        # Always fullscreen on Jetson, windowed mode on other platforms
        is_jetson = connector.is_jetson()
        
        if is_jetson:
            try:
                # Set window flags for kiosk mode (no decorations, always on top)
                self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
                self.showFullScreen()
            except Exception as e:
                import sys
                print(f"ERROR: Failed to show fullscreen: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                # Fallback to windowed mode
                self.resize(800, 600)
                self.show()
        else:
            # Windowed mode for development/testing
            self.resize(800, 600)
            self.show()
        
        # Set up keyboard shortcuts
        self._setup_shortcuts()

    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts for the application."""
        # F11 - Toggle fullscreen
        shortcut_f11 = QShortcut(QKeySequence(Qt.Key_F11), self)
        shortcut_f11.activated.connect(self._toggle_fullscreen)
        
        # Escape - Exit fullscreen (only if in fullscreen)
        shortcut_escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        shortcut_escape.activated.connect(self._exit_fullscreen)
        
        # Ctrl+C - Quit application
        shortcut_quit = QShortcut(QKeySequence("Ctrl+C"), self)
        shortcut_quit.activated.connect(self._quit_application)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events."""
        # Handle F11 for fullscreen toggle
        if event.key() == Qt.Key_F11:
            self._toggle_fullscreen()
            event.accept()
            return
        
        # Handle Escape to exit fullscreen
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self._exit_fullscreen()
                event.accept()
                return
        
        # Handle Ctrl+C to quit
        if event.key() == Qt.Key_C and event.modifiers() == Qt.ControlModifier:
            self._quit_application()
            event.accept()
            return
        
        super().keyPressEvent(event)

    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _exit_fullscreen(self) -> None:
        """Exit fullscreen mode if currently in fullscreen."""
        if self.isFullScreen():
            self.showNormal()

    def _quit_application(self) -> None:
        """Quit the application."""
        self.close()

    def _wire_ui(self) -> None:
        # Home
        self.home.btn_classify.clicked.connect(self.vm.start_classification)
        self.home.btn_voice.clicked.connect(self.vm.start_voice_to_text)
        self.home.btn_trip.clicked.connect(self.vm.open_trip_load)
        self.home.btn_quit.clicked.connect(self._quit_application)

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
        self.trip.notes_list.itemClicked.connect(self._on_voice_note_clicked)

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
        elif state == AppStateType.VOICE_TO_TEXT_LOADING:
            self.stack.setCurrentWidget(self.voice_loading)
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
            # Add date/time to rock classification display
            ts = r.ts
            if ts:
                dt = datetime.datetime.fromtimestamp(ts)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            else:
                time_str = "Unknown"
            item = f"[{time_str}] {label} ({conf}%)"
            self.trip.list.addItem(item)
        self.trip.lbl_totals.setText(
            f"Total volume: {summary.total_volume:.2f}   Total weight: {summary.total_weight:.2f}"
        )
        
        # Update voice notes list
        self.trip.notes_list.clear()
        self.trip._voice_notes_data = []  # Store full note data
        for note in summary.voice_notes:
            ts = note.get("ts", 0)
            if ts:
                dt = datetime.datetime.fromtimestamp(ts)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            else:
                time_str = "Unknown"
            cleaned = note.get("cleaned", note.get("transcript", ""))
            # Truncate long notes for display
            display_text = cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
            item = f"[{time_str}] {display_text}"
            self.trip.notes_list.addItem(item)
            # Store the full note data for access when clicked
            self.trip._voice_notes_data.append(note)

    def _on_error(self, message: str) -> None:
        # MVP: pop a modal, then ViewModel returns to HOME
        QMessageBox.warning(self, "Error", message)
    
    def _on_voice_note_clicked(self, item) -> None:
        """Show full voice note text in a dialog when clicked."""
        index = self.trip.notes_list.row(item)
        if 0 <= index < len(self.trip._voice_notes_data):
            note = self.trip._voice_notes_data[index]
            cleaned = note.get("cleaned", note.get("transcript", ""))
            transcript = note.get("transcript", "")
            
            # Create dialog to show full note
            dialog = QDialog(self)
            dialog.setWindowTitle("Voice Note")
            dialog.setMinimumWidth(600)
            dialog.setMinimumHeight(400)
            
            layout = QVBoxLayout(dialog)
            
            # Show timestamp
            ts = note.get("ts", 0)
            if ts:
                dt = datetime.datetime.fromtimestamp(ts)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = "Unknown"
            time_label = QLabel(f"Date/Time: {time_str}")
            time_label.setStyleSheet("font-size: 14px; font-weight: 600;")
            layout.addWidget(time_label)
            
            # Show full text
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(cleaned if cleaned else transcript)
            text_edit.setStyleSheet("font-size: 14px;")
            layout.addWidget(text_edit, stretch=1)
            
            # Close button
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(dialog.accept)
            layout.addWidget(button_box)
            
            dialog.exec()
