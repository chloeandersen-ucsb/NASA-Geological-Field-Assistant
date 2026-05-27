import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QTimer, QThread, Signal

import connector

# ---------------------------------------------------------
# 1. BACKGROUND LOADER THREAD
# ---------------------------------------------------------
class ModelLoaderThread(QThread):
    models_ready = Signal(object)
    error_occurred = Signal(str)
    status_update = Signal(str)

    def run(self):
        try:
            self.status_update.emit("Importing modules...")
            from core.viewmodel import ViewModel

            self.status_update.emit("Setting up storage...")
            store_dir = connector.ensure_data_dir()
            voice_notes_data_dir = connector.ensure_voice_notes_data_dir()

            self.status_update.emit("Loading models (this may take a moment)...")
            vm = ViewModel(store_dir=str(store_dir), voice_notes_data_dir=str(voice_notes_data_dir))

            self.status_update.emit("Almost ready...")
            # CRITICAL: Transfer ownership of the ViewModel back to the main GUI thread
            # so the AppWindow can use its QTimers and signals safely.
            vm.moveToThread(QApplication.instance().thread())

            self.models_ready.emit(vm)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

# ---------------------------------------------------------
# 2. SPLASH SCREEN WITH PROGRESS BAR
# ---------------------------------------------------------
class SplashScreen(QWidget):
    def __init__(self, is_jetson):
        super().__init__()
        self.setStyleSheet("background-color: #cbd2c5; color: #cad2c5; font-family: 'Courier New';")
        
        if is_jetson:
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        else:
            self.resize(480, 800)

        layout = QVBoxLayout(self)
        
        # Logo setup
        logo = QLabel()
        logo_path = project_root / "led-display" / "ui" / "newlogo.png"
        if logo_path.exists():
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

        # Progress Bar Styling
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #cad2c5;
                border-radius: 5px;
                background-color: #577d6a;
                height: 30px;
                text-align: center;
                font-weight: bold;
                font-size: 16px;
            }
            QProgressBar::chunk {
                background-color: #344f41;
                border-radius: 3px;
            }
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

        # --- High-Precision Animation Trackers ---
        self.exact_progress = 0.0 

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(33) # ~30 frames per second

    def update_progress(self):
        val = self.progress.value()
        # Slowly fill up to 95%, decelerating as it gets closer to the end
        if self.exact_progress < 100.0:
            self.exact_progress += (100.0 - self.exact_progress) * 0.01
            self.progress.setValue(int(self.exact_progress))

    def set_status(self, msg: str):
        self.status_label.setStyleSheet("font-size: 13px; color: #344f41; background: transparent;")
        self.status_label.setText(msg)

    def set_error(self, msg: str):
        self.status_label.setStyleSheet("font-size: 13px; color: #8b0000; background: transparent;")
        self.status_label.setText(f"Error: {msg}")

    def finish_progress(self):
        self.timer.stop()
        self.progress.setValue(100)
        self.loading_text.setText("Ready!")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
                self.resize(480, 800) # Shrinks it back to a window
            event.accept()
        else:
            super().keyPressEvent(event)

# ---------------------------------------------------------
# 3. MAIN APPLICATION LOOP
# ---------------------------------------------------------
def main() -> int:
    try:
        is_jetson = connector.is_jetson()
        if is_jetson:
            jetson_config = connector.get_jetson_config()
            if jetson_config["display_backend"]:
                os.environ.setdefault("QT_QPA_PLATFORM", jetson_config["display_backend"])
    except Exception as e:
        print(f"ERROR: Failed during Jetson initialization: {e}", file=sys.stderr)
        return 1

    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print(f"ERROR: Failed to create QApplication: {e}", file=sys.stderr)
        return 1
    
    # Show the splash screen instantly
    splash = SplashScreen(is_jetson)
    if is_jetson:
        splash.showFullScreen()
    else:
        splash.show()

    # We must keep a reference to the main window in this scope 
    # so it isn't destroyed by Python's garbage collector.
    active_windows = []

    # Define what happens when the background thread finishes
    # Define what happens when the background thread finishes
    def on_models_ready(vm):
        # 1. Check if the user pressed Esc on the splash screen
        was_fullscreen = splash.isFullScreen()
        
        splash.finish_progress()
        
        # Now import the UI and launch it
        from ui.app_window import AppWindow
        win = AppWindow(vm)
        
        # 2. If they exited full screen, force the main app to match!
        if not was_fullscreen:
            win._exit_fullscreen()
            
        active_windows.append(win) 
        
        splash.close()

    def on_error(err_msg):
        splash.loading_text.setText("Error loading models!")
        splash.set_error(err_msg)
        print(f"FATAL ERROR: {err_msg}", file=sys.stderr)

    # Start the background thread
    loader_thread = ModelLoaderThread(app)
    loader_thread.models_ready.connect(on_models_ready)
    loader_thread.error_occurred.connect(on_error)
    loader_thread.status_update.connect(splash.set_status)
    loader_thread.start()

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())