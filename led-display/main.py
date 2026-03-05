import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication

import connector
from core.viewmodel import ViewModel
from ui.app_window import AppWindow


def main() -> int:
    try:
        app = QApplication(sys.argv)
    except Exception as e:
        print(f"ERROR: Failed to create QApplication: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    try:
        is_jetson = connector.is_jetson()
        if is_jetson:
            jetson_config = connector.get_jetson_config()
            if jetson_config["display_backend"]:
                os.environ.setdefault("QT_QPA_PLATFORM", jetson_config["display_backend"])
    except Exception as e:
        print(f"ERROR: Failed during Jetson initialization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    try:
        store_dir = connector.ensure_data_dir()
        voice_notes_data_dir = connector.ensure_voice_notes_data_dir()
    except Exception as e:
        print(f"ERROR: Failed to get data store directory: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    try:
        vm = ViewModel(store_dir=str(store_dir), voice_notes_data_dir=str(voice_notes_data_dir))
    except Exception as e:
        print(f"ERROR: Failed to create ViewModel: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    try:
        win = AppWindow(vm)
    except Exception as e:
        print(f"ERROR: Failed to create AppWindow: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

    try:
        result = app.exec()
        return result
    except Exception as e:
        print(f"ERROR: Event loop failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
