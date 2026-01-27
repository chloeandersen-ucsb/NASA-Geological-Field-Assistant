import os
import sys
from pathlib import Path

# Add project root to path to import connector
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication

import connector
from core.viewmodel import ViewModel
from ui.app_window import AppWindow


def main() -> int:
    app = QApplication(sys.argv)
    
    # Jetson-specific initialization
    if connector.is_jetson():
        jetson_config = connector.get_jetson_config()
        if jetson_config["display_backend"]:
            os.environ.setdefault("QT_QPA_PLATFORM", jetson_config["display_backend"])
    
    store_dir = connector.ensure_data_dir()
    
    vm = ViewModel(store_dir=str(store_dir))
    win = AppWindow(vm)

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
