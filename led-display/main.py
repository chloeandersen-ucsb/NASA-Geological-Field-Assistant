import os
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication

import connector
from core.viewmodel import ViewModel
from ui.app_window import AppWindow


def _run_step(step_name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs); on exception print and return None."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: {step_name}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def main() -> int:
    app = _run_step("create QApplication", lambda: QApplication(sys.argv))
    if app is None:
        return 1

    def jetson_init():
        if connector.is_jetson():
            cfg = connector.get_jetson_config()
            if cfg.get("display_backend"):
                os.environ.setdefault("QT_QPA_PLATFORM", cfg["display_backend"])

    if _run_step("Jetson initialization", jetson_init) is None:
        return 1

    store_dir = _run_step("get data store directory", connector.ensure_data_dir)
    if store_dir is None:
        return 1

    vm = _run_step("create ViewModel", ViewModel, store_dir=str(store_dir))
    if vm is None:
        return 1

    win = _run_step("create AppWindow", AppWindow, vm)
    if win is None:
        return 1

    result = _run_step("event loop", app.exec)
    if result is None:
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
