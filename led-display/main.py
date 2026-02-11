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
        sys.stderr.flush()
        return None


def main() -> int:
    print("Starting SAGE application...", file=sys.stderr)
    sys.stderr.flush()
    
    app = _run_step("create QApplication", lambda: QApplication(sys.argv))
    if app is None:
        print("ERROR: Failed to create QApplication", file=sys.stderr)
        sys.stderr.flush()
        return 1

    def jetson_init():
        if connector.is_jetson():
            cfg = connector.get_jetson_config()
            if cfg.get("display_backend"):
                os.environ.setdefault("QT_QPA_PLATFORM", cfg["display_backend"])

    if _run_step("Jetson initialization", jetson_init) is None:
        print("ERROR: Jetson initialization failed", file=sys.stderr)
        sys.stderr.flush()
        return 1

    store_dir = _run_step("get data store directory", connector.ensure_data_dir)
    if store_dir is None:
        print("ERROR: Failed to get data store directory", file=sys.stderr)
        sys.stderr.flush()
        return 1

    print(f"Creating ViewModel with store_dir={store_dir}...", file=sys.stderr)
    sys.stderr.flush()
    vm = _run_step("create ViewModel", ViewModel, store_dir=str(store_dir))
    if vm is None:
        print("ERROR: Failed to create ViewModel", file=sys.stderr)
        sys.stderr.flush()
        return 1

    print("Creating AppWindow...", file=sys.stderr)
    sys.stderr.flush()
    win = _run_step("create AppWindow", AppWindow, vm)
    if win is None:
        print("ERROR: Failed to create AppWindow", file=sys.stderr)
        sys.stderr.flush()
        return 1

    print("Starting event loop...", file=sys.stderr)
    sys.stderr.flush()
    result = _run_step("event loop", app.exec)
    if result is None:
        print("ERROR: Event loop failed", file=sys.stderr)
        sys.stderr.flush()
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
