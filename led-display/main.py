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
    print("=" * 60, file=sys.stderr)
    print("SAGE Application Starting", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Debug: Print environment variables
    print(f"DEBUG: SAGE_USE_MOCKS = {os.environ.get('SAGE_USE_MOCKS', 'not set')}", file=sys.stderr)
    print(f"DEBUG: SAGE_USE_MOCK_ML = {os.environ.get('SAGE_USE_MOCK_ML', 'not set')}", file=sys.stderr)
    print(f"DEBUG: SAGE_STORE_DIR = {os.environ.get('SAGE_STORE_DIR', 'not set')}", file=sys.stderr)
    print(f"DEBUG: JETSON_PLATFORM = {os.environ.get('JETSON_PLATFORM', 'not set')}", file=sys.stderr)
    print(f"DEBUG: DISPLAY = {os.environ.get('DISPLAY', 'not set')}", file=sys.stderr)
    print(f"DEBUG: QT_QPA_PLATFORM = {os.environ.get('QT_QPA_PLATFORM', 'not set')}", file=sys.stderr)
    
    # Check display availability
    display = os.environ.get('DISPLAY')
    if not display:
        print("WARNING: DISPLAY environment variable not set", file=sys.stderr)
        print("WARNING: This may cause issues with GUI applications", file=sys.stderr)
    else:
        print(f"DEBUG: DISPLAY is set to: {display}", file=sys.stderr)
        # Try to check if X server is accessible
        try:
            import subprocess
            result = subprocess.run(['xdpyinfo'], capture_output=True, timeout=2)
            if result.returncode == 0:
                print("DEBUG: X server is accessible", file=sys.stderr)
            else:
                print(f"WARNING: xdpyinfo failed (return code {result.returncode})", file=sys.stderr)
                print(f"WARNING: stderr: {result.stderr.decode('utf-8', errors='ignore')[:200]}", file=sys.stderr)
        except FileNotFoundError:
            print("DEBUG: xdpyinfo not available (not critical)", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("WARNING: xdpyinfo timed out - X server may not be responding", file=sys.stderr)
        except Exception as e:
            print(f"DEBUG: Could not check X server: {e}", file=sys.stderr)
    
    try:
        print("DEBUG: Creating QApplication...", file=sys.stderr)
        app = QApplication(sys.argv)
        print(f"DEBUG: QApplication created successfully", file=sys.stderr)
        print(f"DEBUG: QApplication instance: {app}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to create QApplication: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    # Jetson-specific initialization
    try:
        print("DEBUG: Checking if Jetson...", file=sys.stderr)
        is_jetson = connector.is_jetson()
        print(f"DEBUG: is_jetson() = {is_jetson}", file=sys.stderr)
        
        if is_jetson:
            print("DEBUG: Jetson detected, getting config...", file=sys.stderr)
            # Set display backend if needed
            jetson_config = connector.get_jetson_config()
            print(f"DEBUG: Jetson config: {jetson_config}", file=sys.stderr)
            if jetson_config["display_backend"]:
                os.environ.setdefault("QT_QPA_PLATFORM", jetson_config["display_backend"])
                print(f"DEBUG: Set QT_QPA_PLATFORM = {jetson_config['display_backend']}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed during Jetson initialization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    # Use connector to get data store directory
    try:
        print("DEBUG: Getting data store directory...", file=sys.stderr)
        store_dir = connector.ensure_data_dir()
        print(f"DEBUG: Data store directory: {store_dir}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to get data store directory: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    try:
        print("DEBUG: Creating ViewModel...", file=sys.stderr)
        vm = ViewModel(store_dir=str(store_dir))
        print("DEBUG: ViewModel created successfully", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to create ViewModel: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    try:
        print("DEBUG: Creating AppWindow...", file=sys.stderr)
        win = AppWindow(vm)
        print("DEBUG: AppWindow created successfully", file=sys.stderr)
        print(f"DEBUG: Window visible: {win.isVisible()}", file=sys.stderr)
        print(f"DEBUG: Window size: {win.size().width()}x{win.size().height()}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to create AppWindow: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

    print("DEBUG: Starting event loop (app.exec())...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    try:
        result = app.exec()
        print(f"DEBUG: Event loop exited with code: {result}", file=sys.stderr)
        return result
    except Exception as e:
        print(f"ERROR: Event loop failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
