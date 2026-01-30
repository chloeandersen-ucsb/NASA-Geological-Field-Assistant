from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ============================================================================
# Platform Detection
# ============================================================================

def is_jetson() -> bool:
    """
    Detect if running on NVIDIA Jetson platform.
    
    Checks for:
    - /etc/nv_tegra_release file (Jetson-specific)
    - JETSON_PLATFORM environment variable
    """
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    if os.environ.get("JETSON_PLATFORM", "").lower() in ("1", "true", "yes"):
        return True
    return False


def is_mac() -> bool:
    """Detect if running on macOS."""
    import platform
    return platform.system() == "Darwin"


# ============================================================================
# Path Resolution
# ============================================================================

def get_project_root() -> Path:
    """
    Get the root directory of the SAGE project (capstone/).
    
    Assumes this file (connector.py) is at the project root.
    """
    return Path(__file__).parent.absolute()


def get_led_display_dir() -> Path:
    """Get the led-display directory path."""
    return get_project_root() / "led-display"


def get_ml_classifications_dir() -> Path:
    """
    Get the ML-classifications directory path.
    
    Can be overridden with SAGE_ML_CLASSIFICATIONS_DIR environment variable.
    """
    env_path = os.environ.get("SAGE_ML_CLASSIFICATIONS_DIR")
    if env_path:
        return Path(env_path).absolute()
    return get_project_root() / "ML-classifications"


def get_voice_to_text_dir() -> Path:
    """
    Get the voiceNotes directory path.
    
    Can be overridden with SAGE_VOICE_TO_TEXT_DIR environment variable.
    """
    env_path = os.environ.get("SAGE_VOICE_TO_TEXT_DIR")
    if env_path:
        return Path(env_path).absolute()
    return get_project_root() / "voiceNotes"


def get_data_store_dir() -> Path:
    """
    Get the data storage directory path.
    
    On Jetson: defaults to /data/sage
    On Mac/other: defaults to ./sage_data (relative to led-display/)
    
    Can be overridden with SAGE_STORE_DIR environment variable.
    """
    env_path = os.environ.get("SAGE_STORE_DIR")
    if env_path:
        return Path(env_path).absolute()
    
    if is_jetson():
        # Use persistent storage on Jetson
        return Path("/data/sage")
    else:
        # Use local directory for development
        return get_led_display_dir() / "sage_data"


def get_rocknet_script_path() -> Path:
    """Get the path to rocknet_infer.py script."""
    return get_ml_classifications_dir() / "rocknet_infer.py"


def get_rocknet_weights_path() -> Path:
    """
    Get the path to best_rocknet.pt model weights.
    
    Can be overridden with SAGE_ROCKNET_WEIGHTS environment variable.
    """
    env_path = os.environ.get("SAGE_ROCKNET_WEIGHTS")
    if env_path:
        return Path(env_path).absolute()
    return get_ml_classifications_dir() / "best_rocknet.pt"


def get_voice_to_text_script_path() -> Path:
    """Get the path to shruthi's rtt_lav.py script."""
    return get_voice_to_text_dir() / "shruthi-files" / "rtt_lav.py"


def get_camera_script_path() -> Path:
    """
    Get the path to camera capture script.
    
    Returns mock script if SAGE_USE_MOCKS or SAGE_USE_MOCK_ML is set,
    otherwise returns the real camera script path from ML-classifications
    (which may not exist yet).
    """
    led_display = get_led_display_dir()
    use_mocks = os.environ.get("SAGE_USE_MOCKS", "").lower() in ("1", "true", "yes")
    use_mock_ml = os.environ.get("SAGE_USE_MOCK_ML", "").lower() in ("1", "true", "yes")
    
    if use_mocks or use_mock_ml:
        return led_display / "scripts" / "mock_camera_capture.py"
    else:
        return get_ml_classifications_dir() / "camera-pipeline"


def get_mock_rocknet_script_path() -> Path:
    """Get the path to mock_rocknet_infer.py script."""
    return get_led_display_dir() / "scripts" / "mock_rocknet_infer.py"


def get_mock_voice_to_text_script_path() -> Path:
    """Get the path to mock_improved2.py script."""
    return get_led_display_dir() / "scripts" / "mock_improved2.py"


# ============================================================================
# Path Validation
# ============================================================================

def validate_path(path: Path, description: str) -> None:
    """
    Validate that a path exists, raising FileNotFoundError if not.
    
    Args:
        path: Path to validate
        description: Human-readable description for error message
    """
    if not path.exists():
        # Provide helpful guidance for missing model weights
        if "weights" in description.lower() or "best_rocknet.pt" in str(path):
            parent_dir = path.parent
            error_msg = f"{description} not found: {path}\n\n"
            if not parent_dir.exists():
                error_msg += f"Directory does not exist: {parent_dir}\n"
            else:
                error_msg += f"Directory exists: {parent_dir}\n"
            error_msg += "\nTo fix this:\n"
            error_msg += "1. Copy your trained model weights file to the ML-classifications directory:\n"
            error_msg += f"   cp /path/to/your/best_rocknet.pt {parent_dir}/\n\n"
            error_msg += "2. Or set the SAGE_ROCKNET_WEIGHTS environment variable:\n"
            error_msg += "   export SAGE_ROCKNET_WEIGHTS=/path/to/your/best_rocknet.pt\n\n"
            error_msg += "3. Or use mock mode for testing:\n"
            error_msg += "   make run-mock\n"
            raise FileNotFoundError(error_msg)
        else:
            raise FileNotFoundError(f"{description} not found: {path}")


def validate_ml_paths() -> None:
    """Validate that ML-classifications paths exist (when not using mocks)."""
    # Skip validation when using mocks (either all mocks or just ML mocks)
    use_mocks = os.environ.get("SAGE_USE_MOCKS", "").lower() in ("1", "true", "yes")
    use_mock_ml = os.environ.get("SAGE_USE_MOCK_ML", "").lower() in ("1", "true", "yes")
    if use_mocks or use_mock_ml:
        return  # Skip validation when using mocks
    
    # Check ML-classifications directory first
    ml_dir = get_ml_classifications_dir()
    if not ml_dir.exists():
        raise FileNotFoundError(
            f"ML-classifications directory not found: {ml_dir}\n\n"
            f"This directory should contain:\n"
            f"  - rocknet_infer.py\n"
            f"  - best_rocknet.pt (model weights)\n\n"
            f"To fix:\n"
            f"1. Ensure the ML-classifications directory exists in your project\n"
            f"2. Or set SAGE_ML_CLASSIFICATIONS_DIR environment variable:\n"
            f"   export SAGE_ML_CLASSIFICATIONS_DIR=/path/to/ML-classifications\n"
        )
    
    validate_path(get_rocknet_script_path(), "RockNet script")
    validate_path(get_rocknet_weights_path(), "Model weights")


def validate_voice_to_text_paths() -> None:
    if os.environ.get("SAGE_USE_MOCKS", "").lower() in ("1", "true", "yes"):
        return  # Skip validation when using mocks
    
    validate_path(get_voice_to_text_script_path(), "voiceNotes script")


# ============================================================================
# Jetson Configuration
# ============================================================================

def get_jetson_config() -> dict:
    """
    Get Jetson-specific configuration.
    
    Returns:
        Dictionary with GPU, display, and path configurations
    """
    config = {
        "is_jetson": is_jetson(),
        "gpu_available": False,
        "cuda_device": "cuda:0",
        "display_backend": os.environ.get("QT_QPA_PLATFORM", "xcb"),
        "data_dir": str(get_data_store_dir()),
    }
    
    if is_jetson():
        # Check for CUDA availability
        try:
            import torch
            config["gpu_available"] = torch.cuda.is_available()
            if config["gpu_available"]:
                config["cuda_device"] = f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"
        except ImportError:
            pass  # PyTorch not available
    
    return config


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_data_dir() -> Path:
    """
    Ensure the data storage directory exists and is writable.
    
    Returns:
        Path to the data directory
    """
    data_dir = get_data_store_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if writable
    test_file = data_dir / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Data directory is not writable: {data_dir} ({e})")
    
    return data_dir


def get_python_executable() -> str:
    return os.environ.get("SAGE_PYTHON", "python3")
