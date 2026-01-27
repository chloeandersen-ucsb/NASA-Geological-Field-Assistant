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
    Get the voice-to-text directory path.
    
    Can be overridden with SAGE_VOICE_TO_TEXT_DIR environment variable.
    """
    env_path = os.environ.get("SAGE_VOICE_TO_TEXT_DIR")
    if env_path:
        return Path(env_path).absolute()
    return get_project_root() / "voice-to-text"


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
    """Get the path to improved2.py script."""
    return get_voice_to_text_dir() / "improved2.py"


def get_camera_script_path() -> Path:
    """
    Get the path to camera capture script.
    
    Returns mock script if SAGE_USE_MOCKS is set, otherwise returns
    the real camera_capture.py path (which may not exist yet).
    """
    led_display = get_led_display_dir()
    if os.environ.get("SAGE_USE_MOCKS", "").lower() in ("1", "true", "yes"):
        return led_display / "scripts" / "mock_camera_capture.py"
    else:
        return led_display / "scripts" / "camera_capture.py"


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
        raise FileNotFoundError(f"{description} not found: {path}")


def validate_ml_paths() -> None:
    """Validate that ML-classifications paths exist (when not using mocks)."""
    if os.environ.get("SAGE_USE_MOCKS", "").lower() in ("1", "true", "yes"):
        return  # Skip validation when using mocks
    
    validate_path(get_rocknet_script_path(), "RockNet script")
    validate_path(get_rocknet_weights_path(), "Model weights")


def validate_voice_to_text_paths() -> None:
    """Validate that voice-to-text paths exist (when not using mocks)."""
    if os.environ.get("SAGE_USE_MOCKS", "").lower() in ("1", "true", "yes"):
        return  # Skip validation when using mocks
    
    validate_path(get_voice_to_text_script_path(), "Voice-to-text script")


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
