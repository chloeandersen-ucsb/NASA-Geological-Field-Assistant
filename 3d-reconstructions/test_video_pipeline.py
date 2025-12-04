#!/usr/bin/env python3
"""
Test script for video-to-3D pipeline dependencies and functionality.

Verifies:
- Python version
- Required packages
- Optional packages
- GPU availability
- Video codecs
- Configuration system

Usage:
    python test_video_pipeline.py
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("=" * 60)
    print("Checking Python Version")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ FAIL: Python 3.8+ required")
        return False
    else:
        print("✓ PASS: Python version OK")
        return True


def check_package(package_name, import_name=None, optional=False):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name:20s} - version {version}")
        return True
    except ImportError:
        if optional:
            print(f"  ⚠ {package_name:20s} - NOT INSTALLED (optional)")
        else:
            print(f"  ❌ {package_name:20s} - NOT INSTALLED (required)")
        return not optional


def check_required_packages():
    """Check required packages."""
    print("\n" + "=" * 60)
    print("Checking Required Packages")
    print("=" * 60)
    
    packages = [
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('pillow', 'PIL'),
        ('torch', 'torch'),
        ('scikit-image', 'skimage'),
        ('PyYAML', 'yaml'),
    ]
    
    all_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_ok = False
    
    if all_ok:
        print("\n✓ All required packages installed")
    else:
        print("\n❌ Some required packages missing")
        print("\nInstall missing packages:")
        print("  pip install opencv-python numpy pillow torch scikit-image PyYAML")
    
    return all_ok


def check_optional_packages():
    """Check optional packages."""
    print("\n" + "=" * 60)
    print("Checking Optional Packages")
    print("=" * 60)
    
    packages = [
        ('pycolmap', 'pycolmap'),
        ('open3d', 'open3d'),
        ('trimesh', 'trimesh'),
    ]
    
    for pkg_name, import_name in packages:
        check_package(pkg_name, import_name, optional=True)


def check_gpu():
    """Check GPU availability."""
    print("\n" + "=" * 60)
    print("Checking GPU")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ CUDA available")
            print(f"  Device: {gpu_name}")
            print(f"  Memory: {gpu_mem:.1f} GB")
            return True
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_opencv_video():
    """Check OpenCV video codec support."""
    print("\n" + "=" * 60)
    print("Checking OpenCV Video Support")
    print("=" * 60)
    
    try:
        import cv2
        
        # Try to create video capture
        cap = cv2.VideoCapture()
        
        # Check backends
        backends = cv2.videoio_registry.getBackends()
        print(f"✓ Available video backends: {backends}")
        
        # Common codecs
        codecs = ['H264', 'XVID', 'MJPG', 'MP4V']
        print("\n  Common codecs:")
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            print(f"    {codec}: {fourcc}")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV video check failed: {e}")
        return False


def check_scripts():
    """Check if pipeline scripts exist."""
    print("\n" + "=" * 60)
    print("Checking Pipeline Scripts")
    print("=" * 60)
    
    scripts = [
        'scripts/extract_frames.py',
        'scripts/generate_scene_from_video.py',
        'scripts/generate_scene_multiview_pycolmap.py',
        'scripts/generate_scene_multiview_dust3r.py',
    ]
    
    all_ok = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ❌ {script} - NOT FOUND")
            all_ok = False
    
    if all_ok:
        print("\n✓ All pipeline scripts present")
    else:
        print("\n❌ Some pipeline scripts missing")
    
    return all_ok


def check_config():
    """Check configuration system."""
    print("\n" + "=" * 60)
    print("Checking Configuration System")
    print("=" * 60)
    
    try:
        from config import Config, PRESETS
        
        print(f"✓ Config module loaded")
        print(f"  Available presets: {list(PRESETS.keys())}")
        
        # Test preset loading
        cfg = Config.preset_balanced()
        print(f"  Test preset: {cfg.reconstruction.backend}, {cfg.video.max_frames} frames")
        
        return True
    except Exception as e:
        print(f"❌ Config system check failed: {e}")
        return False


def run_all_checks():
    """Run all checks."""
    print("\n")
    print("#" * 60)
    print("# Video-to-3D Pipeline Test Suite")
    print("#" * 60)
    print()
    
    results = {}
    
    results['python'] = check_python_version()
    results['packages'] = check_required_packages()
    check_optional_packages()  # Don't fail on optional
    results['gpu'] = check_gpu()
    results['opencv'] = check_opencv_video()
    results['scripts'] = check_scripts()
    results['config'] = check_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {check_name:15s}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ System ready for video-to-3D reconstruction!")
        print("\nNext steps:")
        print("  1. python config.py  # Generate config files")
        print("  2. python scripts/generate_scene_from_video.py video.mp4")
        return 0
    else:
        print("\n❌ Some checks failed - fix issues before using pipeline")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_checks())
