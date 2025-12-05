#!/usr/bin/env python3
"""Quick test for COLMAP installation"""
import subprocess
import os

colmap_path = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'COLMAP', 'COLMAP-3.8-windows-no-cuda', 'bin', 'colmap.exe')

print(f"Testing COLMAP at: {colmap_path}")
print(f"Exists: {os.path.exists(colmap_path)}")

if os.path.exists(colmap_path):
    try:
        result = subprocess.run([colmap_path, '-h'], capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)}")
        print(f"Stderr length: {len(result.stderr)}")
        
        if result.returncode == 0:
            print("\n✓ COLMAP is working!")
            print("\nFirst 500 chars of output:")
            print(result.stdout[:500])
        else:
            print(f"\n✗ COLMAP failed with code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[:500])
    except Exception as e:
        print(f"\n✗ Error running COLMAP: {e}")
else:
    print("\n✗ COLMAP executable not found")
    print("\nPlease run PowerShell as Administrator and install:")
    print("  choco install vcredist140 -y")
