#!/usr/bin/env python
"""Download SAM model weights."""
import os
from segment_anything import sam_model_registry

model_dir = os.path.expanduser("~/.sam_models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "sam_vit_b.pth")

if not os.path.exists(model_path):
    print("Downloading SAM model (base, ~375MB)...")
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    urllib.request.urlretrieve(url, model_path)
    print(f"Downloaded to {model_path}")
else:
    print(f"Model already exists: {model_path}")

print("✓ SAM model ready")
