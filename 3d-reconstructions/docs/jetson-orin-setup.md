# Jetson Orin setup

These steps target Jetson Orin devices running JetPack 5.x or 6.x. You’ll install PyTorch for aarch64, then TripoSR, and run inference on the GPU. Optionally, you can try ONNX/TensorRT if supported by your TripoSR version.

## 0) Prerequisites

- JetPack 5.x/6.x with CUDA and cuDNN installed (part of standard JetPack flash)
- Sufficient swap space (8–16 GB recommended) if RAM is limited
- Python 3.8–3.10 (match what your PyTorch wheel supports)

## 1) Create a Python environment

On device, it’s common to use system Python or venv:

```bash
python3 -m venv ~/envs/triposr
source ~/envs/triposr/bin/activate
python -m pip install --upgrade pip
```

## 2) Install PyTorch for aarch64

Install the wheel matching your JetPack and CUDA. NVIDIA and the Jetson community provide prebuilt wheels. For example, for JetPack 6.x (CUDA 12):

```bash
# Example only – get the exact URL from the official PyTorch-for-Jetson resources
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

For JetPack 5.x, use the corresponding CUDA 11.x wheels. Refer to NVIDIA forums or "PyTorch for Jetson" repo for exact links and versions.

Verify CUDA is visible:

```bash
python - << 'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

## 3) Install TripoSR

Either via pip (if wheels are available for aarch64) or from source:

```bash
# Clone the official repo
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
pip install -r requirements.txt
```

This repo runs via `python run.py` (no separate console command).

## 4) Run inference

Copy or SCP an image to your device, then run:

```bash
python scripts/generate_3d_triposr.py --input /path/to/image.jpg --out /path/to/outputs --device cuda --extra "--bake-texture --texture-resolution 1024"
```

Meshes will appear under `outputs/<image_stem>/`.

## 5) Optional: ONNX/TensorRT path

Some versions of TripoSR or community forks support ONNX export and TensorRT acceleration. If supported:

1. Export the model to ONNX on a desktop (x86_64) or on-device if dependencies allow.
2. Build a TensorRT engine with `trtexec` or the project’s helper script.
3. Run inference with a `--backend onnxrt` or `--backend tensorrt` flag if provided.

Consult the TripoSR repo for canonical instructions; flags and support may change over time.

## Troubleshooting

- If `torch.cuda.is_available()` is False, ensure the LD_LIBRARY_PATH and CUDA versions are correct for your JetPack.
- For RAM pressure, add swap (zram or swapfile) before large model runs.
- If building deps from source, install build-essential, cmake, and Python dev headers.
