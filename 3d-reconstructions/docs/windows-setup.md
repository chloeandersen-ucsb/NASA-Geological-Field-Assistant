# Windows setup (laptop)

Below are step-by-step instructions to get TripoSR running on Windows. Choose GPU if your laptop has an NVIDIA GPU; CPU works but is slower.

## 1) Create a Python environment

You can use conda, uv, or venv. Example with venv:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Install PyTorch (GPU preferred)

Visit https://pytorch.org/get-started/locally/ to get the exact command for your CUDA version.

Common example for CUDA 12.1 (adjust if different):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you don’t have an NVIDIA GPU, install the CPU-only wheels:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 3) Install TripoSR

Two options:

1) pip (if available):

```powershell
pip install triposr
```

2) From source:

```powershell
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
pip install -r requirements.txt
```

This repo runs via `python run.py` (no separate console command).

## 4) Verify with a sample image

Put a test image in `samples/image.jpg` (create the folder). Then run:

```powershell
python .\scripts\generate_3d_triposr.py --input .\samples\image.jpg --out .\outputs --device auto --extra "--save-obj --save-ply"
```

Open the resulting OBJ/PLY in MeshLab or Blender.

## Troubleshooting

- If you see CUDA errors, ensure your NVIDIA driver and CUDA runtime match the PyTorch wheels you installed.
- If `triposr` isn’t recognized, use `python -m triposr` or reinstall the package (`pip install -e .`).
- For permission issues in PowerShell, run `Set-ExecutionPolicy -Scope Process RemoteSigned` in your session.
