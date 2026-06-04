"""
Persistent RockNet inference daemon.

Loads the model once, then processes classify requests from stdin
(one JSON line per request) and writes results to stdout (one JSON line per result).

Protocol:
  startup stdout:  {"status": "ready"}\n
  request  stdin:  {"id": "<uuid>", "image": "<path>", "output_json": "<path>"}\n
  response stdout: {"id": "<uuid>", "status": "ok",    "json_path": "<path>"}\n
                or {"id": "<uuid>", "status": "error",  "message":   "<msg>"}\n
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from rocknet_v2 import load_checkpoint, run_inference


def select_device() -> torch.device:
    # NeMo (ASR) occupies most of the Jetson's unified GPU memory.
    # Running RockNet on CPU avoids CUDA OOM and camera NVMM starvation.
    return torch.device("cpu")


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(Path(__file__).parent.parent / "models" / "best_rocknet_v2.pt"),
    )
    args = parser.parse_args()

    device = select_device()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, metadata = load_checkpoint(args.weights, device=device)
    model.eval()

    # Finish CUDA warm-up before signalling ready so the camera never starts
    # while heavy GPU work is in flight.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 640, 640, device=device)
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    emit({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"id": None, "status": "error", "message": f"Bad JSON: {exc}"})
            continue

        req_id      = req.get("id")
        image_path  = req.get("image")
        output_json = req.get("output_json")

        if not image_path:
            emit({"id": req_id, "status": "error", "message": "missing 'image' field"})
            continue

        try:
            result = run_inference(model, image_path, device)

            if not output_json:
                stem        = Path(image_path).stem
                output_json = str(Path(image_path).parent / f"{stem}_prediction.json")

            with open(output_json, "w") as f:
                json.dump(result, f, indent=2)

            emit({"id": req_id, "status": "ok", "json_path": output_json})

        except Exception as exc:
            print(f"[DAEMON] Inference failed: {exc}", file=sys.stderr, flush=True)
            emit({"id": req_id, "status": "error", "message": str(exc)})


if __name__ == "__main__":
    main()
