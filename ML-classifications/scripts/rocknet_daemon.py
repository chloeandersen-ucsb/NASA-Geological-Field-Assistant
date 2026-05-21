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
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from rocknet_v2 import load_checkpoint, run_inference


def select_device() -> torch.device:
    try:
        import connector
        if connector.is_jetson():
            return torch.device("cuda")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def _cuda_mem_mb() -> str:
    """Return a short CUDA memory summary string, or empty string if unavailable."""
    if not torch.cuda.is_available():
        return ""
    alloc  = torch.cuda.memory_allocated()  / 1024**2
    reserv = torch.cuda.memory_reserved()   / 1024**2
    return f"  [CUDA alloc={alloc:.1f}MB reserved={reserv:.1f}MB]"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(Path(__file__).parent.parent / "models" / "best_rocknet_v2.pt"),
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    device = select_device()
    print(f"[DAEMON] Starting — device={device}", file=sys.stderr, flush=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"[DAEMON] CUDA cache cleared{_cuda_mem_mb()}", file=sys.stderr, flush=True)

    model, metadata = load_checkpoint(args.weights, device=device)
    model.eval()

    load_ms = (time.perf_counter() - t0) * 1000
    print(
        f"[DAEMON] Model loaded in {load_ms:.0f}ms{_cuda_mem_mb()}",
        file=sys.stderr, flush=True,
    )
    if metadata:
        print(f"[DAEMON] Checkpoint metadata: {metadata}", file=sys.stderr, flush=True)

    emit({"status": "ready"})
    inference_count = 0

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

        t1 = time.perf_counter()
        try:
            result = run_inference(model, image_path, device)
            infer_ms = (time.perf_counter() - t1) * 1000
            inference_count += 1

            if not output_json:
                stem        = Path(image_path).stem
                output_json = str(Path(image_path).parent / f"{stem}_prediction.json")

            with open(output_json, "w") as f:
                json.dump(result, f, indent=2)

            primary = result.get("primary", {})
            print(
                f"[DAEMON] #{inference_count} infer={infer_ms:.0f}ms "
                f"result={primary.get('family','?')} conf={primary.get('confidence',0):.3f} "
                f"tier={primary.get('tier','?')}{_cuda_mem_mb()}",
                file=sys.stderr, flush=True,
            )

            emit({"id": req_id, "status": "ok", "json_path": output_json})

        except Exception as exc:
            infer_ms = (time.perf_counter() - t1) * 1000
            print(
                f"[DAEMON] #{inference_count+1} FAILED after {infer_ms:.0f}ms: {exc}",
                file=sys.stderr, flush=True,
            )
            emit({"id": req_id, "status": "error", "message": str(exc)})


if __name__ == "__main__":
    main()
