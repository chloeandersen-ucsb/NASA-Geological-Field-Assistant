"""
Loads trained RockNetV2 checkpoint, runs the two-stage multi-head pipeline,
and writes the locked v2.0 output JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from rocknet_v2 import RockNetV2, load_checkpoint, run_inference

def select_device() -> torch.device:
    try:
        import connector
        if connector.is_jetson():
            print("[ML] Jetson detected — using CUDA", file=sys.stderr)
            return torch.device("cuda")
    except ImportError:
        pass

    if torch.cuda.is_available():
        print("[ML] CUDA available — using GPU", file=sys.stderr)
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        print("[ML] Apple Silicon — using MPS", file=sys.stderr)
        return torch.device("mps")

    print("[ML] Using CPU", file=sys.stderr)
    return torch.device("cpu")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RockNet v2 — two-stage multi-head inference"
    )
    parser.add_argument(
        "--weights",
        default=str(Path(__file__).parent.parent / "models" / "best_rocknet_v2.pt"),
        help="Path to trained RockNetV2 checkpoint (.pt)",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON path (default: <image_stem>_prediction.json next to image)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print JSON result to stdout in addition to writing file",
    )

    args = parser.parse_args()

    device = select_device()

    # Load model
    model, metadata = load_checkpoint(args.weights, device=device)
    if metadata:
        print(f"[ML] Checkpoint metadata: {metadata}", file=sys.stderr)

    # Run inference
    result = run_inference(model, args.image, device)

    # Determine output path
    if args.output_json:
        json_path = args.output_json
    else:
        stem = Path(args.image).stem
        json_path = str(Path(args.image).parent / f"{stem}_prediction.json")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    if args.print:
        print(json.dumps(result, indent=2))
    else:
        print(f"[ML] Wrote prediction to {json_path}", file=sys.stderr)
        fam   = result["primary"]["family"]
        conf  = result["primary"]["confidence"]
        tier  = result["primary"]["tier"]
        print(f"[ML] Primary: {fam} ({conf:.3f}, {tier})", file=sys.stderr)


if __name__ == "__main__":
    main()
