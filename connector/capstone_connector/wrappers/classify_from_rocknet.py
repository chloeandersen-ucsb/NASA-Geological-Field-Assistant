#!/usr/bin/env python3
"""
Run capstone/ML-classifications/rocknet_infer.py and normalize stdout to:
  {"label":"Basalt","confidence":97}
Accepts either JSON or Python-dict repr on stdout.
- If --image is omitted, defaults to <CAPSTONE_ROOT>/ML-classifications/latest.jpg
- If --weights is omitted, defaults to <CAPSTONE_ROOT>/ML-classifications/best_rocknet.pt
You can override CAPSTONE_ROOT in env.
"""
import os, sys, json, argparse, subprocess, ast
from pathlib import Path

def guess_capstone_root():
    here = Path(__file__).resolve()
    return str(here.parents[2])  # .../capstone

def normalize_conf(v):
    try:
        v = float(v)
    except Exception:
        v = 0.0
    if v <= 1.0:
        v = int(round(v*100))
    else:
        v = int(round(v))
    return max(0, min(100, v))

def parse_obj_from_text(s: str):
    s = s.strip()
    # Try JSON then Python literal
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            inner = s[start:end+1]
            return ast.literal_eval(inner)
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default=None, help="Path to rocknet_infer.py")
    parser.add_argument("--weights", default=None, help="Path to model weights .pt")
    parser.add_argument("--image", default=None, help="Path to image to classify")
    args = parser.parse_args()

    cap_root = os.environ.get("CAPSTONE_ROOT", guess_capstone_root())
    script = args.script or os.path.join(cap_root, "ML-classifications", "rocknet_infer.py")
    weights = args.weights or os.path.join(cap_root, "ML-classifications", "best_rocknet.pt")
    image = args.image or os.path.join(cap_root, "ML-classifications", "latest.jpg")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if not os.path.exists(image):
        print(json.dumps({"label":"Unknown","confidence":0,"error":"missing_image","image":image}))
        return 0

    cmd = ["python3", "-u", script, "--weights", weights, "--image", image]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        print(json.dumps({"label":"Unknown","confidence":0,"error":"subprocess","detail":e.output[:200]}))
        return 0

    obj = parse_obj_from_text(out)
    if not isinstance(obj, dict):
        print(json.dumps({"label":"Unknown","confidence":0,"error":"parse"}))
        return 0

    label = str(obj.get("label","Unknown"))
    conf = normalize_conf(obj.get("confidence", 0))
    print(json.dumps({"label": label, "confidence": conf}))

if __name__ == "__main__":
    main()
