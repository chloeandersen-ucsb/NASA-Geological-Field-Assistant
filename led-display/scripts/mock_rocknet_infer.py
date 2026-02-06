import argparse
import json
import random
import sys
import time
from pathlib import Path

ROCK_TYPES = [
    'Basalt', 'Clay', 'Conglomerate', 'Diatomite', 'Shale-(Mudstone)', 
    'Siliceous-sinter', 'chert', 'gypsum', 'olivine-basalt'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Weights path (ignored in mock)")
    parser.add_argument("--image", required=True, help="Image path (ignored in mock)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (ignored in mock)")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()
    
    time.sleep(1.5)
    
    # Generate top 3 classifications (matching real rocknet_infer.py format)
    top3 = []
    used_labels = set()
    for i in range(3):
        # Ensure unique labels
        available_labels = [l for l in ROCK_TYPES if l not in used_labels]
        if not available_labels:
            available_labels = ROCK_TYPES
        
        label = random.choice(available_labels)
        used_labels.add(label)
        
        # First result has highest confidence, others are lower
        if i == 0:
            confidence = round(random.uniform(0.75, 0.99), 2)
        else:
            confidence = round(random.uniform(0.01, 0.20), 2)
        
        top3.append({
            "label": label,
            "confidence": confidence,
        })
    
    # Normalize confidences so they sum to ~1.0 (like real softmax output)
    total_conf = sum(r["confidence"] for r in top3)
    if total_conf > 0:
        for r in top3:
            r["confidence"] = round(r["confidence"] / total_conf, 4)
    
    # Output as list format (matching real rocknet_infer.py)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(top3, f, indent=2)
    
    print(json.dumps(top3))
    sys.exit(0)

if __name__ == "__main__":
    main()
