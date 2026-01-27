import argparse
import json
import random
import sys
import time
from pathlib import Path

# Mock rock types matching the real CLASS_NAMES from rocknet_infer.py
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
    
    # Simulate processing time (similar to real ML inference)
    time.sleep(1.5)
    
    # Generate mock classification result matching real script format
    label = random.choice(ROCK_TYPES)
    confidence = round(random.uniform(0.75, 0.99), 2)
    
    # Create output matching the real script's format
    result = {
        "label": label,
        "confidence": confidence,
    }
    
    # Write to output JSON file (same format as real script)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    # Also print to stdout (though service reads from JSON file)
    print(json.dumps(result))
    sys.exit(0)

if __name__ == "__main__":
    main()
