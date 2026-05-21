#!/usr/bin/env python3
"""
Parse sage_ground_data folder and generate missions.json for ground-UI.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import uuid
import re

def get_image_files(images_dir):
    """Get all JPG files from images directory with their timestamps.
    Returns paths relative to sage_ground_data root: images/YYYYMMDD/filename.jpg
    Key by filename to avoid collisions when multiple images share the same second.
    """
    images = {}
    if not images_dir.exists():
        return images
    
    for date_folder in images_dir.iterdir():
        if date_folder.is_dir():
            for img_file in date_folder.glob("*.jpg"):
                # Extract timestamp from filename: YYYYMMDD_HHMMSS_uuid.jpg
                match = re.match(r'(\d{8})_(\d{6})', img_file.name)
                if match:
                    date_str = match.group(1)
                    time_str = match.group(2)
                    # Parse to timestamp
                    try:
                        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                        ts = dt.timestamp()
                        # Path relative to sage_ground_data root: RUNNAME/images/YYYYMMDD/filename.jpg
                        relative_path = f"{images_dir.parent.name}/images/{date_folder.name}/{img_file.name}"
                        # Key by filename to avoid timestamp collision overwrites
                        images[img_file.name] = {
                            'path': relative_path,
                            'filename': img_file.name,
                            'timestamp': ts,
                            'date_folder': date_folder.name
                        }
                    except ValueError:
                        pass
    
    return images

def parse_mission_data(ground_data_path):
    """Parse sage_ground_data folder structure and return missions dict."""
    
    ground_data_path = Path(ground_data_path)
    missions = {}
    
    # Find all timestamped run folders
    run_folders = sorted([d for d in ground_data_path.iterdir() if d.is_dir()])
    
    for run_folder in run_folders:
        run_name = run_folder.name
        
        sage_store = run_folder / "sage_store"
        rocks_file = sage_store / "rocks.jsonl"
        voice_file = sage_store / "voice_notes.jsonl"
        images_dir = run_folder / "images"
        
        if not rocks_file.exists():
            continue
        
        # Get all available images (keyed by filename to avoid timestamp collisions)
        available_images = get_image_files(images_dir)
        images_by_filename = {filename: img['path'] for filename, img in available_images.items()}
        
        # Parse rocks
        rocks = {}
        with open(rocks_file, 'r') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    rock_id = obj.get('rock_id')
                    ts = obj.get('ts', 0)
                    result = obj.get('result', {})
                    rocks[rock_id] = {
                        'id': rock_id[:12],
                        'full_id': rock_id,
                        'timestamp': ts,
                        'classification': result.get('label', 'Unknown'),
                        'confidence': float(result.get('confidence', 0)),
                        'volume': result.get('estimated_volume'),
                        'image_path': result.get('image_path'),
                        'side_image_path': result.get('side_image_path')
                    }
        
        # Parse voice notes and group by rock; if IDs do not match, fall back to timestamp proximity.
        voice_notes = defaultdict(list)
        unmatched_voice_notes = []
        if voice_file.exists():
            with open(voice_file, 'r') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        rock_id = obj.get('rock_id')
                        ts = obj.get('ts', 0)
                        transcript = obj.get('transcript', '')
                        note = {
                            'timestamp': ts,
                            'transcript': transcript,
                            'session_id': obj.get('session_id')
                        }
                        if rock_id in rocks:
                            voice_notes[rock_id].append(note)
                        else:
                            unmatched_voice_notes.append(note)
        
        # One mission per timestamped run folder (e.g., 20260416_094443)
        mission_id = run_name
        sorted_rock_ids = sorted(rocks.keys(), key=lambda rid: rocks[rid]['timestamp'])
        rock_timestamps = {rock_id: rocks[rock_id]['timestamp'] for rock_id in sorted_rock_ids}

        if unmatched_voice_notes and rock_timestamps:
            for note in unmatched_voice_notes:
                nearest_rock_id = min(rock_timestamps, key=lambda rid: abs(rock_timestamps[rid] - note['timestamp']))
                if abs(rock_timestamps[nearest_rock_id] - note['timestamp']) < 10800:
                    voice_notes[nearest_rock_id].append(note)

        samples = []
        for rock_id in sorted_rock_ids:
            rock = rocks[rock_id]
            dt = datetime.fromtimestamp(rock['timestamp'])
            timestamp_utc = dt.isoformat() + 'Z'

            classification = str(rock.get('classification') or '').strip()
            if not classification or classification.lower() == 'unknown':
                continue

            top_image_name = Path(rock['image_path']).name if rock.get('image_path') else None
            side_image_name = Path(rock['side_image_path']).name if rock.get('side_image_path') else None
            top_image_file = images_by_filename.get(top_image_name) if top_image_name else None
            side_image_file = images_by_filename.get(side_image_name) if side_image_name else None

            if not top_image_file or not side_image_file:
                continue

            sample = {
                'id': rock['id'],
                'full_id': rock['full_id'],
                'timestampUtc': timestamp_utc,
                'classification': classification,
                'confidence': rock['confidence'],
                'predictionTimestamp': timestamp_utc,
                'volumeCm3': rock['volume'] or 0,
                'dimensions': {
                    'length': 0,
                    'width': 0,
                    'height': 0
                },
                'footprintArea': 0,
                'shapeFactor': 0,
                'estimationMethod': 'Stored metadata',
                'scaleTop': 0.12,
                'scaleSide': 0.11,
                'hasImagePair': True,
                'imagePath': top_image_file,
                'topImagePath': top_image_file,
                'sideImagePath': side_image_file,
                'notes': '',
                'voiceNotes': voice_notes.get(rock['full_id'], [])
            }
            samples.append(sample)

        if samples:
            missions[mission_id] = {
                'id': mission_id,
                'operator': 'Ground Analyst',
                'samples': samples,
                'audioFiles': []
            }
    
    return missions

def main():
    """Main entry point.
    
    Usage:
      python parse_mission_data.py [path_to_sage_ground_data]
    
    If no path provided, looks for sage_ground_data in common locations.
    """
    
    # Get path from command line
    if len(sys.argv) > 1:
        ground_data = Path(sys.argv[1])
    else:
        # Look for sage_ground_data in common locations
        possible_paths = [
            Path("sage_ground_data"),
            Path(".") / "sage_ground_data",
            Path("..") / "sage_ground_data",
            Path("..") / ".." / "sage_ground_data"
        ]
        
        ground_data = None
        for p in possible_paths:
            if p.exists():
                ground_data = p
                break
    
    if not ground_data.exists():
        print(f"Error: Could not find sage_ground_data folder")
        print(f"Usage: python parse_mission_data.py /path/to/sage_ground_data")
        return
    
    print(f"Parsing: {ground_data.resolve()}")
    missions = parse_mission_data(ground_data)
    
    output_file = Path(__file__).parent / "missions.json"
    with open(output_file, 'w') as f:
        json.dump(missions, f, indent=2)
    
    # Write config.json to tell app.js where the data is
    config = {
        "sage_ground_data_path": str(ground_data.resolve())
    }
    config_file = Path(__file__).parent / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nGenerated {len(missions)} missions")
    print(f"Saved missions to: {output_file.resolve()}")
    print(f"Saved config to: {config_file.resolve()}")
    
    print("\nMission summary:")
    for mission_id, mission in missions.items():
        samples_with_images = sum(1 for s in mission['samples'] if s['imagePath'])
        print(f"  {mission_id}: {len(mission['samples'])} samples ({samples_with_images} with images)")

if __name__ == "__main__":
    main()
