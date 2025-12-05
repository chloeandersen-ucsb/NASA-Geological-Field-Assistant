# Smart Max Frames Feature

## What Changed

The video-to-3D pipeline now **automatically calculates optimal frame count** based on:
1. Video duration
2. Selected backend (COLMAP/DUSt3R/MASt3R)

## How It Works

**Before:**
```powershell
# Always extracted exactly 20 frames
python scripts/generate_scene_from_video.py video.mp4
```

**Now:**
```powershell
# Auto-calculates based on video length
python scripts/generate_scene_from_video.py video.mp4

# 10s video → 12 frames (MASt3R)
# 30s video → 20 frames (MASt3R)
# 60s video → 30 frames (MASt3R)
```

## Frame Count Rules

### MASt3R (Default Backend)
| Video Length | Max Frames | Reason |
|--------------|------------|--------|
| < 10s | 12 | Small object/quick scan |
| 10-30s | 20 | Room/medium scene |
| 30-60s | 30 | Large room/outdoor |
| > 60s | 40 | Complex environment |

### DUSt3R (Fast Preview)
| Video Length | Max Frames | Reason |
|--------------|------------|--------|
| < 10s | 8 | Quick test |
| 10-30s | 12 | Fast reconstruction |
| > 30s | 15 | Speed priority |

### COLMAP (High Quality)
| Video Length | Max Frames | Reason |
|--------------|------------|--------|
| < 10s | 15 | Min for COLMAP |
| 10-30s | 25 | Room coverage |
| 30-60s | 40 | Detailed scene |
| > 60s | 60 | Maximum quality |

## Override Anytime

```powershell
# Force specific frame count
python scripts/generate_scene_from_video.py video.mp4 --max-frames 50

# System will show:
# [INFO] Video duration: 45.2s → recommended 30 frames for mast3r
# But uses your override of 50
```

## Benefits

✓ **No guessing** - system picks optimal count for your video length
✓ **Backend-aware** - COLMAP gets more frames, DUSt3R gets fewer
✓ **Better quality** - longer videos get proportionally more coverage
✓ **Time efficient** - short videos don't waste time on unnecessary frames
✓ **Still flexible** - override with `--max-frames` if needed

## Examples

**Short object scan (8s):**
```powershell
python scripts/generate_scene_from_video.py object.mp4
# Auto: 12 frames with MASt3R → ~2 min runtime
```

**Medium room walk (25s):**
```powershell
python scripts/generate_scene_from_video.py room.mp4
# Auto: 20 frames with MASt3R → ~4 min runtime
```

**Long outdoor scene (90s):**
```powershell
python scripts/generate_scene_from_video.py landscape.mp4 --backend colmap --device cuda
# Auto: 60 frames with COLMAP → ~15 min runtime
```

## Code Location

Function: `calculate_smart_max_frames()` in `scripts/generate_scene_from_video.py`

Called automatically before frame extraction when `--max-frames` is not specified.
