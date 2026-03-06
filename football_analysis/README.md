# Football Video Analysis – Detection & Tracking Pipeline

A modular Python pipeline that detects and tracks football players in match footage using YOLOv8s and BoT-SORT/ByteTrack, then produces an annotated video, per-player heatmaps, and speed statistics.

---

## Project Structure

```
football_analysis/
├── main.py                  # Pipeline entry point (CLI)
├── config.py                # Dataclass-based configuration
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── detector.py          # YOLOv8 player detection wrapper
│   ├── tracker.py           # Detection + tracking with multi-stage filtering
│   ├── speed_estimator.py   # Pixel-displacement speed estimation with outlier rejection
│   ├── heatmap.py           # Per-player Gaussian heatmap generator
│   ├── annotator.py         # Bounding box / label drawing
│   └── video_io.py          # Video reader & writer utilities
├── input/                   # Place input videos here
└── output/                  # Generated outputs appear here
    ├── annotated_output.mp4
    ├── player_stats.csv
    └── heatmaps/
        ├── heatmap_player_1.png
        ├── heatmap_player_2.png
        └── ...
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- A CUDA-capable GPU is recommended but not required (CPU inference works)

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will automatically download the YOLOv8s model weights (~22 MB for `yolov8s.pt`).

---

## How to Run

### Basic usage

```bash
cd football_analysis
python main.py path/to/football_video.mp4
```

### With options

```bash
python main.py input/match_clip.mp4 \
    --output-dir output \
    --model yolov8s.pt \
    --tracker botsort \
    --conf 0.5 \
    --meters-per-pixel 0.05 \
    --display
```

### CLI Arguments

| Argument              | Default       | Description                                      |
|-----------------------|---------------|--------------------------------------------------|
| `input_video`         | *(required)*  | Path to the input football video                 |
| `--output-dir`        | `output`      | Directory for all output files                   |
| `--model`             | `yolov8s.pt`  | YOLO model name or path                          |
| `--tracker`           | `botsort`     | Tracker type: `botsort` or `bytetrack`           |
| `--conf`              | `0.5`         | Detection confidence threshold                   |
| `--meters-per-pixel`  | `0.05`        | Calibration factor for speed estimation          |
| `--max-frames`        | `0` (all)     | Limit number of frames to process                |
| `--display`           | off           | Show live preview window during processing       |

### Outputs

1. **Annotated video** (`output/annotated_output.mp4`) – Input video with bounding boxes, tracking IDs, and live speed overlaid on each player.
2. **Player statistics** (`output/player_stats.csv`) – CSV with columns: `Player ID`, `Avg Speed (m/s)`, `Avg Speed (km/h)`.
3. **Heatmaps** (`output/heatmaps/`) – One PNG per tracked player showing spatial occupancy.

---

## Design Choices

### Detection & Filtering
- **YOLOv8s** (via Ultralytics) is used by default — the small variant offers a strong balance between accuracy and speed. Larger variants (`yolov8m`, `yolov8l`) can be specified via `--model` for even higher accuracy.
- Detection runs at **1280 px** input resolution for better sensitivity to players at distance.
- Only the COCO `person` class (ID 0) is detected, filtering out balls, goalposts, and other objects.
- A multi-stage post-detection filter eliminates non-player detections:
  1. **Minimum bounding box height** (40 px) — removes tiny crowd/spectator detections.
  2. **Aspect ratio constraint** (1.2–5.0 height/width) — players are taller than wide; this rejects banners, logos, and wide crowd clusters.
  3. **Pitch ROI** — only detections whose vertical centre falls within the middle 85% of the frame are kept (top 10% sky/banners and bottom 5% ad boards are ignored).
- The default **confidence threshold** is 0.5 to reduce false positives.

### Tracking
- **BoT-SORT** is used by default through Ultralytics' integrated tracking API. ByteTrack is available as an alternative (`--tracker bytetrack`). Both are online multi-object trackers that associate detections across frames using motion and appearance cues.
- Decoupling detection from tracking conceptually (separate modules) while using the integrated API for efficiency reflects a pragmatic engineering trade-off.
- A **minimum track lifetime filter** (60 frames, ~2 seconds at 30 FPS) removes transient/noisy tracks from the final statistics and heatmaps. Only players tracked for a sustained duration are included in the output, keeping the player count realistic.

### Speed Estimation
- Speed is approximated from **pixel displacement** between consecutive tracked positions, converted to metres using a configurable `meters_per_pixel` ratio.
- A **sliding-window average** (10 frames) smooths noisy frame-to-frame measurements for the live overlay.
- Individual frame-to-frame segments exceeding **12 m/s** (~43 km/h, faster than any human sprint) are treated as tracker jumps and excluded from both instantaneous and average speed calculations.
- The overall average speed for each player is computed over the entire tracked duration and reported in the CSV.

### Heatmaps
- Tracked centre positions are binned onto a reduced-resolution grid, then **Gaussian-blurred** and colour-mapped (JET) to produce per-player heatmaps.
- This approach works without requiring pitch homography — positions are in pixel space, which is sufficient for relative movement patterns.

---

## Assumptions

1. **Camera is mostly static or slowly panning.** Rapid camera motion would cause pixel-displacement speed estimates to reflect camera movement rather than player movement.
2. **Players are visible and upright.** The COCO-trained YOLO model expects standard person poses; heavily occluded or prone players may be missed.
3. **Uniform scale across the frame.** The single `meters_per_pixel` factor assumes roughly constant depth. A proper homography transform would improve accuracy but is outside the scope of this prototype.
4. **Input video is 1–3 minutes** of broadcast or fixed-camera football footage.

---

## Limitations and Possible Improvements

| Limitation | Possible Improvement |
|---|---|
| Speed estimation is uncalibrated | Use pitch-line detection + homography to project to real-world coordinates |
| Single `meters_per_pixel` ignores perspective | Implement perspective transform from four known pitch points |
| No team classification | Add jersey colour clustering (e.g., K-means on bbox crops) to separate teams |
| No ball detection/tracking | Train or use a ball-specific detector; integrate with player tracking |
| Heatmaps are in pixel space | Project onto a top-down pitch template for tactical analysis |
| No referee filtering | Use appearance features or a specialised classifier to exclude referees |
| Bbox height filter is resolution-dependent | Compute min height as a percentage of frame height instead of fixed pixels |
| Tracking may fragment for long occlusions | Use re-identification features or a more robust tracker like StrongSORT |
| CPU inference is slow for long videos | Use TensorRT or ONNX export for optimised GPU inference |

---

## License

This project is provided for evaluation purposes only.
