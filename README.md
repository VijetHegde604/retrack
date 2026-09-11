# ReTrack

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-11%20passed-brightgreen.svg)](tests/)

**ReTrack** is a real-time object tracker that **remembers who's who** — even after they leave the frame and come back.

Most trackers forget an object the moment it disappears. When it returns, it gets a brand-new ID as if it were a stranger. ReTrack fixes this by combining fast motion-based tracking with a lightweight appearance memory, so returning objects get their original identity back — without needing a GPU.

---

## Demo

<div align="center">
  <video src="https://github.com/user-attachments/assets/80ae595d-dd5b-4080-8588-2adbee7919ce" width="480" controls="controls" muted="muted" loop="loop">
    Your browser does not support the video tag.
  </video>
  <p>
    🎬 <strong>Sample Tracking Output:</strong> <a href="outputs/test.mp4"><code>outputs/test.mp4</code></a>
  </p>
</div>

---

## How It Works

ReTrack processes each video frame in three stages:

### Stage 1 — Match confident detections by position

YOLO detects objects in the frame. High-confidence detections (score ≥ 0.50) are matched to existing tracks using spatial overlap (IoU). If a detection lines up with where a track is expected to be, it's a match.

### Stage 2 — Recover partially hidden objects

Lower-confidence detections (score 0.10–0.49) often come from occluded or blurry objects. Instead of throwing these away, ReTrack tries to match them to any tracks that weren't matched in Stage 1 — rescuing tracks that would otherwise be lost.

### Stage 3 — Re-identify returning objects by appearance

Any confident detection that *still* didn't match a track might be a returning object. ReTrack crops the detection, runs it through a lightweight neural network (MobileNetV3-Small) to get a visual fingerprint, and checks it against a gallery of previously seen objects. If the fingerprint is close enough (cosine similarity ≥ 0.65), the original ID is restored.

```
                      Input Video Frame
                             │
                      YOLO11 Detector
                             │
            ┌────────────────┴────────────────┐
            ▼ (Conf >= 0.50)                  ▼ (0.10 <= Conf < 0.50)
     High-Confidence Dets              Low-Confidence Dets
            │                                 │
            ▼                                 │
   [ Stage 1: Spatial IoU ]                   │
   (Active & Lost Tracks)                     │
       ├── Matched ──────> Update Track       │
       └── Unmatched ─────────┐               │
                              ▼               ▼
                     [ Stage 2: Spatial IoU ]
                         ├── Matched ──> Update Track
                         └── Unmatched ─> Mark Lost (Buffer: 30)
                                                │
   [ Stage 3: Selective ReID ]                  ▼
   (Unmatched High-Conf Dets)          Archive in Gallery
       ├── Cosine Sim >= 0.65
       │     ├── Match ──> Restore ID & Warp Kalman State
       │     └── No Match ─> Allocate New ID & Register
```

---

## The Appearance Gallery

When an object disappears for good (more than 30 frames), its visual fingerprint is archived in a **persistent gallery**. This gallery is what lets ReTrack recognize objects minutes later, not just frames later.

Each object's entry in the gallery stores:

- **Up to 6 diverse snapshots** — front view, side view, etc. — so it can handle appearance changes from different angles. Near-duplicate snapshots are filtered out to keep the gallery useful.
- **A running average** — a smoothly updated summary of the object's appearance over time, which handles gradual changes in lighting or scale.

The key trick: **ReID only runs when needed.** During normal tracking (Stages 1 & 2), the neural network isn't touched at all — zero overhead. It only fires for unmatched detections and periodic gallery updates, which is why ReTrack stays fast.

---

## Key Features

| Feature | What it means |
| :--- | :--- |
| **Persistent Re-ID** | Objects get their original ID back after leaving and returning, even minutes later. |
| **Selective ReID** | The appearance network only runs when needed — most frames have zero ReID cost. |
| **Two-Stage ByteTrack** | High-confidence matches first, then low-confidence recovery for occluded objects. |
| **Multi-View Gallery** | Stores multiple viewpoints per object so angle changes don't break recognition. |
| **Fast Mask Blending** | Segmentation masks are blended within bounding-box ROIs using integer math — 5.1× faster on 4K frames. |
| **Consistent Colors** | Track colors are deterministically hashed from IDs, so returning objects keep their color. |
| **Real-Time on CPU** | 30–45 FPS on modern CPUs, no GPU required. |

---

## Installation

### Prerequisites
- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Option 1: Using `uv` (Recommended)

```bash
git clone https://github.com/VijetHegde604/retrack.git
cd retrack
uv sync
```

### Option 2: Using Nix Flake (Reproducible Dev Environment)

For NixOS or systems using the Nix package manager with system OpenCV / X11 dependencies:

```bash
nix develop
uv sync
```

---

## Quick Start

### Run on a video file
```bash
uv run retrack path/to/video.mp4
```

### Run with live webcam
```bash
uv run retrack --webcam
```

### Save annotated output
```bash
uv run retrack path/to/video.mp4 --output outputs/tracked.mp4
```
*(See [`outputs/test.mp4`](outputs/test.mp4) for an example.)*

### Headless mode (servers / benchmarking)
```bash
uv run retrack path/to/video.mp4 --no-display --output outputs/tracked.mp4
```

---

## CLI Options

```bash
retrack [source] [options]
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `source` | — | Path to input video file. |
| `--webcam` | `False` | Use the default webcam. |
| `--model` | `yolo11s-seg.pt` | YOLO model path (e.g. `yolo11n.pt` for max speed). |
| `--device` | `auto` | Inference device: `auto`, `cpu`, `cuda`, or `mps`. |
| `--imgsz` | `640` | Inference resolution. |
| `--conf` | `0.35` | Minimum detection confidence. |
| `--reid-thresh` | `0.65` | Cosine similarity threshold for re-identification. |
| `--lost-buffer` | `30` | Frames before a lost track moves to the gallery. |
| `--no-reid` | `False` | Disable ReID (motion-only ByteTrack). |
| `--mask-alpha` | `0.45` | Mask opacity (`0.0` = bounding boxes only). |
| `--output` | — | Path to write the annotated output video. |
| `--no-display` | `False` | Disable the display window (headless). |
| `--max-frames` | — | Stop after this many frames. |

---

## Project Structure

```
retrack/
├── README.md                    # You are here
├── pyproject.toml               # Package dependencies & CLI entrypoints
├── flake.nix                    # Nix devShell with system dependencies
├── outputs/
│   └── test.mp4                 # Sample annotated output video
├── src/retrack/
│   ├── __main__.py              # CLI entrypoint & video processing loop
│   ├── byte_tracker.py          # 3-Stage ByteTrack + ReID association engine
│   ├── detector.py              # YOLO detection & instance segmentation wrapper
│   ├── kalman.py                # Kalman filter for motion prediction & IoU math
│   ├── reid.py                  # MobileNetV3-Small extractor & persistent gallery
│   ├── render.py                # ROI-bounded mask blending & visual badging
│   ├── tracker.py               # Track state machine & data structures
│   └── video.py                 # Thread-safe frame generator
└── tests/
    ├── test_kalman.py           # Motion prediction & IoU tests
    ├── test_persistent_tracker.py # Re-entry & disambiguation tests
    ├── test_reid.py             # Feature extraction & gallery matching tests
    └── test_render.py           # Color hashing & mask blending tests
```

---

## Tests

```bash
uv run pytest tests/ -v
```

All 11 tests cover the core behaviors:
- Re-identifying objects after they leave and return, restoring their original ID with the `[RETRACK]` badge.
- Telling apart multiple objects of the same class when they re-enter.
- Falling back to new IDs when ReID is disabled.
- Kalman filter math staying numerically stable.

---

## License

This project is licensed under the MIT License.
