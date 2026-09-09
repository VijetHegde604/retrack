# ReTrack

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-11%20passed-brightgreen.svg)](tests/)

**ReTrack** is a high-performance, real-time Multi-Object Tracking (MOT) system designed for edge devices and CPU-constrained environments. It combines **ByteTrack's two-stage motion association** with **selective appearance Re-Identification (ReID)** and a **persistent multi-view gallery**.

Traditional trackers suffer from **track fragmentation**: when an object leaves the frame, is occluded, or displaced by camera movement, its trajectory is lost and a brand-new ID is assigned upon re-entry. ReTrack restores the original tracking ID without running heavy neural networks on every frame.
---

## Demo

Watch ReTrack in action on a sample indoor sequence demonstrating persistent object tracking, selective appearance ReID, and ROI-bounded mask blending:
<div align="center">
  <video src="[outputs/test.mp4](https://github.com/user-attachments/assets/80ae595d-dd5b-4080-8588-2adbee7919ce)" width="480" controls="controls" muted="muted" loop="loop">
    Your browser does not support the video tag.
  </video>
  <p>
    🎬 <strong>Sample Tracking Output:</strong> <a href="outputs/test.mp4"><code>outputs/test.mp4</code></a>
  </p>
</div>

---

## Key Features

- **Persistent Re-Identification**: Restores original tracking IDs when objects re-enter after long intervals, occlusions, or sudden camera pans.
- **Selective ReID Execution**: Extracts deep embeddings (MobileNetV3-Small) only for unmatched candidates and periodic updates. Regular frames run at **0 ms ReID overhead**.
- **Two-Stage ByteTrack Association**: Associates high-confidence detections first, then recovers occluded objects from low-confidence detections ($0.1 \le s < 0.5$).
- **Multi-View Exemplar Gallery**: Maintains up to 6 diverse viewpoint vectors plus an Exponential Moving Average (EMA) profile per object to prevent appearance drift.
- **ROI-Bounded Mask Blending**: Blends segmentation masks strictly within bounding-box ROIs using integer arithmetic, achieving a **5.1x speedup on 4K frames**.
- **Deterministic Color Consistency**: Colors are deterministically hashed from track IDs, ensuring returning objects retain their original color.
- **Real-Time CPU Throughput**: Runs at 30–45 FPS on modern CPUs without requiring an external GPU.

---

## Architecture at a Glance

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

### 1. Run on a Video File
```bash
uv run retrack path/to/video.mp4
```

### 2. Run with Live Webcam
```bash
uv run retrack --webcam
```

### 3. Save Annotated Output Video
```bash
uv run retrack path/to/video.mp4 --output outputs/tracked.mp4
```
*(See [`outputs/test.mp4`](outputs/test.mp4) for an example annotated output.)*

### 4. Headless Mode (Servers / Benchmarking)
```bash
uv run retrack path/to/video.mp4 --no-display --output outputs/tracked.mp4
```

---

## CLI Options Reference

```bash
retrack [source] [options]
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `source` | `None` | Path to input video file (positional argument). |
| `--webcam` | `False` | Use the default connected webcam. |
| `--model` | `yolo11s-seg.pt` | YOLO model path or name (e.g. `yolo11n.pt` for max speed). |
| `--device` | `auto` | Inference device: `'auto'`, `'cpu'`, `'cuda'`, or `'mps'`. |
| `--imgsz` | `640` | Inference image resolution (e.g. `640` for fast CPU throughput). |
| `--conf` | `0.35` | Minimum object detection confidence threshold. |
| `--reid-thresh` | `0.65` | Cosine similarity threshold for appearance re-identification. |
| `--lost-buffer` | `30` | Missed frames before a track moves to the persistent gallery. |
| `--no-reid` | `False` | Disables ReID (reverts to motion-only ByteTrack). |
| `--mask-alpha` | `0.45` | Opacity of segmentation masks (`0.0` for bounding boxes only). |
| `--output` | `None` | Optional path to write the annotated MP4 video file. |
| `--no-display` | `False` | Disables OpenCV display window for headless runs. |
| `--max-frames` | `None` | Maximum number of frames to process before exiting. |

---

## Project Structure

```
retrack/
├── REPORT.md                    # In-depth technical & architecture report
├── README.md                    # Project overview & quickstart guide
├── pyproject.toml               # Package dependencies & CLI entrypoints
├── flake.nix                    # Nix devShell with system dependencies
├── outputs/
│   └── test.mp4                 # Sample annotated output video
├── src/retrack/
│   ├── __main__.py              # CLI entrypoint & video processing loop
│   ├── byte_tracker.py          # 3-Stage ByteTrack + ReID association engine
│   ├── detector.py              # YOLO detection & instance segmentation wrapper
│   ├── kalman.py                # 8D Kalman filter & vectorized broadcast IoU
│   ├── reid.py                  # MobileNetV3-Small extractor & persistent gallery
│   ├── render.py                # ROI-bounded mask blending & visual badging
│   ├── tracker.py               # Abstract Tracker and Track data structures
│   └── video.py                 # Thread-safe frame generator
└── tests/
    ├── test_kalman.py           # Kalman motion dynamics & IoU math tests
    ├── test_persistent_tracker.py # Re-entry restoration & disambiguation tests
    ├── test_reid.py             # Feature normalization & gallery matching tests
    └── test_render.py           # Color hashing & mask blending tests
```

---

## Automated Verification & Tests

ReTrack includes a full test suite covering Kalman filter mechanics, IoU geometry, appearance normalization, and persistent ID restoration:

```bash
# Run tests with uv
uv run pytest tests/ -v

# Or via Nix develop environment
nix develop --command uv run pytest tests/ -v
```

All 11 unit and integration tests validate core system behaviors, including:
- Re-identifying objects after frame exits and restoring their persistent ID with the `[RETRACK]` badge.
- Disambiguating multiple same-class objects upon re-entry.
- Proper fallback to new IDs when ReID is disabled.
- Mathematical consistency of Kalman covariance bounds and aspect ratio formulations.

---

## Technical Documentation

For the complete theoretical analysis, mathematical derivations (Kalman state transitions, Cholesky solver), performance benchmarks, and interview Q&A, refer to:

👉 **[Read the Full Technical Architecture Report (REPORT.md)](REPORT.md)**

---

## License

This project is licensed under the MIT License.
