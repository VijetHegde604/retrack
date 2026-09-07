# ReTrack: CPU-Efficient Multi-Object Tracking with Persistent Appearance Re-Identification

**ReTrack** is a high-performance, real-time Multi-Object Tracking (MOT) system designed for edge and CPU-constrained environments. It bridges the gap between **motion-based short-term tracking (ByteTrack)** and **deep appearance-based long-term re-identification (ReID)**.

Traditional trackers like SORT, ByteTrack, or pure Kalman-filter trackers suffer from **track fragmentation**: when an object temporarily leaves the camera frame, is occluded for several seconds, or changes trajectory during a camera pan, the tracker loses it and assigns a brand-new tracking ID upon re-appearance. ReTrack solves this by maintaining a **Persistent Multi-View Appearance Gallery** that selectively re-identifies returning objects and restores their original tracking IDs without incurring the heavy compute cost of running deep neural networks on every frame.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Theoretical Foundations & Core Algorithms](#theoretical-foundations--core-algorithms)
   - [The Tracking-by-Detection Paradigm](#the-tracking-by-detection-paradigm)
   - [Kalman Filter Motion Estimation](#kalman-filter-motion-estimation)
   - [The Two-Stage ByteTrack Data Association](#the-two-stage-bytetrack-data-association)
   - [Stage 3: Selective Appearance Re-Identification](#stage-3-selective-appearance-re-identification)
3. [Persistent Appearance Gallery & Feature Extraction](#persistent-appearance-gallery--feature-extraction)
   - [Lightweight Embedding Backbone (MobileNetV3-Small)](#lightweight-embedding-backbone-mobilenetv3-small)
   - [Multi-Exemplar FIFO Gallery + Exponential Moving Average (EMA)](#multi-exemplar-fifo-gallery--exponential-moving-average-ema)
   - [Selective ReID Execution Model](#selective-reid-execution-model)
4. [Engineering & Performance Optimizations](#engineering--performance-optimizations)
   - [ROI-Bounded Mask Blending (5x–10x Speedup on 4K)](#roi-bounded-mask-blending-5x10x-speedup-on-4k)
   - [Vectorized Broadcast IoU Computation](#vectorized-broadcast-iou-computation)
   - [Deterministic Color Consistency](#deterministic-color-consistency)
5. [Interview Deep-Dive & Technical Q&A](#interview-deep-dive--technical-qa)
6. [CLI Usage & Options](#cli-usage--options)
7. [Automated Verification & Testing](#automated-verification--testing)

---

## System Architecture

```mermaid
flowchart TD
    RawFrame[Input Video Frame (up to 4K)] --> Det[YOLO Detector (CUDA/MPS/CPU)]
    Det --> SplitDets[Split Detections: High-Conf vs Low-Conf]
    
    subgraph Engine [ReTrack Association Pipeline]
        KF[Kalman Filter Predict: Active & Lost Tracks]
        
        KF --> Stage1[Stage 1: High-Confidence IoU Hungarian Match]
        SplitDets -->|High Conf (>= 0.5)| Stage1
        
        Stage1 -->|Matched| UpdateTrack[Update Track State & Mask]
        Stage1 -->|Unmatched Tracks| Stage2[Stage 2: Low-Confidence IoU Hungarian Match]
        SplitDets -->|Low Conf (0.1 - 0.5)| Stage2
        
        Stage2 -->|Matched| UpdateTrack
        Stage2 -->|Still Unmatched| MarkLost[Move to Lost Pool (Buffer: 30 Frames)]
        MarkLost -->|Exceeded Buffer| Archive[Archive to Long-Term Gallery]
        
        Stage1 -->|Unmatched Candidates| Stage3{Selective ReID Check}
        Stage3 --> ExtractFeat[Extract 576-D L2 Embedding (MobileNetV3)]
        ExtractFeat --> QueryGallery[Cosine Sim against Persistent Gallery]
        
        QueryGallery -->|Score >= Threshold (0.65)| ReIdentified[RETRACK: Restore Original Persistent ID & Warp Kalman]
        QueryGallery -->|Score < Threshold| NewID[Allocate New Persistent ID & Register to Gallery]
        
        ReIdentified --> UpdateTrack
        NewID --> UpdateTrack
    end
    
    UpdateTrack --> SelectiveEMA[Periodic Gallery Refresh (Interval = 10)]
    UpdateTrack --> FastRender[ROI-Optimized Integer Blending & Badging]
    FastRender --> Output[Display / Video Writer]
```

---

## Theoretical Foundations & Core Algorithms

### The Tracking-by-Detection Paradigm

Multi-Object Tracking (MOT) in modern computer vision follows the **Tracking-by-Detection** paradigm:
1. An object detector (e.g., YOLO11) produces a set of spatial observations $\mathcal{D}_t = \{(\mathbf{b}_i, c_i, s_i)\}_{i=1}^{M}$ for each frame $t$, where $\mathbf{b}_i = (x_1, y_1, x_2, y_2)$ is the bounding box, $c_i$ is the semantic class, and $s_i \in [0, 1]$ is the detection confidence score.
2. The tracker associates these observations across consecutive time steps to reconstruct continuous trajectories $\mathcal{T}_k = \{\mathbf{b}_{t_1}^{(k)}, \mathbf{b}_{t_2}^{(k)}, \dots\}$.

### Kalman Filter Motion Estimation

To predict where an object will be in frame $t$ before detections arrive, ReTrack models the object's spatial dynamics using an **8-dimensional state vector** in a constant-velocity linear dynamical system:

$$\mathbf{x} = [x_c, y_c, a, h, \dot{x}_c, \dot{y}_c, \dot{a}, \dot{h}]^T$$

where:
- $(x_c, y_c)$ is the bounding box center coordinate.
- $a = \frac{w}{h}$ is the aspect ratio of the bounding box.
- $h$ is the bounding box height.
- $(\dot{x}_c, \dot{y}_c, \dot{a}, \dot{h})$ represent their respective first-order temporal derivatives (velocities).

#### Why aspect ratio ($a$) and height ($h$) instead of $(w, h)$?
In real-world camera geometry, an object changing distance from the camera exhibits scale changes where width and height scale proportionally. Parameterizing the state with aspect ratio ($a$) provides structural stability because aspect ratio changes much more slowly and smoothly than absolute pixel width during perspective scale variations.

#### Kalman Prediction Step
The state transition model assumes discrete time step $\Delta t = 1$:

$$\mathbf{x}_{t|t-1} = \mathbf{F} \mathbf{x}_{t-1|t-1}$$

$$\mathbf{P}_{t|t-1} = \mathbf{F} \mathbf{P}_{t-1|t-1} \mathbf{F}^T + \mathbf{Q}$$

where the transition matrix $\mathbf{F} \in \mathbb{R}^{8 \times 8}$ is:

$$\mathbf{F} = \begin{bmatrix} \mathbf{I}_{4 \times 4} & \mathbf{I}_{4 \times 4} \Delta t \\ \mathbf{0}_{4 \times 4} & \mathbf{I}_{4 \times 4} \end{bmatrix}$$

and $\mathbf{Q} \in \mathbb{R}^{8 \times 8}$ is the process noise covariance matrix scaled adaptively by the object's height $h$.

#### Kalman Correction (Update) Step
Given a detection measurement $\mathbf{z}_t = [x_c, y_c, a, h]^T \in \mathbb{R}^4$:

$$\mathbf{y}_t = \mathbf{z}_t - \mathbf{H} \mathbf{x}_{t|t-1} \quad (\text{Measurement Innovation})$$

$$\mathbf{S}_t = \mathbf{H} \mathbf{P}_{t|t-1} \mathbf{H}^T + \mathbf{R} \quad (\text{Innovation Covariance})$$

$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}^T \mathbf{S}_t^{-1} \quad (\text{Kalman Gain})$$

$$\mathbf{x}_{t|t} = \mathbf{x}_{t|t-1} + \mathbf{K}_t \mathbf{y}_t$$

$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_{t|t-1}$$

*Implementation Detail*: To guarantee numerical stability when inverting $\mathbf{S}_t$, ReTrack uses **Cholesky decomposition** (`scipy.linalg.cho_factor` and `cho_solve`) rather than direct matrix inversion.

---

### The Two-Stage ByteTrack Data Association

Standard trackers (such as SORT) discard all detections below a rigid confidence threshold (e.g., $s < 0.5$). However, in complex scenes, partially occluded objects, objects in motion blur, or objects entering shadows frequently have low confidence scores (e.g., $0.15 \le s < 0.45$). Discarding them causes broken trajectories and ID switches.

ByteTrack introduces a **two-stage association strategy**:

1. **Stage 1 (High-Confidence Association)**:
   - Detections with $s \ge \tau_{\text{high}}$ (e.g., 0.5) are matched against all active and recently lost tracks ($\mathcal{T}_{\text{tracked}} \cup \mathcal{T}_{\text{lost}}$).
   - Cost matrix: $C_{i, j} = 1.0 - \text{IoU}(\mathbf{b}_i, \mathbf{b}_j)$. A class-mismatch penalty is applied ($C_{i,j} = 1.0$ if $c_i \ne c_j$).
   - Optimal bipartite matching is solved via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
   - Pairs with $\text{IoU} \ge 0.2$ ($C \le 0.8$) are confirmed as active matches.

2. **Stage 2 (Low-Confidence Recovery)**:
   - Remaining unmatched tracks from Stage 1 are matched against low-confidence detections ($0.1 \le s < \tau_{\text{high}}$).
   - Because these detections have lower confidence, we require a stricter spatial overlap threshold ($\text{IoU} \ge 0.5$, $C \le 0.5$) to prevent hallucinated tracks.
   - This recovers occluded tracks without creating false positive tracks from background noise.

---

### Stage 3: Selective Appearance Re-Identification

**The ByteTrack Limitation**: What happens to unmatched high-confidence detections? In vanilla ByteTrack, every unmatched high-confidence detection immediately receives `next_id()`. If an object left the camera view for 2 seconds and returned, it was guaranteed to receive a new ID.

**The ReTrack Solution**:
1. All unmatched high-confidence detections are treated as **candidate re-entry queries**.
2. If a video frame is available, ReTrack extracts an L2-normalized appearance embedding $\mathbf{e}_{\text{cand}} \in \mathbb{R}^{576}$ using MobileNetV3-Small.
3. The candidate is matched against the **Persistent Appearance Gallery** using cosine similarity:

$$S(i, \text{cand}) = \max_{\mathbf{f} \in \mathcal{G}_i} \left( \frac{\mathbf{e}_{\text{cand}} \cdot \mathbf{f}}{\|\mathbf{e}_{\text{cand}}\| \|\mathbf{f}\|} \right)$$

4. Constraints applied:
   - **Semantic Class Gating**: Only tracks with matching `class_id` are eligible.
   - **Active Object Exclusion**: Tracks currently visible in the frame are masked out.
   - **ReID Threshold**: If $\max_i S(i, \text{cand}) \ge \tau_{\text{reid}}$ (default $0.65$):
     - **Re-identification Success!** The candidate detection inherits the persistent `track_id`.
     - Its Kalman filter state is re-initialized / warped to the new location (clearing accumulated positional uncertainty).
     - It receives a visible `[RETRACK]` badge.
5. Only if no match exceeds $\tau_{\text{reid}}$ is a new global ID allocated.

---

## Persistent Appearance Gallery & Feature Extraction

### Lightweight Embedding Backbone (MobileNetV3-Small)

Instead of running a heavy ResNet-50 or ViT ReID model that would saturate CPU capacity:
- ReTrack uses **MobileNetV3-Small** with its 1000-class classification head replaced by an `Identity()` layer.
- Architecture highlights: Hard-swish activations, squeeze-and-excitation blocks, depthwise-separable convolutions.
- Yields a compact **576-dimensional feature vector**.
- Benchmarked CPU extraction latency: **~5.2 ms** for a batch of 4 crops.

### Multi-Exemplar FIFO Gallery + Exponential Moving Average (EMA)

A single feature vector cannot capture an object viewed from multiple angles (e.g., front vs. back vs. profile). ReTrack combines two representation strategies:

1. **Multi-Exemplar FIFO Gallery**:
   - Stores up to $K = 6$ diverse exemplar embeddings per object.
   - When a new embedding arrives, it is only added if it provides diverse visual information ($\max_{f \in \mathcal{G}} (\mathbf{e} \cdot \mathbf{f}) < 0.95$), preventing redundant copies of identical frames from flushing out distinct viewpoints.
2. **Exponential Moving Average (EMA) Feature**:
   - Maintains a running average vector:

$$\mathbf{e}_{\text{EMA}}^{(t)} = \alpha \mathbf{e}_{\text{EMA}}^{(t-1)} + (1 - \alpha) \mathbf{e}_t \quad (\alpha = 0.85)$$

   - Followed by unit L2 normalization: $\mathbf{e}_{\text{EMA}} \leftarrow \frac{\mathbf{e}_{\text{EMA}}}{\|\mathbf{e}_{\text{EMA}}\|_2}$.

### Selective ReID Execution Model

To achieve real-time performance on CPU, deep feature extraction is **not** executed on every bounding box on every frame. Instead, it is triggered selectively:

| Event | Action | Rationale |
| :--- | :--- | :--- |
| **Normal continuous tracking** | Pure IoU + Kalman Filter | **0 ms ReID overhead**. Motion model handles smooth trajectories. |
| **Unmatched candidate detection** | Feature extraction & Gallery Query | Evaluates if the candidate is a returning object. |
| **Confirmed new track** | Initial gallery registration | Captures baseline appearance profile. |
| **Periodic update ($K = 10$)** | Background EMA update | Adapts to slow lighting/scale changes without burning CPU. |

---

## Engineering & Performance Optimizations

### ROI-Bounded Mask Blending (5x–10x Speedup on 4K)

On 4K video ($3840 \times 2160$ = 8,294,400 pixels per frame):
- **Naive Implementation**: Boolean indexing across the full frame: `frame[mask] = frame[mask] * (1 - alpha) + color * alpha`. This casts up to 8.3 million float32 numbers per detection per frame, reducing playback to 1–2 FPS.
- **ReTrack Optimization**:
  1. Crop the bounding box sub-region: $\text{ROI} = \text{frame}[y_1:y_2, x_1:x_2]$.
  2. Crop the mask sub-region: $\text{sub\_mask} = \text{mask}[y_1:y_2, x_1:x_2]$.
  3. Execute blended overlay using **integer fixed-point arithmetic**:

$$\text{ROI}[\text{sub\_mask}] = \frac{\text{ROI}[\text{sub\_mask}] \times (256 - \alpha_{256}) + \text{color} \times \alpha_{256}}{256}$$

  4. Draw contours localized to the ROI coordinates shifted by $(x_1, y_1)$.

*Benchmark Result*:
- Full-frame blend: **43.5 ms** per mask.
- ROI integer blend: **8.5 ms** per mask (**~5.1x speedup**).

### Vectorized Broadcast IoU Computation

Rather than using nested Python loops to compare $N$ tracks and $M$ detections:
```python
# Broadcast arrays of shape (N, 1, 2) and (1, M, 2)
top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
wh = np.maximum(0.0, bottom_right - top_left)
intersection = wh[:, :, 0] * wh[:, :, 1]
union = area_a[:, None] + area_b[None, :] - intersection
iou_matrix = intersection / np.maximum(union, 1e-6)
```
Executes in under **0.1 ms** in pure NumPy.

### Deterministic Color Consistency

Previous implementations assigned colors based on detection loop index (`index % len(colors)`), which caused an object's color to flicker every frame. ReTrack computes a deterministic hash:

$$\text{color} = \text{PALETTE}[\text{track\_id} \pmod{|\text{PALETTE}|}]$$

When an object re-enters after leaving the scene, because it re-acquires its persistent `track_id`, its visual color remains 100% identical.

---

## Interview Deep-Dive & Technical Q&A

### 1. What is the fundamental difference between SORT, DeepSORT, ByteTrack, and ReTrack?
- **SORT**: Uses Kalman filter + Hungarian algorithm with bounding box IoU. Very fast, but suffers from severe ID switches during occlusions and discards low-confidence detections.
- **DeepSORT**: Adds a CNN feature extractor to SORT and matches using cosine distance. Computes embeddings for all detections on every frame, which is computationally expensive on CPU. Still drops low-confidence detections.
- **ByteTrack**: Introduces two-stage association for high- and low-confidence detections, achieving state-of-the-art short-term tracking using motion alone. However, it has **no long-term memory**—any object leaving the frame gets a new ID upon return.
- **ReTrack (This Project)**: Combines ByteTrack's two-stage low-confidence recovery with a long-term **Persistent ReID Gallery**. Employs **Selective ReID** (computing embeddings only on-demand for unmatched candidates and periodic updates) to achieve long-term persistence at CPU real-time speeds.

### 2. Why is pure motion tracking insufficient for long-term association?
Kalman filters assume a linear constant-velocity motion model. Over long intervals ($> 1$–2 seconds):
- Covariance $\mathbf{P}$ grows unbounded, making spatial predictions meaningless.
- Objects can turn, stop, accelerate, or leave the camera frame entirely and re-enter from a different angle.
- When an object re-enters, its spatial IoU with its old predicted position is $0$. Only visual appearance features can associate it back to its identity.

### 3. How do you prevent identity stealing (ID collision) between two identical-looking objects?
1. **Short-Term Priority**: High-confidence detections are first matched to active tracks via IoU and Kalman filter. If Object A and Object B are both visible, their spatial trajectories prevent them from competing for the same detection.
2. **Active-ID Exclusion**: When querying the persistent gallery, any `track_id` currently visible in the active tracking pool is masked out. An active object's ID cannot be claimed by another detection.
3. **Multi-Object Hungarian Matching**: When multiple candidates query the gallery simultaneously, global 1-to-1 matching prevents two detections from claiming the same gallery ID.

### 4. Why use MobileNetV3-Small instead of an OSNet or ResNet-50 trained specifically on Market-1501?
- **Domain Generality**: Market-1501 models are specialized solely for pedestrian re-identification and fail on general objects (suitcases, laptops, vehicles, pets, furniture). MobileNetV3 pretrained on ImageNet extracts rich generic semantic and texture representations that generalize across all 80 COCO categories.
- **Compute Budget**: MobileNetV3-Small takes ~5ms for 4 crops on CPU, whereas ResNet-50 takes >80ms on CPU, which would bottleneck the pipeline.

### 5. What happens if camera motion is abrupt (camera pan/shake)?
- Pure IoU trackers fail under camera pans because all object bounding boxes jump simultaneously.
- In ReTrack, if a sudden camera pan causes IoU matching to fail in Stage 1 and Stage 2:
  - The object falls through to Stage 3 (ReID check).
  - Its appearance embedding matches its stored gallery profile with high cosine similarity ($\sim 0.85$–$0.92$).
  - ReTrack re-identifies the object, **warps the Kalman filter state** directly to the new coordinates, and maintains track continuity.

### 6. How is appearance drift handled in the gallery?
Appearance drift occurs when lighting changes gradually or the object slowly rotates:
- If we never update the gallery, the initial embedding may no longer match the object viewed from the opposite side.
- If we update naively every frame with 100% weight, a single occluded or corrupted detection will pollute the gallery.
- **ReTrack's Solution**: Uses a multi-view exemplar set ($K = 6$) that only admits diverse views ($\text{sim} < 0.95$) combined with an exponential moving average ($\alpha = 0.85$). This preserves historical identity while accommodating viewpoint shifts.

### 7. What is the computational complexity of the association pipeline?
- **IoU Matrix**: $\mathcal{O}(N \times M)$ floating-point operations, vectorized in NumPy ($< 0.1\text{ ms}$).
- **Hungarian Algorithm**: $\mathcal{O}(\min(N, M)^3)$ using the Jonker-Volgenant algorithm in SciPy. For typical scenes ($N, M \le 50$), this runs in $< 0.5\text{ ms}$.
- **ReID Matching**: For $C$ unmatched candidates and $G$ gallery objects, cosine similarity is a matrix multiplication $\mathcal{O}(C \times G \times 576)$. Even with 1000 gallery objects, this matrix multiplication takes $< 1\text{ ms}$.

---

## CLI Usage & Options

### Run on a Video File
```bash
uv run retrack path/to/video.mp4
```

### Run with Live Webcam
```bash
uv run retrack --webcam
```

### Save Output Video
```bash
uv run retrack path/to/video.mp4 --output outputs/tracked_video.mp4
```

### Headless Mode (for Servers / Benchmarking)
```bash
uv run retrack path/to/video.mp4 --no-display --output outputs/tracked_video.mp4
```

### Advanced Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | `yolo11s-seg.pt` | YOLO model path (e.g., `yolo11n.pt` for ultra-fast speed). |
| `--device` | `auto` | Inference device: `'auto'`, `'cuda'`, `'cpu'`, or `'mps'`. |
| `--imgsz` | `640` | Inference resolution (e.g., `640` for fast CPU throughput). |
| `--conf` | `0.35` | Minimum detection confidence threshold. |
| `--reid-thresh` | `0.65` | Cosine similarity threshold for ReID object recovery. |
| `--lost-buffer` | `30` | Number of missed frames before track is moved to long-term gallery. |
| `--no-reid` | `False` | Disables ReID (falls back to pure motion ByteTrack). |
| `--mask-alpha` | `0.45` | Segmentation mask opacity ($0.0$ for outlines only). |
| `--max-frames` | `None` | Process up to $N$ frames and exit. |

---

## Automated Verification & Testing

The repository contains a test suite covering Kalman filter mechanics, IoU geometry, ReID normalization, gallery matching, continuous tracking, long-term persistence, and multi-object disambiguation.

To run the full test suite:
```bash
pytest tests/ -v
```

### Test Suite Structure:
- `tests/test_kalman.py`: Bounding box transformations $(x_1, y_1, x_2, y_2) \leftrightarrow (x_c, y_c, a, h)$, pairwise IoU matrices, and Kalman predict/update covariance bounds.
- `tests/test_reid.py`: Verifies unit L2 normalization of feature vectors, multi-exemplar cosine matching, semantic class filtering, and active-ID exclusion.
- `tests/test_persistent_tracker.py`:
  - `test_short_term_continuous_tracking`: Verifies steady-state trajectory tracking without ID flicker.
  - `test_persistent_reid_on_reappearance`: Simulates an object exiting the frame for 25 frames ($> \text{buffer}$) and re-entering at a distant coordinate, asserting that ID 1 is successfully restored with `reidentified = True`.
  - `test_non_reid_allocates_new_id_on_reappearance`: Confirms that without ReID, the tracker fails to associate and creates ID 2.
  - `test_two_objects_persistent_disambiguation`: Verifies two distinct objects of the same class never swap IDs upon re-entry.
- `tests/test_render.py`: Deterministic color generation and ROI-bounded mask blending.
