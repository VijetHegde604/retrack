"""OpenCV drawing helpers for visualizing detections, tracks, and segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np

from retrack.detector import Detection
from retrack.tracker import Track

# High-contrast, visually distinct BGR color palette
TRACK_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 99, 71),    # Tomato / Coral
    (60, 180, 75),    # Emerald green
    (255, 191, 0),    # Amber
    (255, 105, 180),  # Hot pink
    (0, 191, 255),    # Deep sky blue
    (138, 43, 226),   # Blue violet
    (0, 215, 255),    # Gold
    (205, 50, 154),   # Medium violet red
    (0, 250, 154),    # Medium spring green
    (238, 130, 238),  # Violet
    (30, 144, 255),   # Dodger blue
    (255, 140, 0),    # Dark orange
)


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for a persistent track ID."""
    return TRACK_COLORS[abs(track_id) % len(TRACK_COLORS)]


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    class_names: dict[int, str] | None = None,
    *,
    mask_alpha: float = 0.45,
) -> None:
    """Draw persistent tracks, bounding boxes, segmentation masks, and ReID badges.

    Optimized to blend masks only within the bounding-box ROI for real-time speed.
    """
    h, w = frame.shape[:2]
    class_names = class_names or {}

    for track in tracks:
        color = get_track_color(track.track_id)
        x1, y1, x2, y2 = map(int, track.bbox)

        # Clamp bounding box coordinates to frame boundaries
        ix1 = max(0, min(w - 1, x1))
        iy1 = max(0, min(h - 1, y1))
        ix2 = max(ix1 + 1, min(w, x2))
        iy2 = max(iy1 + 1, min(h, y2))

        # 1. Draw segmentation mask if present (ROI-optimized blending)
        if track.mask is not None and mask_alpha > 0.0:
            sub_mask = track.mask[iy1:iy2, ix1:ix2]
            if sub_mask.any():
                roi = frame[iy1:iy2, ix1:ix2]
                alpha_int = int(mask_alpha * 256)
                inv_alpha = 256 - alpha_int

                color_arr = np.asarray(color, dtype=np.uint16)
                # Fast integer blending
                roi[sub_mask] = (
                    (roi[sub_mask].astype(np.uint16) * inv_alpha + color_arr * alpha_int) // 256
                ).astype(np.uint8)

                # Draw mask contour outline
                contours, _ = cv2.findContours(
                    sub_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for cnt in contours:
                    cnt[:, :, 0] += ix1
                    cnt[:, :, 1] += iy1
                cv2.drawContours(frame, contours, -1, color, 2, lineType=cv2.LINE_AA)

        # 2. Draw bounding box
        cv2.rectangle(
            frame,
            (ix1, iy1),
            (ix2, iy2),
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        # 3. Label text
        cls_name = class_names.get(track.class_id, f"obj_{track.class_id}")
        reid_tag = " [RETRACK]" if track.reidentified else ""
        label = f"{cls_name} ID:{track.track_id}{reid_tag}"

        # Measure text for background badge with dynamic resolution scaling
        scale = max(0.5, w / 2200.0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(round(scale * 2.2)))
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        pad = max(4, int(6 * scale))

        badge_y1 = max(0, iy1 - th - 2 * pad)
        badge_y2 = badge_y1 + th + 2 * pad
        badge_x2 = min(w, ix1 + tw + 2 * pad)

        # Draw filled background rectangle for readability
        badge_color = (0, 200, 255) if track.reidentified else color
        cv2.rectangle(
            frame,
            (ix1, badge_y1),
            (badge_x2, badge_y2),
            badge_color,
            -1,
        )

        # Text color (black for contrast on bright badge)
        text_color = (0, 0, 0)
        cv2.putText(
            frame,
            label,
            (ix1 + pad, badge_y2 - pad - 2),
            font,
            scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    *,
    mask_alpha: float = 0.45,
) -> None:
    """Draw raw detections and segmentation masks (fallback/debug mode)."""
    h, w = frame.shape[:2]

    for index, detection in enumerate(detections):
        color = TRACK_COLORS[(detection.class_id + index) % len(TRACK_COLORS)]
        x1, y1, x2, y2 = map(int, detection.bbox)
        ix1, iy1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        ix2, iy2 = max(ix1 + 1, min(w, x2)), max(iy1 + 1, min(h, y2))

        if detection.mask is not None and mask_alpha > 0.0:
            sub_mask = detection.mask[iy1:iy2, ix1:ix2]
            if sub_mask.any():
                roi = frame[iy1:iy2, ix1:ix2]
                alpha_int = int(mask_alpha * 256)
                inv_alpha = 256 - alpha_int
                color_arr = np.asarray(color, dtype=np.uint16)
                roi[sub_mask] = (
                    (roi[sub_mask].astype(np.uint16) * inv_alpha + color_arr * alpha_int) // 256
                ).astype(np.uint8)

        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2, lineType=cv2.LINE_AA)
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (ix1, max(iy1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
