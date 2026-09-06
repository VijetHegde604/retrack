"""OpenCV drawing helpers for visualizing detections and segmentation masks."""

import cv2
import numpy as np

from retrack.detector import Detection

# High-contrast BGR colours. Keeping this palette fixed makes an object's
# visualization consistent between frames while still distinguishing instances.
MASK_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 99, 71),
    (60, 180, 75),
    (255, 191, 0),
    (255, 105, 180),
    (0, 191, 255),
    (138, 43, 226),
    (0, 215, 255),
    (205, 50, 154),
)


def detection_color(
    detection: Detection,
    index: int,
) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for a detection instance."""

    return MASK_COLORS[(detection.class_id + index) % len(MASK_COLORS)]


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    *,
    mask_alpha: float = 0.45,
) -> None:
    """Draw segmentation masks, outlines, bounding boxes, and labels.

    Masks are resized to the source frame resolution when necessary.
    """

    height, width = frame.shape[:2]

    for index, detection in enumerate(detections):
        color = detection_color(detection, index)

        # Make sure the mask matches the frame dimensions.
        mask = _frame_mask(
            detection.mask,
            width,
            height,
        )

        # Draw segmentation mask.
        if mask is not None:
            frame[mask] = (
                frame[mask].astype(np.float32) * (1.0 - mask_alpha)
                + np.asarray(color, dtype=np.float32) * mask_alpha
            ).astype(np.uint8)

            # Draw mask outline.
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            cv2.drawContours(
                frame,
                contours,
                -1,
                color,
                2,
                lineType=cv2.LINE_AA,
            )

        # Draw bounding box.
        x1, y1, x2, y2 = map(int, detection.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        # Draw class name and confidence.
        label = f"{detection.class_name}: {detection.confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            lineType=cv2.LINE_AA,
        )


def _frame_mask(
    mask: np.ndarray | None,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Return a boolean mask matching the frame dimensions."""

    if mask is None:
        return None

    target_shape = (height, width)

    # Mask already matches the frame.
    if mask.shape == target_shape:
        return mask.astype(bool)

    # Resize masks emitted at a different inference resolution.
    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized.astype(bool)
