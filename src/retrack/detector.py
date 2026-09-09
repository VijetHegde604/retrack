"""Object detector backed by a YOLO segmentation or detection model."""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass(slots=True)
class Detection:
    """A single object detection."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = None


def resolve_inference_device(device: str = "auto") -> str:
    """Resolve compute device string for YOLO inference."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class Detector:
    """Object detector backed by a YOLO segmentation model."""

    def __init__(
        self,
        model: str = "yolo11s-seg.pt",
        confidence: float = 0.4,
        device: str = "auto",
        imgsz: int = 640,
    ) -> None:
        self.device = resolve_inference_device(device)
        self.model = YOLO(model)
        self.confidence = confidence
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR OpenCV frame."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections: list[Detection] = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            masks = result.masks
            num_boxes = len(boxes)

            for i in range(num_boxes):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                class_id = int(boxes.cls[i])
                class_name = result.names.get(class_id, f"class_{class_id}")
                conf = float(boxes.conf[i])

                mask = None
                if masks is not None and i < len(masks.xy):
                    polygon = masks.xy[i]
                    if polygon is not None and len(polygon) >= 3:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
                        mask = mask.astype(bool)

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=class_id,
                        class_name=class_name,
                        mask=mask,
                    )
                )

        return detections
