from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(slots=True)
class Detection:
    """A single object detection."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = None


class Detector:
    """Object detector backed by a YOLO segmentation model."""

    def __init__(
        self,
        model: str = "yolo11s-seg.pt",
        confidence: float = 0.4,
    ) -> None:
        self.model = YOLO(model)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR OpenCV frame."""

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device="cpu",
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            masks = result.masks

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                class_id = int(boxes.cls[i])
                class_name = result.names[class_id]

                mask = None

                if masks is not None and i < len(masks.xy):
                    polygon = masks.xy[i]

                    mask = np.zeros(
                        frame.shape[:2],
                        dtype=np.uint8,
                    )

                    if polygon is not None and len(polygon) >= 3:
                        cv2.fillPoly(
                            mask,
                            [polygon.astype(np.int32)],
                            1,
                        )

                    mask = mask.astype(bool)

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(boxes.conf[i]),
                        class_id=class_id,
                        class_name=class_name,
                        mask=mask,
                    )
                )

        return detections
