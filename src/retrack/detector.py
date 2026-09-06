from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass(slots=True)
class Detection:
    """A single object detection."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    mask: np.ndarray | None = None


class Detector:
    """Object detector backed by a YOLO model."""

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
                mask = None

                # Segmentation models provide one mask for each box. Detection-only
                # models leave ``result.masks`` unset, so they still work normally.
                if masks is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(boxes.conf[i]),
                        class_id=int(boxes.cls[i]),
                        mask=mask,
                    )
                )

        return detections
