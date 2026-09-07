from dataclasses import dataclass

import numpy as np

from retrack.detector import Detection


@dataclass(slots=True)
class Track:
    """A tracked object."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    mask: np.ndarray | None = None


class Tracker:
    """Base interface for object trackers."""

    def update(
        self,
        detections: list[Detection],
    ) -> list[Track]:
        raise NotImplementedError
