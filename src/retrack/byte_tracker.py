from __future__ import annotations

import numpy as np
import supervision as sv

from retrack.detector import Detection
from retrack.tracker import Track, Tracker


class ByteTrackTracker(Tracker):
    """ByteTrack adapter for ReTrack."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(
        self,
        detections: list[Detection],
    ) -> list[Track]:
        supervision_detections = sv.Detections(
            xyxy=np.asarray(
                [d.bbox for d in detections],
                dtype=np.float32,
            ),
            confidence=np.asarray(
                [d.confidence for d in detections],
                dtype=np.float32,
            ),
            class_id=np.asarray(
                [d.class_id for d in detections],
                dtype=np.int32,
            ),
        )

        tracked = self.tracker.update_with_detections(supervision_detections)

        if tracked.tracker_id is None:
            return []

        tracks: list[Track] = []

        for index, track_id in enumerate(tracked.tracker_id):
            detection = detections[index]

            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    mask=detection.mask,
                )
            )

        return tracks
