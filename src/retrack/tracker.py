from dataclasses import dataclass
from typing import List, Tuple



BBox = Tuple[float, float, float, float]


@dataclass
class Track:
    track_id: int
    bbox: BBox
    class_id: int
    class_name: str


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3):
        self.next_id = 1
        self.tracks: List[Track] = []
        self.iou_threshold = iou_threshold

    def update(self, detections):
        """
        Match new detections with existing tracks using IoU.
        Unmatched detections receive new IDs.
        """

        updated_tracks = []
        used_track_ids = set()

        for detection in detections:
            best_track = None
            best_iou = 0.0

            for track in self.tracks:
                if track.track_id in used_track_ids:
                    continue

                if track.class_id != detection.class_id:
                    continue

                score = self.iou(track.bbox, detection.bbox)

                if score > best_iou:
                    best_iou = score
                    best_track = track

            if best_track is not None and best_iou >= self.iou_threshold:
                track = Track(
                    track_id=best_track.track_id,
                    bbox=detection.bbox,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )

                updated_tracks.append(track)
                used_track_ids.add(best_track.track_id)

            else:
                track = Track(
                    track_id=self.next_id,
                    bbox=detection.bbox,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                )

                self.next_id += 1
                updated_tracks.append(track)

        self.tracks = updated_tracks

        return self.tracks

    @staticmethod
    def iou(box_a: BBox, box_b: BBox) -> float:
        """
        Calculate Intersection over Union between two boxes.
        """

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(0.0, ix2 - ix1)
        intersection_height = max(0.0, iy2 - iy1)

        intersection = (
            intersection_width * intersection_height
        )

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union == 0:
            return 0.0

        return intersection / union