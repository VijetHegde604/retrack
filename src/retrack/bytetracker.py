from dataclasses import dataclass
from typing import List, Tuple


BBox = Tuple[float, float, float, float]


@dataclass
class Track:
    track_id: int
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float


class ByteTracker:
    """
    Simple ByteTrack-style multi-object tracker.

    Main idea:
    1. Split detections into high-confidence and low-confidence.
    2. Match existing tracks with high-confidence detections.
    3. Match remaining tracks with low-confidence detections.
    4. Keep unmatched tracks for a few frames.
    5. Create new IDs for unmatched high-confidence detections.
    """

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        iou_threshold: float = 0.3,
        max_lost: int = 30,
    ) -> None:

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

        self.next_id = 1

        # Currently active tracks
        self.tracks: List[Track] = []

        # Number of frames each track has been missing
        self.lost_frames: dict[int, int] = {}

    def update(self, detections) -> List[Track]:
        """
        Update tracker using detections from the current frame.
        """

        # --------------------------------------------------
        # STEP 1: Split detections by confidence
        # --------------------------------------------------

        high_detections = []
        low_detections = []

        for detection in detections:

            if detection.confidence >= self.high_threshold:
                high_detections.append(detection)

            elif detection.confidence >= self.low_threshold:
                low_detections.append(detection)

        # --------------------------------------------------
        # STEP 2: Match existing tracks with HIGH confidence
        # detections
        # --------------------------------------------------

        matched_tracks = []
        unmatched_tracks = list(self.tracks)
        unmatched_high = list(high_detections)

        matches, unmatched_tracks, unmatched_high = self.match(
            unmatched_tracks,
            unmatched_high,
        )

        for track, detection in matches:

            updated_track = Track(
                track_id=track.track_id,
                bbox=detection.bbox,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
            )

            matched_tracks.append(updated_track)

            # Object was found again
            self.lost_frames[track.track_id] = 0

        # --------------------------------------------------
        # STEP 3: Match remaining tracks with LOW confidence
        # detections
        # --------------------------------------------------

        second_matches, still_unmatched_tracks, _ = self.match(
            unmatched_tracks,
            low_detections,
        )

        for track, detection in second_matches:

            updated_track = Track(
                track_id=track.track_id,
                bbox=detection.bbox,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
            )

            matched_tracks.append(updated_track)

            # Track found using low-confidence detection
            self.lost_frames[track.track_id] = 0

        # --------------------------------------------------
        # STEP 4: Handle tracks that were not matched
        # --------------------------------------------------

        for track in still_unmatched_tracks:

            track_id = track.track_id

            self.lost_frames[track_id] = (
                self.lost_frames.get(track_id, 0) + 1
            )

            # Keep track temporarily
            if self.lost_frames[track_id] <= self.max_lost:
                matched_tracks.append(track)

        # --------------------------------------------------
        # STEP 5: Create new IDs for unmatched HIGH
        # confidence detections
        # --------------------------------------------------

        for detection in unmatched_high:

            new_track = Track(
                track_id=self.next_id,
                bbox=detection.bbox,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
            )

            matched_tracks.append(new_track)

            self.lost_frames[self.next_id] = 0

            self.next_id += 1

        # --------------------------------------------------
        # STEP 6: Remove tracks that have been lost too long
        # --------------------------------------------------

        active_tracks = []

        for track in matched_tracks:

            if self.lost_frames.get(track.track_id, 0) <= self.max_lost:
                active_tracks.append(track)

        self.tracks = active_tracks

        return self.tracks

    def match(
        self,
        tracks: List[Track],
        detections,
    ):
        """
        Match tracks and detections using IoU.

        Returns:
            matches
            unmatched_tracks
            unmatched_detections
        """

        matches = []

        unmatched_tracks = list(tracks)
        unmatched_detections = list(detections)

        # Continue matching until no good match remains
        while True:

            best_track = None
            best_detection = None
            best_iou = 0.0

            for track in unmatched_tracks:

                for detection in unmatched_detections:

                    # Only compare the same object class
                    if track.class_id != detection.class_id:
                        continue

                    score = self.iou(
                        track.bbox,
                        detection.bbox,
                    )

                    if score > best_iou:

                        best_iou = score
                        best_track = track
                        best_detection = detection

            # No acceptable match
            if (
                best_track is None
                or best_detection is None
                or best_iou < self.iou_threshold
            ):
                break

            matches.append(
                (best_track, best_detection)
            )

            unmatched_tracks.remove(best_track)
            unmatched_detections.remove(best_detection)

        return (
            matches,
            unmatched_tracks,
            unmatched_detections,
        )

    @staticmethod
    def iou(
        box_a: BBox,
        box_b: BBox,
    ) -> float:
        """
        Calculate Intersection over Union (IoU).
        """

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        # Intersection box
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(
            0.0,
            ix2 - ix1,
        )

        intersection_height = max(
            0.0,
            iy2 - iy1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        # Area of box A
        area_a = (
            max(0.0, ax2 - ax1)
            * max(0.0, ay2 - ay1)
        )

        # Area of box B
        area_b = (
            max(0.0, bx2 - bx1)
            * max(0.0, by2 - by1)
        )

        # Union
        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union