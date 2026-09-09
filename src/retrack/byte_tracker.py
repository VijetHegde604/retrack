"""High-performance Persistent ByteTrack multi-object tracker with appearance ReID.

Combines ByteTrack's robust two-stage motion/IoU data association with a deep
appearance Re-Identification (ReID) gallery to ensure tracking IDs remain persistent
even when objects leave the camera view, undergo severe occlusion, and re-enter.
"""

from __future__ import annotations

from enum import IntEnum
import numpy as np
from scipy.optimize import linear_sum_assignment

from retrack.detector import Detection
from retrack.kalman import KalmanFilter, compute_iou_matrix, xyah_to_xyxy, xyxy_to_xyah
from retrack.reid import GalleryManager, ReIDExtractor
from retrack.tracker import Track, Tracker


class TrackState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Internal single-object track representation."""

    shared_kalman = KalmanFilter()

    def __init__(
        self,
        detection: Detection,
        track_id: int,
    ) -> None:
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.mask = detection.mask
        self.reidentified = False

        self._bbox = detection.bbox
        self.mean, self.covariance = self.shared_kalman.initiate(xyxy_to_xyah(self._bbox))

        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.time_since_update = 0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self._bbox

    def predict(self) -> None:
        """Run Kalman filter prediction."""
        if self.state != TrackState.Tracked:
            self.mean[7] = 0.0  # Zero out vertical velocity when lost
        self.mean, self.covariance = self.shared_kalman.predict(self.mean, self.covariance)
        self._bbox = xyah_to_xyxy(self.mean)

    def activate(self, frame_id: int) -> None:
        """Activate a new tracklet."""
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0

    def re_activate(
        self,
        new_detection: Detection,
        frame_id: int,
        is_reid: bool = False,
    ) -> None:
        """Re-activate a lost or re-identified track."""
        measurement = xyxy_to_xyah(new_detection.bbox)
        if is_reid:
            # When re-identified after a long gap or camera pan, reset Kalman state to measurement
            self.mean, self.covariance = self.shared_kalman.initiate(measurement)
            self.reidentified = True
        else:
            self.mean, self.covariance = self.shared_kalman.update(
                self.mean, self.covariance, measurement
            )
            self.reidentified = False

        self._bbox = xyah_to_xyxy(self.mean)
        self.confidence = new_detection.confidence
        self.mask = new_detection.mask
        self.class_id = new_detection.class_id
        self.class_name = new_detection.class_name

        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0

    def update(self, new_detection: Detection, frame_id: int) -> None:
        """Update track with matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0
        self.reidentified = False

        measurement = xyxy_to_xyah(new_detection.bbox)
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, measurement
        )
        self._bbox = xyah_to_xyxy(self.mean)
        self.confidence = new_detection.confidence
        self.mask = new_detection.mask
        self.state = TrackState.Tracked
        self.is_activated = True

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve linear sum assignment with cost thresholding."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    unmatched_rows: list[int] = list(range(cost_matrix.shape[0]))
    unmatched_cols: list[int] = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= threshold:
            matches.append((r, c))
            if r in unmatched_rows:
                unmatched_rows.remove(r)
            if c in unmatched_cols:
                unmatched_cols.remove(c)

    return matches, unmatched_rows, unmatched_cols


class ByteTrackTracker(Tracker):
    """High-performance Persistent ByteTrack multi-object tracker with selective ReID."""

    def __init__(
        self,
        track_activation_threshold: float = 0.4,
        high_conf_threshold: float = 0.5,
        low_conf_threshold: float = 0.1,
        match_threshold: float = 0.8,
        match_threshold_second: float = 0.5,
        lost_track_buffer: int = 30,
        frame_rate: int = 30,
        reid_threshold: float = 0.65,
        reid_device: str = "auto",
        enable_reid: bool = True,
        reid_update_interval: int = 10,
    ) -> None:
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.match_threshold = match_threshold
        self.match_threshold_second = match_threshold_second
        self.lost_track_buffer = lost_track_buffer
        self.frame_rate = frame_rate
        self.reid_threshold = reid_threshold
        self.enable_reid = enable_reid
        self.reid_update_interval = reid_update_interval

        self.frame_id = 0
        self._next_id = 1

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        # Long-term persistent gallery
        self.gallery = GalleryManager(similarity_threshold=reid_threshold)

        # Lazy-load ReID extractor
        self._reid_extractor: ReIDExtractor | None = None
        self._reid_device = reid_device

    @property
    def reid_extractor(self) -> ReIDExtractor:
        if self._reid_extractor is None:
            self._reid_extractor = ReIDExtractor(device=self._reid_device)
        return self._reid_extractor

    def _allocate_id(self) -> int:
        allocated = self._next_id
        self._next_id += 1
        return allocated

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        """Update tracker with detections and optional video frame for ReID.

        Args:
            detections: List of Detection objects from the detector.
            frame: Optional BGR image frame used for ReID appearance extraction.

        Returns:
            List of active Track objects.
        """
        self.frame_id += 1

        # 1. Split detections into high and low confidence
        high_detections: list[Detection] = []
        low_detections: list[Detection] = []

        for d in detections:
            if d.confidence >= self.high_conf_threshold:
                high_detections.append(d)
            elif d.confidence >= self.low_conf_threshold:
                low_detections.append(d)

        # 2. Predict next positions with Kalman filter
        for track in self.tracked_stracks:
            track.predict()
            track.time_since_update += 1

        for track in self.lost_stracks:
            track.predict()
            track.time_since_update += 1

        # 3. Association Step 1: High-confidence detections with active + lost tracks
        strack_pool = self.tracked_stracks + self.lost_stracks
        if strack_pool and high_detections:
            pool_boxes = np.array([t.bbox for t in strack_pool], dtype=np.float32)
            det_boxes = np.array([d.bbox for d in high_detections], dtype=np.float32)
            iou_mat = compute_iou_matrix(pool_boxes, det_boxes)
            cost_mat = 1.0 - iou_mat

            # Penalty for class mismatch (prevent matching different object categories)
            for r, track in enumerate(strack_pool):
                for c, det in enumerate(high_detections):
                    if track.class_id != det.class_id:
                        cost_mat[r, c] = 1.0

            matches, unmatched_tracks_idx, unmatched_high_idx = linear_assignment(
                cost_mat, threshold=self.match_threshold
            )
        else:
            matches = []
            unmatched_tracks_idx = list(range(len(strack_pool)))
            unmatched_high_idx = list(range(len(high_detections)))

        activated_stracks: list[STrack] = []
        refind_stracks: list[STrack] = []

        for r_idx, c_idx in matches:
            track = strack_pool[r_idx]
            det = high_detections[c_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, is_reid=False)
                refind_stracks.append(track)

        # 4. Association Step 2: Low-confidence detections with remaining active tracks
        unmatched_active_tracks = [
            strack_pool[i]
            for i in unmatched_tracks_idx
            if strack_pool[i].state == TrackState.Tracked
        ]

        if unmatched_active_tracks and low_detections:
            active_boxes = np.array([t.bbox for t in unmatched_active_tracks], dtype=np.float32)
            low_boxes = np.array([d.bbox for d in low_detections], dtype=np.float32)
            iou_mat = compute_iou_matrix(active_boxes, low_boxes)
            cost_mat = 1.0 - iou_mat

            for r, track in enumerate(unmatched_active_tracks):
                for c, det in enumerate(low_detections):
                    if track.class_id != det.class_id:
                        cost_mat[r, c] = 1.0

            matches_sec, unmatched_sec_tracks_idx, _ = linear_assignment(
                cost_mat, threshold=self.match_threshold_second
            )

            for r_idx, c_idx in matches_sec:
                track = unmatched_active_tracks[r_idx]
                det = low_detections[c_idx]
                track.update(det, self.frame_id)
                activated_stracks.append(track)

            lost_from_active = [unmatched_active_tracks[i] for i in unmatched_sec_tracks_idx]
        else:
            lost_from_active = unmatched_active_tracks

        for track in lost_from_active:
            track.mark_lost()

        # 5. Association Step 3: Persistent ReID for unmatched high-confidence detections
        unmatched_high_detections = [high_detections[i] for i in unmatched_high_idx]
        active_ids = {t.track_id for t in (activated_stracks + refind_stracks)}

        if unmatched_high_detections:
            if self.enable_reid and frame is not None:
                # Extract embeddings for candidate detections
                cand_boxes = [d.bbox for d in unmatched_high_detections]
                embeddings = self.reid_extractor.extract_from_frame(frame, cand_boxes)

                for det, embedding in zip(unmatched_high_detections, embeddings):
                    # Query persistent gallery
                    matched_id, score = self.gallery.match_candidate(
                        candidate_embedding=embedding,
                        class_id=det.class_id,
                        active_ids=active_ids,
                        threshold=self.reid_threshold,
                    )

                    if matched_id is not None:
                        # RE-IDENTIFICATION: Same object has re-entered!
                        # Find existing track or revive from removed/gallery
                        re_track = None
                        for t in self.lost_stracks:
                            if t.track_id == matched_id:
                                re_track = t
                                break

                        if re_track is None:
                            re_track = STrack(det, track_id=matched_id)

                        re_track.re_activate(det, self.frame_id, is_reid=True)
                        refind_stracks.append(re_track)
                        active_ids.add(matched_id)

                        # Update appearance gallery with new view
                        self.gallery.register_or_update(
                            track_id=matched_id,
                            class_id=det.class_id,
                            class_name=det.class_name,
                            embedding=embedding,
                            bbox=det.bbox,
                            frame_idx=self.frame_id,
                        )
                    else:
                        # Genuine new object: allocate new persistent ID
                        new_id = self._allocate_id()
                        new_track = STrack(det, track_id=new_id)
                        new_track.activate(self.frame_id)
                        activated_stracks.append(new_track)
                        active_ids.add(new_id)

                        self.gallery.register_or_update(
                            track_id=new_id,
                            class_id=det.class_id,
                            class_name=det.class_name,
                            embedding=embedding,
                            bbox=det.bbox,
                            frame_idx=self.frame_id,
                        )
            else:
                # ReID disabled or no frame provided: allocate new IDs
                for det in unmatched_high_detections:
                    new_id = self._allocate_id()
                    new_track = STrack(det, track_id=new_id)
                    new_track.activate(self.frame_id)
                    activated_stracks.append(new_track)
                    active_ids.add(new_id)

        # 6. Selective appearance updates for active tracks (to keep gallery fresh)
        if self.enable_reid and frame is not None and activated_stracks:
            tracks_to_update: list[STrack] = []
            for t in activated_stracks:
                # Update if newly confirmed or periodic interval
                if t.tracklet_len == 2 or (t.tracklet_len % self.reid_update_interval == 0):
                    tracks_to_update.append(t)

            if tracks_to_update:
                update_boxes = [t.bbox for t in tracks_to_update]
                update_embeddings = self.reid_extractor.extract_from_frame(frame, update_boxes)
                for t, emb in zip(tracks_to_update, update_embeddings):
                    self.gallery.register_or_update(
                        track_id=t.track_id,
                        class_id=t.class_id,
                        class_name=t.class_name,
                        embedding=emb,
                        bbox=t.bbox,
                        frame_idx=self.frame_id,
                    )

        # 7. Update active and lost track lists
        current_tracked = [
            t for t in self.tracked_stracks
            if t.state == TrackState.Tracked and t.track_id in active_ids
        ]
        # Deduplicate active tracks by track_id
        seen_ids: set[int] = set()
        new_tracked: list[STrack] = []
        for t in activated_stracks + refind_stracks + current_tracked:
            if t.track_id not in seen_ids:
                seen_ids.add(t.track_id)
                new_tracked.append(t)
        self.tracked_stracks = new_tracked

        # Manage lost tracks
        current_lost = [
            t for t in self.lost_stracks
            if t.state == TrackState.Lost and t.track_id not in seen_ids
        ] + lost_from_active

        surviving_lost: list[STrack] = []
        for t in current_lost:
            if t.time_since_update <= self.lost_track_buffer:
                surviving_lost.append(t)
            else:
                t.mark_removed()
                self.removed_stracks.append(t)

        self.lost_stracks = surviving_lost

        # 8. Return public Track objects
        return [
            Track(
                track_id=t.track_id,
                bbox=t.bbox,
                confidence=t.confidence,
                class_id=t.class_id,
                mask=t.mask,
                reidentified=t.reidentified,
            )
            for t in self.tracked_stracks
            if t.is_activated
        ]
