import numpy as np
import pytest

from retrack.byte_tracker import ByteTrackTracker
from retrack.detector import Detection


def make_test_frame(pattern: str, size=(600, 800, 3)) -> np.ndarray:
    """Create a synthetic test frame with distinct textures for testing ReID."""
    frame = np.zeros(size, dtype=np.uint8)
    if pattern == "checker":
        # Alternating black/white check pattern
        frame[::20, :, :] = 255
        frame[:, ::20, :] = 255
    elif pattern == "stripes":
        frame[:, ::10, 0] = 200
        frame[:, ::10, 1] = 50
        frame[:, ::10, 2] = 255
    elif pattern == "green_dots":
        frame[::15, ::15, 1] = 255
    return frame


def test_short_term_continuous_tracking():
    tracker = ByteTrackTracker(enable_reid=True)
    frame = make_test_frame("checker")

    # Object moving linearly across 10 frames
    for f in range(10):
        x = 50.0 + f * 5.0
        det = Detection(
            bbox=(x, 50.0, x + 50.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )
        tracks = tracker.update([det], frame=frame)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1


def test_persistent_reid_on_reappearance():
    """Verify that an object that exits for > lost_track_buffer frames recovers its ID upon re-entry."""
    tracker = ByteTrackTracker(
        lost_track_buffer=10,
        enable_reid=True,
        reid_threshold=0.65,
    )

    frame = make_test_frame("stripes")

    # 1. Object appears for first 3 frames
    for f in range(3):
        det = Detection(
            bbox=(50.0, 50.0, 120.0, 140.0),
            confidence=0.92,
            class_id=0,
            class_name="suitcase",
        )
        tracks = tracker.update([det], frame=frame)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    # 2. Object disappears for 25 frames (> lost_track_buffer=10)
    for _ in range(25):
        tracks = tracker.update([], frame=frame)
        assert len(tracks) == 0

    # Verify track has been removed from active/lost pool to gallery
    assert len(tracker.tracked_stracks) == 0
    assert len(tracker.lost_stracks) == 0
    assert 1 in tracker.gallery.gallery

    # 3. Same object reappears at a completely different position (e.g. re-entering from another side)
    reentering_det = Detection(
        bbox=(400.0, 300.0, 470.0, 390.0),
        confidence=0.91,
        class_id=0,
        class_name="suitcase",
    )

    tracks = tracker.update([reentering_det], frame=frame)
    assert len(tracks) == 1
    # CRITICAL: It MUST maintain and recover its original persistent ID: 1
    assert tracks[0].track_id == 1
    assert tracks[0].reidentified is True


def test_non_reid_allocates_new_id_on_reappearance():
    """Verify that with ReID disabled, reappearing objects incorrectly get new IDs (demonstrating why ReID is essential)."""
    tracker = ByteTrackTracker(
        lost_track_buffer=5,
        enable_reid=False,
    )
    frame = make_test_frame("stripes")

    # Initial appearance
    det1 = Detection(bbox=(50.0, 50.0, 100.0, 100.0), confidence=0.9, class_id=0, class_name="box")
    tracks = tracker.update([det1], frame=frame)
    assert tracks[0].track_id == 1

    # Disappears past buffer
    for _ in range(10):
        tracker.update([], frame=frame)

    # Reappears at new location
    det2 = Detection(bbox=(300.0, 300.0, 350.0, 350.0), confidence=0.9, class_id=0, class_name="box")
    tracks = tracker.update([det2], frame=frame)
    # Without ReID, it creates a new ID
    assert tracks[0].track_id == 2


def test_two_objects_persistent_disambiguation():
    """Verify two distinct objects maintain their own unique IDs and don't cross-match."""
    tracker = ByteTrackTracker(lost_track_buffer=5, enable_reid=True)
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Create two visually distinct regions
    frame[50:150, 50:150, 0] = 255   # Blue square
    frame[50:150, 400:500, 2] = 255  # Red square

    det_a = Detection(bbox=(50.0, 50.0, 150.0, 150.0), confidence=0.9, class_id=0, class_name="box")
    det_b = Detection(bbox=(400.0, 50.0, 500.0, 150.0), confidence=0.9, class_id=0, class_name="box")

    tracks = tracker.update([det_a, det_b], frame=frame)
    assert len(tracks) == 2
    id_a = tracks[0].track_id
    id_b = tracks[1].track_id
    assert id_a != id_b

    # Both objects leave for 10 frames
    for _ in range(10):
        tracker.update([], frame=frame)

    # Object B re-enters at a new location (300, 300) with its red pattern
    frame[300:400, 300:400, 2] = 255
    det_b_reenter = Detection(bbox=(300.0, 300.0, 400.0, 400.0), confidence=0.9, class_id=0, class_name="box")

    tracks = tracker.update([det_b_reenter], frame=frame)
    assert len(tracks) == 1
    # Must re-associate with ID B, not ID A!
    assert tracks[0].track_id == id_b

