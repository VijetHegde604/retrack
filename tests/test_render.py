import numpy as np
import pytest

from retrack.render import draw_tracks, get_track_color
from retrack.tracker import Track


def test_track_color_determinism():
    c1 = get_track_color(1)
    c1_again = get_track_color(1)
    c2 = get_track_color(2)

    assert c1 == c1_again
    assert isinstance(c1, tuple)
    assert len(c1) == 3


def test_draw_tracks_with_roi_mask():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    mask = np.zeros((400, 400), dtype=bool)
    mask[50:150, 50:150] = True

    track = Track(
        track_id=1,
        bbox=(50.0, 50.0, 150.0, 150.0),
        confidence=0.95,
        class_id=0,
        mask=mask,
        reidentified=True,
    )

    # Frame should be modified without error
    draw_tracks(frame, [track], class_names={0: "suitcase"}, mask_alpha=0.5)

    # Pixels inside the mask should no longer be pure black (0, 0, 0)
    assert np.any(frame[50:150, 50:150] > 0)
    # Outside the bounding box should remain untouched
    assert np.all(frame[300:400, 300:400] == 0)
