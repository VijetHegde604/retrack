import numpy as np
import pytest

from retrack.kalman import (
    KalmanFilter,
    compute_iou_matrix,
    xyah_to_xyxy,
    xyxy_to_xyah,
)


def test_bbox_conversions_roundtrip():
    bbox = (100.0, 150.0, 200.0, 350.0)
    xyah = xyxy_to_xyah(bbox)
    recovered = xyah_to_xyxy(xyah)

    assert pytest.approx(recovered[0], abs=1e-3) == bbox[0]
    assert pytest.approx(recovered[1], abs=1e-3) == bbox[1]
    assert pytest.approx(recovered[2], abs=1e-3) == bbox[2]
    assert pytest.approx(recovered[3], abs=1e-3) == bbox[3]


def test_compute_iou_matrix():
    boxes_a = np.array([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
    ], dtype=np.float32)

    boxes_b = np.array([
        [0, 0, 10, 10],    # Perfect match with a[0]
        [0, 0, 5, 10],     # Half overlap with a[0]
        [50, 50, 60, 60],  # No overlap with either
    ], dtype=np.float32)

    iou = compute_iou_matrix(boxes_a, boxes_b)
    assert iou.shape == (2, 3)
    assert pytest.approx(iou[0, 0], abs=1e-4) == 1.0
    assert pytest.approx(iou[0, 1], abs=1e-4) == 0.5
    assert pytest.approx(iou[0, 2], abs=1e-4) == 0.0
    assert pytest.approx(iou[1, 0], abs=1e-4) == 0.0


def test_kalman_filter_lifecycle():
    kf = KalmanFilter()
    bbox = (100.0, 100.0, 200.0, 200.0)
    xyah = xyxy_to_xyah(bbox)

    mean, cov = kf.initiate(xyah)
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)

    # Prediction
    pred_mean, pred_cov = kf.predict(mean, cov)
    assert pred_mean.shape == (8,)

    # Update with new measurement (shifted slightly)
    new_meas = xyxy_to_xyah((105.0, 105.0, 205.0, 205.0))
    upd_mean, upd_cov = kf.update(pred_mean, pred_cov, new_meas)

    # Center position should move towards measurement
    assert upd_mean[0] > mean[0]
    assert upd_mean[1] > mean[1]
