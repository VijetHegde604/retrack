"""Kalman Filter and bounding-box geometry utilities for ByteTrack.

Implements the standard constant-velocity motion model in [x_center, y_center, aspect_ratio, height]
parameter space, with bounding box transformations and vectorized IoU distance computation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def xyxy_to_xyah(bbox: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) bounding box to [center_x, center_y, aspect_ratio, height]."""
    x1, y1, x2, y2 = bbox
    w = max(0.0, float(x2 - x1))
    h = max(1e-6, float(y2 - y1))
    x = float(x1) + w / 2.0
    y = float(y1) + h / 2.0
    a = w / h
    return np.array([x, y, a, h], dtype=np.float32)


def xyah_to_xyxy(xyah: np.ndarray) -> tuple[float, float, float, float]:
    """Convert [center_x, center_y, aspect_ratio, height] to (x1, y1, x2, y2)."""
    x, y, a, h = xyah[:4]
    w = a * h
    x1 = float(x - w / 2.0)
    y1 = float(y - h / 2.0)
    x2 = float(x + w / 2.0)
    y2 = float(y + h / 2.0)
    return (x1, y1, x2, y2)


def compute_iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of (x1, y1, x2, y2) boxes.

    Args:
        boxes_a: Array of shape (N, 4).
        boxes_b: Array of shape (M, 4).

    Returns:
        IoU matrix of shape (N, M) with values in [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    # Area of boxes_a: (N,)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    # Area of boxes_b: (M,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # Pairwise intersections
    # Broadcast (N, 1, 2) and (1, M, 2)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    wh = np.maximum(0.0, bottom_right - top_left)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    # Union
    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-6)

    return (intersection / union).astype(np.float32)


class KalmanFilter:
    """A standard Kalman filter for tracking bounding boxes in image space.

    8-dimensional state space:
        [x, y, a, h, vx, vy, va, vh]
    where (x, y) is the center of the bounding box, a is aspect ratio (w/h),
    and h is height.
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # State transition matrix F (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement projection matrix H (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Motion and measurement uncertainties
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create track from unassociated measurement.

        Args:
            measurement: Bounding box coordinates [x, y, a, h] with shape (4,).

        Returns:
            Initial mean vector (8,) and covariance matrix (8, 8).
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step.

        Args:
            mean: State mean vector (8,).
            covariance: State covariance matrix (8, 8).

        Returns:
            Predicted mean (8,) and covariance (8, 8).
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space.

        Args:
            mean: State mean vector (8,).
            covariance: State covariance matrix (8, 8).

        Returns:
            Projected mean (4,) and covariance (4, 4).
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)

        mean = np.dot(self._update_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
            + innovation_cov
        )
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step.

        Args:
            mean: Predicted state mean (8,).
            covariance: Predicted state covariance (8, 8).
            measurement: Measurement vector [x, y, a, h] (4,).

        Returns:
            Updated mean (8,) and covariance (8, 8).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance
