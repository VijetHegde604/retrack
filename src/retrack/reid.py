"""Deep visual Re-Identification (ReID) module for persistent object tracking.

Extracts appearance embeddings from object crops using a lightweight, CPU-efficient
deep neural network and manages a multi-view persistent gallery to re-identify
objects that leave and re-enter the camera view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T


def get_optimal_device(requested: str = "auto") -> torch.device:
    """Select the best available compute device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class ReIDExtractor:
    """Lightweight visual feature extractor using MobileNetV3-Small backbone."""

    def __init__(
        self,
        device: str | torch.device = "auto",
        crop_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.device = get_optimal_device(str(device)) if isinstance(device, str) else device
        self.crop_size = crop_size

        # Load lightweight MobileNetV3-Small
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Replace classifier with identity to output 576-dim feature vector
        model.classifier = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)

        # Standard ImageNet normalization
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

    @torch.no_grad()
    def extract_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract L2-normalized appearance embeddings from a list of BGR image crops.

        Args:
            crops: List of BGR numpy image patches.

        Returns:
            Numpy array of shape (N, 576) containing L2-normalized feature vectors.
        """
        if not crops:
            return np.empty((0, 576), dtype=np.float32)

        tensors: list[torch.Tensor] = []
        for crop in crops:
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                # Dummy black crop for degenerate boxes
                crop = np.zeros((*self.crop_size, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)

            # BGR to RGB, transpose HWC -> CHW, float in [0, 1]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
            tensors.append(tensor)

        batch = torch.stack(tensors).to(self.device)
        batch = (batch - self._mean) / self._std

        features = self.model(batch)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype(np.float32)

    def extract_from_frame(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> np.ndarray:
        """Extract appearance embeddings for bounding boxes on a full frame."""
        if not bboxes:
            return np.empty((0, 576), dtype=np.float32)

        h, w = frame.shape[:2]
        crops: list[np.ndarray] = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            ix1 = max(0, min(w - 1, int(round(x1))))
            iy1 = max(0, min(h - 1, int(round(y1))))
            ix2 = max(ix1 + 1, min(w, int(round(x2))))
            iy2 = max(iy1 + 1, min(h, int(round(y2))))

            crop = frame[iy1:iy2, ix1:ix2]
            crops.append(crop)

        return self.extract_crops(crops)


@dataclass
class PersistentObject:
    """Stores the persistent identity and appearance gallery of a tracked object."""

    track_id: int
    class_id: int
    class_name: str
    features: list[np.ndarray] = field(default_factory=list)
    ema_feature: np.ndarray | None = None
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_frame: int = 0
    total_detections: int = 0
    reidentified_times: int = 0

    def add_feature(self, feature: np.ndarray, max_exemplars: int = 6) -> None:
        """Add an appearance embedding to the multi-view gallery and update EMA."""
        feature = feature / max(1e-6, float(np.linalg.norm(feature)))

        if self.ema_feature is None:
            self.ema_feature = feature.copy()
        else:
            alpha = 0.85
            self.ema_feature = alpha * self.ema_feature + (1.0 - alpha) * feature
            self.ema_feature = self.ema_feature / max(1e-6, float(np.linalg.norm(self.ema_feature)))

        if len(self.features) >= max_exemplars:
            # Check if this new feature is sufficiently diverse compared to existing exemplars
            similarities = [float(np.dot(feature, f)) for f in self.features]
            if max(similarities) < 0.95:
                # Replace the oldest exemplar with the most similar one or FIFO
                self.features.pop(0)
                self.features.append(feature)
        else:
            self.features.append(feature)


class GalleryManager:
    """Manages long-term appearance identities and ReID matching across frames."""

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        max_exemplars: int = 6,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_exemplars = max_exemplars
        self.gallery: dict[int, PersistentObject] = {}

    def register_or_update(
        self,
        track_id: int,
        class_id: int,
        class_name: str,
        embedding: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_idx: int,
    ) -> None:
        """Register a new object or update its appearance profile."""
        if track_id not in self.gallery:
            self.gallery[track_id] = PersistentObject(
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
            )

        obj = self.gallery[track_id]
        obj.add_feature(embedding, self.max_exemplars)
        obj.last_bbox = bbox
        obj.last_frame = frame_idx
        obj.total_detections += 1

    def match_candidate(
        self,
        candidate_embedding: np.ndarray,
        class_id: int,
        active_ids: set[int],
        threshold: float | None = None,
    ) -> tuple[int | None, float]:
        """Match a candidate embedding against inactive gallery objects of the same class.

        Args:
            candidate_embedding: L2-normalized feature vector (576,).
            class_id: Detection class ID (filters matching to same semantic class).
            active_ids: Set of currently visible track IDs that cannot be re-matched.
            threshold: Minimum cosine similarity threshold (defaults to self.similarity_threshold).

        Returns:
            Tuple of (matched_track_id, similarity_score). If no match, (None, best_score).
        """
        thresh = self.similarity_threshold if threshold is None else threshold
        candidate_embedding = candidate_embedding / max(1e-6, float(np.linalg.norm(candidate_embedding)))

        best_id: int | None = None
        best_score = -1.0

        for track_id, obj in self.gallery.items():
            if track_id in active_ids:
                # Already visible in current frame
                continue
            if obj.class_id != class_id:
                # Different object class
                continue

            # Compute similarity against all stored exemplars + EMA
            scores = [float(np.dot(candidate_embedding, feat)) for feat in obj.features]
            if obj.ema_feature is not None:
                scores.append(float(np.dot(candidate_embedding, obj.ema_feature)))

            if not scores:
                continue

            max_score = max(scores)
            if max_score > best_score:
                best_score = max_score
                best_id = track_id

        if best_score >= thresh:
            return best_id, best_score

        return None, best_score
