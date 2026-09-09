import numpy as np
import pytest

from retrack.reid import GalleryManager, ReIDExtractor


def test_reid_extractor_normalization():
    extractor = ReIDExtractor(device="cpu")
    dummy_crop = np.full((100, 100, 3), 128, dtype=np.uint8)

    embs = extractor.extract_crops([dummy_crop])
    assert embs.shape == (1, 576)
    norm = np.linalg.norm(embs[0])
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_gallery_manager_matching_and_filtering():
    gallery = GalleryManager(similarity_threshold=0.70)

    # Base feature
    f1 = np.random.randn(576).astype(np.float32)
    f1 /= np.linalg.norm(f1)

    # Similar feature (cosine sim ~0.95)
    noise = np.random.randn(576).astype(np.float32)
    noise -= np.dot(noise, f1) * f1
    noise /= np.linalg.norm(noise)
    f1_similar = 0.95 * f1 + 0.31 * noise
    f1_similar /= np.linalg.norm(f1_similar)

    # Orthogonal / dissimilar feature
    f_diff = noise

    # Register object 10 (class 0, 'suitcase')
    gallery.register_or_update(
        track_id=10,
        class_id=0,
        class_name="suitcase",
        embedding=f1,
        bbox=(10, 10, 50, 50),
        frame_idx=1,
    )

    # 1. Match similar feature of same class -> matches ID 10
    matched_id, score = gallery.match_candidate(
        candidate_embedding=f1_similar,
        class_id=0,
        active_ids=set(),
    )
    assert matched_id == 10
    assert score >= 0.70

    # 2. Match dissimilar feature -> returns None
    matched_id, score = gallery.match_candidate(
        candidate_embedding=f_diff,
        class_id=0,
        active_ids=set(),
    )
    assert matched_id is None
    assert score < 0.70

    # 3. Class filter: feature identical to object 10, but class is 1 ('person') -> returns None
    matched_id, score = gallery.match_candidate(
        candidate_embedding=f1_similar,
        class_id=1,
        active_ids=set(),
    )
    assert matched_id is None

    # 4. Active ID filter: if object 10 is currently active in frame -> cannot match
    matched_id, score = gallery.match_candidate(
        candidate_embedding=f1_similar,
        class_id=0,
        active_ids={10},
    )
    assert matched_id is None
