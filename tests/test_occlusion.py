"""
Tests for dfl_torch/occlusion.py (requirements.md Section 6.3: MediaPipe Hands occlusion
detector). Same limitation as tests/test_alignment.py's detector test: real detection accuracy
isn't validated without an actual hand photo; only graceful no-detection behavior is checked.
"""
import numpy as np
import pytest

from dfl_torch.occlusion import hand_landmarks_to_occlusion_mask


def test_hand_landmarks_to_mask_empty_when_no_hands():
    mask = hand_landmarks_to_occlusion_mask((128, 128, 3), [])
    assert mask.shape == (128, 128, 1)
    assert mask.max() == 0.0


def test_hand_landmarks_to_mask_covers_hand_region():
    # A rough hand-shaped cluster of 21 points in the top-left quadrant.
    rng = np.random.RandomState(0)
    hand = rng.uniform(low=[10, 10], high=[50, 50], size=(21, 2)).astype(np.float32)
    mask = hand_landmarks_to_occlusion_mask((128, 128, 3), [hand])

    assert mask.shape == (128, 128, 1)
    assert mask.max() == 1.0
    assert mask[30, 30, 0] == 1.0  # inside the hand cluster's bounding area
    assert mask[110, 110, 0] == 0.0  # far away, untouched


def test_hand_landmarks_to_mask_multiple_hands():
    rng = np.random.RandomState(0)
    hand1 = rng.uniform(low=[10, 10], high=[30, 30], size=(21, 2)).astype(np.float32)
    hand2 = rng.uniform(low=[90, 90], high=[110, 110], size=(21, 2)).astype(np.float32)
    mask = hand_landmarks_to_occlusion_mask((128, 128, 3), [hand1, hand2])

    assert mask[20, 20, 0] == 1.0
    assert mask[100, 100, 0] == 1.0
    assert mask[64, 64, 0] == 0.0  # between the two hands


def test_hand_detector_reports_no_hands_on_non_hand_image():
    pytest.importorskip("mediapipe")
    try:
        from dfl_torch.occlusion import HandLandmarkDetector
        detector = HandLandmarkDetector()
    except Exception as e:
        pytest.skip(f"could not initialize HandLandmarkDetector (likely no network for model download): {e}")

    rng = np.random.RandomState(0)
    non_hand_image = (rng.rand(256, 256, 3) * 255).astype(np.uint8)
    hands = detector.detect(non_hand_image)
    assert hands == []
