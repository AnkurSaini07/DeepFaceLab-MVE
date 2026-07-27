"""
Tests for dfl_torch/masking.py (requirements.md Section 6: two-mask system) and
dfl_torch/occlusion.py (Section 6.3: MediaPipe Hands occlusion detector).
"""
import numpy as np
import pytest

from dfl_torch.masking import combine_masks, feather_combined_mask, feather_mask


def _filled_circle_mask(size, radius, center_rc=None):
    """center_rc: (row, col), matching how callers index the returned mask as mask[row, col]."""
    mask = np.zeros((size, size, 1), dtype=np.float32)
    center_row, center_col = center_rc or (size // 2, size // 2)
    y, x = np.ogrid[:size, :size]
    inside = (x - center_col) ** 2 + (y - center_row) ** 2 <= radius**2
    mask[inside, 0] = 1.0
    return mask


# --- combine_masks ---

def test_combine_masks_no_occlusion_keeps_face_mask():
    face = _filled_circle_mask(64, 20)
    combined = combine_masks(face, np.zeros_like(face))
    np.testing.assert_allclose(combined, face)


def test_combine_masks_full_occlusion_zeros_everything():
    face = _filled_circle_mask(64, 20)
    combined = combine_masks(face, np.ones_like(face))
    np.testing.assert_allclose(combined, np.zeros_like(face))


def test_combine_masks_none_occlusion_keeps_face_mask():
    face = _filled_circle_mask(64, 20)
    combined = combine_masks(face, None)
    np.testing.assert_allclose(combined, face)


def test_combine_masks_partial_occlusion_algebra():
    face = np.full((4, 4, 1), 0.8, dtype=np.float32)
    occlusion = np.full((4, 4, 1), 0.5, dtype=np.float32)
    combined = combine_masks(face, occlusion)
    np.testing.assert_allclose(combined, np.full((4, 4, 1), 0.4, dtype=np.float32), atol=1e-6)


# --- feather_mask ---

def test_feather_mask_preserves_shape_and_range():
    mask = _filled_circle_mask(64, 20)
    feathered = feather_mask(mask)
    assert feathered.shape == mask.shape
    assert feathered.min() >= 0.0 and feathered.max() <= 1.0 + 1e-6


def test_feather_mask_softens_hard_edge():
    mask = _filled_circle_mask(64, 20)
    feathered = feather_mask(mask)
    # A hard-edged mask has only ~0/1 values; feathering should introduce intermediate values.
    n_intermediate_before = np.sum((mask > 0.05) & (mask < 0.95))
    n_intermediate_after = np.sum((feathered > 0.05) & (feathered < 0.95))
    assert n_intermediate_before == 0
    assert n_intermediate_after > 0


def test_feather_mask_empty_mask_returned_unchanged():
    mask = np.zeros((32, 32, 1), dtype=np.float32)
    feathered = feather_mask(mask)
    np.testing.assert_allclose(feathered, mask)


def test_feather_combined_mask_occlusion_tighter_than_face():
    face = _filled_circle_mask(128, 40)
    occlusion = _filled_circle_mask(128, 15, center_rc=(64, 80))  # small occluder inside the face region
    combined = feather_combined_mask(face, occlusion)
    assert combined.shape == face.shape
    assert combined.min() >= -1e-6 and combined.max() <= 1.0 + 1e-6
    # Center of the occluder should be strongly suppressed relative to an unoccluded face region.
    assert combined[64, 80, 0] < 0.3
    assert combined[64, 40, 0] > 0.5  # away from the occluder, inside the face circle


def test_feather_combined_mask_no_occlusion_matches_feathered_face_only():
    face = _filled_circle_mask(64, 20)
    combined = feather_combined_mask(face, None)
    np.testing.assert_allclose(combined, feather_mask(face))
