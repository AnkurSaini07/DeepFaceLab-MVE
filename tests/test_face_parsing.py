"""
Tests for dfl_torch/face_parsing.py (requirements.md Section 6.4: modern face-parsing network).

The pure class-map -> mask logic is fully testable with synthetic class maps. The FaceParser
network itself is only smoke-tested (initializes, downloads+caches its model, produces the right
shape) — real parsing accuracy needs a real face photo, same limitation as the other detector
tests in this suite (test_alignment.py, test_extract.py).
"""
import numpy as np
import pytest

from dfl_torch.face_parsing import (
    FACE_SKIN_CLASSES,
    GLASSES_CLASS,
    HAIR_CLASS,
    class_map_to_mask,
    face_skin_mask,
    glasses_mask,
    hair_mask,
)


def _synthetic_class_map(size=64):
    """Left half is skin (class 1), right half is hair (class 17), a small square in the
    middle is glasses (class 6), everything else background (class 0)."""
    class_map = np.zeros((size, size), dtype=np.int64)
    class_map[:, : size // 2] = 1
    class_map[:, size // 2 :] = 17
    mid = size // 2
    class_map[mid - 4 : mid + 4, mid - 4 : mid + 4] = 6
    return class_map


def test_class_map_to_mask_basic():
    class_map = _synthetic_class_map()
    mask = class_map_to_mask(class_map, [1])
    assert mask.shape == (64, 64, 1)
    assert mask.dtype == np.float32
    assert mask[:, :32].mean() == pytest.approx(1.0, abs=0.02)  # skin half is ~all 1s (minus the glasses square overlap)
    assert mask[:, 32:].max() == 0.0  # hair half has none


def test_class_map_to_mask_multiple_classes():
    class_map = _synthetic_class_map()
    mask = class_map_to_mask(class_map, [1, 17])
    assert mask.mean() == pytest.approx(1.0, abs=0.02)  # everything except the tiny glasses square


def test_face_skin_mask_excludes_hair():
    class_map = _synthetic_class_map()
    mask = face_skin_mask(class_map)
    assert mask[:, :24].mean() == 1.0  # well inside the skin region, away from the glasses square
    assert mask[:, 40:].max() == 0.0  # entirely inside the hair region


def test_face_skin_mask_covers_all_expected_classes():
    for class_id in FACE_SKIN_CLASSES:
        class_map = np.full((8, 8), class_id, dtype=np.int64)
        mask = face_skin_mask(class_map)
        assert mask.mean() == 1.0, f"class {class_id} should count as face skin"

    class_map = np.full((8, 8), HAIR_CLASS, dtype=np.int64)
    assert face_skin_mask(class_map).mean() == 0.0


def test_hair_mask():
    class_map = _synthetic_class_map()
    mask = hair_mask(class_map)
    assert mask[:, 40:].mean() == 1.0
    assert mask[:, :24].max() == 0.0


def test_glasses_mask():
    class_map = _synthetic_class_map()
    mask = glasses_mask(class_map)
    assert mask.sum() == 8 * 8  # the 8x8 square set to GLASSES_CLASS
    mid = 32
    assert mask[mid, mid, 0] == 1.0
    assert mask[0, 0, 0] == 0.0


def test_glasses_class_constant_matches_taxonomy():
    assert GLASSES_CLASS == 6
    assert HAIR_CLASS == 17


# --- FaceParser (network-dependent: downloads+caches BiSeNet + ResNet18 backbone weights) ---

def _make_parser():
    pytest.importorskip("face_parsing")
    from dfl_torch.face_parsing import FaceParser
    try:
        return FaceParser()
    except Exception as e:
        pytest.skip(f"could not initialize FaceParser (likely no network for weight download): {e}")


def test_face_parser_output_shape_and_range():
    parser = _make_parser()
    img = (np.random.RandomState(0).rand(128, 128, 3) * 255).astype(np.uint8)
    class_map = parser.parse(img)
    assert class_map.shape == (128, 128)
    assert class_map.dtype == np.int64
    assert class_map.min() >= 0 and class_map.max() <= 18


def test_face_parser_handles_non_square_input_and_resizes_back():
    parser = _make_parser()
    img = (np.random.RandomState(0).rand(96, 160, 3) * 255).astype(np.uint8)
    class_map = parser.parse(img)
    assert class_map.shape == (96, 160)


def test_face_parser_accepts_float_input():
    parser = _make_parser()
    img_uint8 = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
    img_float = img_uint8.astype(np.float32) / 255.0
    class_map = parser.parse(img_float)
    assert class_map.shape == (64, 64)
