"""
Tests for dfl_torch/augment.py (requirements.md Section 4: color/noise/blur/downsample/HSV/
shadow augmentation, reusing core.imagelib where DFL already factored it out).
"""
import numpy as np

from dfl_torch.augment import (
    apply_hsv_shift,
    apply_pixel_augmentations,
    apply_random_blur,
    apply_random_downsample,
    apply_random_jpeg,
    apply_random_noise,
    apply_random_shadow,
)


def _fixture_image(size=64, seed=0):
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:] = rng.uniform(0.2, 0.5, size=3)
    cv2_center = size // 2
    y, x = np.ogrid[:size, :size]
    inside = (x - cv2_center) ** 2 + (y - cv2_center) ** 2 <= (size // 3) ** 2
    img[inside] = rng.uniform(0.6, 0.9, size=3)
    return img


def test_apply_random_blur_changes_image_and_preserves_shape():
    img = _fixture_image()
    blurred = apply_random_blur(img.copy(), np.random.RandomState(0))
    assert blurred.shape == img.shape
    assert not np.allclose(blurred, img)


def test_apply_random_noise_changes_image():
    img = _fixture_image()
    noisy = apply_random_noise(img.copy(), np.random.RandomState(0))
    assert noisy.shape == img.shape
    assert not np.allclose(noisy, img)


def test_apply_random_jpeg_changes_image_and_preserves_shape_dtype():
    img = _fixture_image()
    compressed = apply_random_jpeg(img.copy(), np.random.RandomState(0))
    assert compressed.shape == img.shape
    assert compressed.dtype == np.float32
    assert 0.0 <= compressed.min() and compressed.max() <= 1.0
    assert not np.allclose(compressed, img)


def test_apply_random_downsample_changes_image_and_preserves_shape():
    img = _fixture_image()
    down = apply_random_downsample(img.copy(), resolution=64, rnd_state=np.random.RandomState(0))
    assert down.shape == img.shape
    assert not np.allclose(down, img)


# --- apply_pixel_augmentations ---

def test_apply_pixel_augmentations_no_flags_is_noop_besides_clipping():
    img = _fixture_image()
    out = apply_pixel_augmentations(img.copy(), resolution=64, rnd_state=np.random.RandomState(0))
    np.testing.assert_allclose(out, img, atol=1e-6)


def test_apply_pixel_augmentations_output_in_valid_range():
    img = _fixture_image()
    out = apply_pixel_augmentations(
        img.copy(), resolution=64, enable_blur=True, enable_noise=True, enable_jpeg=True,
        enable_downsample=True, rnd_state=np.random.RandomState(0),
    )
    assert out.shape == img.shape
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_apply_pixel_augmentations_deterministic_with_seeded_state():
    img = _fixture_image()
    out_a = apply_pixel_augmentations(
        img.copy(), resolution=64, enable_blur=True, enable_noise=True, enable_jpeg=True,
        enable_downsample=True, rnd_state=np.random.RandomState(42),
    )
    out_b = apply_pixel_augmentations(
        img.copy(), resolution=64, enable_blur=True, enable_noise=True, enable_jpeg=True,
        enable_downsample=True, rnd_state=np.random.RandomState(42),
    )
    np.testing.assert_allclose(out_a, out_b)


def test_apply_pixel_augmentations_all_enabled_changes_image():
    img = _fixture_image()
    out = apply_pixel_augmentations(
        img.copy(), resolution=64, enable_blur=True, enable_noise=True, enable_jpeg=True,
        enable_downsample=True, rnd_state=np.random.RandomState(0),
    )
    assert not np.allclose(out, img)


# --- HSV shift ---

def test_apply_hsv_shift_zero_amount_is_noop():
    img = _fixture_image()
    out = apply_hsv_shift(img.copy(), amount=0, rnd_state=np.random.RandomState(0))
    np.testing.assert_array_equal(out, img)


def test_apply_hsv_shift_nonzero_changes_image_and_preserves_shape_range():
    img = _fixture_image()
    out = apply_hsv_shift(img.copy(), amount=0.5, rnd_state=np.random.RandomState(0))
    assert out.shape == img.shape
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert not np.allclose(out, img)


def test_apply_hsv_shift_same_rnd_state_sequence_gives_same_result():
    """Feeding two images through separately-constructed but identically-seeded RandomStates
    (the pattern used to keep warped/target color-consistent) gives the same shift."""
    img_a = _fixture_image(seed=1)
    img_b = _fixture_image(seed=2)
    out_a = apply_hsv_shift(img_a.copy(), amount=0.5, rnd_state=np.random.RandomState(7))
    out_b = apply_hsv_shift(img_a.copy(), amount=0.5, rnd_state=np.random.RandomState(7))
    np.testing.assert_allclose(out_a, out_b)


# --- shadow ---

def test_apply_random_shadow_changes_image_and_preserves_shape():
    img = _fixture_image()
    out = apply_random_shadow(img.copy(), seed=0)
    assert out.shape == img.shape
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert not np.allclose(out, img)


def test_apply_random_shadow_same_seed_gives_same_result():
    img = _fixture_image()
    out_a = apply_random_shadow(img.copy(), seed=3)
    out_b = apply_random_shadow(img.copy(), seed=3)
    np.testing.assert_allclose(out_a, out_b)
