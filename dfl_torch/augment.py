"""
Color/noise/blur/downsample/HSV/shadow augmentation — requirements.md Section 4's remaining data
augmentation ask (geometric warp augmentation lives in dfl_torch/data.py, reusing
core.imagelib.warp unchanged). Reuses core.imagelib's standalone functions
(`LinearMotionBlur`, `shadow_highlights_augmentation`) where DFL already factored them out; the
rest (noise/jpeg/downsample/HSV shift) is reimplemented here matching
`samplelib/SampleProcessor.py`'s exact formulas, since that logic lives inline inside
`SampleProcessor`'s large `process` method rather than as standalone reusable functions.

Matches `models/Model_SAEHD/Model.py`'s actual sample-generator configuration (confirmed by
reading it, not assumed): blur/noise/jpeg/downsample apply ONLY to the encoder's warped input —
its `output_sample_types` entry sets these, the target entry doesn't — the target must stay a
clean ground truth. HSV shift and shadow apply to BOTH warped and target with the SAME random
draw (both entries pass identical `random_hsv_shift_amount`/`random_shadow` values and derive
their randomness from the same per-sample seed), keeping a sample's color grading consistent
between input and target — only geometry/sharpness/noise differs.

**Not implemented:** `ct_mode` (color transfer against a reference face from a different
identity's faceset) — needs an additional data source (a random cross-identity sample), a
materially separate feature from per-image augmentation; noted as a gap, not silently dropped.
"""
import cv2
import numpy as np

from core.imagelib import LinearMotionBlur
from core.imagelib.shadows import shadow_highlights_augmentation


def apply_random_blur(img, rnd_state):
    blur_type = rnd_state.choice(["motion", "gaussian"])
    if blur_type == "motion":
        blur_k = rnd_state.randint(10, 20)
        blur_angle = 360 * rnd_state.random()
        return LinearMotionBlur(img, blur_k, blur_angle)

    blur_sigma = 5 * rnd_state.random() + 3
    kernel_size = 2.9 * blur_sigma if blur_sigma < 5.0 else 2.6 * blur_sigma
    kernel_size = int(kernel_size)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), blur_sigma)


def apply_random_noise(img, rnd_state):
    noise_type = rnd_state.choice(["gaussian", "laplace", "poisson"])
    noise_scale = 20 * rnd_state.random() + 20
    if noise_type == "gaussian":
        noise = rnd_state.normal(scale=noise_scale, size=img.shape)
    elif noise_type == "laplace":
        noise = rnd_state.laplace(scale=noise_scale, size=img.shape)
    else:
        noise_lam = 15 * rnd_state.random() + 15
        noise = rnd_state.poisson(lam=noise_lam, size=img.shape)
    return img + noise / 255.0


def apply_random_jpeg(img, rnd_state):
    img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    quality = rnd_state.randint(50, 85)
    _, enc = cv2.imencode(".jpg", img_u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return dec.astype(np.float32) / 255.0


def apply_random_downsample(img, resolution, rnd_state):
    lo = max(1, int(0.125 * resolution))
    hi = max(lo + 1, int(0.25 * resolution))
    down_res = rnd_state.randint(lo, hi)
    down = cv2.resize(img, (down_res, down_res), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(down, (resolution, resolution), interpolation=cv2.INTER_CUBIC)


def apply_pixel_augmentations(img, resolution, enable_blur=False, enable_noise=False,
                               enable_jpeg=False, enable_downsample=False, rnd_state=None):
    """
    Applies the enabled subset of {blur, noise, jpeg, downsample} in a random order (matching
    `SampleProcessor.py`'s `randomization_order` shuffle). Intended for the warped/input image
    only — never the target, per the module docstring.
    """
    if rnd_state is None:
        rnd_state = np.random
    order = ["blur", "noise", "jpeg", "down"]
    rnd_state.shuffle(order)
    for kind in order:
        if kind == "blur" and enable_blur:
            img = apply_random_blur(img, rnd_state)
        elif kind == "noise" and enable_noise:
            img = apply_random_noise(img, rnd_state)
        elif kind == "jpeg" and enable_jpeg:
            img = apply_random_jpeg(img, rnd_state)
        elif kind == "down" and enable_downsample:
            img = apply_random_downsample(img, resolution, rnd_state)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def apply_hsv_shift(img, amount, rnd_state):
    """amount in [0, 1]; amount=0 is a no-op, matching SampleProcessor's `if
    random_hsv_shift_amount != 0` guard."""
    if amount == 0:
        return img
    h_amount = max(1, int(360 * amount * 0.5))
    img_h, img_s, img_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    img_h = (img_h + rnd_state.randint(-h_amount, h_amount + 1)) % 360
    img_s = np.clip(img_s + (rnd_state.random() - 0.5) * amount, 0, 1)
    img_v = np.clip(img_v + (rnd_state.random() - 0.5) * amount, 0, 1)
    merged = cv2.cvtColor(cv2.merge([img_h, img_s, img_v]), cv2.COLOR_HSV2BGR)
    return np.clip(merged, 0.0, 1.0).astype(np.float32)


def apply_random_shadow(img, seed):
    return shadow_highlights_augmentation(img, seed=seed)
