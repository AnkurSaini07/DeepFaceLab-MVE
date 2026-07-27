"""
Generates a small synthetic DFL-formatted faceset for data-pipeline tests (Section 11.4).

Not a real face — a procedurally generated image with a plausible 68-point landmark layout
embedded via DFLJPG, so DFLIMG/SampleLoader parsing and LandmarksProcessor mask generation have
something structurally valid to operate on. Checked in as tests/fixtures/faceset/*.jpg so tests
don't need to regenerate it, but this script is kept for reproducibility / regenerating with a
different size or count.

Run with a Python that can import this repo's DFLIMG/facelib/samplelib (no TF dependency in that
chain — .venv-torch works, or system Python with cv2/numpy/scipy/pillow/numexpr installed).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from DFLIMG import DFLJPG
from facelib import FaceType

FIXTURE_DIR = Path(__file__).resolve().parent / "faceset"
RESOLUTION = 128
NUM_SAMPLES = 3


def _ellipse_points(cx, cy, rx, ry, angles_deg):
    angles = np.radians(angles_deg)
    x = cx + rx * np.cos(angles)
    y = cy + ry * np.sin(angles)
    return np.stack([x, y], axis=1)


def synthetic_68_landmarks(resolution, jitter_seed=0):
    """Procedural stand-in for a real 68-point dlib-style landmark set (jaw/brows/nose/eyes/mouth)."""
    rng = np.random.RandomState(jitter_seed)
    cx, cy = resolution * 0.5, resolution * 0.55
    jaw = _ellipse_points(cx, cy, resolution * 0.32, resolution * 0.38, np.linspace(200, 340, 17))
    reb = _ellipse_points(cx - resolution * 0.18, cy - resolution * 0.12, resolution * 0.10, resolution * 0.03, np.linspace(160, 20, 5))
    leb = _ellipse_points(cx + resolution * 0.18, cy - resolution * 0.12, resolution * 0.10, resolution * 0.03, np.linspace(160, 20, 5))
    bridge = np.stack([np.full(4, cx), np.linspace(cy - resolution * 0.05, cy + resolution * 0.08, 4)], axis=1)
    base = _ellipse_points(cx, cy + resolution * 0.10, resolution * 0.07, resolution * 0.02, np.linspace(200, 340, 5))
    reye = _ellipse_points(cx - resolution * 0.15, cy - resolution * 0.02, resolution * 0.06, resolution * 0.03, np.linspace(0, 300, 6))
    leye = _ellipse_points(cx + resolution * 0.15, cy - resolution * 0.02, resolution * 0.06, resolution * 0.03, np.linspace(0, 300, 6))
    mouth_outer = _ellipse_points(cx, cy + resolution * 0.22, resolution * 0.12, resolution * 0.05, np.linspace(0, 330, 12))
    mouth_inner = _ellipse_points(cx, cy + resolution * 0.22, resolution * 0.08, resolution * 0.03, np.linspace(0, 315, 8))

    lmrks = np.concatenate([jaw, reb, leb, bridge, base, reye, leye, mouth_outer, mouth_inner])
    assert lmrks.shape == (68, 2), lmrks.shape
    lmrks += rng.uniform(-1.0, 1.0, size=lmrks.shape)  # small jitter so samples aren't identical
    return np.clip(lmrks, 0, resolution - 1)


def make_fixture_image(resolution, seed):
    rng = np.random.RandomState(seed)
    img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    img[:] = rng.randint(60, 180, size=3)
    cv2.circle(img, (resolution // 2, resolution // 2), resolution // 3, tuple(int(v) for v in rng.randint(80, 220, size=3)), -1)
    cv2.circle(img, (resolution // 2 - resolution // 8, resolution // 2 - resolution // 12), resolution // 20, (255, 255, 255), -1)
    cv2.circle(img, (resolution // 2 + resolution // 8, resolution // 2 - resolution // 12), resolution // 20, (255, 255, 255), -1)
    return img


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_SAMPLES):
        path = FIXTURE_DIR / f"{i:05d}.jpg"
        img = make_fixture_image(RESOLUTION, seed=i)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        dflimg = DFLJPG.load(str(path))
        dflimg.set_dict({})
        dflimg.set_face_type(FaceType.toString(FaceType.FULL))
        dflimg.set_landmarks(synthetic_68_landmarks(RESOLUTION, jitter_seed=i).tolist())
        dflimg.set_source_filename(f"synthetic_source_{i:05d}.jpg")
        dflimg.set_eyebrows_expand_mod(1.0)
        dflimg.save()
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
