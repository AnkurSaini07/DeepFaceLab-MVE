"""
Tests for dfl_torch/extract.py (Phase 4 completion: MediaPipe-based extraction pipeline).

Real face detection isn't exercisable here (no real face photo available as a fixture — same
limitation as tests/test_alignment.py's detector tests), so most of these use a stub detector
object matching FaceLandmarkDetector's `.detect(image_rgb) -> (landmarks, matrix)` interface,
returning a fixed synthetic-but-plausible landmark set. This isolates and validates the
extraction pipeline's own logic (alignment via the real, unstubbed
facelib.LandmarksProcessor.get_transform_mat, DFLJPG saving, pose filtering) independent of
whether MediaPipe itself finds a face in a given image.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from dfl_torch.extract import extract_directory, extract_one_image
from facelib import FaceType

FIXTURE_FACESET = Path(__file__).resolve().parent / "fixtures" / "faceset"


def _rotation_matrix_4x4(yaw_deg, pitch_deg, roll_deg):
    def rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    r = rz(math.radians(roll_deg)) @ ry(math.radians(yaw_deg)) @ rx(math.radians(pitch_deg))
    m = np.eye(4)
    m[:3, :3] = r
    return m


class _StubDetector:
    """Matches FaceLandmarkDetector's .detect() interface with fixed, plausible-but-synthetic
    output, so extraction logic can be tested without depending on real face detection."""

    def __init__(self, yaw=0.0, pitch=0.0, roll=0.0, detect_face=True):
        self.matrix = _rotation_matrix_4x4(yaw, pitch, roll)
        self.detect_face = detect_face

    def detect(self, image_rgb):
        if not self.detect_face:
            return None, None
        h, w = image_rgb.shape[:2]
        center = np.array([w / 2.0, h / 2.0])
        rng = np.random.RandomState(0)
        landmarks = center + rng.uniform(-min(w, h) * 0.3, min(w, h) * 0.3, size=(478, 2))
        return landmarks, self.matrix


def _make_test_image(path, size=256):
    img = (np.random.RandomState(0).rand(size, size, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def test_extract_one_image_no_face_returns_false(tmp_path):
    input_path = tmp_path / "frame.jpg"
    _make_test_image(input_path)
    output_dir = tmp_path / "out"

    detector = _StubDetector(detect_face=False)
    result = extract_one_image(input_path, output_dir, detector, resolution=64)
    assert result is False
    assert not any(output_dir.iterdir()) if output_dir.exists() else True


def test_extract_one_image_saves_valid_dflimg_when_face_detected(tmp_path):
    input_path = tmp_path / "frame.jpg"
    _make_test_image(input_path)
    output_dir = tmp_path / "out"

    detector = _StubDetector()
    result = extract_one_image(input_path, output_dir, detector, resolution=64, face_type=FaceType.FULL)
    assert result is True

    out_files = list(output_dir.iterdir())
    assert len(out_files) == 1
    assert out_files[0].suffix == ".jpg"

    from DFLIMG import DFLJPG

    dflimg = DFLJPG.load(str(out_files[0]))
    assert dflimg is not None
    assert dflimg.has_data()
    assert dflimg.get_face_type() == FaceType.toString(FaceType.FULL)
    landmarks = dflimg.get_landmarks()
    assert landmarks.shape == (68, 2)
    assert dflimg.get_source_filename() == str(input_path)
    assert dflimg.get_source_rect() == (0, 0, 256, 256)
    assert dflimg.get_image_to_face_mat() is not None


def test_extract_one_image_output_is_loadable_by_sample_loader(tmp_path):
    """The real compatibility test: extracted output must work with the same SampleLoader path
    dfl_torch/data.py's SAEHDFaceDataset uses -- not just be a structurally-plausible DFLJPG."""
    input_path = tmp_path / "frame.jpg"
    _make_test_image(input_path)
    output_dir = tmp_path / "out"

    detector = _StubDetector()
    extract_one_image(input_path, output_dir, detector, resolution=64, face_type=FaceType.FULL)

    from samplelib import SampleLoader, SampleType

    samples = SampleLoader.load(SampleType.FACE, output_dir)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.landmarks.shape == (68, 2)
    img = sample.load_bgr()
    assert img.shape == (64, 64, 3)


def test_extract_one_image_filters_out_extreme_pose(tmp_path):
    input_path = tmp_path / "frame.jpg"
    _make_test_image(input_path)
    output_dir = tmp_path / "out"

    detector = _StubDetector(yaw=89.0)  # extreme yaw, beyond default max_yaw=75
    result = extract_one_image(input_path, output_dir, detector, resolution=64)
    assert result is False


def test_extract_one_image_keeps_moderate_pose(tmp_path):
    input_path = tmp_path / "frame.jpg"
    _make_test_image(input_path)
    output_dir = tmp_path / "out"

    detector = _StubDetector(yaw=20.0, pitch=10.0, roll=5.0)
    result = extract_one_image(input_path, output_dir, detector, resolution=64)
    assert result is True


def test_extract_one_image_returns_false_for_unreadable_image(tmp_path):
    bad_path = tmp_path / "not_an_image.jpg"
    bad_path.write_bytes(b"not a real jpeg")
    output_dir = tmp_path / "out"

    detector = _StubDetector()
    result = extract_one_image(bad_path, output_dir, detector, resolution=64)
    assert result is False


def test_extract_directory_processes_all_images_and_returns_counts(tmp_path):
    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    for i in range(3):
        _make_test_image(input_dir / f"frame_{i:03d}.jpg")

    output_dir = tmp_path / "out"
    detector = _StubDetector()
    extracted, skipped = extract_directory(input_dir, output_dir, resolution=64, detector=detector)

    assert extracted == 3
    assert skipped == 0
    assert len(list(output_dir.iterdir())) == 3


def test_extract_directory_with_real_detector_on_non_face_fixtures(tmp_path):
    """Smoke test with the real (network-dependent) FaceLandmarkDetector against the checked-in
    synthetic fixture faceset -- not real faces, so every frame should be gracefully skipped
    (no crash), same limitation as tests/test_alignment.py's detector tests."""
    pytest.importorskip("mediapipe")
    try:
        extracted, skipped = extract_directory(FIXTURE_FACESET, tmp_path / "out", resolution=64)
    except Exception as e:
        pytest.skip(f"could not initialize FaceLandmarkDetector (likely no network for model download): {e}")

    assert extracted == 0
    assert skipped == 3
