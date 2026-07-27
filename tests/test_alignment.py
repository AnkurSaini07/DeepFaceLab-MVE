"""
Tests for dfl_torch/alignment.py (requirements.md Section 5.1: landmark detection + automated
quality filtering).

Pose-estimation and quality-filtering predicates are pure math and fully testable without a real
face. The MediaPipe FaceLandmarker wrapper itself is only smoke-tested (runs without crashing,
correctly reports "no face" on non-face input) since validating real detection accuracy needs an
actual face photo, which isn't available as a test fixture here — same category of limitation as
Section 11.6 ("explicitly not validated without [X]"), just for a different X.
"""
import math

import numpy as np
import pytest

from dfl_torch.alignment import (
    compute_landmark_jitter,
    estimate_pose_from_matrix,
    passes_confidence_threshold,
    passes_jitter_threshold,
    passes_pose_range,
)


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


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [(0, 0, 0), (30, 0, 0), (0, 20, 0), (0, 0, 15), (25, -10, 5), (-45, 30, -20)],
)
def test_estimate_pose_from_matrix_recovers_known_angles(yaw, pitch, roll):
    matrix = _rotation_matrix_4x4(yaw, pitch, roll)
    out_yaw, out_pitch, out_roll = estimate_pose_from_matrix(matrix)
    assert out_yaw == pytest.approx(yaw, abs=1e-3)
    assert out_pitch == pytest.approx(pitch, abs=1e-3)
    assert out_roll == pytest.approx(roll, abs=1e-3)


def test_passes_confidence_threshold():
    assert passes_confidence_threshold(0.8, threshold=0.5)
    assert not passes_confidence_threshold(0.3, threshold=0.5)
    assert passes_confidence_threshold(0.5, threshold=0.5)  # boundary is inclusive


def test_passes_pose_range():
    assert passes_pose_range(10, 5, 5, max_yaw=75, max_pitch=60, max_roll=45)
    assert not passes_pose_range(80, 5, 5, max_yaw=75, max_pitch=60, max_roll=45)
    assert not passes_pose_range(10, 65, 5, max_yaw=75, max_pitch=60, max_roll=45)
    assert not passes_pose_range(10, 5, 50, max_yaw=75, max_pitch=60, max_roll=45)


def test_compute_landmark_jitter_zero_for_identical_landmarks():
    lmrks = np.random.RandomState(0).rand(68, 2) * 100
    assert compute_landmark_jitter(lmrks, lmrks) == pytest.approx(0.0)


def test_compute_landmark_jitter_positive_for_shifted_landmarks():
    rng = np.random.RandomState(0)
    lmrks_prev = rng.rand(68, 2) * 100
    lmrks_curr = lmrks_prev + 5.0  # every point shifts by exactly (5, 5)
    jitter = compute_landmark_jitter(lmrks_prev, lmrks_curr)
    assert jitter == pytest.approx(5.0 * math.sqrt(2), abs=1e-6)


def test_compute_landmark_jitter_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        compute_landmark_jitter(np.zeros((68, 2)), np.zeros((60, 2)))


def test_passes_jitter_threshold():
    rng = np.random.RandomState(0)
    lmrks_prev = rng.rand(68, 2) * 100
    small_shift = lmrks_prev + 0.5
    large_shift = lmrks_prev + 50.0
    assert passes_jitter_threshold(lmrks_prev, small_shift, max_jitter_px=5.0)
    assert not passes_jitter_threshold(lmrks_prev, large_shift, max_jitter_px=5.0)


def test_detector_reports_no_face_on_non_face_image():
    pytest.importorskip("mediapipe")
    try:
        from dfl_torch.alignment import FaceLandmarkDetector
        detector = FaceLandmarkDetector()
    except Exception as e:
        pytest.skip(f"could not initialize FaceLandmarkDetector (likely no network for model download): {e}")

    rng = np.random.RandomState(0)
    non_face_image = (rng.rand(256, 256, 3) * 255).astype(np.uint8)
    landmarks, matrix = detector.detect(non_face_image)
    assert landmarks is None
    assert matrix is None
