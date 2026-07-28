"""
Tests for dfl_torch/merge.py (Phase 12a: inference/merge pipeline). Uses a small, randomly
initialized SAEHDModel (no training involved — this validates the merge *pipeline's* mechanics:
crop, swap, blend, paste-back — not swap quality, which needs a real trained model and real
footage neither of which exist in this dev environment).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from dfl_torch.merge import merge_one_frame, merge_video_frames
from dfl_torch.model import SAEHDModel
from facelib import FaceType

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from generate_face_fixture import synthetic_68_landmarks  # noqa: E402

RESOLUTION = 32


def _build_model():
    return SAEHDModel(RESOLUTION, e_dims=8, ae_dims=16, d_dims=8, d_mask_dims=4)


def _synthetic_frame_and_landmarks(frame_size=256, seed=0):
    """Uses the same anatomically-plausible (jaw/brow/nose/eye/mouth-shaped) synthetic landmark
    generator as the extraction fixtures — pure random-scatter points produce a wildly
    ill-conditioned facelib.LandmarksProcessor.get_transform_mat fit (verified: the resulting
    "face" crop region can span thousands of pixels outside a 256px frame), which isn't
    representative of anything merge_one_frame needs to handle in practice."""
    rng = np.random.RandomState(seed)
    frame = rng.uniform(0.2, 0.6, size=(frame_size, frame_size, 3)).astype(np.float32)
    face_span = frame_size * 0.5
    offset = (frame_size - face_span) / 2.0
    landmarks = synthetic_68_landmarks(face_span, jitter_seed=seed) + offset
    return frame, landmarks.astype(np.float32)


def test_merge_one_frame_output_shape_and_range():
    model = _build_model()
    frame, landmarks = _synthetic_frame_and_landmarks()
    out = merge_one_frame(model, frame, landmarks, RESOLUTION, face_type=FaceType.FULL)
    assert out.shape == frame.shape
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_merge_one_frame_preserves_background_far_from_face():
    model = _build_model()
    frame, landmarks = _synthetic_frame_and_landmarks(frame_size=256)
    out = merge_one_frame(model, frame, landmarks, RESOLUTION, face_type=FaceType.FULL)

    # Corners are far outside the landmark cluster (centered, +/-51px spread in a 256px frame) --
    # the warped-back face mask should be exactly 0 there, so output must equal the original.
    corner_slices = [
        (slice(0, 10), slice(0, 10)),
        (slice(-10, None), slice(-10, None)),
    ]
    for ys, xs in corner_slices:
        np.testing.assert_allclose(out[ys, xs], frame[ys, xs], atol=1e-5)


def test_merge_one_frame_without_color_transfer_runs():
    model = _build_model()
    frame, landmarks = _synthetic_frame_and_landmarks()
    out = merge_one_frame(model, frame, landmarks, RESOLUTION, face_type=FaceType.FULL, color_transfer=False)
    assert out.shape == frame.shape
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_merge_one_frame_erode_and_blur_do_not_crash():
    model = _build_model()
    frame, landmarks = _synthetic_frame_and_landmarks()
    out_eroded = merge_one_frame(model, frame, landmarks, RESOLUTION, erode=3)
    out_dilated = merge_one_frame(model, frame, landmarks, RESOLUTION, erode=-3)
    out_blurred = merge_one_frame(model, frame, landmarks, RESOLUTION, blur=5)
    for out in (out_eroded, out_dilated, out_blurred):
        assert out.shape == frame.shape
        assert 0.0 <= out.min() and out.max() <= 1.0


def test_merge_video_frames_writes_output_and_passes_through_missing_landmarks(tmp_path):
    model = _build_model()
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()

    frame_paths = []
    landmarks_by_frame = {}
    for i in range(3):
        frame, landmarks = _synthetic_frame_and_landmarks(seed=i)
        path = frame_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(path), np.clip(frame * 255, 0, 255).astype(np.uint8))
        frame_paths.append(path)
        if i != 1:  # frame 1 has no detected landmarks -- should pass through unchanged
            landmarks_by_frame[str(path)] = landmarks

    output_dir = tmp_path / "merged"
    merged_count = merge_video_frames(
        model, frame_paths, landmarks_by_frame, output_dir, RESOLUTION, face_type=FaceType.FULL,
    )

    assert merged_count == 2
    output_files = sorted(output_dir.iterdir())
    assert len(output_files) == 3

    from core.cv2ex import cv2_imread

    original_frame1 = cv2_imread(str(frame_paths[1]))
    merged_frame1 = cv2_imread(str(output_dir / frame_paths[1].name))
    np.testing.assert_array_equal(original_frame1, merged_frame1)
