"""
Face extraction (Phase 4 completion) — detects, quality-filters, aligns, and saves cropped faces
as DFLJPG files with embedded metadata, using dfl_torch.alignment's MediaPipe detector in place
of the legacy TF FANExtractor/S3FDExtractor.

Clean-room, standalone pipeline (dfl_torch/extract.py), not a patch to mainscripts/Extractor.py —
consistent with this migration's approach throughout: parallel implementation, legacy code
untouched until it's ready to be fully retired (see IMPLEMENTATION_PLAN.md Phase 0.5/4).
mainscripts/Extractor.py's multi-stage subprocessor architecture, debug visualization, and
video-to-frames extraction (a plain ffmpeg call, no TF dependency, doesn't need porting) aren't
replicated here — this covers face detect → quality-filter → align → save for a directory of
already-extracted frame images.

Alignment reuses facelib.LandmarksProcessor.get_transform_mat **unchanged**, via a
MediaPipe→dlib-68 landmark conversion (dfl_torch.alignment.convert_mediapipe_landmarks_to_dlib68
— see that function's docstring for why the conversion table is vendored from a verified source
rather than hand-derived). Output DFLJPG files are written with the same fields
`mainscripts/Extractor.py` itself writes (face_type, landmarks in aligned-crop space,
source_landmarks in original-frame space, source_rect, image_to_face_mat, source_filename), so
they're fully compatible with `samplelib.SampleLoader` / `dfl_torch.data.SAEHDFaceDataset` —
faces extracted this way are ordinary training-ready samples, not a separate format.

**Not implemented:** the temporal smoothing / two-pass alignment primitives already built in
dfl_torch/alignment.py aren't wired in here yet (this processes each frame independently) — see
IMPLEMENTATION_PLAN.md Phase 4.
"""
from pathlib import Path

import cv2

from core import pathex
from core.cv2ex import cv2_imread
from DFLIMG import DFLJPG
from facelib import FaceType, LandmarksProcessor

from dfl_torch.alignment import (
    FaceLandmarkDetector,
    convert_mediapipe_landmarks_to_dlib68,
    estimate_pose_from_matrix,
    passes_pose_range,
)


def extract_one_image(image_path, output_dir, detector, resolution=512, face_type=FaceType.FULL,
                       max_yaw=75.0, max_pitch=60.0, max_roll=45.0):
    """
    Detects, quality-filters, aligns, and saves one image as a DFLJPG file in output_dir (named
    after the input file's stem). Returns True if a face was found, passed pose filtering, and
    saved; False if no face was detected or it was filtered out for extreme pose.
    """
    image_bgr = cv2_imread(str(image_path))
    if image_bgr is None:
        return False
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    landmarks_mp, transform_matrix = detector.detect(image_rgb)
    if landmarks_mp is None:
        return False

    if transform_matrix is not None:
        yaw, pitch, roll = estimate_pose_from_matrix(transform_matrix)
        if not passes_pose_range(yaw, pitch, roll, max_yaw=max_yaw, max_pitch=max_pitch, max_roll=max_roll):
            return False

    image_landmarks = convert_mediapipe_landmarks_to_dlib68(landmarks_mp)

    image_to_face_mat = LandmarksProcessor.get_transform_mat(image_landmarks, resolution, face_type)
    face_image = cv2.warpAffine(image_bgr, image_to_face_mat, (resolution, resolution), flags=cv2.INTER_LANCZOS4)
    face_image_landmarks = LandmarksProcessor.transform_points(image_landmarks, image_to_face_mat)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out_path), face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    dflimg = DFLJPG.load(str(out_path))
    dflimg.set_dict({})
    dflimg.set_face_type(FaceType.toString(face_type))
    dflimg.set_landmarks(face_image_landmarks.tolist())
    dflimg.set_source_filename(str(image_path))
    dflimg.set_source_rect((0, 0, image_bgr.shape[1], image_bgr.shape[0]))
    dflimg.set_source_landmarks(image_landmarks.tolist())
    dflimg.set_image_to_face_mat(image_to_face_mat)
    dflimg.set_eyebrows_expand_mod(1.0)
    dflimg.save()
    return True


def extract_directory(input_dir, output_dir, resolution=512, face_type=FaceType.FULL, detector=None, **filter_kwargs):
    """
    Runs extract_one_image over every image in input_dir. Builds its own FaceLandmarkDetector
    (downloading/caching its model on first use) unless one is passed in — useful for reusing a
    single detector instance across multiple directories. Returns (num_extracted, num_skipped).
    """
    if detector is None:
        detector = FaceLandmarkDetector()

    image_paths = pathex.get_image_paths(input_dir)
    extracted, skipped = 0, 0
    for image_path in image_paths:
        if extract_one_image(image_path, output_dir, detector, resolution=resolution, face_type=face_type, **filter_kwargs):
            extracted += 1
        else:
            skipped += 1
    return extracted, skipped
