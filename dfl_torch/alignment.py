"""
Landmark detection + automated quality filtering — requirements.md Section 5.1.

Detector: MediaPipe Face Mesh (via the Tasks API's FaceLandmarker — the older
`mp.solutions.face_mesh` API isn't available in this mediapipe build). requirements.md lists
InsightFace as preferred and MediaPipe as the alternative; MediaPipe was chosen for this initial
implementation because (a) it's explicitly called out as more robust to partial occlusion, which
is this project's actual problem (mic-occluded mouth region), and (b) its model is a single
self-contained download with no separate ONNX model-zoo dependency, which is more test-friendly
in this CPU-only/no-persistent-GPU dev setup. Swapping in InsightFace later is a contained change
if MediaPipe's accuracy proves insufficient on real footage — nothing downstream depends on which
detector produced the landmarks/pose.

Scope: detector wrapper, pose estimation, per-frame quality-filtering predicates (Section 5.1's
confidence threshold, pose-range filtering, jitter detection), temporal smoothing (moving average
/ Kalman), two-pass alignment primitives, and a MediaPipe->dlib-68 landmark conversion (below) so
this detector's output can drive `facelib.LandmarksProcessor.get_transform_mat` — the actual
alignment-crop wiring lives in `dfl_torch/extract.py`. See IMPLEMENTATION_PLAN.md Phase 4 for
what's still open (temporal smoothing/two-pass aren't wired into `extract.py` yet).
"""
import math
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_MODEL_CACHE_PATH = Path.home() / ".cache" / "dfl_torch" / "face_landmarker.task"


def download_face_landmarker_model(cache_path=DEFAULT_MODEL_CACHE_PATH):
    """Downloads and caches the MediaPipe FaceLandmarker model bundle (~3.7MB) if not present."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MODEL_URL, cache_path)
    return cache_path


class FaceLandmarkDetector:
    """Wraps MediaPipe's FaceLandmarker task. Returns 478 landmarks (468 face mesh + 10 iris)."""

    def __init__(self, model_path=None, min_detection_confidence=0.3):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(model_path) if model_path is not None else download_face_landmarker_model()

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._mp = mp
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb):
        """
        image_rgb: HWC uint8 RGB array.
        Returns (landmarks, transform_matrix) for the first detected face, both None if no face
        was detected. landmarks is (478, 2) in pixel coordinates; transform_matrix is the 4x4
        facial transformation matrix (used by estimate_pose_from_matrix for yaw/pitch/roll).
        """
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return None, None

        h, w = image_rgb.shape[:2]
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]], dtype=np.float32)

        matrix = None
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)

        return landmarks, matrix


def estimate_pose_from_matrix(transform_matrix):
    """
    Decomposes the rotation component of a 4x4 facial transformation matrix into yaw/pitch/roll
    (degrees), using the standard XYZ Euler-angle decomposition. Pure math — independently
    testable against known rotation matrices without needing a real detected face.
    """
    r = transform_matrix[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def passes_confidence_threshold(confidence, threshold=0.5):
    return confidence >= threshold


def passes_pose_range(yaw, pitch, roll, max_yaw=75.0, max_pitch=60.0, max_roll=45.0):
    return abs(yaw) <= max_yaw and abs(pitch) <= max_pitch and abs(roll) <= max_roll


def compute_landmark_jitter(landmarks_prev, landmarks_curr):
    """Mean per-point pixel displacement between consecutive frames' landmarks — large values
    flag likely misdetection (Section 5.1's frame-to-frame jitter check) rather than real motion."""
    if landmarks_prev.shape != landmarks_curr.shape:
        raise ValueError("landmark sets must be the same shape to compare jitter")
    return float(np.linalg.norm(landmarks_curr - landmarks_prev, axis=1).mean())


def passes_jitter_threshold(landmarks_prev, landmarks_curr, max_jitter_px):
    return compute_landmark_jitter(landmarks_prev, landmarks_curr) <= max_jitter_px


# --- Temporal smoothing (Section 5.1: "apply a smoothing pass ... to reduce single-frame jitter") ---

def smooth_landmarks_moving_average(landmarks_sequence, window_size=5):
    """
    landmarks_sequence: (T, N, 2) array, one landmark set per video frame, in order.
    Centered moving average along the time axis; the window shrinks near the sequence edges
    (rather than padding) so it never averages in out-of-sequence data.
    """
    landmarks_sequence = np.asarray(landmarks_sequence, dtype=np.float64)
    t_len = landmarks_sequence.shape[0]
    half = window_size // 2
    smoothed = np.empty_like(landmarks_sequence)
    for t in range(t_len):
        lo = max(0, t - half)
        hi = min(t_len, t + half + 1)
        smoothed[t] = landmarks_sequence[lo:hi].mean(axis=0)
    return smoothed


def smooth_landmarks_kalman(landmarks_sequence, process_var=0.5, measurement_var=4.0):
    """
    Per-coordinate scalar Kalman filter (constant-position model) applied independently to every
    landmark x/y across time. Simpler than a constant-velocity model but sufficient for reducing
    single-frame jitter, and it's the other option Section 5.1 explicitly names ("moving average
    or Kalman filter"). Defaults assume ~2px measurement noise std (measurement_var=4.0) and
    moderate frame-to-frame motion (process_var=0.5) — real face video moves slowly relative to
    frame rate, so the filter should track genuine motion quickly rather than over-smooth it into
    lag; tune both to the actual detector's noise characteristics and footage motion speed.
    """
    landmarks_sequence = np.asarray(landmarks_sequence, dtype=np.float64)
    t_len = landmarks_sequence.shape[0]
    smoothed = np.empty_like(landmarks_sequence)

    x = landmarks_sequence[0].copy()
    p = np.ones_like(x)
    smoothed[0] = x
    for t in range(1, t_len):
        z = landmarks_sequence[t]
        p = p + process_var
        k = p / (p + measurement_var)
        x = x + k * (z - x)
        p = (1 - k) * p
        smoothed[t] = x
    return smoothed


# --- Two-pass alignment (Section 5.1: median reference pose/size, constrained re-alignment) ---

def compute_landmark_span(landmarks):
    """Bounding-box diagonal of a landmark set — a simple proxy for in-frame face size."""
    mins = landmarks.min(axis=0)
    maxs = landmarks.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def compute_reference_pose_and_size(poses, sizes):
    """
    poses: (T, 3) yaw/pitch/roll per frame. sizes: (T,) face size per frame (e.g.
    compute_landmark_span output). Returns (median_pose (3,), median_size) — the clip-level
    reference that individual frames are then checked/constrained against.
    """
    poses = np.asarray(poses, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    return np.median(poses, axis=0), float(np.median(sizes))


def passes_reference_deviation(pose, size, ref_pose, ref_size, max_pose_dev_deg=20.0, max_size_dev_ratio=0.3):
    """Whether a single frame's pose/size is close enough to the clip's reference to keep as-is."""
    pose_dev = float(np.linalg.norm(np.asarray(pose, dtype=np.float64) - ref_pose))
    size_dev_ratio = abs(size - ref_size) / ref_size if ref_size > 0 else float("inf")
    return pose_dev <= max_pose_dev_deg and size_dev_ratio <= max_size_dev_ratio


def clamp_size_to_reference(size, ref_size, max_dev_ratio=0.3):
    """
    Clamps a frame's crop size to within max_dev_ratio of the clip's reference size, instead of
    discarding the frame outright — this is the "re-run alignment constrained to reasonable
    deviation" half of Section 5.1's two-pass approach (the filtering half is
    passes_reference_deviation above). Wiring this into the actual crop-transform computation
    (facelib.LandmarksProcessor.get_transform_mat) is mainscripts/Extractor.py integration work,
    not yet done — see IMPLEMENTATION_PLAN.md Phase 4.
    """
    lo = ref_size * (1.0 - max_dev_ratio)
    hi = ref_size * (1.0 + max_dev_ratio)
    return float(np.clip(size, lo, hi))


# --- MediaPipe -> dlib-68 conversion ---
#
# facelib.LandmarksProcessor.get_transform_mat (DFL's alignment-crop transform, reused unchanged
# by dfl_torch/extract.py) fits a similarity transform against a specific 33-point subset of
# dlib's 68-point landmark scheme. MediaPipe's FaceLandmarker returns 478 points in an entirely
# different topology (no shared index numbering), so using it to drive DFL's alignment code
# needs a correspondence table between the two.
#
# Getting that table wrong would silently misalign every extracted frame -- a high-blast-radius,
# hard-to-detect failure mode with no real face photo available in this dev environment to
# visually verify a hand-derived mapping against. Rather than guess from memory, this table is
# vendored from a maintained, purpose-built, MIT-licensed conversion:
#   https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks (mp2dlib.py, commit as of 2024-09-18)
#   Copyright (c) 2024 Peizhi Yan (Matthew), MIT License.
# Each dlib index maps to one or two MediaPipe indices; where there are two, their coordinates
# are averaged to approximate the dlib landmark position. This is an *approximation* (MediaPipe's
# underlying model and topology differ from dlib's), not a bit-exact equivalent of what a real
# dlib detector would produce on the same face -- but it lands in the same coordinate convention
# and index scheme that every downstream consumer (get_transform_mat, get_image_hull_mask,
# samplelib) expects, which is what actually matters for pipeline compatibility.
_MEDIAPIPE_TO_DLIB68_INDICES = [
    # Face contour (1-17)
    [127], [234], [93], [132, 58], [58, 172], [136], [150], [176], [152],
    [400], [379], [365], [397, 288], [361], [323], [454], [356],
    # Right brow (18-22)
    [70], [63], [105], [66], [107],
    # Left brow (23-27)
    [336], [296], [334], [293], [300],
    # Nose (28-36)
    [168, 6], [197, 195], [5], [4], [75], [97], [2], [326], [305],
    # Right eye (37-42)
    [33], [160], [158], [133], [153], [144],
    # Left eye (43-48)
    [362], [385], [387], [263], [373], [380],
    # Upper lip contour top (49-55)
    [61], [39], [37], [0], [267], [269], [291],
    # Lower lip contour bottom (56-60)
    [321], [314], [17], [84], [91],
    # Upper lip contour bottom (61-65)
    [78], [82], [13], [312], [308],
    # Lower lip contour top (66-68)
    [317], [14], [87],
]
assert len(_MEDIAPIPE_TO_DLIB68_INDICES) == 68


def convert_mediapipe_landmarks_to_dlib68(landmarks):
    """
    landmarks: (478, 2) or (468, 2) array (or list), MediaPipe FaceLandmarker output in pixel
    coordinates (as returned by FaceLandmarkDetector.detect). Returns a (68, 2) float32 array in
    dlib's 68-point index convention, suitable for facelib.LandmarksProcessor.get_transform_mat
    and everything downstream that expects it (see module note above).
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    dlib68 = np.empty((68, landmarks.shape[1]), dtype=np.float64)
    for i, idxs in enumerate(_MEDIAPIPE_TO_DLIB68_INDICES):
        dlib68[i] = landmarks[idxs].mean(axis=0)
    return dlib68.astype(np.float32)
