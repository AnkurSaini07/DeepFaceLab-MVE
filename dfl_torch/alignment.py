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

Scope note: this covers the detector wrapper, pose estimation, and per-frame quality-filtering
predicates (Section 5.1's confidence threshold, pose-range filtering, jitter detection). Temporal
smoothing (moving average / Kalman) and two-pass alignment are not yet implemented — see
IMPLEMENTATION_PLAN.md Phase 4 for what's left.
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
