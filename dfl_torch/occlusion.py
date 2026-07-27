"""
Hand-specific occlusion detection — requirements.md Section 6.3: "Hand-specific occlusion:
MediaPipe Hands as a fast specific-case detector."

The other two occlusion sources named in Section 6.3 — a custom-trained mic detector (needs a
few dozen manually boxed real examples from this project's actual footage, which don't exist
yet) and SAM as a general point/box-prompted fallback (a much heavier dependency+checkpoint) —
are not implemented here; this covers only the hand case, which is self-contained and, like
dfl_torch/alignment.py's face landmarker, ships with a single bundled model download.
"""
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_CACHE_PATH = Path.home() / ".cache" / "dfl_torch" / "hand_landmarker.task"


def download_hand_landmarker_model(cache_path=DEFAULT_MODEL_CACHE_PATH):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MODEL_URL, cache_path)
    return cache_path


class HandLandmarkDetector:
    """Wraps MediaPipe's HandLandmarker task. Returns per-hand 21-point landmark sets."""

    def __init__(self, model_path=None, min_detection_confidence=0.5, num_hands=2):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(model_path) if model_path is not None else download_hand_landmarker_model()

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._mp = mp
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, image_rgb):
        """
        image_rgb: HWC uint8 RGB array.
        Returns a list of (21, 2) pixel-coordinate landmark arrays, one per detected hand
        (possibly empty if no hand was detected).
        """
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)

        h, w = image_rgb.shape[:2]
        return [
            np.array([[lm.x * w, lm.y * h] for lm in hand], dtype=np.float32)
            for hand in result.hand_landmarks
        ]


def hand_landmarks_to_occlusion_mask(image_shape, hands_landmarks, dilate_ratio=0.15):
    """
    Converts detected hand landmark sets into a binary occlusion mask (convex hull per hand,
    filled, then dilated a bit since the 21 landmark points trace the hand skeleton, not its
    silhouette, and a hand covering the mouth typically occludes a somewhat larger area than the
    bare point hull). Returns an all-zero mask if no hands were detected.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w, 1), dtype=np.float32)

    for hand in hands_landmarks:
        hull = cv2.convexHull(hand.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, (1.0,))

    if hands_landmarks:
        span = max(1, int(min(w, h) * dilate_ratio))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)))
        if mask.ndim == 2:
            mask = mask[..., None]

    return mask
