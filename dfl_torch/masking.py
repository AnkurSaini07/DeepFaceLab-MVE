"""
Two-mask system — requirements.md Section 6.

final_swap_mask = face_mask * (1 - occlusion_mask)

| Mask            | Training                              | Merge/Inference                        |
|-----------------|----------------------------------------|-----------------------------------------|
| Face mask       | Defines region contributing to loss    | Defines blend region into final frame   |
| Occlusion mask  | Excludes region from loss calculation  | Excludes region from swap                |

Feathering reuses DFL's existing proportional erode+blur approach
(facelib.LandmarksProcessor.blur_image_hull_mask, generalized here to any mask, not just the
landmarks-derived face hull) — kernel sizes scale with the mask's own bounding-box extent rather
than a fixed pixel radius, matching that convention. Section 6.2 calls for the occlusion boundary
to be feathered tighter than the outer face-mask edge, hence the separate (smaller default)
occlusion blur ratio in feather_combined_mask.
"""
import cv2
import numpy as np


def feather_mask(mask, erode_ratio=0.085, blur_ratio=0.10):
    """
    mask: (H, W, 1) or (H, W) float32 mask in [0, 1]. A mask with no nonzero pixels (nothing
    detected) is returned unchanged.
    """
    squeeze = mask.ndim == 3 and mask.shape[-1] == 1
    m = mask[..., 0] if squeeze else mask

    region = np.argwhere(m > 0)
    if region.size == 0:
        return mask.copy()

    miny, minx = region.min(axis=0)[:2]
    maxy, maxx = region.max(axis=0)[:2]
    lowest_len = min(maxx - minx, maxy - miny)
    if lowest_len <= 0:
        return mask.copy()

    ero = max(1, int(lowest_len * erode_ratio))
    blur = max(1, int(lowest_len * blur_ratio))

    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero, ero)), iterations=1)
    m = cv2.blur(m, (blur, blur))

    return m[..., None] if squeeze else m


def combine_masks(face_mask, occlusion_mask):
    """final_swap_mask = face_mask * (1 - occlusion_mask). occlusion_mask=None (nothing
    detected) is equivalent to an all-zero occlusion mask."""
    if occlusion_mask is None:
        return face_mask.copy()
    return face_mask * (1.0 - occlusion_mask)


def feather_combined_mask(face_mask, occlusion_mask, face_blur_ratio=0.10, occlusion_blur_ratio=0.04):
    """Feathers face and occlusion masks independently (occlusion tighter, per Section 6.2)
    before combining."""
    feathered_face = feather_mask(face_mask, blur_ratio=face_blur_ratio)
    feathered_occlusion = (
        feather_mask(occlusion_mask, blur_ratio=occlusion_blur_ratio) if occlusion_mask is not None else None
    )
    return combine_masks(feathered_face, feathered_occlusion)
