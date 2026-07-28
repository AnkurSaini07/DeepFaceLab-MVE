"""
Inference/merge pipeline (Phase 12a) — runs a trained SAEHDModel's face swap on one aligned crop
and blends the result back into the original full frame using the model's own predicted mask.

Clean-room, standalone pipeline (dfl_torch/merge.py), not a patch to mainscripts/Merger.py /
merger/MergeMasked.py — consistent with this migration's approach throughout: parallel
implementation, legacy code untouched until ready to fully retire. DFL's actual merge logic
(merger/MergeMasked.py) supports many optional modes (raw-rgb/raw-predict, hist-match, seamless
clone, seven XSeg-based mask-combination variants, two-pass merging, super-resolution via a face
enhancer, seven color-transfer algorithms, a separate `output_face_scale`/custom-face-type path).
This covers the single most common path — overlay blending with the model's own learned mask and
Reinhard color transfer (DFL's default `color_transfer_mode`) — not every mode; the rest are
tracked as not-yet-done in IMPLEMENTATION_PLAN.md, not silently dropped.

`facelib.LandmarksProcessor.get_transform_mat` is reused unchanged for the alignment-crop
transform, same as dfl_torch/extract.py.
"""
import cv2
import numpy as np
import torch

from core import imagelib
from facelib import FaceType, LandmarksProcessor


def merge_one_frame(model, frame_bgr, face_landmarks, resolution, face_type=FaceType.FULL,
                     erode=0, blur=0, color_transfer=True, device="cpu"):
    """
    model: a dfl_torch.model.SAEHDModel (trained; called in eval mode here, but the model's own
    `.eval()`/`.train()` state is the caller's responsibility — see the note in merge_frames).
    frame_bgr: HWC float32 [0, 1] BGR full frame. face_landmarks: (68, 2) dlib-scheme landmarks
    in frame_bgr's coordinate space (e.g. a Sample's `source_landmarks`, or freshly detected).
    Returns the merged full frame (HWC float32 [0, 1] BGR) — `frame_bgr` outside the face region,
    the swapped face blended in within it.
    """
    img_size = (frame_bgr.shape[1], frame_bgr.shape[0])

    face_mat = LandmarksProcessor.get_transform_mat(face_landmarks, resolution, face_type=face_type)
    dst_face_bgr = cv2.warpAffine(frame_bgr, face_mat, (resolution, resolution), flags=cv2.INTER_CUBIC)
    dst_face_bgr = np.clip(dst_face_bgr, 0.0, 1.0).astype(np.float32)

    with torch.no_grad():
        input_t = torch.from_numpy(dst_face_bgr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        pred_bgr_t, pred_mask_t = model.swap(input_t)
        pred_bgr = pred_bgr_t[0].permute(1, 2, 0).cpu().numpy()
        pred_mask = pred_mask_t[0, 0].cpu().numpy()

    pred_bgr = np.clip(pred_bgr, 0.0, 1.0).astype(np.float32)
    pred_mask = np.clip(pred_mask, 0.0, 1.0).astype(np.float32)

    if color_transfer:
        mask_binary = (pred_mask > 0).astype(np.float32)[..., None]
        pred_bgr = imagelib.reinhard_color_transfer(pred_bgr, dst_face_bgr, target_mask=mask_binary, source_mask=mask_binary)
        pred_bgr = np.clip(pred_bgr, 0.0, 1.0).astype(np.float32)

    wrk_mask = pred_mask.copy()
    wrk_mask[wrk_mask < (1.0 / 255.0)] = 0.0
    if erode > 0:
        wrk_mask = cv2.erode(wrk_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)))
    elif erode < 0:
        wrk_mask = cv2.dilate(wrk_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-erode, -erode)))
    if blur > 0:
        blur_k = blur + (1 - blur % 2)
        wrk_mask = cv2.GaussianBlur(wrk_mask, (blur_k, blur_k), 0)
    wrk_mask = np.clip(wrk_mask, 0.0, 1.0).astype(np.float32)

    img_face_mask = cv2.warpAffine(wrk_mask, face_mat, img_size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    img_face_mask = np.clip(img_face_mask, 0.0, 1.0).astype(np.float32)
    img_face_mask[img_face_mask < (1.0 / 255.0)] = 0.0
    if img_face_mask.ndim == 2:
        img_face_mask = img_face_mask[..., None]

    img_face_pred = cv2.warpAffine(pred_bgr, face_mat, img_size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    img_face_pred = np.clip(img_face_pred, 0.0, 1.0).astype(np.float32)

    out = frame_bgr * (1.0 - img_face_mask) + img_face_pred * img_face_mask
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def merge_video_frames(model, frame_paths, landmarks_by_frame, output_dir, resolution,
                        face_type=FaceType.FULL, erode=0, blur=0, color_transfer=True, device="cpu"):
    """
    Merges a sequence of already-extracted frames (frame_paths, in order) using per-frame
    landmarks (landmarks_by_frame: dict mapping frame path -> (68, 2) landmarks, or None/missing
    for frames with no detected face, which are passed through unchanged). Writes one merged
    image per input frame to output_dir, same filenames. Returns the number of frames merged
    (as opposed to passed through unchanged).

    Assembling merged frames into an actual video file is a separate step (ffmpeg, same as DFL's
    own `main.py videoed` commands) — not done here, this only produces the merged frame images.
    """
    from pathlib import Path

    from core.cv2ex import cv2_imread

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    merged_count = 0
    for frame_path in frame_paths:
        frame_path = Path(frame_path)
        landmarks = landmarks_by_frame.get(str(frame_path))
        if landmarks is None:
            landmarks = landmarks_by_frame.get(frame_path)
        frame_bgr = cv2_imread(str(frame_path)).astype(np.float32) / 255.0

        if landmarks is None:
            out = frame_bgr
        else:
            out = merge_one_frame(
                model, frame_bgr, np.asarray(landmarks), resolution, face_type=face_type,
                erode=erode, blur=blur, color_transfer=color_transfer, device=device,
            )
            merged_count += 1

        out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / frame_path.name), out_u8)

    return merged_count
