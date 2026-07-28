"""
Modern face-parsing network — requirements.md Section 6.4: "Consider replacing/supplementing
DFL's default mask generation with a modern face-parsing network (BiSeNet-based) for better
boundary precision around hair/glasses/edges, reducing manual XSeg labeling needs."

Wraps `zllrunning/face-parsing.PyTorch`'s BiSeNet architecture (via the `face-parsing` PyPI
package's model definitions — vendored architecture only, not its CUDA-only inference script,
which this reimplements for CPU portability) with the original pretrained CelebAMask-HQ weights.
The original checkpoint is hosted on Google Drive, which isn't reliably automatable (no stable
direct-download URL, confirmation-token dance); this uses a Hugging Face mirror instead —
verified to share the original's SHA256 across multiple independent HF repos before using it.

19 output classes (CelebAMask-HQ taxonomy): 0 background, 1 skin, 2 l_brow, 3 r_brow, 4 l_eye,
5 r_eye, 6 eye_g (glasses), 7 l_ear, 8 r_ear, 9 ear_r (earring), 10 nose, 11 mouth, 12 u_lip,
13 l_lip, 14 neck, 15 neck_l (necklace), 16 cloth, 17 hair, 18 hat.

Supplements (per Section 6.4's "replacing/supplementing"), doesn't replace, DFL's existing mask
generation (`facelib.LandmarksProcessor.get_image_hull_mask`, `dfl_torch.masking`) — finer
boundary precision around hair/glasses at the cost of a real network forward pass + its own
~53MB (+ ~45MB ResNet18 ImageNet backbone, downloaded automatically by torchvision) model.
"""
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch

MODEL_URL = "https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth"
DEFAULT_MODEL_CACHE_PATH = Path.home() / ".cache" / "dfl_torch" / "bisenet_face_parsing.pth"

# CelebAMask-HQ 19-class taxonomy (see module docstring).
FACE_SKIN_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)  # skin, brows, eyes, nose, mouth/lips
HAIR_CLASS = 17
GLASSES_CLASS = 6
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def download_bisenet_model(cache_path=DEFAULT_MODEL_CACHE_PATH):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MODEL_URL, cache_path)
    return cache_path


class FaceParser:
    """Wraps BiSeNet for 19-class face parsing on an aligned face crop of any resolution."""

    def __init__(self, model_path=None, device="cpu"):
        from face_parsing.model import BiSeNet

        model_path = Path(model_path) if model_path is not None else download_bisenet_model()
        self.device = device
        self.model = BiSeNet(n_classes=19)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def parse(self, image_bgr):
        """
        image_bgr: HWC BGR array, uint8 [0, 255] or float [0, 1]. Returns an (H, W) int64
        class-id map at the *same* resolution as the input — internally resizes to the
        network's native 512x512, then resizes the class map back with nearest-neighbor
        (never linear/cubic — interpolating between class IDs would invent nonsense classes).
        """
        h, w = image_bgr.shape[:2]
        if image_bgr.dtype != np.float32 and image_bgr.dtype != np.float64:
            image_bgr = image_bgr.astype(np.float32) / 255.0

        resized = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = resized[..., ::-1]
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD

        img_t = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float()
        img_t = img_t.unsqueeze(0).to(self.device)
        out = self.model(img_t)[0]  # main head only; the other two are training-only aux heads
        class_map = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

        if (h, w) != (512, 512):
            class_map = cv2.resize(class_map, (w, h), interpolation=cv2.INTER_NEAREST)
        return class_map.astype(np.int64)


def class_map_to_mask(class_map, class_ids):
    """Binary (H, W, 1) float32 mask where class_map is any of class_ids."""
    mask = np.isin(class_map, class_ids).astype(np.float32)
    return mask[..., None]


def face_skin_mask(class_map):
    """The 'face' region per Section 6.4: skin + brows + eyes + nose + mouth/lips — excludes
    hair, ears, neck, clothing, hat. A finer-boundary alternative to
    facelib.LandmarksProcessor.get_image_hull_mask's landmark-hull approach."""
    return class_map_to_mask(class_map, FACE_SKIN_CLASSES)


def hair_mask(class_map):
    return class_map_to_mask(class_map, [HAIR_CLASS])


def glasses_mask(class_map):
    return class_map_to_mask(class_map, [GLASSES_CLASS])
