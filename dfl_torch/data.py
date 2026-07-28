"""
torch.utils.data.Dataset wrapper around samplelib's Sample objects.

Reuses the existing framework-agnostic DFL/MVE data layer (samplelib.SampleLoader, DFLJPG
metadata parsing, facelib.LandmarksProcessor mask generation, core.imagelib.warp's random-warp
augmentation) rather than reimplementing any of it — see IMPLEMENTATION_PLAN.md Phase 2 and
requirements.md Section 5.

Note: despite requirements.md describing "PNG-embedded metadata," this MVE fork's
DFLIMG.load() (DFLIMG/DFLIMG.py) only recognizes .jpg files — the on-disk aligned-faceset format
here is JPG with embedded metadata, not PNG. The data-pipeline code below reuses that format
unchanged either way; the file-extension detail doesn't change anything downstream.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core import imagelib
from facelib import LandmarksProcessor
from samplelib import SampleLoader, SampleType


class SAEHDFaceDataset(Dataset):
    """
    Wraps a directory of aligned faces (as produced by DFL/MVE's Extractor) for SAEHD training.
    Each item is a (warped, target, mask) triple of CHW float32 [0, 1] tensors:
      - `warped`: the face image with DFL's actual random-warp augmentation applied (elastic grid
        distortion + random affine rotation/scale/translation/flip) — what the encoder sees.
      - `target`: the same sample with only the affine/flip part of that same augmentation
        applied (no elastic distortion) — what the reconstruction is compared against. Sharing
        the affine/flip params with `warped` (not independently randomized) keeps the two aligned
        in position; only the elastic warp differs, matching
        `models/Model_SAEHD/Model.py`'s actual `warp=True` vs. `warp=False` sample configuration.
      - `mask`: the face mask (XSeg if present, else a landmarks hull mask), warped by the same
        affine/flip params as `target` (no elastic distortion, matching DFL's `SampleType.FACE_MASK`
        handling — masks never get the elastic warp, only images do).
    Pass `warp_augment=False` to disable the elastic distortion (`warped` becomes identical to
    `target`, same affine/flip augmentation still applied) — e.g. for non-SAEHD uses of this
    dataset that just want one consistently-augmented image, or for eval/preview passes.

    `cache_in_ram=True` eagerly decodes+resizes every sample once at construction time (the
    expensive, cacheable part — JPEG decode, mask generation) and holds that in memory; warp
    augmentation itself is always re-applied fresh on every `__getitem__` call (cheap, and it
    would defeat the point of augmentation to cache it) — appropriate at the ~1-5k-frame faceset
    sizes this project's src/dst are (see IMPLEMENTATION_PLAN.md's resolved decision #3). At that
    scale, DataLoader worker parallelism (Section 4) matters less than avoiding repeat JPEG
    decode/mask-generation cost; pass num_workers=0 when using this to avoid needlessly
    pickling the in-RAM cache to worker processes.
    """

    def __init__(self, samples_path, resolution, subdirs=False, cache_in_ram=True, warp_augment=True,
                 random_flip=True, rotation_range=(-2, 2), scale_range=(-0.5, 0.5),
                 tx_range=(-0.05, 0.05), ty_range=(-0.05, 0.05)):
        self.resolution = resolution
        self.samples = SampleLoader.load(SampleType.FACE, Path(samples_path), subdirs=subdirs)
        if len(self.samples) == 0:
            raise ValueError(f"No aligned-face samples found under {samples_path}")

        self.warp_augment = warp_augment
        self.random_flip = random_flip
        self.rotation_range = list(rotation_range)
        self.scale_range = list(scale_range)
        self.tx_range = list(tx_range)
        self.ty_range = list(ty_range)

        self.cache_in_ram = cache_in_ram
        self._cache = None
        if cache_in_ram:
            self._cache = [self._load_decoded(i) for i in range(len(self.samples))]

    def __len__(self):
        return len(self.samples)

    def _load_decoded(self, index):
        """Decodes + resizes the raw image and mask (the cacheable part) — no augmentation."""
        sample = self.samples[index]
        img = sample.load_bgr()  # HWC, float32, [0, 1], BGR (DFL convention)

        if sample.has_xseg_mask():
            mask = sample.get_xseg_mask()
        else:
            mask = LandmarksProcessor.get_image_hull_mask(img.shape, sample.landmarks, sample.eyebrows_expand_mod)

        if img.shape[0] != self.resolution or img.shape[1] != self.resolution:
            img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)

        # INTER_CUBIC can overshoot [0, 1] near hard edges (ringing) — clamp both back in range.
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if mask.ndim == 2:
            mask = mask[..., None]
        return img, mask

    def _augment(self, img, mask):
        """Applies DFL's actual random-warp augmentation (core.imagelib.warp, reused unchanged —
        framework-agnostic, per requirements.md Section 2/5.1) — fresh random params every call."""
        warp_params = imagelib.gen_warp_params(
            self.resolution, self.random_flip,
            rotation_range=self.rotation_range, scale_range=self.scale_range,
            tx_range=self.tx_range, ty_range=self.ty_range,
        )
        target = imagelib.warp_by_params(warp_params, img, False, True, can_flip=True, border_replicate=True)
        warped = (
            imagelib.warp_by_params(warp_params, img, True, True, can_flip=True, border_replicate=True)
            if self.warp_augment else target
        )
        target_mask = imagelib.warp_by_params(warp_params, mask, False, True, can_flip=True, border_replicate=False)

        target = np.clip(target, 0.0, 1.0).astype(np.float32)
        warped = np.clip(warped, 0.0, 1.0).astype(np.float32)
        target_mask = np.clip(target_mask, 0.0, 1.0).astype(np.float32)
        if target_mask.ndim == 2:
            target_mask = target_mask[..., None]
        return warped, target, target_mask

    @staticmethod
    def _to_chw_tensor(arr):
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()

    def __getitem__(self, index):
        img, mask = self._cache[index] if self.cache_in_ram else self._load_decoded(index)
        warped, target, target_mask = self._augment(img, mask)
        return self._to_chw_tensor(warped), self._to_chw_tensor(target), self._to_chw_tensor(target_mask)


def build_dataloader(samples_path, resolution, batch_size, num_workers=0, subdirs=False,
                      cache_in_ram=True, shuffle=True, warp_augment=True):
    """DataLoader per requirements.md Section 4: pin_memory, persistent_workers, prefetch_factor=4."""
    dataset = SAEHDFaceDataset(
        samples_path, resolution, subdirs=subdirs, cache_in_ram=cache_in_ram, warp_augment=warp_augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
