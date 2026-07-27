"""
torch.utils.data.Dataset wrapper around samplelib's Sample objects.

Reuses the existing framework-agnostic DFL/MVE data layer (samplelib.SampleLoader, DFLJPG
metadata parsing, facelib.LandmarksProcessor mask generation) rather than reimplementing
metadata parsing — see IMPLEMENTATION_PLAN.md Phase 2 and requirements.md Section 5.

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

from facelib import LandmarksProcessor
from samplelib import SampleLoader, SampleType


class SAEHDFaceDataset(Dataset):
    """
    Wraps a directory of aligned faces (as produced by DFL/MVE's Extractor) for SAEHD training.
    Each item is the face image + its face mask (XSeg mask if present, else a landmarks-derived
    hull mask), resized to `resolution` and returned as CHW float32 tensors in [0, 1].

    `cache_in_ram=True` eagerly decodes every sample once at construction time and holds the
    result in memory for the lifetime of the dataset — appropriate at the ~1-5k-frame faceset
    sizes this project's src/dst are (see IMPLEMENTATION_PLAN.md's resolved decision #3). At that
    scale, DataLoader worker parallelism (Section 4) matters less than avoiding repeat JPEG
    decode/mask-generation cost; pass num_workers=0 when using this to avoid needlessly
    pickling the in-RAM cache to worker processes.
    """

    def __init__(self, samples_path, resolution, subdirs=False, cache_in_ram=True):
        self.resolution = resolution
        self.samples = SampleLoader.load(SampleType.FACE, Path(samples_path), subdirs=subdirs)
        if len(self.samples) == 0:
            raise ValueError(f"No aligned-face samples found under {samples_path}")

        self.cache_in_ram = cache_in_ram
        self._cache = None
        if cache_in_ram:
            self._cache = [self._load_item(i) for i in range(len(self.samples))]

    def __len__(self):
        return len(self.samples)

    def _load_item(self, index):
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
        img = np.clip(img, 0.0, 1.0)
        mask = np.clip(mask, 0.0, 1.0)
        if mask.ndim == 2:
            mask = mask[..., None]

        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1))).float()
        return img_t, mask_t

    def __getitem__(self, index):
        if self.cache_in_ram:
            return self._cache[index]
        return self._load_item(index)


def build_dataloader(samples_path, resolution, batch_size, num_workers=0, subdirs=False,
                      cache_in_ram=True, shuffle=True):
    """DataLoader per requirements.md Section 4: pin_memory, persistent_workers, prefetch_factor=4."""
    dataset = SAEHDFaceDataset(samples_path, resolution, subdirs=subdirs, cache_in_ram=cache_in_ram)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
