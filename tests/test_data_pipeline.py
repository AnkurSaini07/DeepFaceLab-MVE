"""
Section 11.4 data pipeline tests: DFLJPG metadata read (via samplelib.SampleLoader), landmark
parsing, and mask extraction, against the checked-in synthetic fixture faceset
(tests/fixtures/faceset/, see tests/fixtures/generate_face_fixture.py). Pure CPU/NumPy/OpenCV,
no TF dependency — run with .venv-torch.
"""
from pathlib import Path

import torch

from dfl_torch.data import SAEHDFaceDataset, build_dataloader

FIXTURE_FACESET = Path(__file__).resolve().parent / "fixtures" / "faceset"
RESOLUTION = 96  # deliberately different from the fixture's native 128 to exercise the resize path


def test_dataset_loads_all_fixture_samples():
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION)
    assert len(dataset) == 3


def test_dataset_item_shapes_and_dtype():
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION)
    img, mask = dataset[0]
    assert img.shape == (3, RESOLUTION, RESOLUTION)
    assert mask.shape == (1, RESOLUTION, RESOLUTION)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32


def test_dataset_value_ranges():
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION)
    for i in range(len(dataset)):
        img, mask = dataset[i]
        assert 0.0 <= img.min() and img.max() <= 1.0
        assert 0.0 <= mask.min() and mask.max() <= 1.0


def test_mask_is_not_degenerate():
    """The landmarks-derived hull mask should cover a plausible, non-trivial fraction of the face."""
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION)
    for i in range(len(dataset)):
        _, mask = dataset[i]
        coverage = mask.mean().item()
        assert 0.05 < coverage < 0.95, f"sample {i}: implausible mask coverage {coverage}"


def test_cache_in_ram_matches_uncached():
    cached = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION, cache_in_ram=True)
    uncached = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION, cache_in_ram=False)
    for i in range(len(cached)):
        img_c, mask_c = cached[i]
        img_u, mask_u = uncached[i]
        assert torch.equal(img_c, img_u)
        assert torch.equal(mask_c, mask_u)


def test_dataloader_produces_correctly_shaped_batches():
    loader = build_dataloader(FIXTURE_FACESET, resolution=RESOLUTION, batch_size=2, num_workers=0, shuffle=False)
    batch_img, batch_mask = next(iter(loader))
    assert batch_img.shape == (2, 3, RESOLUTION, RESOLUTION)
    assert batch_mask.shape == (2, 1, RESOLUTION, RESOLUTION)


def test_dataset_raises_on_empty_directory(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        SAEHDFaceDataset(tmp_path, resolution=RESOLUTION)
