"""
Section 11.4 data pipeline tests: DFLJPG metadata read (via samplelib.SampleLoader), landmark
parsing, mask extraction, and DFL's random-warp augmentation (core.imagelib.warp, reused
unchanged), against the checked-in synthetic fixture faceset (tests/fixtures/faceset/, see
tests/fixtures/generate_face_fixture.py). Pure CPU/NumPy/OpenCV, no TF dependency — run with
.venv-torch.
"""
from pathlib import Path

import pytest
import torch

from dfl_torch.data import SAEHDFaceDataset, build_dataloader, build_train_val_dataloaders

FIXTURE_FACESET = Path(__file__).resolve().parent / "fixtures" / "faceset"
RESOLUTION = 96  # deliberately different from the fixture's native 128 to exercise the resize path


def test_dataset_loads_all_fixture_samples():
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION)
    assert len(dataset) == 3


def test_dataset_item_shapes_and_dtype():
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION)
    warped, target, mask = dataset[0]
    assert warped.shape == (3, RESOLUTION, RESOLUTION)
    assert target.shape == (3, RESOLUTION, RESOLUTION)
    assert mask.shape == (1, RESOLUTION, RESOLUTION)
    assert warped.dtype == torch.float32
    assert target.dtype == torch.float32
    assert mask.dtype == torch.float32


def test_dataset_value_ranges():
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION)
    for i in range(len(dataset)):
        warped, target, mask = dataset[i]
        for t in (warped, target, mask):
            assert 0.0 <= t.min() and t.max() <= 1.0


def test_mask_is_not_degenerate():
    """The landmarks-derived hull mask should cover a plausible, non-trivial fraction of the face."""
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION)
    for i in range(len(dataset)):
        _, _, mask = dataset[i]
        coverage = mask.mean().item()
        assert 0.02 < coverage < 0.98, f"sample {i}: implausible mask coverage {coverage}"


def test_cached_and_uncached_decode_produce_identical_base_data():
    """Caching must not change *what* gets decoded — compared before augmentation, since warp
    augmentation is randomized fresh on every access regardless of caching (see next tests)."""
    cached = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION, cache_in_ram=True)
    uncached = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION, cache_in_ram=False)
    for i in range(len(cached)):
        img_c, mask_c = cached._load_decoded(i)
        img_u, mask_u = uncached._load_decoded(i)
        assert (img_c == img_u).all()
        assert (mask_c == mask_u).all()


def test_warp_augment_false_makes_warped_equal_target():
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION, warp_augment=False)
    warped, target, _mask = dataset[0]
    assert torch.equal(warped, target)


def test_warp_augment_true_makes_warped_differ_from_target():
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION, warp_augment=True)
    warped, target, _mask = dataset[0]
    assert not torch.equal(warped, target)


def test_repeated_getitem_calls_apply_fresh_random_augmentation():
    """Warp augmentation must not be cached alongside the decoded image — every access should
    draw fresh random warp params."""
    dataset = SAEHDFaceDataset(FIXTURE_FACESET, resolution=RESOLUTION, warp_augment=True)
    warped_a, _, _ = dataset[0]
    warped_b, _, _ = dataset[0]
    assert not torch.equal(warped_a, warped_b)


def test_dataloader_produces_correctly_shaped_batches():
    loader = build_dataloader(FIXTURE_FACESET, resolution=RESOLUTION, batch_size=2, num_workers=0, shuffle=False)
    batch_warped, batch_target, batch_mask = next(iter(loader))
    assert batch_warped.shape == (2, 3, RESOLUTION, RESOLUTION)
    assert batch_target.shape == (2, 3, RESOLUTION, RESOLUTION)
    assert batch_mask.shape == (2, 1, RESOLUTION, RESOLUTION)


def test_dataset_raises_on_empty_directory(tmp_path):
    with pytest.raises(ValueError):
        SAEHDFaceDataset(tmp_path, resolution=RESOLUTION)


# --- build_train_val_dataloaders ---

def test_train_val_dataloaders_split_sizes():
    train_loader, val_loader = build_train_val_dataloaders(
        FIXTURE_FACESET, RESOLUTION, batch_size=2, val_fraction=0.3, num_workers=0,
    )
    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 1


def test_train_val_dataloaders_disjoint_and_cover_all_samples():
    train_loader, val_loader = build_train_val_dataloaders(
        FIXTURE_FACESET, RESOLUTION, batch_size=1, val_fraction=0.3, num_workers=0,
    )
    train_indices = set(train_loader.dataset.indices)
    val_indices = set(val_loader.dataset.indices)
    assert train_indices.isdisjoint(val_indices)
    assert train_indices | val_indices == {0, 1, 2}


def test_train_val_dataloaders_val_has_no_warp_augmentation():
    """Validation should evaluate on real (unaugmented) faces — warped should equal target."""
    _, val_loader = build_train_val_dataloaders(
        FIXTURE_FACESET, RESOLUTION, batch_size=1, val_fraction=0.3, num_workers=0,
    )
    for warped, target, _mask in val_loader:
        assert torch.equal(warped, target)


def test_train_val_dataloaders_train_has_warp_augmentation():
    train_loader, _ = build_train_val_dataloaders(
        FIXTURE_FACESET, RESOLUTION, batch_size=1, val_fraction=0.3, num_workers=0,
    )
    saw_difference = False
    for warped, target, _mask in train_loader:
        if not torch.equal(warped, target):
            saw_difference = True
    assert saw_difference


def test_train_val_dataloaders_batch_shapes():
    train_loader, val_loader = build_train_val_dataloaders(
        FIXTURE_FACESET, RESOLUTION, batch_size=2, val_fraction=0.3, num_workers=0,
    )
    train_batch = next(iter(train_loader))
    assert train_batch[0].shape == (2, 3, RESOLUTION, RESOLUTION)
    val_batch = next(iter(val_loader))
    assert val_batch[0].shape == (1, 3, RESOLUTION, RESOLUTION)
