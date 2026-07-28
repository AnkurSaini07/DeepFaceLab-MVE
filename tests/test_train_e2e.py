"""
End-to-end orchestration test for dfl_torch/train.py — proves every individual dfl_torch piece
(data loading, model, precision, losses, training-loop utilities), each already unit-tested in
isolation, actually composes into a working training run. Uses the checked-in synthetic fixture
faceset (tests/fixtures/faceset/, 3 images) for both src and dst, small dims/resolution for
speed, and runs entirely on CPU.
"""
from pathlib import Path

import torch

from dfl_torch.model import SAEHDModel
from dfl_torch.train import train

FIXTURE_FACESET = Path(__file__).resolve().parent / "fixtures" / "faceset"
RESOLUTION = 32


def test_train_runs_end_to_end_and_writes_checkpoints(tmp_path):
    model, ema = train(
        src_dir=FIXTURE_FACESET,
        dst_dir=FIXTURE_FACESET,
        output_dir=tmp_path,
        resolution=RESOLUTION,
        e_dims=8,
        ae_dims=16,
        d_dims=8,
        d_mask_dims=4,
        batch_size=2,
        total_steps=6,
        warmup_steps=2,
        lr=1e-3,
        checkpoint_every=3,
        log_every=1,
    )

    assert isinstance(model, SAEHDModel)
    assert (tmp_path / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert any((tmp_path / "logs").iterdir())

    checkpoint = torch.load(tmp_path / "checkpoints" / "latest.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 6
    # loaded state dict should apply cleanly to a freshly constructed model of the same shape
    reloaded = SAEHDModel(RESOLUTION, e_dims=8, ae_dims=16, d_dims=8, d_mask_dims=4)
    reloaded.load_state_dict(checkpoint["model"])


def test_train_with_gan_loss_runs_end_to_end(tmp_path):
    model, ema = train(
        src_dir=FIXTURE_FACESET,
        dst_dir=FIXTURE_FACESET,
        output_dir=tmp_path,
        resolution=RESOLUTION,
        e_dims=8,
        ae_dims=16,
        d_dims=8,
        d_mask_dims=4,
        gan_dims=4,
        batch_size=2,
        total_steps=4,
        warmup_steps=1,
        lr=1e-3,
        gan_power=0.1,
        checkpoint_every=2,
        log_every=1,
    )
    assert isinstance(model, SAEHDModel)


def test_train_reduces_reconstruction_loss_over_more_steps(tmp_path):
    """Not a strict monotonic-decrease check (batches are shuffled from a 3-image dataset, so
    there's some step-to-step noise) — checks the model actually learns something over a longer
    run by comparing average loss in the first vs. last third of training."""
    torch.manual_seed(0)
    losses = []

    from dfl_torch.data import build_dataloader
    from dfl_torch.losses import masked_reconstruction_loss
    from dfl_torch.training import build_lr_scheduler

    src_loader = build_dataloader(FIXTURE_FACESET, RESOLUTION, batch_size=2, num_workers=0)
    model = SAEHDModel(RESOLUTION, e_dims=8, ae_dims=16, d_dims=8, d_mask_dims=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=3, total_steps=60)

    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite(src_loader)
    for _ in range(60):
        img, mask = next(data_iter)
        optimizer.zero_grad()
        pred, _ = model.forward_src(img)
        loss = masked_reconstruction_loss(pred, img, mask)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    first_third = sum(losses[:20]) / 20
    last_third = sum(losses[-20:]) / 20
    assert last_third < first_third * 0.5, f"did not learn: {first_third} -> {last_third}"
