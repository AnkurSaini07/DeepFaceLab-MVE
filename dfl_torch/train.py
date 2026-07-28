"""
End-to-end training orchestration — assembles every dfl_torch piece (Phases 1-8) into one
runnable pipeline: data loading (dfl_torch.data) -> DF-variant SAEHD model (dfl_torch.model) ->
BF16 autocast (dfl_torch.precision) -> masked reconstruction + optional GAN loss
(dfl_torch.losses) -> LR schedule/EMA/grad accumulation/checkpointing/logging
(dfl_torch.training). Every individual piece has its own unit tests; this is what proves they
actually compose into a working training loop (tests/test_train_e2e.py runs this for real, on
CPU, against the checked-in fixture faceset).

Uses DFL's actual random-warp augmentation (`dfl_torch.data.SAEHDFaceDataset`, reusing
`core.imagelib.warp` unchanged): the encoder sees an elastically-warped `warped` image, and the
reconstruction is compared against `target` — the same sample with the same affine/flip
augmentation but no elastic distortion, matching `models/Model_SAEHD/Model.py`'s actual
`warp=True`/`warp=False` sample pair. (`tests/test_training.py`'s overfit-one-sample test doesn't
go through this data pipeline at all — it trains directly on a synthetic in-memory tensor to
validate loop wiring in isolation, so warp augmentation doesn't apply there.)
"""
import argparse
from pathlib import Path

import torch

from dfl_torch.data import build_dataloader
from dfl_torch.discriminator import UNetPatchDiscriminator
from dfl_torch.losses import discriminator_gan_loss, generator_adversarial_loss, masked_reconstruction_loss
from dfl_torch.model import SAEHDModel
from dfl_torch.precision import autocast_context
from dfl_torch.training import CheckpointManager, EMA, GradientAccumulator, TrainingLogger, build_lr_scheduler


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def train(
    src_dir,
    dst_dir,
    output_dir,
    resolution=128,
    e_dims=64,
    ae_dims=256,
    d_dims=64,
    d_mask_dims=22,
    batch_size=4,
    total_steps=1000,
    warmup_steps=100,
    lr=5e-5,
    gan_power=0.0,
    gan_dims=16,
    accumulation_steps=1,
    device_type="cpu",
    checkpoint_every=100,
    log_every=10,
    num_workers=0,
):
    """Runs `total_steps` training steps and returns (model, ema) — mainly for tests/callers that
    want to inspect the result directly rather than just reading checkpoints off disk."""
    device = torch.device(device_type)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    src_loader = build_dataloader(src_dir, resolution, batch_size, num_workers=num_workers, cache_in_ram=True)
    dst_loader = build_dataloader(dst_dir, resolution, batch_size, num_workers=num_workers, cache_in_ram=True)
    src_iter = _infinite(src_loader)
    dst_iter = _infinite(dst_loader)

    model = SAEHDModel(resolution, e_dims=e_dims, ae_dims=ae_dims, d_dims=d_dims, d_mask_dims=d_mask_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    ema = EMA(model, decay=0.999)
    accumulator = GradientAccumulator(accumulation_steps=accumulation_steps)
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    logger = TrainingLogger(log_dir)

    discriminator = None
    disc_optimizer = None
    if gan_power > 0:
        discriminator = UNetPatchDiscriminator(patch_size=max(1, resolution // 8), in_ch=3, base_ch=gan_dims).to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    optimizer.zero_grad()
    last_recon_loss = None
    for step in range(total_steps):
        # warped_*: elastically-distorted input the encoder sees. target_*/*_mask: the same
        # sample with only the shared affine/flip augmentation (no elastic warp) — what the
        # reconstruction is compared against. See dfl_torch/data.py's SAEHDFaceDataset docstring.
        src_warped, src_target, src_mask = next(src_iter)
        dst_warped, dst_target, dst_mask = next(dst_iter)
        src_warped, src_target, src_mask = src_warped.to(device), src_target.to(device), src_mask.to(device)
        dst_warped, dst_target, dst_mask = dst_warped.to(device), dst_target.to(device), dst_mask.to(device)

        with autocast_context(device_type):
            pred_src_src, pred_src_mask = model.forward_src(src_warped)
            pred_dst_dst, pred_dst_mask = model.forward_dst(dst_warped)

            recon_loss = masked_reconstruction_loss(pred_src_src, src_target, src_mask) \
                + masked_reconstruction_loss(pred_dst_dst, dst_target, dst_mask)
            mask_loss = (pred_src_mask - src_mask).pow(2).mean() + (pred_dst_mask - dst_mask).pow(2).mean()
            gen_loss = recon_loss + mask_loss

            if discriminator is not None:
                _, fake_logits = discriminator(pred_src_src * src_mask)
                gen_loss = gen_loss + gan_power * generator_adversarial_loss(fake_logits)

        accumulator.scale_loss(gen_loss).backward()
        if accumulator.step(optimizer):
            scheduler.step()
            ema.update(model)

        if discriminator is not None:
            disc_optimizer.zero_grad()
            with autocast_context(device_type):
                _, real_logits = discriminator(src_target * src_mask)
                _, fake_logits_detached = discriminator((pred_src_src * src_mask).detach())
                disc_loss = discriminator_gan_loss(real_logits, fake_logits_detached)
            disc_loss.backward()
            disc_optimizer.step()

        last_recon_loss = recon_loss.item()
        if step % log_every == 0:
            logger.log_scalars({"loss/gen": gen_loss.item(), "loss/recon": last_recon_loss}, step)

        if step % checkpoint_every == 0 and step > 0:
            checkpoint_manager.maybe_save(
                metric=last_recon_loss,
                state_dict={"model": model.state_dict(), "ema": ema.state_dict(), "step": step},
            )
            checkpoint_manager.save_latest({"model": model.state_dict(), "step": step})

    checkpoint_manager.maybe_save(
        metric=last_recon_loss,
        state_dict={"model": model.state_dict(), "ema": ema.state_dict(), "step": total_steps},
    )
    checkpoint_manager.save_latest({"model": model.state_dict(), "step": total_steps})
    logger.close()
    return model, ema


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch SAEHD (DF-variant) model.")
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dst-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gan-power", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device-type", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(
        args.src_dir, args.dst_dir, args.output_dir,
        resolution=args.resolution, batch_size=args.batch_size, total_steps=args.total_steps,
        warmup_steps=args.warmup_steps, lr=args.lr, gan_power=args.gan_power,
        num_workers=args.num_workers, device_type=args.device_type,
    )


if __name__ == "__main__":
    main()
