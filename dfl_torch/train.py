"""
End-to-end training orchestration — assembles every dfl_torch piece (Phases 1-8) into one
runnable pipeline: data loading (dfl_torch.data) -> DF-variant SAEHD model (dfl_torch.model) ->
BF16 autocast (dfl_torch.precision) -> masked reconstruction + LPIPS + optional GAN loss
(dfl_torch.losses) -> LR schedule/EMA/grad accumulation/checkpointing/logging
(dfl_torch.training). Every individual piece has its own unit tests; this is what proves they
actually compose into a working training loop (tests/test_train_e2e.py runs this for real, on
CPU, against the checked-in fixture faceset).

Uses DFL's actual random-warp augmentation (`dfl_torch.data.SAEHDFaceDataset`, reusing
`core.imagelib.warp` unchanged): the encoder sees an elastically-warped `warped` image, and the
reconstruction is compared against `target` — the same sample with the same affine/flip
augmentation but no elastic distortion, matching `models/Model_SAEHD/Model.py`'s actual
`warp=True`/`warp=False` sample pair.

**Adversarial loss applies only to the src reconstruction, not dst** — this matches
`models/Model_SAEHD/Model.py`'s actual behavior (confirmed by reading it, not assumed): despite
the discriminator's internal variable name (`D_src_dst_loss`), the GAN loss there is only ever
computed against `pred_src_src`/`target_src`. At inference/swap time it's `decoder_src`'s output
quality that matters (dst input decoded through the src decoder), so only that path gets
adversarial sharpening. DFL also uses noisy/smoothed real-fake labels for GAN stability
(`gan_smoothing`/`gan_noise` options) — not implemented here, a minor stabilization detail.

Validation (Section 10): a held-out slice (`dfl_torch.data.build_train_val_dataloaders`, never
trained on, warp augmentation disabled) is evaluated periodically; `CheckpointManager` saves the
best checkpoint by *validation* reconstruction loss, not training loss — training loss can keep
dropping from memorization while validation loss stalls or rises, and Section 10 specifically
asks for best-by-validation-metric checkpointing to avoid overwriting a good model with a later
overfit one.
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from dfl_torch.data import build_train_val_dataloaders
from dfl_torch.discriminator import UNetPatchDiscriminator
from dfl_torch.losses import (
    IdentityLoss,
    LPIPSLoss,
    discriminator_gan_loss,
    generator_adversarial_loss,
    masked_reconstruction_loss,
)
from dfl_torch.model import SAEHDModel
from dfl_torch.precision import autocast_context
from dfl_torch.training import CheckpointManager, EMA, GradientAccumulator, TrainingLogger, build_lr_scheduler


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _compute_losses(model, discriminator, warped, target, mask, gan_power, lpips_loss_fn=None, lpips_weight=1.0,
                     identity_loss_fn=None, identity_weight=1.0, is_src=False):
    """
    Shared forward+loss computation for one src or dst batch. `is_src` gates the adversarial and
    identity-preservation terms — GAN loss only applies to the src reconstruction (see module
    docstring); identity loss follows the same convention (it's the src decoder's output whose
    identity fidelity matters, both here and at swap() inference time — dst's reconstruction
    identity isn't the objective SAEHD is actually optimizing for).
    """
    pred, pred_mask = model.forward_src(warped) if is_src else model.forward_dst(warped)

    recon_loss = masked_reconstruction_loss(pred, target, mask)
    mask_loss = (pred_mask - mask).pow(2).mean()
    loss = recon_loss + mask_loss

    lpips_loss = None
    if lpips_loss_fn is not None:
        lpips_loss = lpips_loss_fn(pred, target, mask=mask)
        loss = loss + lpips_weight * lpips_loss

    identity_loss = None
    if is_src and identity_loss_fn is not None:
        identity_loss = identity_loss_fn(pred, target, mask=mask)
        loss = loss + identity_weight * identity_loss

    if is_src and discriminator is not None and gan_power > 0:
        _, fake_logits = discriminator(pred * mask)
        loss = loss + gan_power * generator_adversarial_loss(fake_logits)

    return pred, pred_mask, loss, recon_loss, lpips_loss, identity_loss


@torch.no_grad()
def _evaluate(model, src_val_loader, dst_val_loader, device, lpips_loss_fn=None, identity_loss_fn=None):
    """Mean masked-reconstruction (+ LPIPS/identity, if enabled) loss over the held-out
    validation split, with the model in eval mode and no warp augmentation (see
    build_train_val_dataloaders). Identity loss only applies to the src side, matching
    _compute_losses' training-time convention (see its docstring)."""
    model.eval()
    total_loss, n = 0.0, 0
    for loader, is_src in [(src_val_loader, True), (dst_val_loader, False)]:
        for warped, target, mask in loader:
            warped, target, mask = warped.to(device), target.to(device), mask.to(device)
            pred, _ = model.forward_src(warped) if is_src else model.forward_dst(warped)
            loss = masked_reconstruction_loss(pred, target, mask)
            if lpips_loss_fn is not None:
                loss = loss + lpips_loss_fn(pred, target, mask=mask)
            if is_src and identity_loss_fn is not None:
                loss = loss + identity_loss_fn(pred, target, mask=mask)
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(1, n)


@torch.no_grad()
def _save_preview(model, src_batch, dst_batch, device, path):
    """Saves a grid: [src target | src recon | dst target | dst recon | dst->src swap] for the
    first few samples of a fixed batch — a static preview image on disk in place of DFL's live
    preview window (this pipeline has no GUI)."""
    model.eval()
    src_warped, src_target, _ = [t.to(device) for t in src_batch]
    dst_warped, dst_target, _ = [t.to(device) for t in dst_batch]

    n = min(4, src_target.shape[0], dst_target.shape[0])
    pred_src_src, _ = model.forward_src(src_warped[:n])
    pred_dst_dst, _ = model.forward_dst(dst_warped[:n])
    swapped, _ = model.swap(dst_warped[:n])

    rows = torch.cat([
        src_target[:n], pred_src_src.clamp(0, 1),
        dst_target[:n], pred_dst_dst.clamp(0, 1),
        swapped.clamp(0, 1),
    ], dim=0)
    grid = make_grid(rows, nrow=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
    model.train()


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
    lpips_weight=0.0,
    identity_weight=0.0,
    accumulation_steps=1,
    device_type="cpu",
    val_fraction=0.05,
    checkpoint_every=100,
    log_every=10,
    preview_every=0,
    num_workers=0,
    resume_from=None,
    random_blur=False,
    random_noise=False,
    random_jpeg=False,
    random_downsample=False,
    random_hsv_shift_amount=0.0,
    random_shadow=False,
):
    """
    Runs training from `resume_from`'s saved step (or 0) up to `total_steps`, returning
    (model, ema). `lpips_weight=0`/`identity_weight=0` (both default) skip building the
    corresponding network entirely — each is a real download+model on first use (~230MB LPIPS,
    ~107MB identity), not something to pay for unconditionally.

    `random_blur`/`random_noise`/`random_jpeg`/`random_downsample`/`random_hsv_shift_amount`/
    `random_shadow` (all off/0 by default): dfl_torch.augment's pixel-level augmentations, applied
    to both src and dst training data (never validation — see build_train_val_dataloaders).
    """
    device = torch.device(device_type)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    preview_dir = output_dir / "previews"

    augment_kwargs = dict(
        random_blur=random_blur, random_noise=random_noise, random_jpeg=random_jpeg,
        random_downsample=random_downsample, random_hsv_shift_amount=random_hsv_shift_amount,
        random_shadow=random_shadow,
    )
    src_train_loader, src_val_loader = build_train_val_dataloaders(
        src_dir, resolution, batch_size, val_fraction=val_fraction, num_workers=num_workers, **augment_kwargs,
    )
    dst_train_loader, dst_val_loader = build_train_val_dataloaders(
        dst_dir, resolution, batch_size, val_fraction=val_fraction, num_workers=num_workers, **augment_kwargs,
    )
    src_iter = _infinite(src_train_loader)
    dst_iter = _infinite(dst_train_loader)
    preview_src_batch = next(iter(src_train_loader)) if preview_every > 0 else None
    preview_dst_batch = next(iter(dst_train_loader)) if preview_every > 0 else None

    model = SAEHDModel(resolution, e_dims=e_dims, ae_dims=ae_dims, d_dims=d_dims, d_mask_dims=d_mask_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    ema = EMA(model, decay=0.999)
    accumulator = GradientAccumulator(accumulation_steps=accumulation_steps)
    checkpoint_manager = CheckpointManager(checkpoint_dir, higher_is_better=False)  # lower val loss is better
    logger = TrainingLogger(log_dir)

    discriminator = None
    disc_optimizer = None
    if gan_power > 0:
        discriminator = UNetPatchDiscriminator(patch_size=max(1, resolution // 8), in_ch=3, base_ch=gan_dims).to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    lpips_loss_fn = LPIPSLoss().to(device) if lpips_weight > 0 else None
    identity_loss_fn = IdentityLoss().to(device) if identity_weight > 0 else None

    start_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        if discriminator is not None and "discriminator" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator"])
            disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
        start_step = checkpoint["step"]

    def _checkpoint_state():
        # step + 1: `step` is the last *completed* loop index (0-based); the checkpoint records
        # how many steps are done / where to resume (range(start_step, total_steps)), so it must
        # be the next index to run, not the last one — off by one otherwise, and resuming would
        # silently repeat the last step of the previous run.
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "ema": ema.state_dict(), "step": step + 1,
        }
        if discriminator is not None:
            state["discriminator"] = discriminator.state_dict()
            state["disc_optimizer"] = disc_optimizer.state_dict()
        return state

    optimizer.zero_grad()
    last_recon_loss = None
    step = start_step
    for step in range(start_step, total_steps):
        src_warped, src_target, src_mask = next(src_iter)
        dst_warped, dst_target, dst_mask = next(dst_iter)
        src_warped, src_target, src_mask = src_warped.to(device), src_target.to(device), src_mask.to(device)
        dst_warped, dst_target, dst_mask = dst_warped.to(device), dst_target.to(device), dst_mask.to(device)

        with autocast_context(device_type):
            pred_src_src, _, src_loss, src_recon_loss, src_lpips, src_identity = _compute_losses(
                model, discriminator, src_warped, src_target, src_mask, gan_power,
                lpips_loss_fn, lpips_weight, identity_loss_fn, identity_weight, is_src=True,
            )
            _, _, dst_loss, dst_recon_loss, dst_lpips, _dst_identity = _compute_losses(
                model, discriminator, dst_warped, dst_target, dst_mask, gan_power,
                lpips_loss_fn, lpips_weight, identity_loss_fn, identity_weight, is_src=False,
            )
            gen_loss = src_loss + dst_loss
            recon_loss = src_recon_loss + dst_recon_loss

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
            log_dict = {"loss/gen": gen_loss.item(), "loss/recon": last_recon_loss}
            if lpips_loss_fn is not None:
                log_dict["loss/lpips"] = (src_lpips.item() + dst_lpips.item())
            if identity_loss_fn is not None:
                log_dict["loss/identity"] = src_identity.item()
            logger.log_scalars(log_dict, step)

        if preview_every > 0 and step % preview_every == 0:
            _save_preview(model, preview_src_batch, preview_dst_batch, device, preview_dir / f"step_{step:07d}.png")

        if step % checkpoint_every == 0 and step > 0:
            val_loss = _evaluate(model, src_val_loader, dst_val_loader, device, lpips_loss_fn, identity_loss_fn)
            logger.log_scalar("loss/val", val_loss, step)
            checkpoint_manager.maybe_save(metric=val_loss, state_dict=_checkpoint_state())
            checkpoint_manager.save_latest(_checkpoint_state())

    val_loss = _evaluate(model, src_val_loader, dst_val_loader, device, lpips_loss_fn, identity_loss_fn)
    logger.log_scalar("loss/val", val_loss, total_steps)
    checkpoint_manager.maybe_save(metric=val_loss, state_dict=_checkpoint_state())
    checkpoint_manager.save_latest(_checkpoint_state())
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
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--identity-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--preview-every", type=int, default=0, help="0 disables preview saving")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-from", default=None, help="Path to a checkpoint (e.g. output_dir/checkpoints/latest.pt) to resume from")
    parser.add_argument("--device-type", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random-blur", action="store_true", default=False)
    parser.add_argument("--random-noise", action="store_true", default=False)
    parser.add_argument("--random-jpeg", action="store_true", default=False)
    parser.add_argument("--random-downsample", action="store_true", default=False)
    parser.add_argument("--random-hsv-shift-amount", type=float, default=0.0)
    parser.add_argument("--random-shadow", action="store_true", default=False)
    args = parser.parse_args()
    train(
        args.src_dir, args.dst_dir, args.output_dir,
        resolution=args.resolution, batch_size=args.batch_size, total_steps=args.total_steps,
        warmup_steps=args.warmup_steps, lr=args.lr, gan_power=args.gan_power, lpips_weight=args.lpips_weight,
        identity_weight=args.identity_weight,
        checkpoint_every=args.checkpoint_every, log_every=args.log_every, preview_every=args.preview_every,
        num_workers=args.num_workers, resume_from=args.resume_from, device_type=args.device_type,
        random_blur=args.random_blur, random_noise=args.random_noise, random_jpeg=args.random_jpeg,
        random_downsample=args.random_downsample, random_hsv_shift_amount=args.random_hsv_shift_amount,
        random_shadow=args.random_shadow,
    )


if __name__ == "__main__":
    main()
