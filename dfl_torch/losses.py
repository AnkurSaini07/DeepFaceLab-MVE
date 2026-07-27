"""
Loss functions — requirements.md Section 9.

**Masking convention (matches DFL's actual code, not a new design):** `models/Model_SAEHD/Model.py`
pre-multiplies both the target and prediction by the mask *before* computing SSIM/pixel loss or
feeding the discriminator (e.g. `gpu_target_src_masked_opt = gpu_target_src*gpu_target_srcm_blur`,
then `DLoss(..., self.D_src(gpu_pred_src_src_masked_opt))`), then averages over *all* pixels —
not just the masked-in ones. That average-over-everything (rather than average-over-masked-only)
is what makes a heavily-occluded frame contribute proportionally less to the loss automatically,
without a separate downweighting step (Section 7's severity-based downweighting falls out of this
for free). This module follows the same convention: callers mask pred/target *before* calling
these functions, mirroring DFL rather than doing map-level masking inside a windowed SSIM.

**Correction to the common "SSIM + L1" description** (requirements.md Section 9, and DFL's own
docs): the actual reconstruction loss in `models/Model_SAEHD/Model.py` is MS-SSIM +
*squared* error (`tf.square`), not L1/absolute error — `masked_reconstruction_loss` below matches
what the code actually does, not the shorthand name.

Adds LPIPS (via the `lpips` package, AlexNet backbone — the lightest available, though its
ImageNet-pretrained weights are still a ~230MB download cached to `~/.cache/torch/hub/`) and
GAN adversarial loss (BCE-with-logits, matching DFL's actual `DLoss`, not hinge — see
`models/Model_SAEHD/Model.py`'s `DLoss` for the source of this convention) using the Phase 1
discriminator (`dfl_torch.discriminator.UNetPatchDiscriminator`).

**Not implemented:** identity-preservation loss (ArcFace embedding similarity) — same
heavier-dependency deferral as Phase 5's mic detector and Phase 6's ArcFace dedup signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred, target, max_val=1.0, window_size=11, sigma=1.5):
    """
    Single-scale SSIM (Wang et al. 2004), windowed via Gaussian filtering — a clean-room
    reimplementation of the same family of metric as DFL's `tf.image.ssim_multiscale` (not
    multi-scale here, for simplicity; see module docstring for the masking convention this is
    meant to be used under). Returns one value per batch item, in [-1, 1] (1 = identical).
    """
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_t = F.conv2d(target, window, padding=pad, groups=channels)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

    sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_p2
    sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_t2
    sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pt

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2))
    return ssim_map.mean(dim=[1, 2, 3])


def masked_reconstruction_loss(pred, target, mask, ssim_weight=10.0, pixel_weight=10.0, max_val=1.0):
    """
    pred, target: (N, C, H, W) in [0, 1]. mask: (N, 1, H, W) in [0, 1], broadcast over channels.
    Mask is applied here (multiplies pred/target) — pass unmasked images in, matching DFL's
    `masked_training` option where this masking is the actual mask application point.
    """
    pred_m = pred * mask
    target_m = target * mask
    ssim_loss = 1.0 - ssim(pred_m, target_m, max_val=max_val)
    pixel_loss = (pred_m - target_m).pow(2).mean(dim=[1, 2, 3])
    return (ssim_weight * ssim_loss + pixel_weight * pixel_loss).mean()


class LPIPSLoss(nn.Module):
    """Wraps the `lpips` package's perceptual loss. lpips expects inputs in [-1, 1]; this
    rescales from this codebase's [0, 1] convention internally. Mask (if given) is applied at
    the image level before the backbone, same convention as masked_reconstruction_loss."""

    def __init__(self, net="alex"):
        super().__init__()
        import lpips

        self.model = lpips.LPIPS(net=net)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred * mask
            target = target * mask
        return self.model(pred * 2 - 1, target * 2 - 1).mean()


def discriminator_gan_loss(real_logits, fake_logits):
    """BCE-with-logits GAN loss for the discriminator (real -> 1, fake -> 0), matching DFL's
    `DLoss`. Each argument is a single logits tensor or an iterable of them (e.g. the
    (center_out, out) tuple UNetPatchDiscriminator returns) — summed over all of them."""
    real_logits = real_logits if isinstance(real_logits, (list, tuple)) else [real_logits]
    fake_logits = fake_logits if isinstance(fake_logits, (list, tuple)) else [fake_logits]
    loss = 0.0
    for r in real_logits:
        loss = loss + F.binary_cross_entropy_with_logits(r, torch.ones_like(r))
    for f in fake_logits:
        loss = loss + F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))
    return loss


def generator_adversarial_loss(fake_logits):
    """The generator's adversarial term: wants the discriminator to score fakes as real (-> 1)."""
    fake_logits = fake_logits if isinstance(fake_logits, (list, tuple)) else [fake_logits]
    loss = 0.0
    for f in fake_logits:
        loss = loss + F.binary_cross_entropy_with_logits(f, torch.ones_like(f))
    return loss
