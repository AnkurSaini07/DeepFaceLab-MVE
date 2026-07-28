"""
Loss functions — requirements.md Section 9, revised by Section 14c ("Masked-Loss Implementation
Detail").

**Masking convention — two different rules for two different kinds of loss, per Section 14c:**

For **LPIPS and SSIM specifically**, the occlusion mask must NOT be applied as a pre-multiplication
on the input images (`pred * mask`, `target * mask`) before they go through a receptive-field-based
computation (a conv-windowed SSIM, or a CNN backbone). Section 14c's own reasoning, confirmed by
inspection here: zeroing out an occluded region and then computing SSIM over the *whole* map gives
that region a perfect ssim≈1 (two identical all-zero patches look "identical"), and the windowed
convolution near the mask boundary blends zeroed and real pixels together, corrupting the
statistics on the visible side too. The masked-out region's fake-perfect score would then get
averaged into the final loss right alongside the real ones — the opposite of exclusion. The fix
(Section 14c): compute the loss **map** (SSIM map, LPIPS spatial map) from the *unmasked* images,
then average only over the masked-in (visible) pixels: `loss = (error * mask).sum() / mask.sum()`.
That's what `masked_reconstruction_loss` and `LPIPSLoss` do below — see `masked_mean`.

For the **adversarial (GAN) loss**, this pre-multiplication concern doesn't apply the same way —
this follows `models/Model_SAEHD/Model.py`'s actual convention of feeding a mask-multiplied image
into the discriminator (`self.D_src(gpu_pred_src_src_masked_opt)`) directly; Section 14c's revision
is explicitly scoped to "LPIPS and SSIM specifically."

**Correction to the common "SSIM + L1" description** (requirements.md Section 9, and DFL's own
docs): the actual reconstruction loss in `models/Model_SAEHD/Model.py` is MS-SSIM +
*squared* error (`tf.square`), not L1/absolute error — `masked_reconstruction_loss` below matches
what the code actually does, not the shorthand name.

Adds LPIPS (via the `lpips` package, AlexNet backbone — the lightest available, though its
ImageNet-pretrained weights are still a ~230MB download cached to `~/.cache/torch/hub/`) and
GAN adversarial loss (BCE-with-logits, matching DFL's actual `DLoss`, not hinge — see
`models/Model_SAEHD/Model.py`'s `DLoss` for the source of this convention) using the Phase 1
discriminator (`dfl_torch.discriminator.UNetPatchDiscriminator`).

Per Section 14c: LPIPS (and ArcFace, when implemented) must stay frozen and in `.eval()` mode —
`LPIPSLoss` enforces this by overriding `train()` so a parent module's `.train()` call can't
accidentally flip it into training mode.

**Not implemented:** identity-preservation loss (ArcFace embedding similarity) — same
heavier-dependency deferral as Phase 5's mic detector and Phase 6's ArcFace dedup signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x, mask, eps=1e-6):
    """
    x: (N, C, H, W). mask: (N, 1, H, W) (or already (N, C, H, W)), in [0, 1]. Mean over
    masked-in elements only, computed per batch item first (so one sample's occlusion severity
    doesn't skew another's contribution), then averaged across the batch — Section 14c's
    `(error * mask).sum() / mask.sum()`, generalized to a per-sample-then-batch reduction.
    """
    mask = mask.expand_as(x) if mask.shape[1] == 1 and x.shape[1] != 1 else mask
    numer = (x * mask).sum(dim=[1, 2, 3])
    denom = mask.sum(dim=[1, 2, 3]).clamp_min(eps)
    return (numer / denom).mean()


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_map(pred, target, max_val=1.0, window_size=11, sigma=1.5):
    """
    Single-scale SSIM (Wang et al. 2004), windowed via Gaussian filtering — a clean-room
    reimplementation of the same family of metric as DFL's `tf.image.ssim_multiscale` (not
    multi-scale here, for simplicity). Returns the full per-pixel, per-channel map (N, C, H, W)
    in [-1, 1] (1 = identical) — callers that need masking should reduce this with masked_mean
    rather than computing SSIM on pre-masked inputs (Section 14c).
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

    return ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2))


def ssim(pred, target, max_val=1.0, window_size=11, sigma=1.5):
    """Unmasked, whole-image SSIM — one value per batch item. For masked use, use ssim_map +
    masked_mean instead (see masked_reconstruction_loss)."""
    return ssim_map(pred, target, max_val=max_val, window_size=window_size, sigma=sigma).mean(dim=[1, 2, 3])


def masked_reconstruction_loss(pred, target, mask, ssim_weight=10.0, pixel_weight=10.0, max_val=1.0):
    """
    pred, target: (N, C, H, W) in [0, 1]. mask: (N, 1, H, W) in [0, 1].
    Per Section 14c: the SSIM map and per-pixel squared error are computed from the *unmasked*
    pred/target (so occluded-region windows don't produce fake-perfect scores), then averaged
    only over the masked-in region via masked_mean — not by pre-multiplying pred/target by mask.
    """
    s_map = ssim_map(pred, target, max_val=max_val)
    ssim_loss = 1.0 - masked_mean(s_map, mask)
    pixel_loss = masked_mean((pred - target).pow(2), mask)
    return ssim_weight * ssim_loss + pixel_weight * pixel_loss


class LPIPSLoss(nn.Module):
    """
    Wraps the `lpips` package's perceptual loss in spatial mode (`spatial=True`, returns a
    (N, 1, H, W) map matching input resolution) so masking can be applied to the *output* map via
    masked_mean, not to the input images before the backbone (Section 14c). lpips expects inputs
    in [-1, 1]; this rescales from this codebase's [0, 1] convention internally.
    """

    def __init__(self, net="alex"):
        super().__init__()
        import lpips

        self.model = lpips.LPIPS(net=net, spatial=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        # Stays frozen/eval regardless of a parent module's .train() call (Section 14c).
        return super().train(False)

    def forward(self, pred, target, mask=None):
        lpips_map = self.model(pred * 2 - 1, target * 2 - 1)
        if mask is None:
            return lpips_map.mean()
        return masked_mean(lpips_map, mask)


def discriminator_gan_loss(real_logits, fake_logits):
    """BCE-with-logits GAN loss for the discriminator (real -> 1, fake -> 0), matching DFL's
    `DLoss`. Each argument is a single logits tensor or an iterable of them (e.g. the
    (center_out, out) tuple UNetPatchDiscriminator returns) — summed over all of them. Mask, if
    used, is applied by the caller to the image fed into the discriminator (see module docstring
    — this loss isn't in Section 14c's SSIM/LPIPS-specific revision)."""
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
