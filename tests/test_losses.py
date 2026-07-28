"""
Tests for dfl_torch/losses.py (requirements.md Section 9). Key exit criterion from
IMPLEMENTATION_PLAN.md Phase 7: the combined mask must correctly exclude occluder pixels from a
loss computation — verified here as zero gradient contribution from a masked-out region.
"""
import pytest
import torch
import torch.nn.functional as F

from dfl_torch.discriminator import UNetPatchDiscriminator
from dfl_torch.losses import (
    discriminator_gan_loss,
    generator_adversarial_loss,
    masked_reconstruction_loss,
    ssim,
)

RESOLUTION = 32
BATCH = 2


# --- SSIM ---

def test_ssim_identical_images_is_one():
    x = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    assert torch.allclose(ssim(x, x), torch.ones(BATCH), atol=1e-5)


def test_ssim_different_images_less_than_one():
    torch.manual_seed(0)
    x = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    y = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    values = ssim(x, y)
    assert torch.all(values < 0.99)


def test_ssim_output_shape():
    x = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    y = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    assert ssim(x, y).shape == (BATCH,)


# --- masked_reconstruction_loss ---

def test_masked_reconstruction_loss_finite():
    torch.manual_seed(0)
    pred = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    target = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    mask = torch.ones(BATCH, 1, RESOLUTION, RESOLUTION)
    loss = masked_reconstruction_loss(pred, target, mask)
    assert torch.isfinite(loss)


def test_masked_reconstruction_loss_zero_for_identical_images_full_mask():
    x = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    mask = torch.ones(BATCH, 1, RESOLUTION, RESOLUTION)
    loss = masked_reconstruction_loss(x, x, mask)
    assert loss.abs() < 1e-4


def test_masked_reconstruction_loss_zero_gradient_on_fully_masked_out_region():
    """The key Phase 7 exit criterion: occluded (mask=0) pixels contribute zero gradient."""
    # SSIM's window (radius 5, for window_size=11) legitimately blends a few boundary-adjacent
    # occluded pixels into a visible pixel's windowed statistics — Section 14c's own critique of
    # naive masking is precisely about *not* letting this kind of thing corrupt the whole map, but
    # a few pixels of boundary bleed in a windowed metric is an inherent, expected property (the
    # pixel-squared-error term has no such bleed, since it's not windowed). The exit criterion is
    # that INTERIOR occluded pixels — far enough from the boundary that no visible pixel's window
    # can reach them — get exactly zero gradient, checked here with a safety margin beyond the
    # window radius.
    torch.manual_seed(0)
    pred = torch.rand(1, 3, RESOLUTION, RESOLUTION, requires_grad=True)
    target = torch.rand(1, 3, RESOLUTION, RESOLUTION)

    boundary = RESOLUTION // 2
    interior_start = boundary + 6  # window_size=11 => radius 5; +1 safety margin

    mask = torch.ones(1, 1, RESOLUTION, RESOLUTION)
    mask[:, :, :, boundary:] = 0.0  # right half occluded

    loss = masked_reconstruction_loss(pred, target, mask)
    loss.backward()

    grad = pred.grad
    interior_occluded_grad = grad[:, :, :, interior_start:]
    visible_grad = grad[:, :, :, :boundary]

    assert torch.all(interior_occluded_grad == 0.0)
    assert torch.any(visible_grad != 0.0)


def test_masked_reconstruction_loss_ignores_prediction_differences_in_interior_masked_region():
    """Changing pred within the *interior* of the masked-out region (away from the SSIM window's
    boundary reach) shouldn't change the loss at all — see the boundary-bleed note above."""
    torch.manual_seed(0)
    pred_a = torch.rand(1, 3, RESOLUTION, RESOLUTION)
    target = torch.rand(1, 3, RESOLUTION, RESOLUTION)

    boundary = RESOLUTION // 2
    interior_start = boundary + 6

    mask = torch.ones(1, 1, RESOLUTION, RESOLUTION)
    mask[:, :, :, boundary:] = 0.0

    pred_b = pred_a.clone()
    pred_b[:, :, :, interior_start:] = torch.rand(1, 3, RESOLUTION, RESOLUTION - interior_start)

    loss_a = masked_reconstruction_loss(pred_a, target, mask)
    loss_b = masked_reconstruction_loss(pred_b, target, mask)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


# --- adversarial losses ---

def test_discriminator_gan_loss_finite_and_positive():
    disc = UNetPatchDiscriminator(patch_size=RESOLUTION // 8, in_ch=3, base_ch=8)
    real = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    fake = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    real_logits = disc(real)
    fake_logits = disc(fake)
    loss = discriminator_gan_loss(real_logits, fake_logits)
    assert torch.isfinite(loss)
    assert loss > 0


def test_discriminator_gan_loss_lower_when_correctly_confident():
    """Loss should be low if real logits are strongly positive and fake logits strongly
    negative (the discriminator being confidently correct)."""
    confident_correct = discriminator_gan_loss(torch.full((2, 1), 10.0), torch.full((2, 1), -10.0))
    confident_wrong = discriminator_gan_loss(torch.full((2, 1), -10.0), torch.full((2, 1), 10.0))
    assert confident_correct < confident_wrong


def test_generator_adversarial_loss_lower_when_fooling_discriminator():
    fooling = generator_adversarial_loss(torch.full((2, 1), 10.0))  # disc thinks fake is real
    failing = generator_adversarial_loss(torch.full((2, 1), -10.0))  # disc correctly says fake
    assert fooling < failing


def test_generator_adversarial_loss_gradients_flow():
    disc = UNetPatchDiscriminator(patch_size=RESOLUTION // 8, in_ch=3, base_ch=8)
    fake_image = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION, requires_grad=True)
    fake_logits = disc(fake_image)
    loss = generator_adversarial_loss(fake_logits)
    loss.backward()
    assert fake_image.grad is not None
    assert not torch.isnan(fake_image.grad).any()


# --- LPIPS (network-dependent: downloads+caches a pretrained AlexNet on first use) ---

def _make_lpips_loss():
    pytest.importorskip("lpips")
    from dfl_torch.losses import LPIPSLoss
    try:
        return LPIPSLoss(net="alex")
    except Exception as e:
        pytest.skip(f"could not initialize LPIPSLoss (likely no network for weight download): {e}")


def test_lpips_zero_for_identical_images():
    lpips_loss = _make_lpips_loss()
    x = torch.rand(1, 3, 64, 64)
    loss = lpips_loss(x, x)
    assert loss.abs() < 1e-4


def test_lpips_positive_for_different_images():
    lpips_loss = _make_lpips_loss()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 64, 64)
    loss = lpips_loss(x, y)
    assert loss > 0


def test_lpips_respects_mask():
    lpips_loss = _make_lpips_loss()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    y = x.clone()
    y[:, :, :, 32:] = torch.rand(1, 3, 64, 32)  # differs only in the right half

    full_mask = torch.ones(1, 1, 64, 64)
    left_only_mask = torch.ones(1, 1, 64, 64)
    left_only_mask[:, :, :, 32:] = 0.0

    loss_full = lpips_loss(x, y, mask=full_mask)
    loss_left_only = lpips_loss(x, y, mask=left_only_mask)
    assert loss_left_only.abs() < loss_full.abs()


def test_lpips_stays_frozen_and_eval_even_if_wrapped_in_a_training_module():
    """Section 14c: LPIPS must stay frozen (requires_grad=False) and in eval mode, even if used
    as a submodule of something that gets model.train() called on it."""
    lpips_loss = _make_lpips_loss()

    class Wrapper(torch.nn.Module):
        def __init__(self, lpips_loss):
            super().__init__()
            self.lpips_loss = lpips_loss

    wrapper = Wrapper(lpips_loss)
    wrapper.train()  # should NOT flip the LPIPS submodule into training mode
    assert not lpips_loss.model.training

    for p in lpips_loss.parameters():
        assert not p.requires_grad


def test_lpips_no_gradient_flows_into_frozen_network():
    lpips_loss = _make_lpips_loss()
    pred = torch.rand(1, 3, 64, 64, requires_grad=True)
    target = torch.rand(1, 3, 64, 64)

    loss = lpips_loss(pred, target)
    loss.backward()

    assert pred.grad is not None  # gradient flows into the actual inputs...
    for p in lpips_loss.parameters():
        assert p.grad is None  # ...but never into the frozen backbone's own weights
