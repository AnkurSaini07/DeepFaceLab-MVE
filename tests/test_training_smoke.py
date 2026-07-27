"""
Section 11.2 smoke test + Phase 3 (BF16 autocast) validation: single forward + backward +
optimizer step on dummy data, wrapped in dfl_torch.precision.autocast_context, asserting loss is
not NaN and gradients flow into FP32 master weights.

Loss functions are the DFL SSIM+L1 baseline's L1 term only (Section 9's fuller loss stack —
LPIPS/adversarial/identity — is Phase 7); the point here is validating the training-step wiring
and precision handling, not the loss design.

Runs on CPU (device_type='cpu') since that's this repo's development constraint (requirements.md
"initial implementation and testing must be doable on CPU only"). PyTorch's CPU autocast supports
bfloat16 (via oneDNN), so this actually exercises the autocast code path, not just a structural
stand-in for it — see IMPLEMENTATION_PLAN.md Phase 3.
"""
import torch
import torch.nn.functional as F

from dfl_torch.df_archi import Decoder, Encoder, Inter
from dfl_torch.discriminator import UNetPatchDiscriminator
from dfl_torch.precision import autocast_context

RESOLUTION = 64
E_DIMS = 32
AE_DIMS = 64
D_DIMS = 32
D_MASK_DIMS = 12
INPUT_CH = 3
BATCH = 2
DEVICE_TYPE = "cpu"


def build_models():
    encoder = Encoder(in_ch=INPUT_CH, e_ch=E_DIMS)
    encoder_out_res = encoder.get_out_res(RESOLUTION)
    encoder_out_ch = encoder.get_out_ch() * encoder_out_res**2
    inter = Inter(in_ch=encoder_out_ch, ae_ch=AE_DIMS, ae_out_ch=AE_DIMS, lowest_dense_res=RESOLUTION // 16)
    decoder = Decoder(in_ch=inter.get_out_ch(), d_ch=D_DIMS, d_mask_ch=D_MASK_DIMS)
    disc = UNetPatchDiscriminator(patch_size=RESOLUTION // 8, in_ch=INPUT_CH, base_ch=8)
    return encoder, inter, decoder, disc


def all_params(*modules):
    params = []
    for m in modules:
        params += list(m.parameters())
    return params


def test_master_weights_are_fp32_before_training():
    encoder, inter, decoder, disc = build_models()
    for p in all_params(encoder, inter, decoder, disc):
        assert p.dtype == torch.float32


def test_bf16_autocast_training_step_no_nan_and_grads_flow():
    encoder, inter, decoder, disc = build_models()
    params = all_params(encoder, inter, decoder, disc)
    optimizer = torch.optim.Adam(params, lr=1e-4)

    x = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION)
    target_rgb = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION)
    target_mask = torch.rand(BATCH, 1, RESOLUTION, RESOLUTION)

    optimizer.zero_grad()
    with autocast_context(DEVICE_TYPE):
        rgb, mask = decoder(inter(encoder(x)))
        disc_center_out, disc_out = disc(rgb)
        loss = (
            F.l1_loss(rgb, target_rgb) + F.l1_loss(mask, target_mask)
            + disc_out.float().mean() + disc_center_out.float().mean()
        )

    # No GradScaler needed for bf16 (unlike fp16) — loss is finite in fp32 and backward() runs
    # directly (Section 4).
    assert loss.dtype == torch.float32
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    loss.backward()
    optimizer.step()

    for name, p in list(encoder.named_parameters()) + list(inter.named_parameters()) \
            + list(decoder.named_parameters()) + list(disc.named_parameters()):
        assert p.grad is not None, f"no grad for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"
        assert p.dtype == torch.float32, f"{name} was not kept in FP32 (master weights should stay FP32)"


def test_ops_run_in_bf16_inside_autocast():
    encoder, inter, decoder, _ = build_models()
    x = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION)
    with autocast_context(DEVICE_TYPE):
        rgb, mask = decoder(inter(encoder(x)))
    assert rgb.dtype == torch.bfloat16
    assert mask.dtype == torch.bfloat16
