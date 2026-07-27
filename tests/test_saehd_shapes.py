"""
Section 11.1 shape tests: forward-pass shape assertions for the DF-variant SAEHD modules and the
U-Net PatchGAN discriminator, against dummy tensors. CPU-only, no TF/leras dependency — pure
PyTorch, run with .venv-torch.
"""
import torch

from dfl_torch.df_archi import Decoder, Encoder, Inter
from dfl_torch.discriminator import UNetPatchDiscriminator

RESOLUTION = 128
E_DIMS = 64
AE_DIMS = 256
D_DIMS = 64
D_MASK_DIMS = 22
INPUT_CH = 3
BATCH = 2


def build_generator():
    encoder = Encoder(in_ch=INPUT_CH, e_ch=E_DIMS)
    encoder_out_res = encoder.get_out_res(RESOLUTION)
    encoder_out_ch = encoder.get_out_ch() * encoder_out_res**2

    inter = Inter(
        in_ch=encoder_out_ch, ae_ch=AE_DIMS, ae_out_ch=AE_DIMS,
        lowest_dense_res=RESOLUTION // 16,
    )
    decoder = Decoder(in_ch=inter.get_out_ch(), d_ch=D_DIMS, d_mask_ch=D_MASK_DIMS)
    return encoder, inter, decoder


def test_encoder_output_shape():
    encoder, _, _ = build_generator()
    x = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION)
    out = encoder(x)
    expected_ch = encoder.get_out_ch() * encoder.get_out_res(RESOLUTION) ** 2
    assert out.shape == (BATCH, expected_ch)


def test_inter_output_shape():
    encoder, inter, _ = build_generator()
    x = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION)
    encoder_out = encoder(x)
    inter_out = inter(encoder_out)
    expected_res = inter.get_out_res()
    assert inter_out.shape == (BATCH, AE_DIMS, expected_res, expected_res)


def test_decoder_output_shapes():
    encoder, inter, decoder = build_generator()
    x = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION)
    inter_out = inter(encoder(x))
    rgb, mask = decoder(inter_out)
    assert rgb.shape == (BATCH, 3, RESOLUTION, RESOLUTION)
    assert mask.shape == (BATCH, 1, RESOLUTION, RESOLUTION)


def test_decoder_outputs_are_valid_sigmoid_range():
    encoder, inter, decoder = build_generator()
    x = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION)
    inter_out = inter(encoder(x))
    rgb, mask = decoder(inter_out)
    assert torch.all((rgb >= 0) & (rgb <= 1))
    assert torch.all((mask >= 0) & (mask <= 1))


def test_full_generator_pipeline_end_to_end():
    encoder, inter, decoder = build_generator()
    x = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION)
    rgb, mask = decoder(inter(encoder(x)))
    assert not torch.isnan(rgb).any()
    assert not torch.isnan(mask).any()


def test_discriminator_output_shapes():
    disc = UNetPatchDiscriminator(patch_size=RESOLUTION // 8, in_ch=INPUT_CH, base_ch=16)
    x = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION)
    center_out, out = disc(x)
    assert center_out.shape[0] == BATCH
    assert center_out.shape[1] == 1
    assert out.shape == (BATCH, 1, RESOLUTION, RESOLUTION)
    assert not torch.isnan(out).any()


def test_gradients_flow_through_full_pipeline():
    encoder, inter, decoder = build_generator()
    disc = UNetPatchDiscriminator(patch_size=RESOLUTION // 8, in_ch=INPUT_CH, base_ch=16)

    x = torch.rand(BATCH, INPUT_CH, RESOLUTION, RESOLUTION, requires_grad=True)
    rgb, mask = decoder(inter(encoder(x)))
    center_out, disc_out = disc(rgb)
    loss = disc_out.mean() + center_out.mean() + rgb.mean() + mask.mean()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    for module in (encoder, inter, decoder, disc):
        for name, param in module.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad for {name}"
