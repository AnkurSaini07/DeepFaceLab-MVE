"""
Tests for dfl_torch/model.py's SAEHDModel — the full DF-variant assembly (shared Encoder/Inter +
decoder_src/decoder_dst), as opposed to tests/test_saehd_shapes.py which exercises a single
decoder in isolation.
"""
import torch

from dfl_torch.model import SAEHDModel

RESOLUTION = 32
BATCH = 2


def _build_model():
    return SAEHDModel(RESOLUTION, e_dims=8, ae_dims=16, d_dims=8, d_mask_dims=4)


def test_forward_src_and_dst_output_shapes():
    model = _build_model()
    x = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)

    rgb_src, mask_src = model.forward_src(x)
    rgb_dst, mask_dst = model.forward_dst(x)

    assert rgb_src.shape == (BATCH, 3, RESOLUTION, RESOLUTION)
    assert mask_src.shape == (BATCH, 1, RESOLUTION, RESOLUTION)
    assert rgb_dst.shape == (BATCH, 3, RESOLUTION, RESOLUTION)
    assert mask_dst.shape == (BATCH, 1, RESOLUTION, RESOLUTION)


def test_swap_output_shape():
    model = _build_model()
    dst_image = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    swapped_rgb, swapped_mask = model.swap(dst_image)
    assert swapped_rgb.shape == (BATCH, 3, RESOLUTION, RESOLUTION)
    assert swapped_mask.shape == (BATCH, 1, RESOLUTION, RESOLUTION)


def test_src_and_dst_decoders_are_independent_modules():
    """forward_src and forward_dst must use different decoder weights (the whole point of the
    DF-variant dual-decoder design) — sanity-checked by confirming decoder_src and decoder_dst
    don't share parameters."""
    model = _build_model()
    src_params = {id(p) for p in model.decoder_src.parameters()}
    dst_params = {id(p) for p in model.decoder_dst.parameters()}
    assert src_params.isdisjoint(dst_params)


def test_encoder_and_inter_are_shared_between_src_and_dst_paths():
    model = _build_model()
    x = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    # encode() is used by both forward_src/forward_dst/swap -- same encoder+inter weights.
    latent_from_encode = model.encode(x)
    latent_manual = model.inter(model.encoder(x))
    assert torch.equal(latent_from_encode, latent_manual)


def test_gradients_flow_through_both_decoder_paths_independently():
    model = _build_model()
    x = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION, requires_grad=True)

    # Decoder has two disconnected output branches (rgb and mask) -- both must be in the loss
    # for every decoder_src param to receive a gradient (the rgb-only and mask-only conv stacks
    # are otherwise independent paths from the shared latent).
    rgb_src, mask_src = model.forward_src(x)
    (rgb_src.mean() + mask_src.mean()).backward()

    assert x.grad is not None
    for name, p in model.decoder_src.named_parameters():
        assert p.grad is not None, f"no grad for decoder_src.{name}"
    # decoder_dst wasn't used in this forward pass -- it should have no gradient at all.
    for name, p in model.decoder_dst.named_parameters():
        assert p.grad is None, f"unexpected grad for decoder_dst.{name}"


def test_no_nan_outputs():
    model = _build_model()
    x = torch.rand(BATCH, 3, RESOLUTION, RESOLUTION)
    for rgb, mask in [model.forward_src(x), model.forward_dst(x), model.swap(x)]:
        assert not torch.isnan(rgb).any()
        assert not torch.isnan(mask).any()
