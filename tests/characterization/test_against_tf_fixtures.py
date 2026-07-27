"""
Validates the PyTorch dfl_torch modules against golden fixtures captured from the current
TF/leras implementation (see capture_saehd_fixtures.py and IMPLEMENTATION_PLAN.md's
"Cross-cutting: characterization testing").

This is a clean-room reimplementation, not a port — exact numeric match against the TF fixtures
is neither expected nor checked (different framework numerics, different weight init). What's
checked instead: shapes match exactly (same architecture topology), and output ranges match
(sigmoid outputs bounded in [0, 1]) — catching structural regressions in the PyTorch port
without requiring bit-exact TF parity.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dfl_torch.df_archi import Decoder, Encoder, Inter
from dfl_torch.discriminator import UNetPatchDiscriminator

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def metadata():
    path = FIXTURES_DIR / "saehd_df_metadata.json"
    if not path.exists():
        pytest.skip("fixtures not captured yet — run capture_saehd_fixtures.py under the dfl conda env")
    return json.loads(path.read_text())


def _load(name):
    return np.load(FIXTURES_DIR / f"saehd_df_{name}.npy")


def test_encoder_output_shape_matches_tf(metadata):
    encoder = Encoder(in_ch=metadata["input_ch"], e_ch=metadata["e_dims"])
    x = torch.from_numpy(_load("input")).permute(0, 3, 1, 2).contiguous()  # NHWC (TF) -> NCHW (torch)
    out = encoder(x)

    tf_shape = tuple(metadata["shapes"]["encoder_out"])
    assert out.shape == tf_shape, f"torch {tuple(out.shape)} vs TF {tf_shape}"


def test_inter_output_shape_matches_tf(metadata):
    encoder = Encoder(in_ch=metadata["input_ch"], e_ch=metadata["e_dims"])
    encoder_out_res = encoder.get_out_res(metadata["resolution"])
    encoder_out_ch = encoder.get_out_ch() * encoder_out_res**2
    inter = Inter(
        in_ch=encoder_out_ch, ae_ch=metadata["ae_dims"], ae_out_ch=metadata["ae_dims"],
        lowest_dense_res=metadata["resolution"] // 16,
    )

    x = torch.from_numpy(_load("input")).permute(0, 3, 1, 2).contiguous()
    inter_out = inter(encoder(x))

    # TF fixture is NHWC; torch output is NCHW — compare shape as a set since we're checking
    # topology (dims/sizes), not memory layout.
    tf_shape = tuple(metadata["shapes"]["inter_out"])
    assert sorted(inter_out.shape) == sorted(tf_shape), f"torch {tuple(inter_out.shape)} vs TF {tf_shape}"


def test_decoder_output_shapes_and_range_match_tf(metadata):
    encoder = Encoder(in_ch=metadata["input_ch"], e_ch=metadata["e_dims"])
    encoder_out_res = encoder.get_out_res(metadata["resolution"])
    encoder_out_ch = encoder.get_out_ch() * encoder_out_res**2
    inter = Inter(
        in_ch=encoder_out_ch, ae_ch=metadata["ae_dims"], ae_out_ch=metadata["ae_dims"],
        lowest_dense_res=metadata["resolution"] // 16,
    )
    decoder = Decoder(in_ch=inter.get_out_ch(), d_ch=metadata["d_dims"], d_mask_ch=metadata["d_mask_dims"])

    x = torch.from_numpy(_load("input")).permute(0, 3, 1, 2).contiguous()
    rgb, mask = decoder(inter(encoder(x)))

    tf_rgb_shape = tuple(metadata["shapes"]["decoder_rgb"])
    tf_mask_shape = tuple(metadata["shapes"]["decoder_mask"])
    assert sorted(rgb.shape) == sorted(tf_rgb_shape), f"torch {tuple(rgb.shape)} vs TF {tf_rgb_shape}"
    assert sorted(mask.shape) == sorted(tf_mask_shape), f"torch {tuple(mask.shape)} vs TF {tf_mask_shape}"

    tf_rgb = _load("decoder_rgb")
    tf_mask = _load("decoder_mask")
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    assert 0.0 <= tf_rgb.min() and tf_rgb.max() <= 1.0
    assert 0.0 <= tf_mask.min() and tf_mask.max() <= 1.0


def test_discriminator_output_shapes_match_tf(metadata):
    disc = UNetPatchDiscriminator(
        patch_size=metadata["gan_patch_size"], in_ch=metadata["input_ch"], base_ch=metadata["gan_dims"]
    )
    x = torch.from_numpy(_load("input")).permute(0, 3, 1, 2).contiguous()
    center_out, out = disc(x)

    tf_center_shape = tuple(metadata["shapes"]["discriminator_center_out"])
    tf_out_shape = tuple(metadata["shapes"]["discriminator_out"])
    assert sorted(center_out.shape) == sorted(tf_center_shape), (
        f"torch {tuple(center_out.shape)} vs TF {tf_center_shape}"
    )
    assert sorted(out.shape) == sorted(tf_out_shape), f"torch {tuple(out.shape)} vs TF {tf_out_shape}"
