"""
Tests for dfl_torch/init.py (requirements.md Section 14a point 3): Conv2d/ConvTranspose2d/Linear
weights should be Xavier/Glorot-uniform (matching leras' actual empirically-verified default, not
PyTorch's own Kaiming-uniform default), biases zero.
"""
import math

import torch.nn as nn

from dfl_torch.df_archi import Decoder, Encoder, Inter
from dfl_torch.discriminator import UNetPatchDiscriminator
from dfl_torch.xseg import XSegNet


def _xavier_uniform_bound(m):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
    return math.sqrt(6.0 / (fan_in + fan_out))


def _assert_xavier_and_zero_bias(module):
    checked = 0
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            checked += 1
            bound = _xavier_uniform_bound(m)
            assert m.weight.min().item() >= -bound - 1e-6
            assert m.weight.max().item() <= bound + 1e-6
            if m.bias is not None:
                assert (m.bias == 0).all()
    assert checked > 0, "no Conv2d/ConvTranspose2d/Linear layers found to check"


def test_encoder_weights_xavier_uniform():
    _assert_xavier_and_zero_bias(Encoder(in_ch=3, e_ch=8))


def test_inter_weights_xavier_uniform():
    _assert_xavier_and_zero_bias(Inter(in_ch=512, ae_ch=32, ae_out_ch=32, lowest_dense_res=4))


def test_decoder_weights_xavier_uniform():
    _assert_xavier_and_zero_bias(Decoder(in_ch=32, d_ch=8, d_mask_ch=4))


def test_discriminator_weights_xavier_uniform():
    _assert_xavier_and_zero_bias(UNetPatchDiscriminator(patch_size=16, in_ch=3, base_ch=8))


def test_xseg_weights_xavier_uniform():
    _assert_xavier_and_zero_bias(XSegNet(in_ch=3, base_ch=8, out_ch=1))
