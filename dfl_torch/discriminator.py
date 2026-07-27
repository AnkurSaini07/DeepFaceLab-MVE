"""
U-Net PatchGAN discriminator — clean-room PyTorch reimplementation.

Functional spec: core/leras/models/PatchDiscriminator.py's UNetPatchDiscriminator
(https://arxiv.org/abs/2002.12655). `find_archi`/`calc_receptive_field_size` are pure
layer-topology math (no TF dependency in the original either), so they're carried over verbatim;
only the layer construction/forward pass is reimplemented in PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_receptive_field_size(layers):
    rf = 0
    ts = 1
    for i, (k, s) in enumerate(layers):
        if i == 0:
            rf = k
        else:
            rf += (k - 1) * ts
        ts *= s
    return rf


def find_archi(target_patch_size, max_layers=9):
    """Finds the best configuration of 3x3-conv-only layers for a target patch size."""
    s = {}
    for layers_count in range(1, max_layers + 1):
        val = 1 << (layers_count - 1)
        while True:
            val -= 1

            layers = [[3, 2]]
            sum_st = 2
            for i in range(layers_count - 1):
                st = 1 + (1 if val & (1 << i) != 0 else 0)
                layers.append([3, st])
                sum_st += st

            rf = calc_receptive_field_size(layers)

            s_rf = s.get(rf, None)
            if s_rf is None:
                s[rf] = (layers_count, sum_st, layers)
            else:
                if layers_count < s_rf[0] or (layers_count == s_rf[0] and sum_st > s_rf[1]):
                    s[rf] = (layers_count, sum_st, layers)

            if val == 0:
                break

    keys = sorted(s.keys())
    q = keys[int(np.abs(np.array(keys) - target_patch_size).argmin())]
    return s[q][2]


class UNetPatchDiscriminator(nn.Module):
    def __init__(self, patch_size, in_ch, base_ch=16):
        super().__init__()
        layers = find_archi(patch_size)
        level_chs = {i - 1: min(base_ch * (2**i), 512) for i in range(len(layers) + 1)}

        self.in_conv = nn.Conv2d(in_ch, level_chs[-1], kernel_size=1)

        self.convs = nn.ModuleList()
        upconvs = []
        for i, (kernel_size, stride) in enumerate(layers):
            self.convs.append(
                nn.Conv2d(level_chs[i - 1], level_chs[i], kernel_size, stride=stride, padding=kernel_size // 2)
            )
            up_in_ch = level_chs[i] * (2 if i != len(layers) - 1 else 1)
            upconvs.insert(
                0,
                nn.ConvTranspose2d(
                    up_in_ch, level_chs[i - 1], kernel_size, stride=stride,
                    padding=kernel_size // 2, output_padding=stride - 1,
                ),
            )
        self.upconvs = nn.ModuleList(upconvs)

        self.out_conv = nn.Conv2d(level_chs[-1] * 2, 1, kernel_size=1)
        self.center_out = nn.Conv2d(level_chs[len(layers) - 1], 1, kernel_size=1)
        self.center_conv = nn.Conv2d(level_chs[len(layers) - 1], level_chs[len(layers) - 1], kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(self.in_conv(x), 0.2)

        encs = []
        for conv in self.convs:
            encs.insert(0, x)
            x = F.leaky_relu(conv(x), 0.2)

        center_out = self.center_out(x)
        x = F.leaky_relu(self.center_conv(x), 0.2)

        for upconv, enc in zip(self.upconvs, encs):
            x = F.leaky_relu(upconv(x), 0.2)
            x = torch.cat([enc, x], dim=1)

        x = self.out_conv(x)
        return center_out, x
