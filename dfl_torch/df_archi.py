"""
DF-variant SAEHD encoder/inter/decoder — clean-room PyTorch reimplementation.

Functional spec: core/leras/archis/DeepFakeArchi.py, DF branch, default opts (no 't'/'d'/'c'/'u'
modifiers — the configuration models/Model_SAEHD/Model.py builds for a plain '-df' architecture).
Tensors are NCHW (standard PyTorch layout), unlike the TF version's NHWC-on-CPU default; this is
an intentional divergence, not a bug — see IMPLEMENTATION_PLAN.md Phase 1 for why this is a
clean-room design rather than a literal port.
"""
import torch
import torch.nn as nn

from dfl_torch.init import apply_xavier_init
from dfl_torch.layers import DownscaleBlock, ResidualBlock, Upscale


class Encoder(nn.Module):
    def __init__(self, in_ch, e_ch, n_downscales=4, kernel_size=5):
        super().__init__()
        self.n_downscales = n_downscales
        self.e_ch = e_ch
        self.down1 = DownscaleBlock(in_ch, e_ch, n_downscales=n_downscales, kernel_size=kernel_size)
        apply_xavier_init(self)

    def forward(self, x):
        x = self.down1(x)
        return torch.flatten(x, start_dim=1)

    def get_out_res(self, res):
        return res // (2**self.n_downscales)

    def get_out_ch(self):
        return self.e_ch * 8


class Inter(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, lowest_dense_res):
        super().__init__()
        self.ae_out_ch = ae_out_ch
        self.lowest_dense_res = lowest_dense_res
        self.dense1 = nn.Linear(in_ch, ae_ch)
        self.dense2 = nn.Linear(ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch)
        self.upscale1 = Upscale(ae_out_ch, ae_out_ch)
        apply_xavier_init(self)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.view(-1, self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res)
        return self.upscale1(x)

    def get_out_res(self):
        return self.lowest_dense_res * 2

    def get_out_ch(self):
        return self.ae_out_ch


class Decoder(nn.Module):
    def __init__(self, in_ch, d_ch, d_mask_ch):
        super().__init__()
        self.upscale0 = Upscale(in_ch, d_ch * 8)
        self.res0 = ResidualBlock(d_ch * 8)
        self.upscale1 = Upscale(d_ch * 8, d_ch * 4)
        self.res1 = ResidualBlock(d_ch * 4)
        self.upscale2 = Upscale(d_ch * 4, d_ch * 2)
        self.res2 = ResidualBlock(d_ch * 2)
        self.out_conv = nn.Conv2d(d_ch * 2, 3, kernel_size=1)

        self.upscalem0 = Upscale(in_ch, d_mask_ch * 8)
        self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * 4)
        self.upscalem2 = Upscale(d_mask_ch * 4, d_mask_ch * 2)
        self.out_convm = nn.Conv2d(d_mask_ch * 2, 1, kernel_size=1)
        apply_xavier_init(self)

    def forward(self, z):
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)
        rgb = torch.sigmoid(self.out_conv(x))

        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        mask = torch.sigmoid(self.out_convm(m))

        return rgb, mask
