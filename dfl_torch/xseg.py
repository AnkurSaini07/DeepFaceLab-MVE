"""
XSeg face-mask segmentation network — clean-room PyTorch reimplementation.

Functional spec: core/leras/models/XSeg.py (a 6-level U-Net with a dense bottleneck, FRN+TLU
activations instead of BatchNorm+ReLU, and BlurPool anti-aliased downsampling), as constructed by
facelib/XSegNet.py: `nn.XSeg(in_ch=3, base_ch=32, out_ch=1)` at resolution=256 (256 = 4 * 2^6,
matching the 6 stride-2 downsamples down to a 4x4 dense bottleneck).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dfl_torch.init import apply_xavier_init


class FRNorm2D(nn.Module):
    """Filter Response Normalization (https://arxiv.org/abs/1911.09737) — per-channel RMS norm,
    no batch statistics."""

    def __init__(self, num_ch):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ch))
        self.bias = nn.Parameter(torch.zeros(num_ch))
        self.eps = nn.Parameter(torch.tensor([1e-6]))

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class TLU(nn.Module):
    """Thresholded Linear Unit — paired with FRN in the same paper, replaces ReLU+bias."""

    def __init__(self, num_ch):
        super().__init__()
        self.tau = nn.Parameter(torch.zeros(num_ch))

    def forward(self, x):
        return torch.maximum(x, self.tau.view(1, -1, 1, 1))


class BlurPool(nn.Module):
    """Anti-aliased stride-2 downsampling (fixed binomial blur kernel, no learnable params) —
    https://arxiv.org/abs/1904.11486."""

    _FILTERS = {
        1: [1.0],
        2: [1.0, 1.0],
        3: [1.0, 2.0, 1.0],
        4: [1.0, 3.0, 3.0, 1.0],
        5: [1.0, 4.0, 6.0, 4.0, 1.0],
        6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
        7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
    }

    def __init__(self, channels, filt_size=3, stride=2):
        super().__init__()
        self.stride = stride
        a = torch.tensor(self._FILTERS[filt_size])
        kernel_2d = a[:, None] * a[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer("kernel", kernel_2d[None, None].repeat(channels, 1, 1, 1))
        pad_lo = (filt_size - 1) // 2
        pad_hi = int(math.ceil((filt_size - 1) / 2))
        self.padding = (pad_lo, pad_hi, pad_lo, pad_hi)  # (left, right, top, bottom)

    def forward(self, x):
        x = F.pad(x, self.padding, mode="constant", value=0)
        return F.conv2d(x, self.kernel, stride=self.stride, groups=self.kernel.shape[0])


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.frn = FRNorm2D(out_ch)
        self.tlu = TLU(out_ch)

    def forward(self, x):
        return self.tlu(self.frn(self.conv(x)))


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.frn = FRNorm2D(out_ch)
        self.tlu = TLU(out_ch)

    def forward(self, x):
        return self.tlu(self.frn(self.conv(x)))


class XSegNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, out_ch=1):
        super().__init__()
        self.base_ch = base_ch

        self.conv01 = ConvBlock(in_ch, base_ch)
        self.conv02 = ConvBlock(base_ch, base_ch)
        self.bp0 = BlurPool(base_ch, filt_size=4)

        self.conv11 = ConvBlock(base_ch, base_ch * 2)
        self.conv12 = ConvBlock(base_ch * 2, base_ch * 2)
        self.bp1 = BlurPool(base_ch * 2, filt_size=3)

        self.conv21 = ConvBlock(base_ch * 2, base_ch * 4)
        self.conv22 = ConvBlock(base_ch * 4, base_ch * 4)
        self.bp2 = BlurPool(base_ch * 4, filt_size=2)

        self.conv31 = ConvBlock(base_ch * 4, base_ch * 8)
        self.conv32 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv33 = ConvBlock(base_ch * 8, base_ch * 8)
        self.bp3 = BlurPool(base_ch * 8, filt_size=2)

        self.conv41 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv42 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv43 = ConvBlock(base_ch * 8, base_ch * 8)
        self.bp4 = BlurPool(base_ch * 8, filt_size=2)

        self.conv51 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv52 = ConvBlock(base_ch * 8, base_ch * 8)
        self.conv53 = ConvBlock(base_ch * 8, base_ch * 8)
        self.bp5 = BlurPool(base_ch * 8, filt_size=2)

        self.dense1 = nn.Linear(4 * 4 * base_ch * 8, 512)
        self.dense2 = nn.Linear(512, 4 * 4 * base_ch * 8)

        self.up5 = UpConvBlock(base_ch * 8, base_ch * 4)
        self.uconv53 = ConvBlock(base_ch * 12, base_ch * 8)
        self.uconv52 = ConvBlock(base_ch * 8, base_ch * 8)
        self.uconv51 = ConvBlock(base_ch * 8, base_ch * 8)

        self.up4 = UpConvBlock(base_ch * 8, base_ch * 4)
        self.uconv43 = ConvBlock(base_ch * 12, base_ch * 8)
        self.uconv42 = ConvBlock(base_ch * 8, base_ch * 8)
        self.uconv41 = ConvBlock(base_ch * 8, base_ch * 8)

        self.up3 = UpConvBlock(base_ch * 8, base_ch * 4)
        self.uconv33 = ConvBlock(base_ch * 12, base_ch * 8)
        self.uconv32 = ConvBlock(base_ch * 8, base_ch * 8)
        self.uconv31 = ConvBlock(base_ch * 8, base_ch * 8)

        self.up2 = UpConvBlock(base_ch * 8, base_ch * 4)
        self.uconv22 = ConvBlock(base_ch * 8, base_ch * 4)
        self.uconv21 = ConvBlock(base_ch * 4, base_ch * 4)

        self.up1 = UpConvBlock(base_ch * 4, base_ch * 2)
        self.uconv12 = ConvBlock(base_ch * 4, base_ch * 2)
        self.uconv11 = ConvBlock(base_ch * 2, base_ch * 2)

        self.up0 = UpConvBlock(base_ch * 2, base_ch)
        self.uconv02 = ConvBlock(base_ch * 2, base_ch)
        self.uconv01 = ConvBlock(base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1)
        apply_xavier_init(self)

    def forward(self, x, pretrain=False):
        x = self.conv01(x)
        x = x0 = self.conv02(x)
        x = self.bp0(x)

        x = self.conv11(x)
        x = x1 = self.conv12(x)
        x = self.bp1(x)

        x = self.conv21(x)
        x = x2 = self.conv22(x)
        x = self.bp2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = x3 = self.conv33(x)
        x = self.bp3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = x4 = self.conv43(x)
        x = self.bp4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = x5 = self.conv53(x)
        x = self.bp5(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.view(-1, self.base_ch * 8, 4, 4)

        x = self.up5(x)
        if pretrain:
            x5 = torch.zeros_like(x5)
        x = self.uconv53(torch.cat([x, x5], dim=1))
        x = self.uconv52(x)
        x = self.uconv51(x)

        x = self.up4(x)
        if pretrain:
            x4 = torch.zeros_like(x4)
        x = self.uconv43(torch.cat([x, x4], dim=1))
        x = self.uconv42(x)
        x = self.uconv41(x)

        x = self.up3(x)
        if pretrain:
            x3 = torch.zeros_like(x3)
        x = self.uconv33(torch.cat([x, x3], dim=1))
        x = self.uconv32(x)
        x = self.uconv31(x)

        x = self.up2(x)
        if pretrain:
            x2 = torch.zeros_like(x2)
        x = self.uconv22(torch.cat([x, x2], dim=1))
        x = self.uconv21(x)

        x = self.up1(x)
        if pretrain:
            x1 = torch.zeros_like(x1)
        x = self.uconv12(torch.cat([x, x1], dim=1))
        x = self.uconv11(x)

        x = self.up0(x)
        if pretrain:
            x0 = torch.zeros_like(x0)
        x = self.uconv02(torch.cat([x, x0], dim=1))
        x = self.uconv01(x)

        logits = self.out_conv(x)
        return logits, torch.sigmoid(logits)
