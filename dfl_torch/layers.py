"""
Building blocks shared by the DF-variant SAEHD architecture (dfl_torch/df_archi.py) and the
PatchGAN discriminator (dfl_torch/discriminator.py).

Functional spec: core/leras/archis/DeepFakeArchi.py (the default, opts='' branch). This is a
clean-room reimplementation, not a line-by-line port — see IMPLEMENTATION_PLAN.md Phase 1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2)

    def forward(self, x):
        return F.leaky_relu(self.conv1(x), 0.1)


class DownscaleBlock(nn.Module):
    def __init__(self, in_ch, ch, n_downscales, kernel_size=5):
        super().__init__()
        self.downs = nn.ModuleList()
        last_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * min(2**i, 8)
            self.downs.append(Downscale(last_ch, cur_ch, kernel_size=kernel_size))
            last_ch = cur_ch
        self.out_ch = last_ch

    def forward(self, x):
        for down in self.downs:
            x = down(x)
        return x


class Upscale(nn.Module):
    """Conv + LeakyReLU + pixel-shuffle upsample (x2). Equivalent to TF's Conv2D + depth_to_space."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch * 4, kernel_size, stride=1, padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1)
        return self.pixel_shuffle(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        inp = x
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.conv2(x)
        return F.leaky_relu(inp + x, 0.2)
