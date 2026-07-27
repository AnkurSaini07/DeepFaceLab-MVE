"""
Section 11.1/11.2 tests for dfl_torch/xseg.py: shape assertions, sigmoid output range, and
gradient flow through the full U-Net (including the dense bottleneck and skip connections).
"""
import torch

from dfl_torch.xseg import XSegNet

RESOLUTION = 256
IN_CH = 3
BASE_CH = 8  # smaller than the real base_ch=32 for a fast test; topology is unaffected
OUT_CH = 1
BATCH = 1


def test_output_shapes():
    model = XSegNet(in_ch=IN_CH, base_ch=BASE_CH, out_ch=OUT_CH)
    x = torch.rand(BATCH, IN_CH, RESOLUTION, RESOLUTION)
    logits, probs = model(x)
    assert logits.shape == (BATCH, OUT_CH, RESOLUTION, RESOLUTION)
    assert probs.shape == (BATCH, OUT_CH, RESOLUTION, RESOLUTION)


def test_probs_are_valid_sigmoid_range():
    model = XSegNet(in_ch=IN_CH, base_ch=BASE_CH, out_ch=OUT_CH)
    x = torch.rand(BATCH, IN_CH, RESOLUTION, RESOLUTION)
    _, probs = model(x)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_pretrain_mode_zeros_skip_connections_but_still_runs():
    model = XSegNet(in_ch=IN_CH, base_ch=BASE_CH, out_ch=OUT_CH)
    x = torch.rand(BATCH, IN_CH, RESOLUTION, RESOLUTION)
    logits_normal, _ = model(x, pretrain=False)
    logits_pretrain, _ = model(x, pretrain=True)
    assert logits_normal.shape == logits_pretrain.shape
    assert not torch.equal(logits_normal, logits_pretrain)


def test_no_nan_in_forward_pass():
    model = XSegNet(in_ch=IN_CH, base_ch=BASE_CH, out_ch=OUT_CH)
    x = torch.rand(BATCH, IN_CH, RESOLUTION, RESOLUTION)
    logits, probs = model(x)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(probs).any()


def test_gradients_flow_through_full_unet():
    model = XSegNet(in_ch=IN_CH, base_ch=BASE_CH, out_ch=OUT_CH)
    x = torch.rand(BATCH, IN_CH, RESOLUTION, RESOLUTION, requires_grad=True)
    _, probs = model(x)
    probs.mean().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"
