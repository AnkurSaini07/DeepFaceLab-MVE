"""
BF16 autocast helper — requirements.md Section 4: BF16 autocast (not FP16), no GradScaler needed,
model weights stay FP32 (master copy), optimizer step in FP32.

`device_type='cuda'` is the actual target (Ada Lovelace has native BF16 Tensor Core support).
`autocast_context` also accepts 'cpu' — PyTorch's CPU autocast supports bfloat16 too (via oneDNN),
which is what makes the Section 11.2 smoke test in tests/test_training_smoke.py able to actually
exercise this code path on CPU, not just shape/structure-check around it.
"""
import torch


def autocast_context(device_type):
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
