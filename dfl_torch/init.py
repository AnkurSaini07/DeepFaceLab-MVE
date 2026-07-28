"""
Weight initialization matching leras' actual defaults — requirements.md Section 14a point 3:
"leras uses specific default initializers (typically He/Glorot uniform). Native PyTorch nn.Module
defaults differ — set initializers explicitly and deliberately... to avoid early-training
divergence from expected behavior."

Empirically verified (via the `dfl` conda env, not just read from source) that
core/leras/layers/{Conv2D,Dense}.py's default `kernel_initializer`, when left unset — which is
how DeepFakeArchi.py/XSeg.py construct every layer — falls through to `tf.get_variable`'s
implicit default of **glorot_uniform** (Xavier uniform): a constructed Conv2D/Dense's weight
min/max matched `sqrt(6/(fan_in+fan_out))` exactly. This is neither the `CA` initializer
Conv2D.py's docstring mentions (that code path is commented out) nor PyTorch's own
`nn.Conv2d`/`nn.Linear` default (Kaiming uniform). Bias initializer is explicitly
`tf.initializers.zeros()`.
"""
import torch.nn as nn


def apply_xavier_init(module):
    """Re-initializes every Conv2d/ConvTranspose2d/Linear in `module` (recursively, including
    `module` itself) with Xavier/Glorot-uniform weights and zero biases, matching leras' actual
    default behavior. Call at the end of a network's __init__."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
