"""
Full DF-variant SAEHD model: shared Encoder/Inter + two decoders (decoder_src, decoder_dst) —
the actual face-swap architecture, not just the individual encoder/inter/decoder pieces tested in
isolation in tests/test_saehd_shapes.py. Matches models/Model_SAEHD/Model.py's DF-branch
construction (`if 'df' in archi_type: ...`).

Training: encoder(warped_src) -> inter -> decoder_src -> reconstructs src (compared against
target_src); same for dst via decoder_dst. Face swap at inference: encoder(dst) -> inter ->
decoder_src (the *src* decoder applied to a *dst* input) -> dst's pose/expression rendered with
src's decoder, i.e. the swapped face. This cross-decoder swap is only meaningful once both
decoders are actually trained — `swap()` below is provided for completeness/inference-time use,
not exercised by training.
"""
import torch.nn as nn

from dfl_torch.df_archi import Decoder, Encoder, Inter


class SAEHDModel(nn.Module):
    def __init__(self, resolution, in_ch=3, e_dims=64, ae_dims=256, d_dims=64, d_mask_dims=22):
        super().__init__()
        self.resolution = resolution

        self.encoder = Encoder(in_ch=in_ch, e_ch=e_dims)
        encoder_out_res = self.encoder.get_out_res(resolution)
        encoder_out_ch = self.encoder.get_out_ch() * encoder_out_res**2

        self.inter = Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, lowest_dense_res=resolution // 16)

        self.decoder_src = Decoder(in_ch=self.inter.get_out_ch(), d_ch=d_dims, d_mask_ch=d_mask_dims)
        self.decoder_dst = Decoder(in_ch=self.inter.get_out_ch(), d_ch=d_dims, d_mask_ch=d_mask_dims)

    def encode(self, x):
        return self.inter(self.encoder(x))

    def forward_src(self, x):
        """Reconstructs `x` through the src decoder — used during training on src samples."""
        return self.decoder_src(self.encode(x))

    def forward_dst(self, x):
        """Reconstructs `x` through the dst decoder — used during training on dst samples."""
        return self.decoder_dst(self.encode(x))

    def swap(self, dst_image):
        """Face swap: encode a dst frame, decode with the *src* decoder — dst's pose/expression
        rendered as src's identity. Inference-time use; only meaningful after training."""
        return self.decoder_src(self.encode(dst_image))
