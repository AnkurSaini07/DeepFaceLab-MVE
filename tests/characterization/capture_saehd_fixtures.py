"""
Captures golden output fixtures from the current TF1/leras SAEHD (DF variant) implementation.

This is a one-time capture step, run BEFORE the TF/leras code is removed (see
IMPLEMENTATION_PLAN.md, Phase 1 / "Cross-cutting: characterization testing"). The PyTorch
reimplementation is validated against these fixtures rather than against the TF source directly,
since the TF/leras stack (Python 3.7, TF1-style graph mode) is not expected to keep running
alongside the new code.

Requires the `dfl` conda env (TensorFlow + this repo's dependencies):
    /Users/ankurs/miniconda3/envs/dfl/bin/python tests/characterization/capture_saehd_fixtures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

SEED = 42
RESOLUTION = 128
E_DIMS = 64
AE_DIMS = 256
D_DIMS = 64
D_MASK_DIMS = 22  # matches Model_SAEHD default: (d_dims // 3), rounded up to even
GAN_PATCH_SIZE = RESOLUTION // 8
GAN_DIMS = 16
INPUT_CH = 3


def main():
    from core.leras import nn

    nn.initialize_main_env()
    nn.initialize(data_format="NHWC", device_config=nn.DeviceConfig.CPU())
    tf = nn.tf

    rng = np.random.RandomState(SEED)
    dummy_input = rng.uniform(0.0, 1.0, size=(1, RESOLUTION, RESOLUTION, INPUT_CH)).astype(np.float32)

    model_archi = nn.DeepFakeArchi(RESOLUTION, opts=None)

    encoder = model_archi.Encoder(in_ch=INPUT_CH, e_ch=E_DIMS, name="encoder")
    encoder_out_res = encoder.get_out_res(RESOLUTION)
    encoder_out_ch = encoder.get_out_ch() * encoder_out_res**2

    inter = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=AE_DIMS, ae_out_ch=AE_DIMS, name="inter")
    inter_out_ch = inter.get_out_ch()

    decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=D_DIMS, d_mask_ch=D_MASK_DIMS, name="decoder_src")

    discriminator = nn.UNetPatchDiscriminator(
        patch_size=GAN_PATCH_SIZE, in_ch=INPUT_CH, base_ch=GAN_DIMS, name="D_src"
    )

    input_t = tf.placeholder(nn.floatx, (1, RESOLUTION, RESOLUTION, INPUT_CH), name="input")
    encoder_out_t = encoder(input_t)
    inter_out_t = inter(encoder_out_t)
    decoder_rgb_t, decoder_mask_t = decoder_src(inter_out_t)
    disc_center_out_t, disc_out_t = discriminator(input_t)

    with nn.tf_sess.as_default() as sess:
        sess.run(tf.global_variables_initializer())
        encoder_out, inter_out, decoder_rgb, decoder_mask, disc_center_out, disc_out = sess.run(
            [encoder_out_t, inter_out_t, decoder_rgb_t, decoder_mask_t, disc_center_out_t, disc_out_t],
            feed_dict={input_t: dummy_input},
        )

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURES_DIR / "saehd_df_input.npy", dummy_input)
    np.save(FIXTURES_DIR / "saehd_df_encoder_out.npy", encoder_out)
    np.save(FIXTURES_DIR / "saehd_df_inter_out.npy", inter_out)
    np.save(FIXTURES_DIR / "saehd_df_decoder_rgb.npy", decoder_rgb)
    np.save(FIXTURES_DIR / "saehd_df_decoder_mask.npy", decoder_mask)
    np.save(FIXTURES_DIR / "saehd_df_discriminator_center_out.npy", disc_center_out)
    np.save(FIXTURES_DIR / "saehd_df_discriminator_out.npy", disc_out)

    metadata = {
        "source": "core/leras/archis/DeepFakeArchi.py DF variant, via models/Model_SAEHD/Model.py construction pattern",
        "tensorflow_version": tf.__version__,
        "seed": SEED,
        "resolution": RESOLUTION,
        "e_dims": E_DIMS,
        "ae_dims": AE_DIMS,
        "d_dims": D_DIMS,
        "d_mask_dims": D_MASK_DIMS,
        "gan_patch_size": GAN_PATCH_SIZE,
        "gan_dims": GAN_DIMS,
        "input_ch": INPUT_CH,
        "shapes": {
            "input": list(dummy_input.shape),
            "encoder_out": list(encoder_out.shape),
            "inter_out": list(inter_out.shape),
            "decoder_rgb": list(decoder_rgb.shape),
            "decoder_mask": list(decoder_mask.shape),
            "discriminator_center_out": list(disc_center_out.shape),
            "discriminator_out": list(disc_out.shape),
        },
        "note": (
            "Discriminator weights are randomly initialized (no training occurred), so "
            "discriminator_out is only useful for shape/range characterization, not a "
            "semantic correctness reference."
        ),
    }
    (FIXTURES_DIR / "saehd_df_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Captured fixtures to", FIXTURES_DIR)
    for name, shape in metadata["shapes"].items():
        print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
