"""
Captures golden output fixtures from the current TF1/leras XSeg implementation
(core/leras/models/XSeg.py), matching the construction facelib/XSegNet.py uses:
nn.XSeg(in_ch=3, base_ch=32, out_ch=1) at resolution=256.

One-time capture, run BEFORE the TF/leras code is removed — see
tests/characterization/capture_saehd_fixtures.py for the same methodology applied to SAEHD, and
IMPLEMENTATION_PLAN.md's "Cross-cutting: characterization testing".

Requires the `dfl` conda env:
    /Users/ankurs/miniconda3/envs/dfl/bin/python tests/characterization/capture_xseg_fixtures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

SEED = 42
RESOLUTION = 256
IN_CH = 3
BASE_CH = 32
OUT_CH = 1


def main():
    from core.leras import nn

    nn.initialize_main_env()
    nn.initialize(data_format="NHWC", device_config=nn.DeviceConfig.CPU())
    tf = nn.tf

    rng = np.random.RandomState(SEED)
    dummy_input = rng.uniform(0.0, 1.0, size=(1, RESOLUTION, RESOLUTION, IN_CH)).astype(np.float32)

    xseg = nn.XSeg(IN_CH, BASE_CH, OUT_CH, name="xseg")

    input_t = tf.placeholder(nn.floatx, (1, RESOLUTION, RESOLUTION, IN_CH), name="input")
    logits_t, probs_t = xseg(input_t)

    with nn.tf_sess.as_default() as sess:
        sess.run(tf.global_variables_initializer())
        logits, probs = sess.run([logits_t, probs_t], feed_dict={input_t: dummy_input})

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURES_DIR / "xseg_input.npy", dummy_input)
    np.save(FIXTURES_DIR / "xseg_logits.npy", logits)
    np.save(FIXTURES_DIR / "xseg_probs.npy", probs)

    metadata = {
        "source": "core/leras/models/XSeg.py, via facelib/XSegNet.py construction pattern",
        "tensorflow_version": tf.__version__,
        "seed": SEED,
        "resolution": RESOLUTION,
        "in_ch": IN_CH,
        "base_ch": BASE_CH,
        "out_ch": OUT_CH,
        "shapes": {
            "input": list(dummy_input.shape),
            "logits": list(logits.shape),
            "probs": list(probs.shape),
        },
        "note": "Weights are randomly initialized (no training occurred) — shape/range reference only.",
    }
    (FIXTURES_DIR / "xseg_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Captured fixtures to", FIXTURES_DIR)
    for name, shape in metadata["shapes"].items():
        print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
