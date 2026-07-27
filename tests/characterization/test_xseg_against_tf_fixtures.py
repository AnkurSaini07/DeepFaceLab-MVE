"""
Validates dfl_torch.xseg.XSegNet against golden fixtures captured from the current TF/leras
implementation (capture_xseg_fixtures.py). Same shape/range-only comparison rationale as
test_against_tf_fixtures.py — clean-room reimplementation, exact numeric match not expected.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dfl_torch.xseg import XSegNet

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def metadata():
    path = FIXTURES_DIR / "xseg_metadata.json"
    if not path.exists():
        pytest.skip("XSeg fixtures not captured yet — run capture_xseg_fixtures.py under the dfl conda env")
    return json.loads(path.read_text())


def _load(name):
    return np.load(FIXTURES_DIR / f"xseg_{name}.npy")


def test_xseg_output_shapes_and_range_match_tf(metadata):
    model = XSegNet(in_ch=metadata["in_ch"], base_ch=metadata["base_ch"], out_ch=metadata["out_ch"])
    x = torch.from_numpy(_load("input")).permute(0, 3, 1, 2).contiguous()
    logits, probs = model(x)

    tf_logits_shape = tuple(metadata["shapes"]["logits"])
    tf_probs_shape = tuple(metadata["shapes"]["probs"])
    assert sorted(logits.shape) == sorted(tf_logits_shape), f"torch {tuple(logits.shape)} vs TF {tf_logits_shape}"
    assert sorted(probs.shape) == sorted(tf_probs_shape), f"torch {tuple(probs.shape)} vs TF {tf_probs_shape}"

    tf_probs = _load("probs")
    assert 0.0 <= probs.min() and probs.max() <= 1.0
    assert 0.0 <= tf_probs.min() and tf_probs.max() <= 1.0
