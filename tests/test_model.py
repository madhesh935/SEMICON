from __future__ import annotations

import numpy as np
import pytest
import torch

from restoration.io import save_restored_npy
from restoration.model import RangeAwareLiteNAFSR


@pytest.fixture(scope="module")
def tiny_model():
    return RangeAwareLiteNAFSR(width=8, num_blocks=1, num_hr_blocks=0).eval()


@pytest.mark.parametrize("height,width", [(128, 128), (256, 256)])
def test_supported_forward_shapes_and_finiteness(tiny_model, height, width):
    raw = torch.linspace(-0.3, 1.7, height * width).reshape(1, 1, height, width)
    with torch.inference_mode():
        output = tiny_model(raw)
    assert output.shape == (1, 1, 2 * height, 2 * width)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_raw_out_of_range_path_affects_learned_branch(tiny_model):
    with torch.no_grad():
        torch.nn.init.normal_(tiny_model.head.weight, std=0.02)
    one = torch.full((1, 1, 16, 16), 1.2)
    two = torch.full((1, 1, 16, 16), 2.0)
    with torch.inference_mode():
        first, second = tiny_model(one), tiny_model(two)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.allclose(first, second)


def test_saved_prediction_contract(tiny_model, tmp_path):
    raw = torch.randn(1, 1, 16, 16) * 0.5 + 0.5
    with torch.inference_mode():
        output = tiny_model(raw).squeeze().numpy()
    path = tmp_path / "output.npy"
    save_restored_npy(path, output, (32, 32))
    saved = np.load(path, allow_pickle=False)
    assert saved.dtype == np.float32
    assert saved.shape == (32, 32)
    assert np.isfinite(saved).all()
    assert 0.0 <= float(saved.min()) <= float(saved.max()) <= 1.0


@pytest.mark.parametrize(
    "representation,expected",
    [
        ("raw", [-0.5, 0.5, 1.5]),
        ("raw_clipped", [-0.5, 0.5, 1.5, 0.0, 0.5, 1.0]),
        ("raw_clipped_oor", [-0.5, 0.5, 1.5, 0.0, 0.5, 1.0, -0.5, 0.0, 0.5]),
    ],
)
def test_range_representations_preserve_raw_values(representation, expected):
    model = RangeAwareLiteNAFSR(
        width=8,
        num_blocks=0,
        num_hr_blocks=0,
        input_representation=representation,
    )
    raw = torch.tensor([-0.5, 0.5, 1.5], dtype=torch.float32).reshape(1, 1, 1, 3)
    encoded = model.encode_input(raw)
    assert encoded.shape[1] == len(expected) // 3
    assert encoded.flatten().tolist() == pytest.approx(expected)


def test_context_branch_and_unusual_batch_are_spatially_dynamic():
    model = RangeAwareLiteNAFSR(
        width=8,
        num_blocks=1,
        num_hr_blocks=0,
        input_representation="raw_clipped",
        context_kernel=7,
    ).eval()
    raw = torch.tensor([-0.5, -0.1, 0.0, 0.5, 1.0, 1.1, 1.5]).repeat(3 * 17 * 19 // 7 + 1)
    raw = raw[: 3 * 17 * 19].reshape(3, 1, 17, 19)
    with torch.inference_mode():
        output = model(raw)
    assert output.shape == (3, 1, 34, 38)
    assert torch.isfinite(output).all()


def test_restoration_model_has_no_batchnorm():
    model = RangeAwareLiteNAFSR(width=8, num_blocks=1, num_hr_blocks=1)
    assert not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules())
