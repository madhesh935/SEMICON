from __future__ import annotations

import pytest
import torch

from restoration.losses import CompositeRestorationLoss, PixelOnlyLoss, build_loss


@pytest.mark.parametrize(
    "name,expected_type,zero_components",
    [
        ("charbonnier", PixelOnlyLoss, set()),
        ("charbonnier_ssim", CompositeRestorationLoss, {"edge", "fft"}),
        ("charbonnier_ssim_edge", CompositeRestorationLoss, {"fft"}),
        ("composite", CompositeRestorationLoss, set()),
    ],
)
def test_loss_ablation_variants_are_finite_and_differentiable(name, expected_type, zero_components):
    prediction = torch.rand(2, 1, 16, 16, requires_grad=True)
    target = torch.rand(2, 1, 16, 16)
    criterion = build_loss(name)
    assert isinstance(criterion, expected_type)
    total, components = criterion(prediction, target)
    assert total.ndim == 0 and torch.isfinite(total)
    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    for component in zero_components:
        assert components[component] == 0.0


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss("adversarial")
