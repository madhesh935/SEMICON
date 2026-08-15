"""Training losses for faithful technical-image restoration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channels, 1, size, size).contiguous()


def ssim_torch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Differentiable channel-wise SSIM averaged over batch and pixels."""

    if prediction.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {prediction.shape} vs {target.shape}")
    channels = prediction.shape[1]
    window = _gaussian_window(window_size, sigma, channels, prediction.device, prediction.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(prediction.square(), window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target.square(), window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    return score.mean()


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.epsilon**2).mean()


class SobelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        ky = kx.t()
        self.register_buffer("kernels", torch.stack((kx, ky)).unsqueeze(1), persistent=False)

    def gradients(self, image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, self.kernels.to(dtype=image.dtype), padding=1)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.gradients(prediction), self.gradients(target))


class FFTMagnitudeLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))
        return F.l1_loss(pred_mag, target_mag)


@dataclass(frozen=True)
class LossWeights:
    charbonnier: float = 0.70
    ssim: float = 0.15
    edge: float = 0.10
    fft: float = 0.05


class CompositeRestorationLoss(nn.Module):
    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.pixel = CharbonnierLoss()
        self.edge = SobelLoss()
        self.frequency = FFTMagnitudeLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pixel = self.pixel(prediction, target)
        structural = (
            1.0 - ssim_torch(prediction.float(), target.float())
            if self.weights.ssim
            else prediction.new_zeros(())
        )
        edge = self.edge(prediction, target) if self.weights.edge else prediction.new_zeros(())
        frequency = self.frequency(prediction, target) if self.weights.fft else prediction.new_zeros(())
        total = (
            self.weights.charbonnier * pixel
            + self.weights.ssim * structural
            + self.weights.edge * edge
            + self.weights.fft * frequency
        )
        components = {
            "charbonnier": float(pixel.detach()),
            "ssim_loss": float(structural.detach()),
            "edge": float(edge.detach()),
            "fft": float(frequency.detach()),
        }
        return total, components


class PixelOnlyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pixel = CharbonnierLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss = self.pixel(prediction, target)
        return loss, {"charbonnier": float(loss.detach())}


def build_loss(name: str) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized in {"composite", "default"}:
        return CompositeRestorationLoss()
    if normalized in {"charbonnier_ssim", "pixel_ssim"}:
        return CompositeRestorationLoss(LossWeights(charbonnier=0.85, ssim=0.15, edge=0.0, fft=0.0))
    if normalized in {"charbonnier_ssim_edge", "pixel_ssim_edge"}:
        return CompositeRestorationLoss(LossWeights(charbonnier=0.75, ssim=0.15, edge=0.10, fft=0.0))
    if normalized in {"pixel", "pixel_only", "charbonnier"}:
        return PixelOnlyLoss()
    raise ValueError(
        f"unknown loss '{name}'; choose charbonnier, charbonnier_ssim, "
        "charbonnier_ssim_edge, or composite"
    )
