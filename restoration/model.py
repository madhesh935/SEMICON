"""Range-Aware LiteNAF-SR: lightweight grayscale denoising and 2x SR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Per-pixel channel LayerNorm, independent of batch statistics."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (x.shape[1],), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    """Activation-free restoration block adapted for compact grayscale SR."""

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        expanded = channels * expansion
        if expanded % 2:
            raise ValueError("expanded channels must be even")
        gated = expanded // 2
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.gate1 = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(gated, gated, 1))
        self.conv2 = nn.Conv2d(gated, channels, 1)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, expanded, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(gated, channels, 1)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.gate1(self.depthwise(self.conv1(y)))
        y = y * self.sca(y)
        y = self.dropout1(self.conv2(y))
        x = x + y * self.beta
        y = self.norm2(x)
        y = self.dropout2(self.conv4(self.gate2(self.conv3(y))))
        return x + y * self.gamma


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    num_blocks: int = 24
    num_hr_blocks: int = 1
    expansion: int = 2
    dropout: float = 0.0
    scale: int = 2
    input_representation: str = "raw_clipped_oor"
    context_kernel: int = 0
    context_dilation: int = 1


class RangeAwareLiteNAFSR(nn.Module):
    """Fast residual 2x restoration with an unclipped raw feature path.

    The stem sees three channels: raw input, a clipped copy, and the signed
    out-of-range residual (`raw - clipped`). The image skip uses bicubic
    interpolation of only the clipped copy. The learned branch remains able to
    exploit meaningful values below zero and above one.
    """

    architecture_name = "Range-Aware LiteNAF-SR"

    def __init__(
        self,
        width: int = 64,
        num_blocks: int = 24,
        num_hr_blocks: int = 1,
        expansion: int = 2,
        dropout: float = 0.0,
        scale: int = 2,
        input_representation: str = "raw_clipped_oor",
        context_kernel: int = 0,
        context_dilation: int = 1,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("This competition model supports exactly 2x output")
        if width < 4:
            raise ValueError("width must be at least 4")
        input_channels = {
            "raw": 1,
            "raw_clipped": 2,
            "raw_clipped_oor": 3,
        }
        if input_representation not in input_channels:
            raise ValueError(
                "input_representation must be one of raw, raw_clipped, or raw_clipped_oor"
            )
        if context_kernel not in {0, 5, 7}:
            raise ValueError("context_kernel must be 0 (disabled), 5, or 7")
        if context_dilation < 1:
            raise ValueError("context_dilation must be at least 1")
        self.config = ModelConfig(
            width,
            num_blocks,
            num_hr_blocks,
            expansion,
            dropout,
            scale,
            input_representation,
            context_kernel,
            context_dilation,
        )
        self.scale = scale
        self.input_representation = input_representation
        self.stem = nn.Conv2d(input_channels[input_representation], width, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(width, expansion, dropout) for _ in range(num_blocks)])
        self.body_end = nn.Conv2d(width, width, 3, padding=1)
        if context_kernel:
            padding = context_dilation * (context_kernel // 2)
            self.context = nn.Sequential(
                LayerNorm2d(width),
                nn.Conv2d(
                    width,
                    width,
                    context_kernel,
                    padding=padding,
                    dilation=context_dilation,
                    groups=width,
                ),
                nn.Conv2d(width, width, 1),
            )
            self.context_scale = nn.Parameter(torch.zeros(1, width, 1, 1))
        else:
            self.context = None
            self.register_parameter("context_scale", None)
        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        self.hr_refine = nn.Sequential(
            *[NAFBlock(width, expansion, dropout) for _ in range(num_hr_blocks)]
        )
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def model_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def encode_input(self, raw: torch.Tensor) -> torch.Tensor:
        """Build the selected range representation without discarding raw values."""

        if self.input_representation == "raw":
            return raw
        clipped = raw.clamp(0.0, 1.0)
        if self.input_representation == "raw_clipped":
            return torch.cat((raw, clipped), dim=1)
        return torch.cat((raw, clipped, raw - clipped), dim=1)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 4 or raw.shape[1] != 1:
            raise ValueError(f"expected NCHW single-channel input, got {tuple(raw.shape)}")
        clipped = raw.clamp(0.0, 1.0)
        stem_features = self.stem(self.encode_input(raw))
        features = stem_features + self.body_end(self.body(stem_features))
        if self.context is not None:
            features = features + self.context_scale * self.context(stem_features)
        features = self.upsample(features)
        features = self.hr_refine(features)
        residual = self.head(features)
        skip = F.interpolate(clipped, scale_factor=2.0, mode="bicubic", align_corners=False)
        return skip + residual


def create_model(config: dict[str, Any] | ModelConfig | None = None) -> RangeAwareLiteNAFSR:
    if config is None:
        config = ModelConfig()
    if isinstance(config, ModelConfig):
        config = asdict(config)
    allowed = {field.name for field in ModelConfig.__dataclass_fields__.values()}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown model configuration keys: {sorted(unknown)}")
    return RangeAwareLiteNAFSR(**config)
