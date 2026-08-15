"""Restoration metrics, summaries, MAC counts, and synchronized latency."""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch import nn


def psnr(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"PSNR shape mismatch: {pred.shape} vs {truth.shape}")
    mse = float(np.mean((pred - truth) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * math.log10((data_range**2) / mse)


def ssim(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    if pred.shape != truth.shape:
        raise ValueError(f"SSIM shape mismatch: {pred.shape} vs {truth.shape}")
    return float(structural_similarity(truth, pred, data_range=data_range, channel_axis=None))


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def grouped_metric_summary(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["size_group"]].append(row)
    groups["combined"] = rows
    # The model contract supports both official sizes. Keep both sections in
    # every report even when the provided validation split contains only one.
    groups.setdefault("128x128_to_256x256", [])
    groups.setdefault("256x256_to_512x512", [])
    all_metric_names = [
        name for name in ("psnr", "ssim", "lpips") if any(name in row and row[name] is not None for row in rows)
    ]
    output = {}
    for group, items in groups.items():
        metric_names = [name for name in all_metric_names if not items or any(name in item and item[name] is not None for item in items)]
        output[group] = {name: summarize(item[name] for item in items if item.get(name) is not None) for name in metric_names}
        output[group]["image_count"] = len(items)
    return output


class LPIPSMetric:
    """Lazy LPIPS evaluator; never imported by official inference."""

    def __init__(self, device: torch.device, net: str = "alex") -> None:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("LPIPS requested but package 'lpips' is not installed") from exc
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        pred = prediction.clamp(0, 1).repeat(1, 3, 1, 1).mul(2).sub(1)
        truth = target.clamp(0, 1).repeat(1, 3, 1, 1).mul(2).sub(1)
        return float(self.model(pred.to(self.device), truth.to(self.device)).mean().item())


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.inference_mode()
def estimate_conv_macs(model: nn.Module, input_shape: tuple[int, int, int, int], device: torch.device) -> int:
    total = 0
    hooks = []

    def hook(module: nn.Conv2d, _inputs, output) -> None:
        nonlocal total
        batch, channels_out, height, width = output.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total += int(batch * channels_out * height * width * kernel_ops)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook))
    training = model.training
    model.eval()
    try:
        model(torch.zeros(input_shape, device=device))
    finally:
        for handle in hooks:
            handle.remove()
        model.train(training)
    return total


@torch.inference_mode()
def benchmark_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    *,
    warmup: int = 20,
    iterations: int = 100,
    amp: bool = True,
) -> dict[str, float | int | str]:
    sample = torch.zeros(input_shape, device=device)
    use_amp = bool(amp and device.type == "cuda")
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "device": str(device),
        "batch_size": input_shape[0],
        "input_height": input_shape[2],
        "input_width": input_shape[3],
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "latency_ms_mean": statistics.fmean(timings),
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p95": float(np.percentile(np.asarray(timings, dtype=np.float64), 95)),
        "latency_ms_std": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        "throughput_images_per_second": 1000.0 * input_shape[0] / statistics.fmean(timings),
        "peak_gpu_memory_bytes": int(peak),
    }


def checkpoint_size_bytes(path: str | Path) -> int:
    return Path(path).stat().st_size
