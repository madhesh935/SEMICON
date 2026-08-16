#!/usr/bin/env python
"""Qualitative check for the untested 256->512 path on real semiconductor images.

Companion to score_256_to_512_proxy.py. That script gives numeric proxies;
this one gives a visual one, on the same real held-out GT_256 images, with
synthetic Gaussian noise added (std matched to the measured LR-discrepancy
std of 0.0905 from reports/degradation_analysis/summary.md) so the input
resembles a degraded, not clean, image. No real 512x512 ground truth exists
in the provided dataset, so there is no GT panel here -- only noisy input,
bicubic baseline, and model output, side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from restoration.io import load_grayscale_npy
from restoration.utils import (
    PROJECT_ROOT,
    load_model_from_checkpoint,
    refuse_existing_outputs,
    select_device,
    set_deterministic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, default=PROJECT_ROOT / "Dataset" / "train" / "train" / "GT")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.0905)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "docs" / "manual_256_to_512_visual_check.png"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --output image")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refuse_existing_outputs([args.output], force=args.force, script="visualize_256_to_512.py")
    set_deterministic(args.seed)
    device = select_device(args.device)

    model, _ = load_model_from_checkpoint(args.weights, device, prefer_ema=True)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    keys = split["validation_keys"][: args.num_images]

    fig, axes = plt.subplots(len(keys), 3, figsize=(12, 4 * len(keys)))
    if len(keys) == 1:
        axes = axes[None, :]

    for row, key in enumerate(keys):
        gt = load_grayscale_npy(args.gt_dir / f"{key}.npy", mmap_mode=None)
        gt_t = torch.from_numpy(np.asarray(gt, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        noisy = gt_t + torch.randn_like(gt_t) * args.noise_std

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                restored = model(noisy).float().clamp(0.0, 1.0)
        bicubic = F.interpolate(noisy.clamp(0, 1), scale_factor=2.0, mode="bicubic", align_corners=False).clamp(0, 1)

        panels = (
            (noisy[0, 0].clamp(0, 1).cpu().numpy(), f"{key}: synthetic-noisy input 256x256"),
            (bicubic[0, 0].cpu().numpy(), "bicubic 512x512 (baseline)"),
            (restored[0, 0].cpu().numpy(), "model restored 512x512"),
        )
        for col, (image, title) in enumerate(panels):
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=130)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
