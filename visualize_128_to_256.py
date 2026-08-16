#!/usr/bin/env python
"""Qualitative check for the trained, GT-scored 128->256 path.

Companion to visualize_256_to_512.py, but unlike that script this one uses
real degraded NoisyLR_128 input and real GT_256 -- both exist as paired data
for this scale, so no synthetic noise or proxy protocol is needed here.
Produces one combined grid: real noisy input | bicubic baseline | model
output | real ground truth, for a few real held-out validation images.
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
from restoration.utils import PROJECT_ROOT, load_model_from_checkpoint, select_device, set_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy-dir", type=Path, default=PROJECT_ROOT / "Dataset" / "train" / "train" / "NoisyLR")
    parser.add_argument("--gt-dir", type=Path, default=PROJECT_ROOT / "Dataset" / "train" / "train" / "GT")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--keys", nargs="*", help="Optional exact validation keys")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "docs" / "manual_128_to_256_visual_check.png"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_deterministic(args.seed)

    split = json.loads(args.split.read_text(encoding="utf-8"))
    val_keys = split["validation_keys"]
    if args.keys:
        missing = set(args.keys) - set(val_keys)
        if missing:
            raise ValueError(f"requested keys are not in validation split: {sorted(missing)}")
        selected_keys = args.keys
    else:
        indices = np.linspace(0, len(val_keys) - 1, min(args.num_images, len(val_keys)), dtype=int)
        selected_keys = [val_keys[int(index)] for index in indices]

    device = select_device(args.device)
    model, _ = load_model_from_checkpoint(args.weights, device, prefer_ema=True)

    fig, axes = plt.subplots(len(selected_keys), 4, figsize=(16, 4 * len(selected_keys)))
    if len(selected_keys) == 1:
        axes = axes[None, :]

    for row, key in enumerate(selected_keys):
        noisy_raw = load_grayscale_npy(args.noisy_dir / f"{key}.npy", mmap_mode=None)
        gt = load_grayscale_npy(args.gt_dir / f"{key}.npy", mmap_mode=None)
        noisy_t = torch.from_numpy(np.array(noisy_raw, dtype=np.float32, copy=True)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                restored = model(noisy_t).float().clamp(0.0, 1.0)
            bicubic = F.interpolate(noisy_t.clamp(0, 1), scale_factor=2.0, mode="bicubic", align_corners=False).clamp(0, 1)

        panels = (
            (noisy_t.clamp(0, 1)[0, 0].cpu().numpy(), f"{key}: real NoisyLR input 128x128\nraw [{noisy_raw.min():.3f}, {noisy_raw.max():.3f}]"),
            (bicubic[0, 0].cpu().numpy(), "bicubic 256x256 (baseline)"),
            (restored[0, 0].cpu().numpy(), "model restored 256x256"),
            (gt, "real ground truth 256x256"),
        )
        for col, (image, title) in enumerate(panels):
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis("off")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=130)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
