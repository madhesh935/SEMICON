#!/usr/bin/env python
"""Create presentation-ready full-image and detail comparison figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep Matplotlib fully inside the project on restricted Windows hosts.
_MPL_CACHE = Path(__file__).resolve().parent / ".tmp" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch.nn import functional as F

from restoration.data import build_pair_index, create_or_load_split, discover_dataset, records_from_split
from restoration.io import load_grayscale_npy
from restoration.utils import PROJECT_ROOT, load_model_from_checkpoint, select_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--noisy-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--keys", nargs="*", help="Optional exact validation keys")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def select_zoom(gt: np.ndarray, size: int) -> tuple[slice, slice]:
    gy, gx = np.gradient(gt.astype(np.float32))
    energy = gx * gx + gy * gy
    score = uniform_filter(energy, size=size, mode="nearest")
    margin = size // 2
    if gt.shape[0] > size and gt.shape[1] > size:
        cropped = score[margin : gt.shape[0] - margin, margin : gt.shape[1] - margin]
        y, x = np.unravel_index(int(np.argmax(cropped)), cropped.shape)
        y += margin
        x += margin
    else:
        y, x = gt.shape[0] // 2, gt.shape[1] // 2
    top = min(max(0, y - size // 2), gt.shape[0] - size)
    left = min(max(0, x - size // 2), gt.shape[1] - size)
    return slice(top, top + size), slice(left, left + size)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if (args.noisy_dir is None) != (args.gt_dir is None):
        raise SystemExit("--noisy-dir and --gt-dir must be supplied together")
    if args.noisy_dir:
        noisy_dir, gt_dir = args.noisy_dir.resolve(), args.gt_dir.resolve()
    else:
        locations = discover_dataset(root)
        noisy_dir, gt_dir = locations.noisy_dir, locations.gt_dir
    pairs, _, _ = build_pair_index(noisy_dir, gt_dir, verify_shapes=True)
    split = create_or_load_split(pairs, args.split, root, seed=42, train_fraction=0.85)
    _, records = records_from_split(split, pairs)
    if args.keys:
        wanted = set(args.keys)
        selected = [record for record in records if record.key in wanted]
        missing = wanted - {record.key for record in selected}
        if missing:
            raise ValueError(f"requested keys are not in validation split: {sorted(missing)}")
    else:
        indices = np.linspace(0, len(records) - 1, min(args.num_images, len(records)), dtype=int)
        selected = [records[int(index)] for index in indices]
    device = select_device(args.device)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    model, _ = load_model_from_checkpoint(args.weights, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for record in selected:
        noisy_raw = load_grayscale_npy(record.noisy)
        gt = load_grayscale_npy(record.gt)
        noisy_tensor = torch.from_numpy(np.array(noisy_raw, dtype=np.float32, copy=True)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            baseline_tensor = F.interpolate(noisy_tensor.clamp(0, 1), scale_factor=2, mode="bicubic", align_corners=False)
            prediction_tensor = model(noisy_tensor)
        baseline = baseline_tensor.float().clamp(0, 1).squeeze().cpu().numpy()
        prediction = prediction_tensor.float().clamp(0, 1).squeeze().cpu().numpy()
        noisy_display = baseline.copy()
        images = [noisy_display, baseline, prediction, gt]
        titles = [
            f"NoisyLR (bicubic display)\nraw [{noisy_raw.min():.3f}, {noisy_raw.max():.3f}]",
            "Bicubic baseline",
            "Range-Aware LiteNAF-SR",
            "Ground truth",
        ]
        zoom_size = max(32, min(gt.shape) // 4)
        zoom_y, zoom_x = select_zoom(gt, zoom_size)
        figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        for column, (image, title) in enumerate(zip(images, titles)):
            axes[0, column].imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[0, column].set_title(title, fontsize=10)
            axes[0, column].add_patch(
                plt.Rectangle(
                    (zoom_x.start, zoom_y.start),
                    zoom_x.stop - zoom_x.start,
                    zoom_y.stop - zoom_y.start,
                    edgecolor="#ffb000",
                    facecolor="none",
                    linewidth=1.5,
                )
            )
            axes[1, column].imshow(image[zoom_y, zoom_x], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[1, column].set_title("Detail crop", fontsize=9)
            for axis in (axes[0, column], axes[1, column]):
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(f"Validation sample {record.key} — display-only visualization", fontsize=13)
        output_path = args.output_dir / f"comparison_{Path(record.key).name}.png"
        figure.savefig(output_path, dpi=180, facecolor="white")
        plt.close(figure)
        figures.append(
            {
                "key": record.key,
                "path": str(output_path),
                "zoom_box_hr": [zoom_y.start, zoom_x.start, zoom_y.stop, zoom_x.stop],
                "note": "Display arrays are clipped only for visualization; metric arrays are unchanged.",
            }
        )
        print(f"Saved {output_path}")
    write_json(args.output_dir / "figure_index.json", {"weights": str(args.weights), "figures": figures})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
