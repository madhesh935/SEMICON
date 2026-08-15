#!/usr/bin/env python
"""Compare incumbent/candidate fidelity and measure periodic PixelShuffle artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from restoration.data import build_pair_index, create_or_load_split, discover_dataset, records_from_split
from restoration.io import load_grayscale_npy
from restoration.metrics import psnr, ssim, summarize
from restoration.utils import PROJECT_ROOT, load_model_from_checkpoint, select_device, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--noisy-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument("--incumbent", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--incumbent-metrics",
        type=Path,
        default=PROJECT_ROOT / "reports" / "validation_metrics.csv",
    )
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument(
        "--ood-scores",
        type=Path,
        default=PROJECT_ROOT / "reports" / "degradation_analysis" / "ood_proxy_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "model_comparison",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "artifact_analysis.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--zoom-size", type=int, default=64, help="HR crop size")
    return parser.parse_args()


def read_keyed_csv(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["key"]: row for row in csv.DictReader(handle)}


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(image, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(image, axis=0, mode="reflect") / 8.0
    return np.hypot(gx, gy)


def artifact_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    phases = [error[row::2, column::2] for row in range(2) for column in range(2)]
    phase_means = np.asarray([phase.mean() for phase in phases])
    phase_rmse = np.asarray([np.sqrt(np.mean(np.square(phase))) for phase in phases])
    rows = np.arange(error.shape[0])[:, None]
    columns = np.arange(error.shape[1])[None, :]
    row_pattern = np.where(rows % 2, -1.0, 1.0)
    column_pattern = np.where(columns % 2, -1.0, 1.0)
    checker_pattern = row_pattern * column_pattern
    pred_gradient = gradient_magnitude(prediction)
    target_gradient = gradient_magnitude(target)
    return {
        "phase_mean_spread": float(phase_means.max() - phase_means.min()),
        "phase_rmse_spread": float(phase_rmse.max() - phase_rmse.min()),
        "row_nyquist_projection": float(abs(np.mean(error * row_pattern))),
        "column_nyquist_projection": float(abs(np.mean(error * column_pattern))),
        "checkerboard_projection": float(abs(np.mean(error * checker_pattern))),
        "gradient_mae": float(np.mean(np.abs(pred_gradient - target_gradient))),
        "false_edge_fraction": float(np.mean((pred_gradient > 0.10) & (target_gradient < 0.03))),
        "missing_edge_fraction": float(np.mean((pred_gradient < 0.03) & (target_gradient > 0.10))),
        "zero_saturation_fraction": float(np.mean(prediction <= 0.0)),
        "one_saturation_fraction": float(np.mean(prediction >= 1.0)),
    }


def best_zoom(error: np.ndarray, size: int) -> tuple[slice, slice]:
    height, width = error.shape
    size = min(size, height, width)
    local = ndimage.uniform_filter(np.asarray(error, dtype=np.float32), size=size, mode="nearest")
    center_y, center_x = np.unravel_index(int(np.argmax(local)), local.shape)
    top = min(max(0, center_y - size // 2), height - size)
    left = min(max(0, center_x - size // 2), width - size)
    return slice(top, top + size), slice(left, left + size)


def choose_examples(
    incumbent_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    ood_rows: dict[str, dict],
    count: int,
) -> list[tuple[str, str]]:
    common = sorted(set(incumbent_rows) & set(candidate_rows))
    differences = {
        key: float(candidate_rows[key]["psnr"]) - float(incumbent_rows[key]["psnr"])
        for key in common
    }
    categories: list[tuple[str, list[str]]] = [
        ("hard_incumbent", sorted(common, key=lambda key: float(incumbent_rows[key]["psnr"]))),
        ("candidate_gain", sorted(common, key=lambda key: -differences[key])),
        ("candidate_regression", sorted(common, key=lambda key: differences[key])),
        (
            "structural_ood",
            sorted(
                (key for key in common if key in ood_rows),
                key=lambda key: -float(ood_rows[key]["ood_score"]),
            ),
        ),
    ]
    selected: list[tuple[str, str]] = []
    used = set()
    cursor = defaultdict(int)
    while len(selected) < min(count, len(common)):
        progress = False
        for category, ordered in categories:
            while cursor[category] < len(ordered) and ordered[cursor[category]] in used:
                cursor[category] += 1
            if cursor[category] < len(ordered):
                key = ordered[cursor[category]]
                cursor[category] += 1
                used.add(key)
                selected.append((key, category))
                progress = True
                if len(selected) >= count:
                    break
        if not progress:
            break
    return selected


def main() -> int:
    args = parse_args()
    if (args.noisy_dir is None) != (args.gt_dir is None):
        raise SystemExit("--noisy-dir and --gt-dir must be supplied together")
    root = args.root.resolve()
    if args.noisy_dir:
        noisy_dir, gt_dir = args.noisy_dir.resolve(), args.gt_dir.resolve()
    else:
        locations = discover_dataset(root)
        noisy_dir, gt_dir = locations.noisy_dir, locations.gt_dir
    pairs, missing_gt, missing_noisy = build_pair_index(noisy_dir, gt_dir, verify_shapes=True)
    if missing_gt or missing_noisy:
        raise RuntimeError("paired dataset is incomplete")
    split = create_or_load_split(pairs, args.split, root, seed=42, train_fraction=0.85)
    _, records = records_from_split(split, pairs)
    incumbent_report = read_keyed_csv(args.incumbent_metrics)
    candidate_report = read_keyed_csv(args.candidate_metrics)
    ood_report = read_keyed_csv(args.ood_scores)
    selected = choose_examples(incumbent_report, candidate_report, ood_report, args.num_images)
    selected_by_key = dict(selected)

    device = select_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    incumbent, incumbent_checkpoint = load_model_from_checkpoint(args.incumbent, device)
    candidate, candidate_checkpoint = load_model_from_checkpoint(args.candidate, device)
    method_rows = []
    selected_arrays = {}
    print(f"Artifact analysis on {len(records)} validation pairs; AMP={use_amp}, device={device}")
    with torch.inference_mode():
        for index, record in enumerate(records):
            raw = load_grayscale_npy(record.noisy)
            gt = load_grayscale_npy(record.gt)
            tensor = torch.from_numpy(raw.copy()).view(1, 1, *raw.shape).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                bicubic_tensor = F.interpolate(
                    tensor.clamp(0.0, 1.0), scale_factor=2.0, mode="bicubic", align_corners=False
                )
                incumbent_tensor = incumbent(tensor)
                candidate_tensor = candidate(tensor)
            outputs = {
                "bicubic": bicubic_tensor.float().clamp(0.0, 1.0).squeeze().cpu().numpy(),
                "incumbent": incumbent_tensor.float().clamp(0.0, 1.0).squeeze().cpu().numpy(),
                "candidate": candidate_tensor.float().clamp(0.0, 1.0).squeeze().cpu().numpy(),
            }
            for method, prediction in outputs.items():
                method_rows.append(
                    {
                        "key": record.key,
                        "method": method,
                        "psnr": psnr(prediction, gt),
                        "ssim": ssim(prediction, gt),
                        **artifact_metrics(prediction, gt),
                    }
                )
            if record.key in selected_by_key:
                selected_arrays[record.key] = {"raw": raw, "gt": gt, **outputs}
            if (index + 1) % 120 == 0:
                print(f"  {index + 1}/{len(records)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, category in selected:
        arrays = selected_arrays[key]
        display_lr = np.repeat(np.repeat(np.clip(arrays["raw"], 0.0, 1.0), 2, axis=0), 2, axis=1)
        panels = [display_lr, arrays["bicubic"], arrays["incumbent"], arrays["candidate"], arrays["gt"]]
        labels = ["NoisyLR (nearest display)", "Bicubic", "Incumbent", "Candidate", "Ground truth"]
        zoom = best_zoom(np.abs(arrays["bicubic"] - arrays["gt"]), args.zoom_size)
        figure, axes = plt.subplots(2, 5, figsize=(16, 6.5))
        for column, (panel, label) in enumerate(zip(panels, labels)):
            axes[0, column].imshow(panel, cmap="gray", vmin=0, vmax=1)
            title = label
            if label in {"Bicubic", "Incumbent", "Candidate"}:
                title += f"\n{psnr(panel, arrays['gt']):.2f} dB / {ssim(panel, arrays['gt']):.3f} SSIM"
            axes[0, column].set_title(title)
            axes[0, column].axis("off")
            axes[1, column].imshow(panel[zoom], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[1, column].set_title("Zoomed region")
            axes[1, column].axis("off")
        figure.suptitle(f"Validation {key} — {category}; arrays used for metrics are unchanged", fontsize=13)
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{key}_{category}.png", dpi=180)
        plt.close(figure)

    write_csv(args.report.with_name("artifact_metrics.csv"), method_rows)
    summary = {
        "official_test_data_used": False,
        "validation_count": len(records),
        "selected_examples": [{"key": key, "reason": reason} for key, reason in selected],
        "incumbent": {
            "weights": str(args.incumbent.name),
            "model_config": incumbent_checkpoint["model_config"],
        },
        "candidate": {
            "weights": str(args.candidate.name),
            "model_config": candidate_checkpoint["model_config"],
        },
        "methods": {},
        "interpretation": {
            "phase_metrics": "Lower phase spread/Nyquist projections indicate less systematic 2x periodic residual structure.",
            "edge_metrics": "Fixed thresholds are diagnostic only; lower false/missing-edge fractions are preferred.",
        },
    }
    metric_names = [
        "psnr",
        "ssim",
        "phase_mean_spread",
        "phase_rmse_spread",
        "row_nyquist_projection",
        "column_nyquist_projection",
        "checkerboard_projection",
        "gradient_mae",
        "false_edge_fraction",
        "missing_edge_fraction",
        "zero_saturation_fraction",
        "one_saturation_fraction",
    ]
    for method in ("bicubic", "incumbent", "candidate"):
        rows = [row for row in method_rows if row["method"] == method]
        summary["methods"][method] = {
            metric: summarize(row[metric] for row in rows) for metric in metric_names
        }
    write_json(args.report, summary)
    print(f"Wrote {args.report} and {len(selected)} comparison figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
