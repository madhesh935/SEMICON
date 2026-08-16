#!/usr/bin/env python
"""Measure fixed-validation robustness at 0, 90, 180, and 270 degrees."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from restoration.data import FullImageDataset, build_pair_index, create_or_load_split, discover_dataset, records_from_split
from restoration.metrics import psnr, ssim, summarize
from restoration.utils import (
    PROJECT_ROOT,
    load_model_from_checkpoint,
    refuse_existing_outputs,
    relative_or_absolute,
    select_device,
    synchronize,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--noisy-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-images", type=int, help="Diagnostic subset only")
    parser.add_argument("--force", action="store_true", help="Overwrite existing report files at --report-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refuse_existing_outputs(
        [args.report_dir / "robustness_metrics.csv", args.report_dir / "robustness_summary.json"],
        force=args.force,
        script="robustness_test.py",
    )
    root = args.root.resolve()
    if (args.noisy_dir is None) != (args.gt_dir is None):
        raise SystemExit("--noisy-dir and --gt-dir must be supplied together")
    if args.noisy_dir:
        noisy_dir, gt_dir = args.noisy_dir.resolve(), args.gt_dir.resolve()
    else:
        locations = discover_dataset(root)
        noisy_dir, gt_dir = locations.noisy_dir, locations.gt_dir
    pairs, missing_gt, missing_noisy = build_pair_index(noisy_dir, gt_dir, verify_shapes=True)
    if missing_gt or missing_noisy:
        raise RuntimeError(f"incomplete pairing: missing GT={len(missing_gt)}, missing noisy={len(missing_noisy)}")
    split = create_or_load_split(pairs, args.split, root, seed=42, train_fraction=0.85)
    _, records = records_from_split(split, pairs)
    if args.max_images:
        records = records[: args.max_images]
    device = select_device(args.device)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    model, _ = load_model_from_checkpoint(args.weights, device)
    rows = []
    total_started = time.perf_counter()
    for rotation_k, angle in enumerate((0, 90, 180, 270)):
        loader = DataLoader(
            FullImageDataset(records, rotation_k=rotation_k),
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        angle_psnr, angle_ssim = [], []
        with torch.inference_mode():
            for noisy, gt, keys in loader:
                noisy = noisy.to(device, non_blocking=True)
                synchronize(device)
                started = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    prediction = model(noisy).float().clamp(0.0, 1.0)
                synchronize(device)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                pred_np = prediction.squeeze().cpu().numpy()
                gt_np = gt.squeeze().numpy()
                row = {
                    "key": keys[0],
                    "angle_degrees": angle,
                    "psnr": psnr(pred_np, gt_np, data_range=1.0),
                    "ssim": ssim(pred_np, gt_np, data_range=1.0),
                    "model_inference_ms": elapsed_ms,
                }
                rows.append(row)
                angle_psnr.append(row["psnr"])
                angle_ssim.append(row["ssim"])
        print(f"{angle:3d} degrees: PSNR={np.mean(angle_psnr):.4f}, SSIM={np.mean(angle_ssim):.6f}")
    by_angle = {}
    for angle in (0, 90, 180, 270):
        selected = [row for row in rows if row["angle_degrees"] == angle]
        by_angle[str(angle)] = {
            "image_count": len(selected),
            "psnr": summarize(row["psnr"] for row in selected),
            "ssim": summarize(row["ssim"] for row in selected),
            "model_inference_ms": summarize(row["model_inference_ms"] for row in selected),
        }
    summary = {
        "weights": relative_or_absolute(args.weights, PROJECT_ROOT),
        "device": str(device),
        "amp": use_amp,
        "subset": bool(args.max_images),
        "orientation_contract": "LR and GT receive the same rotation; output remains in the rotated input orientation",
        "split_seed": split["seed"],
        "dataset_fingerprint": split["dataset_fingerprint"],
        "by_angle_degrees": by_angle,
        "wall_seconds": time.perf_counter() - total_started,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "robustness_metrics.csv", rows)
    write_json(args.report_dir / "robustness_summary.json", summary)
    print(f"Wrote {args.report_dir / 'robustness_metrics.csv'}")
    print(f"Wrote {args.report_dir / 'robustness_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
