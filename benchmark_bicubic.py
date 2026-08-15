#!/usr/bin/env python
"""Evaluate the fixed-split clipped bicubic 2x restoration baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from restoration.data import build_pair_index, create_or_load_split, discover_dataset, records_from_split
from restoration.io import load_grayscale_npy
from restoration.metrics import grouped_metric_summary, psnr, ssim
from restoration.utils import PROJECT_ROOT, relative_or_absolute, select_device, synchronize, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--noisy-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument(
        "--validation-subsets",
        type=Path,
        default=PROJECT_ROOT / "splits" / "validation_subsets_seed42.json",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--max-images", type=int, help="Diagnostic subset only; omit for the official fixed split")
    return parser.parse_args()


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
    pairs, missing_gt, missing_noisy = build_pair_index(noisy_dir, gt_dir, verify_shapes=True)
    if missing_gt or missing_noisy:
        raise RuntimeError(f"incomplete pairing: missing GT={len(missing_gt)}, missing noisy={len(missing_noisy)}")
    split = create_or_load_split(pairs, args.split, root, seed=42, train_fraction=0.85)
    _, records = records_from_split(split, pairs)
    if args.max_images:
        records = records[: args.max_images]
    device = select_device(args.device)
    rows = []
    total_time = 0.0
    print(f"Bicubic baseline on {len(records)} validation images using {device}")
    for record in records:
        noisy = load_grayscale_npy(record.noisy)
        gt = load_grayscale_npy(record.gt)
        source = torch.from_numpy(np.clip(noisy, 0.0, 1.0)).unsqueeze(0).unsqueeze(0).to(device)
        synchronize(device)
        started = time.perf_counter()
        prediction = F.interpolate(source, scale_factor=2.0, mode="bicubic", align_corners=False)
        synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        pred = prediction.squeeze().float().cpu().numpy().clip(0.0, 1.0)
        rows.append(
            {
                "key": record.key,
                "input_height": noisy.shape[0],
                "input_width": noisy.shape[1],
                "output_height": gt.shape[0],
                "output_width": gt.shape[1],
                "size_group": f"{noisy.shape[0]}x{noisy.shape[1]}_to_{gt.shape[0]}x{gt.shape[1]}",
                "psnr": psnr(pred, gt, data_range=1.0),
                "ssim": ssim(pred, gt, data_range=1.0),
                "inference_time_ms": elapsed_ms,
            }
        )
        total_time += elapsed_ms
    validation_subsets = {}
    if args.validation_subsets.is_file() and not args.max_images:
        subset_payload = json.loads(args.validation_subsets.read_text(encoding="utf-8"))
        if subset_payload.get("dataset_fingerprint") != split["dataset_fingerprint"]:
            raise RuntimeError("validation subset dataset fingerprint does not match the active split")
        if subset_payload.get("source_split_sha256") != hashlib.sha256(args.split.read_bytes()).hexdigest():
            raise RuntimeError("validation subset source-split hash does not match the active split")
        rows_by_key = {row["key"]: row for row in rows}
        for subset_name in ("val_random", "val_ood_proxy"):
            keys = subset_payload[subset_name]["keys"]
            validation_subsets[subset_name] = grouped_metric_summary([rows_by_key[key] for key in keys])["combined"]
    timings = np.asarray([row["inference_time_ms"] for row in rows], dtype=np.float64)
    summary = {
        "baseline": "clipped_input_bicubic_2x_torch_align_corners_false",
        "device": str(device),
        "split_path": relative_or_absolute(args.split.resolve(), PROJECT_ROOT),
        "dataset_fingerprint": split["dataset_fingerprint"],
        "subset": bool(args.max_images),
        "evaluated_images": len(rows),
        "metrics": grouped_metric_summary(rows),
        "validation_subsets": validation_subsets,
        "timing": {
            "total_model_inference_ms": total_time,
            "mean_model_inference_ms": total_time / len(rows) if rows else None,
            "median_model_inference_ms": float(np.median(timings)) if len(timings) else None,
            "p95_model_inference_ms": float(np.percentile(timings, 95)) if len(timings) else None,
            "throughput_images_per_second": 1000.0 * len(rows) / total_time if total_time else None,
            "note": "Per-image interpolation only; CUDA synchronized before and after every timed operation.",
        },
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "bicubic_metrics.csv", rows)
    write_json(args.report_dir / "bicubic_summary.json", summary)
    combined = summary["metrics"]["combined"]
    print(f"PSNR: {combined['psnr']['mean']:.4f} dB")
    print(f"SSIM: {combined['ssim']['mean']:.6f}")
    print(f"Mean synchronized interpolation time: {summary['timing']['mean_model_inference_ms']:.3f} ms/image")
    print(f"Wrote {args.report_dir / 'bicubic_metrics.csv'}")
    print(f"Wrote {args.report_dir / 'bicubic_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
