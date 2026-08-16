#!/usr/bin/env python
"""Full held-out validation with PSNR, SSIM, LPIPS, complexity, and latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from restoration.data import FullImageDataset, build_pair_index, create_or_load_split, discover_dataset, records_from_split
from restoration.io import save_restored_npy
from restoration.metrics import (
    LPIPSMetric,
    benchmark_latency,
    checkpoint_size_bytes,
    count_parameters,
    estimate_conv_macs,
    grouped_metric_summary,
    psnr,
    ssim,
    summarize,
)
from restoration.utils import (
    PROJECT_ROOT,
    inference_with_tta,
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
    parser.add_argument("--experiment", default="final")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument(
        "--validation-subsets",
        type=Path,
        default=PROJECT_ROOT / "splits" / "validation_subsets_seed42.json",
        help="Optional train-fitted val_random/val_ood_proxy definition",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tta", action="store_true", help="4-rotation self ensemble; off by default")
    parser.add_argument("--no-lpips", action="store_true", help="Diagnostic only; official report should include LPIPS")
    parser.add_argument("--lpips-cache", type=Path, default=PROJECT_ROOT / "weights" / "metric_cache")
    parser.add_argument("--max-images", type=int, help="Diagnostic subset only")
    parser.add_argument("--no-save-outputs", action="store_true", help="Do not retain restored validation arrays")
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-iterations", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Overwrite existing report files at --report-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refuse_existing_outputs(
        [args.report_dir / "validation_metrics.csv", args.report_dir / "validation_summary.json"],
        force=args.force,
        script="validate.py",
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
    model, checkpoint = load_model_from_checkpoint(args.weights, device)
    output_dir = args.run_root.resolve() / args.experiment / "validation_outputs"
    if not args.no_save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    lpips_metric = None
    if not args.no_lpips:
        args.lpips_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(args.lpips_cache.resolve())
        try:
            lpips_metric = LPIPSMetric(device)
        except Exception as exc:
            raise RuntimeError(
                "LPIPS initialization failed. Prepare pretrained AlexNet resources once while online, "
                f"then rerun; cache={args.lpips_cache}. Original error: {exc}"
            ) from exc
    loader = DataLoader(
        FullImageDataset(records),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    rows = []
    print(f"Validating {len(records)} images on {device}; AMP={use_amp}, TTA={args.tta}, LPIPS={lpips_metric is not None}")
    wall_started = time.perf_counter()
    with torch.inference_mode():
        for noisy, gt, keys in loader:
            noisy = noisy.to(device, non_blocking=True)
            synchronize(device)
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = inference_with_tta(model, noisy, args.tta)
            synchronize(device)
            inference_ms = (time.perf_counter() - started) * 1000.0
            prediction = prediction.float().clamp(0.0, 1.0)
            pred_np = prediction.squeeze(0).squeeze(0).cpu().numpy()
            gt_np = gt.squeeze(0).squeeze(0).numpy()
            key = keys[0]
            input_hw = tuple(int(value) for value in noisy.shape[-2:])
            output_hw = tuple(int(value) for value in gt.shape[-2:])
            lpips_value = lpips_metric(prediction, gt.to(device)) if lpips_metric else None
            row = {
                "key": key,
                "input_height": input_hw[0],
                "input_width": input_hw[1],
                "output_height": output_hw[0],
                "output_width": output_hw[1],
                "size_group": f"{input_hw[0]}x{input_hw[1]}_to_{output_hw[0]}x{output_hw[1]}",
                "psnr": psnr(pred_np, gt_np, data_range=1.0),
                "ssim": ssim(pred_np, gt_np, data_range=1.0),
                "lpips": lpips_value,
                "model_inference_ms": inference_ms,
            }
            rows.append(row)
            if not args.no_save_outputs:
                save_restored_npy(output_dir / f"{key}.npy", pred_np, output_hw)
    wall_seconds = time.perf_counter() - wall_started

    shapes_to_profile = [(128, 128), (256, 256)]
    macs = {}
    latency = {}
    for height, width in shapes_to_profile:
        key = f"{height}x{width}_to_{height * 2}x{width * 2}"
        macs[key] = estimate_conv_macs(model, (1, 1, height, width), device)
        latency[key] = benchmark_latency(
            model,
            (1, 1, height, width),
            device,
            warmup=args.latency_warmup,
            iterations=args.latency_iterations,
            amp=use_amp,
        )
    validation_subsets = {}
    if args.validation_subsets and args.validation_subsets.is_file() and not args.max_images:
        subset_payload = json.loads(args.validation_subsets.read_text(encoding="utf-8"))
        if subset_payload.get("dataset_fingerprint") != split["dataset_fingerprint"]:
            raise RuntimeError("validation subset dataset fingerprint does not match the active split")
        digest = hashlib.sha256(args.split.read_bytes()).hexdigest()
        if subset_payload.get("source_split_sha256") != digest:
            raise RuntimeError("validation subset source-split hash does not match the active split")
        rows_by_key = {row["key"]: row for row in rows}
        for subset_name in ("val_random", "val_ood_proxy"):
            keys = subset_payload[subset_name]["keys"]
            missing = sorted(set(keys) - set(rows_by_key))
            if missing:
                raise RuntimeError(f"{subset_name} references missing validation rows: {missing[:5]}")
            validation_subsets[subset_name] = grouped_metric_summary([rows_by_key[key] for key in keys])["combined"]

    summary = {
        "weights": relative_or_absolute(args.weights.resolve(), PROJECT_ROOT),
        "architecture": checkpoint.get("architecture"),
        "model_config": checkpoint.get("model_config"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "amp": use_amp,
        "tta": args.tta,
        "subset": bool(args.max_images),
        "split_identity": {
            "path": relative_or_absolute(args.split.resolve(), PROJECT_ROOT),
            "seed": split["seed"],
            "dataset_fingerprint": split["dataset_fingerprint"],
        },
        "evaluated_images": len(rows),
        "metrics": grouped_metric_summary(rows),
        "validation_subsets": validation_subsets,
        "model": {
            "parameter_count": count_parameters(model),
            "checkpoint_size_bytes": checkpoint_size_bytes(args.weights),
            "conv_macs": macs,
            "mac_note": "Multiply-accumulate count includes Conv2d operations; interpolation, normalization, gates, and elementwise operations are excluded.",
        },
        "latency": latency,
        "validation_timing": {
            "wall_seconds_including_metrics_and_saves": wall_seconds,
            "per_image_model_inference_ms": summarize(row["model_inference_ms"] for row in rows),
            "timing_note": "CUDA synchronized around each model call; standalone benchmark includes warm-up.",
        },
        "validation_outputs": None if args.no_save_outputs else relative_or_absolute(output_dir, PROJECT_ROOT),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "validation_metrics.csv", rows)
    write_json(args.report_dir / "validation_summary.json", summary)
    combined = summary["metrics"]["combined"]
    lpips_text = f", LPIPS={combined['lpips']['mean']:.6f}" if "lpips" in combined else ""
    print(f"PSNR={combined['psnr']['mean']:.4f} dB, SSIM={combined['ssim']['mean']:.6f}{lpips_text}")
    print(f"Parameters={summary['model']['parameter_count']:,}; checkpoint={summary['model']['checkpoint_size_bytes'] / 1e6:.3f} MB")
    for key, value in latency.items():
        print(
            f"Latency {key}: mean={value['latency_ms_mean']:.3f} ms, "
            f"median={value['latency_ms_median']:.3f} ms, p95={value['latency_ms_p95']:.3f} ms, "
            f"{value['throughput_images_per_second']:.2f} images/s"
        )
    if "val_ood_proxy" in validation_subsets:
        ood = validation_subsets["val_ood_proxy"]
        print(f"OOD proxy ({ood['image_count']}): PSNR={ood['psnr']['mean']:.4f}, SSIM={ood['ssim']['mean']:.6f}")
    print(f"Wrote {args.report_dir / 'validation_metrics.csv'}")
    print(f"Wrote {args.report_dir / 'validation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
