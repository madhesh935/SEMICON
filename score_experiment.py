#!/usr/bin/env python
"""Score one trained checkpoint and upsert the controlled experiment registry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from restoration.data import FullImageDataset, build_pair_index, create_or_load_split, discover_dataset, records_from_split
from restoration.metrics import (
    LPIPSMetric,
    benchmark_latency,
    count_parameters,
    estimate_conv_macs,
    grouped_metric_summary,
    psnr,
    ssim,
)
from restoration.utils import (
    PROJECT_ROOT,
    load_model_from_checkpoint,
    refuse_existing_outputs,
    select_device,
    synchronize,
    write_csv,
    write_json,
)


EXPERIMENT_FIELDS = [
    "experiment_id",
    "model_name",
    "input_representation",
    "width",
    "num_blocks",
    "parameters",
    "MACs",
    "loss_configuration",
    "augmentation",
    "patch_size",
    "batch_size",
    "learning_rate",
    "epochs_or_steps",
    "validation_PSNR",
    "validation_SSIM",
    "validation_LPIPS",
    "OOD_PSNR",
    "OOD_SSIM",
    "latency_ms_128",
    "latency_ms_256",
    "throughput",
    "peak_VRAM_MB",
    "checkpoint_size_MB",
    "pareto_optimal",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--noisy-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument(
        "--validation-subsets",
        type=Path,
        default=PROJECT_ROOT / "splits" / "validation_subsets_seed42.json",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--reports-root", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--prefer-raw-weights", action="store_true", help="Score model_state instead of EMA weights")
    parser.add_argument("--lpips", action="store_true", help="Enable validation-only LPIPS")
    parser.add_argument(
        "--lpips-metrics-csv",
        type=Path,
        help="Reuse already measured per-key LPIPS values (for unchanged checkpoint reproduction only)",
    )
    parser.add_argument("--lpips-cache", type=Path, default=PROJECT_ROOT / "weights" / "metric_cache")
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-iterations", type=int, default=30)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing registry entry / detail files for --experiment-id",
    )
    return parser.parse_args()


def read_registry(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "null"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mark_pareto(rows: list[dict]) -> None:
    objectives = (
        ("validation_PSNR", True),
        ("OOD_PSNR", True),
        ("latency_ms_128", False),
        ("parameters", False),
    )
    for candidate in rows:
        candidate["pareto_optimal"] = ""
        if any(number(candidate, key) is None for key, _ in objectives):
            continue
        dominated = False
        for challenger in rows:
            if challenger is candidate or any(number(challenger, key) is None for key, _ in objectives):
                continue
            no_worse = True
            strictly_better = False
            for key, maximize in objectives:
                left = number(challenger, key)
                right = number(candidate, key)
                if maximize:
                    no_worse &= left >= right
                    strictly_better |= left > right
                else:
                    no_worse &= left <= right
                    strictly_better |= left < right
            if no_worse and strictly_better:
                dominated = True
                break
        candidate["pareto_optimal"] = "no" if dominated else "yes"


def markdown_value(row: dict, key: str, digits: int = 4) -> str:
    value = number(row, key)
    return "—" if value is None else f"{value:.{digits}f}"


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Controlled Experiment Registry",
        "",
        "All learned screening runs use the immutable seed-42 training/validation split. OOD metrics use the input-only, train-fitted proxy in `splits/validation_subsets_seed42.json`. Blank LPIPS cells mean the expensive natural-image perceptual metric was intentionally deferred during screening; PSNR/SSIM are always measured on the complete 480-image validation fold. Latency comparisons are CUDA-synchronized and should be compared only within similar current-hardware sessions.",
        "",
        "| ID | Representation | Width/blocks | Loss | Patch | Params | PSNR | SSIM | OOD PSNR | OOD SSIM | 128 ms | 256 ms | Pareto |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {experiment_id} | {input_representation} | {width}/{num_blocks} | {loss_configuration} | {patch_size} | {parameters} | {psnr} | {ssim} | {ood_psnr} | {ood_ssim} | {lat128} | {lat256} | {pareto_optimal} |".format(
                **row,
                psnr=markdown_value(row, "validation_PSNR", 4),
                ssim=markdown_value(row, "validation_SSIM", 5),
                ood_psnr=markdown_value(row, "OOD_PSNR", 4),
                ood_ssim=markdown_value(row, "OOD_SSIM", 5),
                lat128=markdown_value(row, "latency_ms_128", 2),
                lat256=markdown_value(row, "latency_ms_256", 2),
            )
        )
    lines.extend(
        [
            "",
            "Pareto marking uses validation PSNR, OOD-proxy PSNR, 128-input latency, and parameter count. It is a screening aid, not the final selection rule; SSIM, LPIPS for finalists, artifact inspection, and checkpoint training completeness remain mandatory.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.lpips and args.lpips_metrics_csv:
        raise SystemExit("choose --lpips or --lpips-metrics-csv, not both")
    registry_path = args.reports_root / "experiments.csv"
    detail_dir = args.reports_root / "experiment_details"
    existing_registry = read_registry(registry_path)
    if any(item.get("experiment_id") == args.experiment_id for item in existing_registry):
        refuse_existing_outputs(
            [
                registry_path,
                detail_dir / f"{args.experiment_id}_metrics.csv",
                detail_dir / f"{args.experiment_id}.json",
            ],
            force=args.force,
            script=f"score_experiment.py (--experiment-id {args.experiment_id} already scored)",
        )
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
        raise RuntimeError(f"incomplete pairing: missing GT={len(missing_gt)}, missing noisy={len(missing_noisy)}")
    split = create_or_load_split(pairs, args.split, root, seed=42, train_fraction=0.85)
    _, records = records_from_split(split, pairs)
    subset_payload = json.loads(args.validation_subsets.read_text(encoding="utf-8"))
    if subset_payload.get("dataset_fingerprint") != split["dataset_fingerprint"]:
        raise RuntimeError("OOD proxy fingerprint differs from active dataset split")
    ood_keys = set(subset_payload["val_ood_proxy"]["keys"])

    device = select_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    model, checkpoint = load_model_from_checkpoint(
        args.weights.resolve(), device, prefer_ema=not args.prefer_raw_weights
    )
    lpips_metric = None
    lpips_by_key = {}
    if args.lpips:
        args.lpips_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(args.lpips_cache.resolve())
        lpips_metric = LPIPSMetric(device)
    elif args.lpips_metrics_csv:
        with args.lpips_metrics_csv.open(encoding="utf-8", newline="") as handle:
            lpips_by_key = {
                row["key"]: float(row["lpips"])
                for row in csv.DictReader(handle)
                if row.get("lpips") not in (None, "")
            }
        missing_lpips = sorted({record.key for record in records} - set(lpips_by_key))
        if missing_lpips:
            raise RuntimeError(f"LPIPS metrics CSV is missing validation keys: {missing_lpips[:5]}")
    loader = DataLoader(FullImageDataset(records), batch_size=1, shuffle=False, num_workers=0)
    rows = []
    started_wall = time.perf_counter()
    print(
        f"Scoring {args.experiment_id}: {len(records)} validation images, device={device}, "
        f"AMP={use_amp}, weights={'raw' if args.prefer_raw_weights else 'EMA-preferred'}"
    )
    with torch.inference_mode():
        for index, (noisy, gt, keys) in enumerate(loader):
            noisy = noisy.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(noisy)
            prediction = prediction.float().clamp(0.0, 1.0)
            pred_np = prediction.squeeze().cpu().numpy()
            gt_np = gt.squeeze().numpy()
            key = keys[0]
            rows.append(
                {
                    "key": key,
                    "psnr": psnr(pred_np, gt_np, data_range=1.0),
                    "ssim": ssim(pred_np, gt_np, data_range=1.0),
                    "lpips": (
                        lpips_metric(prediction, gt.to(device))
                        if lpips_metric
                        else lpips_by_key.get(key)
                    ),
                    "is_ood_proxy": key in ood_keys,
                }
            )
            if (index + 1) % 120 == 0:
                print(f"  {index + 1}/{len(records)}")
    synchronize(device)
    wall_seconds = time.perf_counter() - started_wall
    random_summary = grouped_metric_summary(
        [
            {
                **row,
                "size_group": "128x128_to_256x256",
            }
            for row in rows
        ]
    )["combined"]
    ood_summary = grouped_metric_summary(
        [
            {
                **row,
                "size_group": "128x128_to_256x256",
            }
            for row in rows
            if row["is_ood_proxy"]
        ]
    )["combined"]
    latency_128 = benchmark_latency(
        model,
        (1, 1, 128, 128),
        device,
        warmup=args.latency_warmup,
        iterations=args.latency_iterations,
        amp=use_amp,
    )
    latency_256 = benchmark_latency(
        model,
        (1, 1, 256, 256),
        device,
        warmup=args.latency_warmup,
        iterations=args.latency_iterations,
        amp=use_amp,
    )
    macs = estimate_conv_macs(model, (1, 1, 128, 128), device)
    config = checkpoint["model_config"]
    train_config = checkpoint.get("train_config") or {}
    data_config = checkpoint.get("data_config") or {}
    context_kernel = int(config.get("context_kernel", 0))
    model_name = args.model_name or (
        "RangeAwareLiteNAFSR" if not context_kernel else f"RangeAwareLiteNAFSR-Context{context_kernel}"
    )
    global_step = checkpoint.get("global_step")
    epoch = checkpoint.get("epoch")
    if global_step is None and args.weights.resolve() == (PROJECT_ROOT / "weights" / "best.pt").resolve():
        global_step, epoch = 6800, 19
    epochs_or_steps = ""
    if epoch is not None or global_step is not None:
        epochs_or_steps = f"{int(epoch) + 1 if epoch is not None else '?'}e/{global_step if global_step is not None else '?'}s"
    notes = args.notes
    if args.prefer_raw_weights:
        notes = (notes + "; " if notes else "") + "scored raw model_state (EMA disabled for evaluation)"
    row = {
        "experiment_id": args.experiment_id,
        "model_name": model_name,
        "input_representation": config.get("input_representation", "raw_clipped_oor"),
        "width": config.get("width", ""),
        "num_blocks": config.get("num_blocks", ""),
        "parameters": count_parameters(model),
        "MACs": macs,
        "loss_configuration": train_config.get("loss", "composite"),
        "augmentation": data_config.get("augmentation", "joint H/V flip and rotations k*90 degrees"),
        "patch_size": data_config.get("patch_size_lr", train_config.get("patch_size", 64)),
        "batch_size": train_config.get("batch_size", 8 if global_step == 6800 else ""),
        "learning_rate": train_config.get("lr", 0.0002 if global_step == 6800 else ""),
        "epochs_or_steps": epochs_or_steps,
        "validation_PSNR": random_summary["psnr"]["mean"],
        "validation_SSIM": random_summary["ssim"]["mean"],
        "validation_LPIPS": random_summary.get("lpips", {}).get("mean", ""),
        "OOD_PSNR": ood_summary["psnr"]["mean"],
        "OOD_SSIM": ood_summary["ssim"]["mean"],
        "latency_ms_128": latency_128["latency_ms_mean"],
        "latency_ms_256": latency_256["latency_ms_mean"],
        "throughput": latency_128["throughput_images_per_second"],
        "peak_VRAM_MB": max(latency_128["peak_gpu_memory_bytes"], latency_256["peak_gpu_memory_bytes"]) / (1024**2),
        "checkpoint_size_MB": args.weights.stat().st_size / 1_000_000,
        "pareto_optimal": "",
        "notes": notes,
    }
    details = {
        "experiment": row,
        "quality": {"val_random": random_summary, "val_ood_proxy": ood_summary},
        "latency": {
            "128x128_to_256x256": latency_128,
            "256x256_to_512x512": latency_256,
        },
        "model_config": config,
        "checkpoint_metadata": {
            key: checkpoint.get(key)
            for key in ("training_complete", "experiment_name", "epoch", "global_step", "best_psnr", "best_ssim")
        },
        "wall_seconds": wall_seconds,
        "official_test_data_used": False,
    }
    detail_dir = args.reports_root / "experiment_details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    write_csv(detail_dir / f"{args.experiment_id}_metrics.csv", rows)
    write_json(detail_dir / f"{args.experiment_id}.json", details)

    registry_path = args.reports_root / "experiments.csv"
    registry = [item for item in read_registry(registry_path) if item.get("experiment_id") != args.experiment_id]
    registry.append(row)
    registry.sort(key=lambda item: item.get("experiment_id", ""))
    mark_pareto(registry)
    write_csv(registry_path, registry, EXPERIMENT_FIELDS)
    write_markdown(args.reports_root / "experiments.md", registry)
    print(
        f"{args.experiment_id}: PSNR={row['validation_PSNR']:.4f}, SSIM={row['validation_SSIM']:.6f}, "
        f"OOD PSNR={row['OOD_PSNR']:.4f}, 128 latency={row['latency_ms_128']:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
