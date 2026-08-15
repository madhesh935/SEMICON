#!/usr/bin/env python
"""Reproducible training for Range-Aware LiteNAF-SR."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

# The default cuDNN v8 execution-plan cache can consume gigabytes on a
# no-pagefile Windows training host when patch and full-validation shapes are
# mixed. A bounded cache avoids observed host `bad allocation` failures without
# changing model mathematics. Users can override this before launching Python.
os.environ.setdefault("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "100")

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from restoration.data import (
    FullImageDataset,
    PairedPatchDataset,
    build_pair_index,
    create_or_load_split,
    discover_dataset,
    records_from_split,
)
from restoration.losses import build_loss
from restoration.metrics import count_parameters, psnr, ssim
from restoration.model import create_model
from restoration.utils import (
    ExponentialMovingAverage,
    PROJECT_ROOT,
    append_csv,
    json_ready,
    relative_or_absolute,
    select_device,
    set_deterministic,
    torch_load_checkpoint,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--noisy-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument("--experiment", default="litenaF_base")
    parser.add_argument("--run-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run-kind", choices=("full", "smoke", "overfit", "ablation"), default="full")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=96, help="LR patch size; GT patches are exactly 2x")
    parser.add_argument("--steps-per-epoch", type=int, help="Override steps per epoch for smoke/overfit runs")
    parser.add_argument("--max-train-pairs", type=int, help="Diagnostic/ablation subset")
    parser.add_argument("--max-val-pairs", type=int, help="Diagnostic/ablation subset")
    parser.add_argument("--overfit-pairs", type=int, default=0, help="Train and validate on the same first N training pairs")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--hr-blocks", type=int, default=1)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--input-representation",
        choices=("raw", "raw_clipped", "raw_clipped_oor"),
        default="raw_clipped_oor",
    )
    parser.add_argument("--context-kernel", type=int, choices=(0, 5, 7), default=0)
    parser.add_argument("--context-dilation", type=int, default=1)
    parser.add_argument(
        "--loss",
        choices=("pixel", "charbonnier", "charbonnier_ssim", "charbonnier_ssim_edge", "composite"),
        default="composite",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="Windows-safe default is 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--resume", nargs="?", const="auto", help="Resume from a checkpoint path or this run's last.pt")
    parser.add_argument("--finalize", action="store_true", help="Publish weights/best.pt; allowed only for a full run that beats bicubic")
    parser.add_argument("--final-weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    return parser.parse_args()


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_ratio: float) -> LambdaLR:
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, schedule)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    records,
    device: torch.device,
    use_amp: bool,
    workers: int,
) -> tuple[float, float, float]:
    loader = DataLoader(
        FullImageDataset(records),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    psnr_values, ssim_values = [], []
    started = time.perf_counter()
    for noisy, gt, _key in loader:
        noisy = noisy.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(noisy).clamp(0.0, 1.0)
        pred_np = prediction.squeeze(0).squeeze(0).float().cpu().numpy()
        gt_np = gt.squeeze(0).squeeze(0).numpy()
        psnr_values.append(psnr(pred_np, gt_np, data_range=1.0))
        ssim_values.append(ssim(pred_np, gt_np, data_range=1.0))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return float(np.mean(psnr_values)), float(np.mean(ssim_values)), time.perf_counter() - started


@torch.inference_mode()
def probe_loss(model: nn.Module, criterion: nn.Module, record, patch_size: int, device: torch.device, use_amp: bool) -> float:
    dataset = PairedPatchDataset([record], patch_size=patch_size, augment=False, seed=0, samples_per_epoch=1)
    noisy, gt, _ = dataset[0]
    noisy, gt = noisy.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        prediction = model(noisy)
    # SSIM/Sobel/FFT are deliberately evaluated in float32. Autocasting their
    # small variance terms to float16 can overflow the backward pass.
    with torch.autocast(device_type=device.type, enabled=False):
        loss, _ = criterion(prediction, gt)
    return float(loss.item())


def checkpoint_payload(
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage | None,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_psnr: float,
    best_ssim: float,
    model_config: dict,
    data_config: dict,
    train_config: dict,
    split: dict,
    split_sha256: str,
) -> dict:
    return {
        "format_version": 1,
        "architecture": "RangeAwareLiteNAFSR",
        "training_complete": False,
        "experiment_name": train_config.get("experiment"),
        "model_config": model_config,
        "model_state": model.state_dict(),
        "ema_model_state": ema.state_dict() if ema else None,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "data_config": data_config,
        "train_config": train_config,
        "split_identity": {
            "seed": split["seed"],
            "dataset_fingerprint": split["dataset_fingerprint"],
            "train_count": split["train_count"],
            "validation_count": split["validation_count"],
            "split_sha256": split_sha256,
        },
        "versions": {
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
        },
    }


def main() -> int:
    args = parse_args()
    if args.finalize and args.run_kind != "full":
        raise SystemExit("--finalize is allowed only with --run-kind full; smoke/overfit/ablation checkpoints are never final weights")
    if args.finalize and (args.max_train_pairs or args.max_val_pairs or args.overfit_pairs):
        raise SystemExit("--finalize refuses diagnostic/subset training; use the complete saved train/validation split")
    if args.overfit_pairs and args.run_kind != "overfit":
        raise SystemExit("--overfit-pairs requires --run-kind overfit")
    root = args.root.resolve()
    set_deterministic(args.seed, args.deterministic_algorithms)
    device = select_device(args.device)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    split = create_or_load_split(pairs, args.split, root, seed=args.seed, train_fraction=0.85)
    train_records, val_records = records_from_split(split, pairs)
    if args.overfit_pairs:
        train_records = train_records[: args.overfit_pairs]
        val_records = list(train_records)
    else:
        if args.max_train_pairs:
            train_records = train_records[: args.max_train_pairs]
        if args.max_val_pairs:
            val_records = val_records[: args.max_val_pairs]
    if not train_records or not val_records:
        raise RuntimeError("training and validation record lists must both be non-empty")

    run_dir = args.run_root.resolve() / args.experiment
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "width": args.width,
        "num_blocks": args.blocks,
        "num_hr_blocks": args.hr_blocks,
        "expansion": args.expansion,
        "dropout": args.dropout,
        "scale": 2,
        "input_representation": args.input_representation,
        "context_kernel": args.context_kernel,
        "context_dilation": args.context_dilation,
    }
    model = create_model(model_config).to(device)
    criterion = build_loss(args.loss).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    samples_per_epoch = args.steps_per_epoch * args.batch_size if args.steps_per_epoch else len(train_records)
    train_dataset = PairedPatchDataset(
        train_records,
        patch_size=args.patch_size,
        augment=True,
        seed=args.seed,
        samples_per_epoch=samples_per_epoch,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= args.batch_size,
        generator=generator,
        persistent_workers=args.workers > 0,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(round(args.warmup_epochs * len(train_loader)))
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps, args.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0, growth_interval=2000)
    ema = None if args.no_ema else ExponentialMovingAverage(model, args.ema_decay)

    start_epoch, global_step, best_psnr, best_ssim = 0, 0, -math.inf, -math.inf
    if args.resume:
        resume_path = checkpoint_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        checkpoint = torch_load_checkpoint(resume_path, map_location="cpu")
        resume_config = create_model(checkpoint.get("model_config", {})).model_config()
        if resume_config != model_config:
            raise RuntimeError(f"resume model config mismatch: {checkpoint.get('model_config')} vs {model_config}")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        if ema and checkpoint.get("ema_model_state"):
            ema.load_state_dict(checkpoint["ema_model_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_psnr = float(checkpoint.get("best_psnr", -math.inf))
        best_ssim = float(checkpoint.get("best_ssim", -math.inf))
        print(f"Resumed {resume_path} at epoch {start_epoch + 1}")

    data_config = {
        "noisy_dir": relative_or_absolute(noisy_dir, root),
        "gt_dir": relative_or_absolute(gt_dir, root),
        "split_path": relative_or_absolute(args.split, root),
        "patch_size_lr": args.patch_size,
        "augmentation": "joint H/V flip and rotations k*90 degrees",
        "raw_noisy_clipped": False,
        "input_representation": args.input_representation,
        "train_records_used": len(train_records),
        "validation_records_used": len(val_records),
    }
    train_config = {
        key: json_ready(value)
        for key, value in vars(args).items()
        if key not in {"root", "noisy_dir", "gt_dir", "split", "run_root", "final_weights"}
    }
    train_config.update(
        {
            "optimizer": "AdamW",
            "schedule": "linear warmup plus cosine decay",
            "amp_enabled": use_amp,
            "device": str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "run_root": relative_or_absolute(args.run_root.resolve(), PROJECT_ROOT),
        }
    )
    write_json(
        run_dir / "run_config.json",
        {
            "model_config": model_config,
            "data_config": data_config,
            "train_config": train_config,
            "split_identity": {
                "seed": split["seed"],
                "dataset_fingerprint": split["dataset_fingerprint"],
            },
        },
    )
    initial_probe = probe_loss(model, criterion, train_records[0], args.patch_size, device, use_amp)
    split_sha256 = hashlib.sha256(args.split.read_bytes()).hexdigest()
    print(
        f"Training {model.__class__.__name__}: {count_parameters(model):,} parameters, "
        f"{len(train_records)} train / {len(val_records)} validation, device={device}, AMP={use_amp}"
    )
    print(f"Initial fixed-patch {args.loss} loss: {initial_probe:.6f}")
    started_training = time.perf_counter()
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        epoch_started = time.perf_counter()
        train_dataset.set_epoch(epoch)
        model.train()
        loss_total = 0.0
        component_totals: dict[str, float] = {}
        batches = 0
        skipped_amp_steps = 0
        for noisy, gt, _keys in train_loader:
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(noisy)
            with torch.autocast(device_type=device.type, enabled=False):
                loss, components = criterion(prediction, gt)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, step {global_step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Explicit foreach=False avoids a Windows/PyTorch foreach-list
            # allocation failure observed after long CUDA epochs on hosts with
            # no pagefile. The scalar implementation is deterministic and the
            # parameter set is small enough that overhead is negligible.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=False)
            if not torch.isfinite(grad_norm):
                if not use_amp:
                    raise FloatingPointError(f"non-finite gradient norm at epoch {epoch + 1}, step {global_step}")
                # GradScaler records the overflow during unscale_ and skips the
                # optimizer update. Do not advance EMA or the LR schedule.
                scaler.step(optimizer)
                scaler.update()
                skipped_amp_steps += 1
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema:
                ema.update(model)
            global_step += 1
            batches += 1
            loss_total += float(loss.detach())
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + value

        should_validate = (epoch + 1) % args.validation_interval == 0 or epoch + 1 == args.epochs
        validation_psnr = validation_ssim = validation_seconds = None
        improved = False
        if should_validate:
            eval_model = ema.model if ema else model
            validation_psnr, validation_ssim, validation_seconds = evaluate_model(
                eval_model, val_records, device, use_amp, args.workers
            )
            improved = validation_psnr > best_psnr
            if improved:
                best_psnr, best_ssim = validation_psnr, validation_ssim
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        payload = checkpoint_payload(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_psnr=best_psnr,
            best_ssim=best_ssim,
            model_config=model_config,
            data_config=data_config,
            train_config=train_config,
            split=split,
            split_sha256=split_sha256,
        )
        atomic_torch_save(payload, checkpoint_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, checkpoint_dir / "best_psnr.pt")
        del payload
        gc.collect()
        elapsed = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": loss_total / max(1, batches),
            "charbonnier": component_totals.get("charbonnier", 0.0) / max(1, batches),
            "ssim_loss": component_totals.get("ssim_loss", 0.0) / max(1, batches),
            "edge_loss": component_totals.get("edge", 0.0) / max(1, batches),
            "fft_loss": component_totals.get("fft", 0.0) / max(1, batches),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_psnr": validation_psnr,
            "validation_ssim": validation_ssim,
            "validation_seconds": validation_seconds,
            "epoch_seconds": elapsed,
            "best_psnr": best_psnr,
            "improved": improved,
            "skipped_amp_steps": skipped_amp_steps,
        }
        append_csv(run_dir / "training_log.csv", row)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}: loss={row['train_loss']:.6f}, "
            f"PSNR={validation_psnr:.4f}, SSIM={validation_ssim:.6f}, "
            f"lr={row['learning_rate']:.3e}, {elapsed:.1f}s"
            if should_validate
            else f"Epoch {epoch + 1:03d}/{args.epochs}: loss={row['train_loss']:.6f}, lr={row['learning_rate']:.3e}, {elapsed:.1f}s"
        )
        if should_validate and args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping after {epochs_without_improvement} validation intervals without PSNR improvement")
            break

    final_probe = probe_loss(ema.model if ema else model, criterion, train_records[0], args.patch_size, device, use_amp)
    elapsed_training = time.perf_counter() - started_training
    completion = {
        "run_kind": args.run_kind,
        "epochs_completed_this_invocation": max(0, last_epoch - start_epoch + 1),
        "last_epoch": last_epoch + 1,
        "global_step": global_step,
        "initial_fixed_patch_loss": initial_probe,
        "final_fixed_patch_loss": final_probe,
        "fixed_patch_loss_ratio": final_probe / initial_probe if initial_probe else None,
        "best_validation_psnr": best_psnr,
        "best_validation_ssim": best_ssim,
        "training_wall_seconds": elapsed_training,
        "best_checkpoint": "checkpoints/best_psnr.pt",
        "last_checkpoint": "checkpoints/last.pt",
        "published_final_weights": False,
    }
    if args.run_kind == "overfit":
        write_json(run_dir / "overfit_result.json", completion)

    if args.finalize:
        bicubic_path = root / "reports" / "bicubic_summary.json"
        if not bicubic_path.is_file():
            raise RuntimeError("refusing to publish final weights: reports/bicubic_summary.json is missing")
        bicubic = json.loads(bicubic_path.read_text(encoding="utf-8"))
        baseline_psnr = float(bicubic["metrics"]["combined"]["psnr"]["mean"])
        if not best_psnr > baseline_psnr:
            raise RuntimeError(
                f"refusing to publish final weights: model PSNR {best_psnr:.4f} does not beat bicubic {baseline_psnr:.4f}"
            )
        best_checkpoint = torch_load_checkpoint(checkpoint_dir / "best_psnr.pt", map_location="cpu")
        inference_bundle = {
            "format_version": 1,
            "architecture": "RangeAwareLiteNAFSR",
            "training_complete": True,
            "experiment_name": args.experiment,
            "epoch": int(best_checkpoint["epoch"]) + 1,
            "global_step": int(best_checkpoint.get("global_step", 0)),
            "model_config": best_checkpoint["model_config"],
            "model_state": best_checkpoint.get("ema_model_state") or best_checkpoint["model_state"],
            "ema_model_state": None,
            "training_kind": "full",
            "validation_metrics": {
                "psnr": best_checkpoint["best_psnr"],
                "ssim": best_checkpoint["best_ssim"],
                "bicubic_psnr": baseline_psnr,
            },
            "split_identity": best_checkpoint["split_identity"],
            "data_config": best_checkpoint["data_config"],
            "train_config": best_checkpoint["train_config"],
            "versions": best_checkpoint["versions"],
        }
        atomic_torch_save(inference_bundle, args.final_weights)
        completion["published_final_weights"] = True
        completion["final_weights"] = relative_or_absolute(args.final_weights, root)
        print(f"Published inference weights: {args.final_weights}")
    write_json(run_dir / "training_summary.json", completion)
    print(
        f"Training complete: best PSNR={best_psnr:.4f}, best SSIM={best_ssim:.6f}, "
        f"fixed-patch loss {initial_probe:.6f} -> {final_probe:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
