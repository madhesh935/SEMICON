#!/usr/bin/env python
"""Manual proxy scoring for the untested 256x512 path (no real 512x512 GT exists).

The provided KLA dataset contains zero 512x512 arrays: every train GT file is
256x256 and every train/test NoisyLR file is 128x128 (see
reports/degradation_analysis/summary.md, "Paired 256x256 -> 512x512 samples:
0"). A literal PSNR/SSIM against real ground truth is therefore impossible at
this scale, and fabricating one (e.g. bicubic-upscaling a 256 GT to pretend
it is a 512 target) would just reward blurriness, not accuracy.

This script instead runs two honest proxies against real 256x256 GT images
from the held-out seed-42 validation split -- it never invents a fake 512
ground truth.

Protocol A - clean cycle-consistency (no synthetic noise):
    real GT_256 --model(256->512)--> Restored_512
    --area-downsample--> Restored_256, scored against the same real GT_256.
  Baseline: bicubic-up(GT_256) --area-downsample--> vs GT_256.
  Tests whether the model corrupts/hallucinates on an input scale it was
  never trained on. Does not test denoising (input has no noise).

Protocol B - synthetic-noise proxy (explicitly synthetic, not official):
    real GT_256 + synthetic Gaussian noise (std matched to the measured
    LR-discrepancy std of 0.0905 from degradation_analysis/summary.md)
    --model(256->512)--> Restored_512 --area-downsample--> Restored_256,
    scored against real GT_256.
  Baseline: same noisy input through plain bicubic 2x up + area-downsample.
  Exercises joint denoise+SR at the untested scale, but the noise is
  synthetic, so this is a proxy, not a measurement against real degradation.

Neither protocol is a substitute for the real, GT-scored 128->256 numbers in
reports/validation_summary.json. Results are a generalization signal only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from restoration.io import load_grayscale_npy
from restoration.metrics import psnr, ssim
from restoration.utils import (
    PROJECT_ROOT,
    load_model_from_checkpoint,
    refuse_existing_outputs,
    select_device,
    set_deterministic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-dir", type=Path, default=PROJECT_ROOT / "Dataset" / "train" / "train" / "GT")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "reports" / "manual_256_to_512_proxy.json")
    parser.add_argument("--max-images", type=int, default=None, help="Diagnostic subset only; omit for all 480")
    parser.add_argument("--noise-std", type=float, default=0.0905, help="Protocol B synthetic noise std")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --report-path")
    return parser.parse_args()


def area_downsample(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="area")


def summarize(values: list[float]) -> dict[str, float]:
    finite = [v for v in values if np.isfinite(v)]
    return {
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def main() -> int:
    args = parse_args()
    refuse_existing_outputs([args.report_path], force=args.force, script="score_256_to_512_proxy.py")
    set_deterministic(args.seed)
    device = select_device(args.device)

    split = json.loads(args.split.read_text(encoding="utf-8"))
    val_keys = split["validation_keys"]
    if args.max_images:
        val_keys = val_keys[: args.max_images]

    model, _ = load_model_from_checkpoint(args.weights, device, prefer_ema=True)

    @torch.inference_mode()
    def run_model(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
        return out.float().clamp(0.0, 1.0)

    rows = []
    print(f"Scoring {len(val_keys)} real 256x256 GT images through the untested 256->512 path on {device}")
    for i, key in enumerate(val_keys):
        gt = load_grayscale_npy(args.gt_dir / f"{key}.npy", mmap_mode=None)
        gt_t = torch.from_numpy(np.asarray(gt, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        restored_512_a = run_model(gt_t)
        restored_256_a = area_downsample(restored_512_a, 256).clamp(0, 1)
        bicubic_512_a = F.interpolate(gt_t, scale_factor=2.0, mode="bicubic", align_corners=False).clamp(0, 1)
        bicubic_256_a = area_downsample(bicubic_512_a, 256).clamp(0, 1)

        noisy = gt_t + torch.randn_like(gt_t) * args.noise_std
        restored_512_b = run_model(noisy)
        restored_256_b = area_downsample(restored_512_b, 256).clamp(0, 1)
        bicubic_512_b = F.interpolate(noisy.clamp(0, 1), scale_factor=2.0, mode="bicubic", align_corners=False).clamp(0, 1)
        bicubic_256_b = area_downsample(bicubic_512_b, 256).clamp(0, 1)

        gt_np = np.asarray(gt, dtype=np.float32)
        rows.append(
            {
                "key": key,
                "a_model_psnr": psnr(restored_256_a[0, 0].cpu().numpy(), gt_np),
                "a_model_ssim": ssim(restored_256_a[0, 0].cpu().numpy(), gt_np),
                "a_bicubic_psnr": psnr(bicubic_256_a[0, 0].cpu().numpy(), gt_np),
                "a_bicubic_ssim": ssim(bicubic_256_a[0, 0].cpu().numpy(), gt_np),
                "b_model_psnr": psnr(restored_256_b[0, 0].cpu().numpy(), gt_np),
                "b_model_ssim": ssim(restored_256_b[0, 0].cpu().numpy(), gt_np),
                "b_bicubic_psnr": psnr(bicubic_256_b[0, 0].cpu().numpy(), gt_np),
                "b_bicubic_ssim": ssim(bicubic_256_b[0, 0].cpu().numpy(), gt_np),
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(val_keys)}")

    fields = (
        "a_model_psnr", "a_model_ssim", "a_bicubic_psnr", "a_bicubic_ssim",
        "b_model_psnr", "b_model_ssim", "b_bicubic_psnr", "b_bicubic_ssim",
    )
    summary = {
        "protocol": {
            "a": "clean cycle-consistency: real GT_256 -> model(256->512) -> area-downsample(256) vs real GT_256",
            "b": "synthetic-noise proxy: real GT_256 + Gaussian(std=noise_std) -> model(256->512) -> area-downsample(256) vs real GT_256",
        },
        "count": len(rows),
        "noise_std_protocol_b": args.noise_std,
        "not_a_substitute_for": "reports/validation_summary.json (real GT-scored 128->256 numbers)",
        **{field: summarize([row[field] for row in rows]) for field in fields},
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps({"summary": summary, "per_image": rows}, indent=2) + "\n", encoding="utf-8")

    print(f"\nProtocol A (clean cycle) - model PSNR {summary['a_model_psnr']['mean']:.3f} dB "
          f"vs bicubic {summary['a_bicubic_psnr']['mean']:.3f} dB")
    print(f"Protocol B (synthetic-noise proxy) - model PSNR {summary['b_model_psnr']['mean']:.3f} dB "
          f"vs bicubic {summary['b_bicubic_psnr']['mean']:.3f} dB "
          f"(gain {summary['b_model_psnr']['mean'] - summary['b_bicubic_psnr']['mean']:+.3f} dB)")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
