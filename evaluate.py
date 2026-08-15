#!/usr/bin/env python
"""Standalone offline inference for KLA PS01 grayscale `.npy` images."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Inference only uses NumPy for array I/O. Prevent OpenBLAS from reserving a
# large worker pool during import on memory-constrained evaluator hosts.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch

from restoration.io import inspect_npy_shape, is_metadata_path, load_grayscale_npy, save_restored_npy
from restoration.utils import PROJECT_ROOT, inference_with_tta, load_model_from_checkpoint, select_device, synchronize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir_pos", nargs="?", type=Path, help="Test-images directory")
    parser.add_argument("output_dir_pos", nargs="?", type=Path, help="Restored-output directory")
    parser.add_argument("--input_dir", "--input-dir", dest="input_dir", type=Path, help="Test-images directory")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, help="Restored-output directory")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--batch-size", type=int, default=8, help="Batches contain only identical input shapes")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA float16 autocast")
    parser.add_argument("--tta", action="store_true", help="Enable slower four-rotation self-ensemble")
    return parser.parse_args()


def resolve_cli_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_dir = args.input_dir or args.input_dir_pos
    output_dir = args.output_dir or args.output_dir_pos
    if input_dir is None or output_dir is None:
        raise ValueError("provide input and output directories either positionally or with --input_dir/--output_dir")
    if args.input_dir and args.input_dir_pos and args.input_dir.resolve() != args.input_dir_pos.resolve():
        raise ValueError("conflicting positional and --input_dir values")
    if args.output_dir and args.output_dir_pos and args.output_dir.resolve() != args.output_dir_pos.resolve():
        raise ValueError("conflicting positional and --output_dir values")
    weights = args.weights
    if not weights.is_absolute() and not weights.exists():
        candidate = PROJECT_ROOT / weights
        if candidate.exists():
            weights = candidate
    return input_dir.resolve(), output_dir.resolve(), weights.resolve()


def collect_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    return sorted(
        path
        for path in input_dir.rglob("*.npy")
        if path.is_file() and not is_metadata_path(path)
    )


def run() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    input_dir, output_dir, weights = resolve_cli_paths(args)
    wall_started = time.perf_counter()
    device = select_device(args.device)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    model_load_started = time.perf_counter()
    model, checkpoint = load_model_from_checkpoint(weights, device)
    model_load_seconds = time.perf_counter() - model_load_started
    paths = collect_inputs(input_dir)
    if not paths:
        raise FileNotFoundError(f"no .npy inputs found recursively under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    failures: list[tuple[Path, str]] = []
    for path in paths:
        try:
            groups[inspect_npy_shape(path)].append(path)
        except Exception as exc:
            failures.append((path, str(exc)))

    processed = 0
    model_seconds = 0.0
    with torch.inference_mode():
        for shape in sorted(groups):
            group = groups[shape]
            for start in range(0, len(group), args.batch_size):
                batch_paths = group[start : start + args.batch_size]
                loaded_paths, arrays = [], []
                for path in batch_paths:
                    try:
                        image = load_grayscale_npy(path, mmap_mode="r", require_finite=True)
                        if tuple(image.shape) != shape:
                            raise ValueError(f"shape changed between header inspection and load: {image.shape} vs {shape}")
                        arrays.append(image)
                        loaded_paths.append(path)
                    except Exception as exc:
                        failures.append((path, str(exc)))
                if not arrays:
                    continue
                tensor = torch.from_numpy(np.stack(arrays, axis=0)).unsqueeze(1).to(device)
                synchronize(device)
                inference_started = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    prediction = inference_with_tta(model, tensor, args.tta)
                synchronize(device)
                model_seconds += time.perf_counter() - inference_started
                prediction = prediction.float().clamp(0.0, 1.0).cpu().numpy()[:, 0]
                expected_hw = (shape[0] * 2, shape[1] * 2)
                if prediction.shape[1:] != expected_hw:
                    for path in loaded_paths:
                        failures.append((path, f"model returned {prediction.shape[1:]}, expected {expected_hw}"))
                    continue
                for path, restored in zip(loaded_paths, prediction):
                    try:
                        relative = path.relative_to(input_dir)
                        save_restored_npy(output_dir / relative, restored, expected_hw)
                        processed += 1
                    except Exception as exc:
                        failures.append((path, str(exc)))

    wall_seconds = time.perf_counter() - wall_started
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"Model: {weights}")
    print(f"Architecture: {checkpoint.get('architecture')}; AMP={use_amp}; TTA={args.tta}")
    print(f"Inputs discovered: {len(paths)}; processed: {processed}; failures: {len(failures)}")
    print(f"Model load time: {model_load_seconds:.3f} s")
    print(f"Total wall time: {wall_seconds:.3f} s")
    print(f"Model inference time: {model_seconds:.3f} s")
    print(f"Average model time: {(1000.0 * model_seconds / processed):.3f} ms/image" if processed else "Average model time: n/a")
    print(f"Output directory: {output_dir}")
    if failures:
        for path, message in failures[:20]:
            print(f"FAILED {path}: {message}", file=sys.stderr)
        if len(failures) > 20:
            print(f"... {len(failures) - 20} additional failures", file=sys.stderr)
        return 1
    if processed != len(paths):
        print(f"FAILED output count mismatch: processed {processed} of {len(paths)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
