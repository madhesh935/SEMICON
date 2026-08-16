#!/usr/bin/env python
"""Discover and fully audit KLA PS01 NumPy image data without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from restoration.data import build_pair_index, create_or_load_split, discover_dataset
from restoration.io import is_metadata_path, normalize_array_shape
from restoration.utils import PROJECT_ROOT, refuse_existing_outputs, relative_or_absolute, write_csv, write_json


PERCENTILES = (0.0, 0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project/data search root")
    parser.add_argument("--noisy-dir", type=Path, help="Optional explicit training NoisyLR directory")
    parser.add_argument("--gt-dir", type=Path, help="Optional explicit training GT directory")
    parser.add_argument("--test-dir", type=Path, action="append", help="Optional test directory; repeatable")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument(
        "--global-samples-per-file",
        type=int,
        default=1024,
        help="Deterministic evenly spaced samples per file for approximate global percentiles",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing report files at --report-dir")
    return parser.parse_args()


def valid_npy_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*.npy")
        if path.is_file() and not is_metadata_path(path)
    )


def array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def scan_role(
    role: str,
    directory: Path,
    root: Path,
    sample_count: int,
    partner_by_path: dict[Path, str] | None = None,
) -> tuple[list[dict], dict, dict[str, list[str]]]:
    rows: list[dict] = []
    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    hashes: dict[str, list[str]] = {}
    samples: list[np.ndarray] = []
    finite_count = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    global_min, global_max = math.inf, -math.inf
    outside_files = outside_values = nan_total = inf_total = 0
    below_zero_files = below_zero_values = 0
    above_one_files = above_one_values = 0
    gt_tolerance_violations = 0
    files = valid_npy_files(directory)
    for path in files:
        relative = relative_or_absolute(path, root)
        base = {
            "role": role,
            "path": relative,
            "partner": (partner_by_path or {}).get(path.resolve(), ""),
            "valid": False,
            "error": "",
        }
        try:
            raw = np.load(path, mmap_mode="r", allow_pickle=False)
            original_shape = tuple(int(value) for value in raw.shape)
            original_dtype = str(raw.dtype)
            image = normalize_array_shape(raw, path)
            shapes[str(original_shape)] += 1
            dtypes[original_dtype] += 1
            flat = np.asarray(image).reshape(-1)
            nan_count = int(np.isnan(flat).sum()) if np.issubdtype(flat.dtype, np.floating) else 0
            inf_count = int(np.isinf(flat).sum()) if np.issubdtype(flat.dtype, np.floating) else 0
            finite_mask = np.isfinite(flat)
            finite = flat[finite_mask].astype(np.float64, copy=False)
            if finite.size == 0:
                raise ValueError("array has no finite values")
            file_min, file_max = float(finite.min()), float(finite.max())
            file_sum = float(finite.sum(dtype=np.float64))
            file_sum_sq = float(np.square(finite).sum(dtype=np.float64))
            file_mean = file_sum / finite.size
            file_std = math.sqrt(max(0.0, file_sum_sq / finite.size - file_mean**2))
            percentiles = np.percentile(finite, PERCENTILES)
            below_zero = int(np.count_nonzero(finite < 0.0))
            above_one = int(np.count_nonzero(finite > 1.0))
            outside = below_zero + above_one
            tolerance_outside = int(np.count_nonzero((finite < -1e-6) | (finite > 1.0 + 1e-6)))
            if role == "train_gt" and tolerance_outside:
                gt_tolerance_violations += 1
            if outside:
                outside_files += 1
            if below_zero:
                below_zero_files += 1
            if above_one:
                above_one_files += 1
            outside_values += outside
            below_zero_values += below_zero
            above_one_values += above_one
            nan_total += nan_count
            inf_total += inf_count
            finite_count += int(finite.size)
            total_sum += file_sum
            total_sum_sq += file_sum_sq
            global_min = min(global_min, file_min)
            global_max = max(global_max, file_max)
            take = min(sample_count, finite.size)
            if take:
                indices = np.linspace(0, finite.size - 1, take, dtype=np.int64)
                samples.append(finite[indices].astype(np.float32, copy=True))
            digest = array_hash(np.asarray(image))
            hashes.setdefault(digest, []).append(relative)
            row = {
                **base,
                "valid": nan_count == 0 and inf_count == 0,
                "shape": str(original_shape),
                "canonical_shape": str(tuple(image.shape)),
                "dtype": original_dtype,
                "values": int(flat.size),
                "min": file_min,
                "max": file_max,
                "mean": file_mean,
                "std": file_std,
                "p0": float(percentiles[0]),
                "p0_1": float(percentiles[1]),
                "p1": float(percentiles[2]),
                "p5": float(percentiles[3]),
                "p25": float(percentiles[4]),
                "p50": float(percentiles[5]),
                "p75": float(percentiles[6]),
                "p95": float(percentiles[7]),
                "p99": float(percentiles[8]),
                "p99_9": float(percentiles[9]),
                "p100": float(percentiles[10]),
                "outside_0_1": outside,
                "below_zero": below_zero,
                "above_one": above_one,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "sha256_canonical": digest,
            }
            if nan_count or inf_count:
                row["error"] = f"non-finite values: NaN={nan_count}, Inf={inf_count}"
            rows.append(row)
        except Exception as exc:
            base["error"] = str(exc)
            rows.append(base)
    mean = total_sum / finite_count if finite_count else math.nan
    std = math.sqrt(max(0.0, total_sum_sq / finite_count - mean**2)) if finite_count else math.nan
    sampled = np.concatenate(samples) if samples else np.array([], dtype=np.float32)
    sampled_percentiles = {}
    if sampled.size:
        values = np.percentile(sampled, PERCENTILES)
        sampled_percentiles = {str(p): float(v) for p, v in zip(PERCENTILES, values)}
        sampled_percentiles["0.0"] = global_min
        sampled_percentiles["100.0"] = global_max
    summary = {
        "directory": relative_or_absolute(directory, root),
        "file_count": len(files),
        "valid_file_count": sum(bool(row.get("valid")) for row in rows),
        "invalid_file_count": sum(not bool(row.get("valid")) for row in rows),
        "shape_distribution": dict(shapes),
        "dtype_distribution": dict(dtypes),
        "finite_value_count": finite_count,
        "min": global_min if finite_count else None,
        "max": global_max if finite_count else None,
        "mean": mean if finite_count else None,
        "std": std if finite_count else None,
        "global_percentiles": sampled_percentiles,
        "global_percentile_method": f"deterministic evenly spaced sample, up to {sample_count} values per file; extrema are exact",
        "files_with_values_outside_0_1": outside_files,
        "values_outside_0_1": outside_values,
        "files_with_values_below_zero": below_zero_files,
        "values_below_zero": below_zero_values,
        "files_with_values_above_one": above_one_files,
        "values_above_one": above_one_values,
        "nan_count": nan_total,
        "inf_count": inf_total,
        "gt_files_outside_tolerance_minus1e-6_to_1plus1e-6": gt_tolerance_violations if role == "train_gt" else None,
        "duplicate_extra_copies": sum(len(group) - 1 for group in hashes.values() if len(group) > 1),
    }
    return rows, summary, hashes


def main() -> int:
    args = parse_args()
    refuse_existing_outputs(
        [args.report_dir / "dataset_audit.json", args.report_dir / "dataset_audit.csv"],
        force=args.force,
        script="audit_dataset.py",
    )
    root = args.root.resolve()
    if (args.noisy_dir is None) != (args.gt_dir is None):
        raise SystemExit("--noisy-dir and --gt-dir must be provided together")
    if args.noisy_dir:
        noisy_dir, gt_dir = args.noisy_dir.resolve(), args.gt_dir.resolve()
        test_dirs = tuple(path.resolve() for path in (args.test_dir or []))
    else:
        locations = discover_dataset(root)
        noisy_dir, gt_dir = locations.noisy_dir, locations.gt_dir
        test_dirs = tuple(path.resolve() for path in args.test_dir) if args.test_dir else locations.test_dirs

    pairs, missing_gt, missing_noisy = build_pair_index(noisy_dir, gt_dir, verify_shapes=True)
    noisy_partner = {Path(pair.noisy).resolve(): relative_or_absolute(Path(pair.gt), root) for pair in pairs}
    gt_partner = {Path(pair.gt).resolve(): relative_or_absolute(Path(pair.noisy), root) for pair in pairs}
    all_rows: list[dict] = []
    noisy_rows, noisy_summary, train_hashes = scan_role(
        "train_noisy", noisy_dir, root, args.global_samples_per_file, noisy_partner
    )
    gt_rows, gt_summary, _ = scan_role("train_gt", gt_dir, root, args.global_samples_per_file, gt_partner)
    all_rows.extend(noisy_rows)
    all_rows.extend(gt_rows)
    test_summaries = []
    test_hashes: dict[str, list[str]] = {}
    for index, test_dir in enumerate(test_dirs):
        rows, summary, hashes = scan_role(
            f"test_noisy_{index + 1}", test_dir, root, args.global_samples_per_file
        )
        all_rows.extend(rows)
        test_summaries.append(summary)
        for digest, paths in hashes.items():
            test_hashes.setdefault(digest, []).extend(paths)

    cross_hashes = sorted(set(train_hashes) & set(test_hashes))
    train_stems = {Path(path).stem.casefold() for group in train_hashes.values() for path in group}
    test_stems = {Path(path).stem.casefold() for group in test_hashes.values() for path in group}
    archives = []
    ignored_walk_dirs = {".venv", ".git", ".tmp", "__pycache__", ".pytest_cache", "runs", "restored_test_outputs"}
    archive_suffixes = (".zip", ".7z", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
    for current, directory_names, file_names in os.walk(root):
        directory_names[:] = [name for name in directory_names if name not in ignored_walk_dirs and name != "__MACOSX"]
        current_path = Path(current)
        for name in file_names:
            path = current_path / name
            if any(name.lower().endswith(suffix) for suffix in archive_suffixes):
                archives.append(relative_or_absolute(path, root))
    archives.sort()
    split = create_or_load_split(
        pairs,
        args.split,
        root,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    report = {
        "format_version": 1,
        "project_root": ".",
        "raw_data_modified": False,
        "archives_discovered": archives,
        "training_noisy_directories": [relative_or_absolute(noisy_dir, root)],
        "ground_truth_directories": [relative_or_absolute(gt_dir, root)],
        "test_directories": [relative_or_absolute(path, root) for path in test_dirs],
        "pairing": {
            "method": "exact case-insensitive relative path stem plus verified 2x shape",
            "matched_pairs": len(pairs),
            "missing_gt_count": len(missing_gt),
            "missing_gt": missing_gt,
            "missing_noisy_count": len(missing_noisy),
            "missing_noisy": missing_noisy,
            "all_pairs_exactly_2x": True,
        },
        "roles": {
            "train_noisy": noisy_summary,
            "train_gt": gt_summary,
            "test": test_summaries,
        },
        "leakage_and_duplicates": {
            "exact_train_test_canonical_hash_matches": len(cross_hashes),
            "exact_train_test_examples": [
                {"train": train_hashes[digest], "test": test_hashes[digest]}
                for digest in cross_hashes[:20]
            ],
            "overlapping_filename_stem_count": len(train_stems & test_stems),
            "note": "stem overlap alone is not content leakage; canonical hashes are compared",
        },
        "split": {
            "path": relative_or_absolute(args.split, root),
            "seed": split["seed"],
            "strategy": split["strategy"],
            "train_count": split["train_count"],
            "validation_count": split["validation_count"],
            "dataset_fingerprint": split["dataset_fingerprint"],
        },
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.report_dir / "dataset_audit.json", report)
    fields = [
        "role", "path", "partner", "valid", "error", "shape", "canonical_shape", "dtype", "values",
        "min", "max", "mean", "std", "p0", "p0_1", "p1", "p5", "p25", "p50", "p75", "p95",
        "p99", "p99_9", "p100", "outside_0_1", "nan_count", "inf_count", "sha256_canonical",
    ]
    write_csv(args.report_dir / "dataset_audit.csv", all_rows, fields)

    print(f"Project root: {root}")
    print(f"Training NoisyLR: {noisy_dir} ({noisy_summary['file_count']} files)")
    print(f"Training GT:      {gt_dir} ({gt_summary['file_count']} files)")
    print(f"Matched pairs: {len(pairs)}; missing GT: {len(missing_gt)}; missing NoisyLR: {len(missing_noisy)}")
    print(f"Pair geometry: all {len(pairs)} pairs are exactly 128/256-compatible 2x relationships")
    print(
        "Noisy range: "
        f"[{noisy_summary['min']:.6f}, {noisy_summary['max']:.6f}], "
        f"out-of-range files={noisy_summary['files_with_values_outside_0_1']}, "
        f"NaN={noisy_summary['nan_count']}, Inf={noisy_summary['inf_count']}"
    )
    print(
        "GT range:    "
        f"[{gt_summary['min']:.6f}, {gt_summary['max']:.6f}], "
        f"tolerance violations={gt_summary['gt_files_outside_tolerance_minus1e-6_to_1plus1e-6']}, "
        f"NaN={gt_summary['nan_count']}, Inf={gt_summary['inf_count']}"
    )
    print(f"Test files: {sum(item['file_count'] for item in test_summaries)} across {len(test_dirs)} director(ies)")
    print(f"Exact train/test duplicates: {len(cross_hashes)}")
    print(f"Split: {split['train_count']} train / {split['validation_count']} validation, seed {split['seed']}")
    print(f"Wrote {args.report_dir / 'dataset_audit.json'}")
    print(f"Wrote {args.report_dir / 'dataset_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
