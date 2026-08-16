#!/usr/bin/env python
"""Measure training-pair degradations and build a train-fitted OOD proxy split.

The official test set is never opened by this script. Descriptor normalization,
covariance, PCA, and nearest-neighbour reference distributions are fitted on the
saved training fold only. Validation input descriptors are used only to select a
structurally unusual validation subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree

from restoration.data import build_pair_index, create_or_load_split, discover_dataset, records_from_split
from restoration.io import load_grayscale_npy
from restoration.utils import PROJECT_ROOT, select_device


DESCRIPTOR_NAMES = [
    "mean",
    "std",
    "p01",
    "p50",
    "p99",
    "below_zero_fraction",
    "above_one_fraction",
    "oor_abs_mean",
    "gradient_mean",
    "gradient_std",
    "gradient_p90",
    "edge_density_005",
    "edge_density_010",
    *[f"orientation_{index}" for index in range(8)],
    "fft_band_0",
    "fft_band_1",
    "fft_band_2",
    "fft_band_3",
    "fft_anisotropy",
    "fft_peak_fraction",
    "foreground_fraction_025",
    "foreground_fraction_050",
    "foreground_fraction_075",
    "components_per_megapixel",
    "component_area_mean",
    "component_area_max_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Data search root")
    parser.add_argument("--noisy-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits" / "split_seed42.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "degradation_analysis",
    )
    parser.add_argument(
        "--validation-subsets",
        type=Path,
        default=None,
        help=(
            "Where to write the OOD-proxy validation subset JSON. Defaults to "
            "<output-dir>/validation_subsets_seed42.json, so a diagnostic run "
            "(e.g. a different --output-dir) never touches the committed "
            "splits/validation_subsets_seed42.json unless you point here explicitly."
        ),
    )
    parser.add_argument("--ood-fraction", type=float, default=0.25)
    parser.add_argument("--fft-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@dataclass
class Moments:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not array.size:
            return
        self.count += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.total_sq += float(np.square(array).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def summary(self) -> dict:
        if not self.count:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class PairMoments:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64).reshape(-1)
        y64 = np.asarray(y, dtype=np.float64).reshape(-1)
        if x64.size != y64.size:
            raise ValueError("paired moment arrays have different sizes")
        if not x64.size:
            return
        self.count += int(x64.size)
        self.sum_x += float(x64.sum(dtype=np.float64))
        self.sum_y += float(y64.sum(dtype=np.float64))
        self.sum_xx += float(np.square(x64).sum(dtype=np.float64))
        self.sum_yy += float(np.square(y64).sum(dtype=np.float64))
        self.sum_xy += float(np.multiply(x64, y64).sum(dtype=np.float64))

    def correlation(self) -> float | None:
        if self.count < 2:
            return None
        covariance = self.sum_xy - self.sum_x * self.sum_y / self.count
        variance_x = self.sum_xx - self.sum_x * self.sum_x / self.count
        variance_y = self.sum_yy - self.sum_y * self.sum_y / self.count
        denominator = math.sqrt(max(0.0, variance_x) * max(0.0, variance_y))
        return covariance / denominator if denominator > 0 else None


class BinnedMoments:
    def __init__(self, edges: np.ndarray):
        self.edges = np.asarray(edges, dtype=np.float64)
        bins = len(self.edges) - 1
        self.count = np.zeros(bins, dtype=np.int64)
        self.total = np.zeros(bins, dtype=np.float64)
        self.total_sq = np.zeros(bins, dtype=np.float64)

    def update(self, signal: np.ndarray, residual: np.ndarray) -> None:
        x = np.asarray(signal, dtype=np.float64).reshape(-1)
        y = np.asarray(residual, dtype=np.float64).reshape(-1)
        indices = np.searchsorted(self.edges, x, side="right") - 1
        indices = np.clip(indices, 0, len(self.count) - 1)
        self.count += np.bincount(indices, minlength=len(self.count))
        self.total += np.bincount(indices, weights=y, minlength=len(self.count))
        self.total_sq += np.bincount(indices, weights=y * y, minlength=len(self.count))

    def rows(self) -> list[dict]:
        result = []
        for index, count in enumerate(self.count):
            mean = self.total[index] / count if count else math.nan
            variance = max(0.0, self.total_sq[index] / count - mean * mean) if count else math.nan
            result.append(
                {
                    "bin_low": float(self.edges[index]),
                    "bin_high": float(self.edges[index + 1]),
                    "bin_center": float((self.edges[index] + self.edges[index + 1]) / 2),
                    "count": int(count),
                    "mean": float(mean),
                    "variance": float(variance),
                    "std": float(math.sqrt(variance)) if count else math.nan,
                }
            )
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_descriptor(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    clipped = np.clip(raw, 0.0, 1.0)
    percentiles = np.percentile(clipped, [1, 50, 99])
    oor = raw - clipped
    gx = ndimage.sobel(clipped, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(clipped, axis=0, mode="reflect") / 8.0
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    orientation_hist, _ = np.histogram(
        orientation,
        bins=np.linspace(0.0, np.pi, 9),
        weights=magnitude,
    )
    orientation_hist = orientation_hist.astype(np.float64)
    orientation_hist /= max(float(orientation_hist.sum()), 1e-12)

    centered = clipped.astype(np.float64) - float(clipped.mean())
    spectrum = np.fft.fft2(centered)
    power = np.abs(spectrum) ** 2
    fy = np.fft.fftfreq(clipped.shape[0])[:, None]
    fx = np.fft.fftfreq(clipped.shape[1])[None, :]
    radius = np.hypot(fx, fy)
    theta = np.arctan2(fy, fx)
    power[0, 0] = 0.0
    power_total = max(float(power.sum()), 1e-12)
    band_edges = (0.0, 0.08, 0.16, 0.30, math.inf)
    band_ratios = [
        float(power[(radius >= low) & (radius < high)].sum() / power_total)
        for low, high in zip(band_edges[:-1], band_edges[1:])
    ]
    anisotropy = abs(np.sum(power * np.exp(2j * theta))) / power_total
    peak_fraction = float(power.max() / power_total)

    smoothed = ndimage.gaussian_filter(clipped, sigma=1.0, mode="reflect")
    mask = smoothed > 0.5
    labels, component_count = ndimage.label(mask)
    areas = np.bincount(labels.reshape(-1))[1:] if component_count else np.empty(0, dtype=np.int64)
    component_area_mean = float(areas.mean()) if areas.size else 0.0
    component_area_max_fraction = float(areas.max() / mask.size) if areas.size else 0.0
    oor_values = np.abs(oor[oor != 0])

    values = [
        float(clipped.mean()),
        float(clipped.std()),
        *[float(value) for value in percentiles],
        float(np.mean(raw < 0.0)),
        float(np.mean(raw > 1.0)),
        float(oor_values.mean()) if oor_values.size else 0.0,
        float(magnitude.mean()),
        float(magnitude.std()),
        float(np.percentile(magnitude, 90)),
        float(np.mean(magnitude > 0.05)),
        float(np.mean(magnitude > 0.10)),
        *orientation_hist.tolist(),
        *band_ratios,
        float(anisotropy),
        peak_fraction,
        float(np.mean(smoothed > 0.25)),
        float(np.mean(mask)),
        float(np.mean(smoothed > 0.75)),
        float(component_count * 1_000_000.0 / mask.size),
        component_area_mean,
        component_area_max_fraction,
    ]
    descriptor = np.asarray(values, dtype=np.float64)
    if descriptor.size != len(DESCRIPTOR_NAMES) or not np.isfinite(descriptor).all():
        raise RuntimeError("invalid structural descriptor")
    return descriptor


def fit_weighted_polynomials(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["count"] > 0 and math.isfinite(row["variance"])]
    x = np.asarray([row["bin_center"] for row in valid], dtype=np.float64)
    y = np.asarray([row["variance"] for row in valid], dtype=np.float64)
    weights = np.asarray([row["count"] for row in valid], dtype=np.float64)
    weights /= max(float(weights.sum()), 1.0)
    weighted_mean = float(np.sum(weights * y))
    baseline_sse = float(np.sum(weights * np.square(y - weighted_mean)))
    models = {}
    for degree, name in ((0, "constant"), (1, "linear"), (2, "quadratic")):
        coefficients = np.polyfit(x, y, degree, w=np.sqrt(weights))
        prediction = np.polyval(coefficients, x)
        sse = float(np.sum(weights * np.square(y - prediction)))
        models[name] = {
            "degree": degree,
            "coefficients_high_to_low": coefficients.tolist(),
            "weighted_rmse": math.sqrt(max(0.0, sse)),
            "weighted_r2_vs_constant": 1.0 - sse / baseline_sse if baseline_sse > 0 else 0.0,
        }
    positive = y[y > 0]
    variance_ratio = float(positive.max() / positive.min()) if positive.size else None
    return {"models": models, "max_to_min_binned_variance_ratio": variance_ratio}


def radial_spectrum(image: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, float, float]:
    centered = np.asarray(image, dtype=np.float64) - float(np.mean(image))
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    magnitude = np.abs(spectrum)
    power = magnitude * magnitude
    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))[None, :]
    radius = np.hypot(fx, fy)
    theta = np.arctan2(fy, fx)
    center = (image.shape[0] // 2, image.shape[1] // 2)
    power[center] = 0.0
    magnitude[center] = 0.0
    indices = np.clip(np.searchsorted(edges, radius.reshape(-1), side="right") - 1, 0, len(edges) - 2)
    sums = np.bincount(indices, weights=np.log1p(magnitude).reshape(-1), minlength=len(edges) - 1)
    counts = np.bincount(indices, minlength=len(edges) - 1)
    profile = sums / np.maximum(counts, 1)
    total = max(float(power.sum()), 1e-12)
    high_ratio = float(power[radius >= 0.25].sum() / total)
    anisotropy = float(abs(np.sum(power * np.exp(2j * theta))) / total)
    return profile, high_ratio, anisotropy


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, values, side="right") / max(1, ordered.size)


def summarize_array(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.ood_fraction < 1.0:
        raise SystemExit("--ood-fraction must be between 0 and 1")
    if (args.noisy_dir is None) != (args.gt_dir is None):
        raise SystemExit("--noisy-dir and --gt-dir must be supplied together")
    root = args.root.resolve()
    if args.noisy_dir:
        noisy_dir = args.noisy_dir.resolve()
        gt_dir = args.gt_dir.resolve()
    else:
        locations = discover_dataset(root)
        noisy_dir, gt_dir = locations.noisy_dir, locations.gt_dir
    pairs, missing_gt, missing_noisy = build_pair_index(noisy_dir, gt_dir, verify_shapes=True)
    if missing_gt or missing_noisy:
        raise RuntimeError(f"incomplete pairing: missing GT={len(missing_gt)}, missing noisy={len(missing_noisy)}")
    split = create_or_load_split(pairs, args.split, root, seed=args.seed, train_fraction=0.85)
    train_records, validation_records = records_from_split(split, pairs)
    pair_by_key = {record.key: record for record in pairs}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_subsets_path = (
        args.validation_subsets.resolve()
        if args.validation_subsets is not None
        else output_dir / "validation_subsets_seed42.json"
    )
    device = select_device(args.device)

    print(f"Computing input-only structural descriptors for {len(pairs)} paired samples")
    descriptors = np.empty((len(pairs), len(DESCRIPTOR_NAMES)), dtype=np.float64)
    keys = []
    descriptor_rows = []
    split_labels = {}
    split_labels.update({record.key: "train" for record in train_records})
    split_labels.update({record.key: "validation" for record in validation_records})
    for index, record in enumerate(pairs):
        raw = load_grayscale_npy(record.noisy, mmap_mode="r")
        descriptor = image_descriptor(raw)
        descriptors[index] = descriptor
        keys.append(record.key)
        descriptor_rows.append(
            {"key": record.key, "split": split_labels[record.key], **dict(zip(DESCRIPTOR_NAMES, descriptor.tolist()))}
        )
        if (index + 1) % 400 == 0:
            print(f"  descriptors: {index + 1}/{len(pairs)}")
    write_rows(output_dir / "structural_descriptors.csv", descriptor_rows)

    key_to_index = {key: index for index, key in enumerate(keys)}
    train_indices = np.asarray([key_to_index[record.key] for record in train_records], dtype=np.int64)
    val_indices = np.asarray([key_to_index[record.key] for record in validation_records], dtype=np.int64)
    train_x = descriptors[train_indices]
    val_x = descriptors[val_indices]
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    train_z = (train_x - center) / scale
    val_z = (val_x - center) / scale
    covariance = np.cov(train_z, rowvar=False)
    shrinkage = 0.10
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.eye(covariance.shape[0])
    precision = np.linalg.pinv(covariance, hermitian=True)
    train_mahalanobis = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", train_z, precision, train_z)))
    val_mahalanobis = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", val_z, precision, val_z)))
    tree = cKDTree(train_z)
    train_nn = tree.query(train_z, k=2, workers=1)[0][:, 1]
    val_nn = tree.query(val_z, k=1, workers=1)[0]
    val_mahalanobis_percentile = empirical_percentile(train_mahalanobis, val_mahalanobis)
    val_nn_percentile = empirical_percentile(train_nn, val_nn)
    ood_score = 0.5 * (val_mahalanobis_percentile + val_nn_percentile)
    ood_count = max(1, int(round(args.ood_fraction * len(validation_records))))
    ranked_val = sorted(
        range(len(validation_records)),
        key=lambda index: (-float(ood_score[index]), validation_records[index].key.casefold()),
    )
    selected_indices = set(ranked_val[:ood_count])
    ood_keys = sorted((validation_records[index].key for index in selected_indices), key=str.casefold)
    ood_rows = []
    for index, record in enumerate(validation_records):
        ood_rows.append(
            {
                "key": record.key,
                "mahalanobis": float(val_mahalanobis[index]),
                "mahalanobis_train_percentile": float(val_mahalanobis_percentile[index]),
                "nearest_train_distance": float(val_nn[index]),
                "nearest_train_distance_percentile": float(val_nn_percentile[index]),
                "ood_score": float(ood_score[index]),
                "selected": index in selected_indices,
            }
        )
    write_rows(output_dir / "ood_proxy_scores.csv", ood_rows)

    split_hash = sha256_file(args.split.resolve())
    validation_subsets = {
        "format_version": 1,
        "seed": args.seed,
        "source_split": str(args.split.name),
        "source_split_sha256": split_hash,
        "dataset_fingerprint": split["dataset_fingerprint"],
        "selection_scope": "validation inputs only; descriptor fitting and reference distributions use train inputs only",
        "official_test_data_used": False,
        "group_metadata": "no reliable source/group metadata was present; retained deterministic filename-random split",
        "descriptor_names": DESCRIPTOR_NAMES,
        "descriptor_fit": {
            "train_count": len(train_records),
            "center_mean": center.tolist(),
            "scale_standard_deviation": scale.tolist(),
            "covariance_shrinkage": shrinkage,
            "ood_score": "mean of train-empirical percentiles for shrinkage-Mahalanobis distance and nearest-train distance",
        },
        "val_random": {
            "count": len(validation_records),
            "keys": [record.key for record in validation_records],
        },
        "val_ood_proxy": {
            "fraction_of_validation": args.ood_fraction,
            "count": len(ood_keys),
            "selection": "highest input-structure OOD score",
            "minimum_selected_score": float(min(ood_score[index] for index in selected_indices)),
            "keys": ood_keys,
        },
    }
    write_json(validation_subsets_path, validation_subsets)

    # Fit PCA using the training fold only for visualization/diversity reporting.
    _, singular_values, components = np.linalg.svd(train_z, full_matrices=False)
    explained = singular_values * singular_values
    explained /= max(float(explained.sum()), 1e-12)
    train_embedding = train_z @ components[:2].T
    val_embedding = val_z @ components[:2].T

    intensity_edges = np.linspace(0.0, 1.0, 21)
    lr_noise_bins = BinnedMoments(intensity_edges)
    hr_residual_bins = BinnedMoments(intensity_edges)
    lr_noise_stats = Moments()
    hr_residual_stats = Moments()
    unclipped_skip_residual_stats = Moments()
    signal_noise_square_correlation = PairMoments()
    oor_gt_correlation = PairMoments()
    residual_hist_edges = np.linspace(-1.0, 1.0, 401)
    residual_hist = np.zeros(len(residual_hist_edges) - 1, dtype=np.int64)
    lr_noise_hist_edges = np.linspace(-1.5, 1.5, 401)
    lr_noise_hist = np.zeros(len(lr_noise_hist_edges) - 1, dtype=np.int64)
    spatial_sum = np.zeros((256, 256), dtype=np.float64)
    spatial_sum_sq = np.zeros((256, 256), dtype=np.float64)
    below_map = np.zeros((128, 128), dtype=np.int64)
    above_map = np.zeros((128, 128), dtype=np.int64)
    below_magnitudes = []
    above_magnitudes = []
    signal_hist_edges = np.linspace(0.0, 1.0, 51)
    signal_hist_all = np.zeros(50, dtype=np.int64)
    signal_hist_below = np.zeros(50, dtype=np.int64)
    signal_hist_above = np.zeros(50, dtype=np.int64)
    pair_rows = []

    fft_count = min(args.fft_samples, len(train_records))
    fft_indices = set(np.linspace(0, len(train_records) - 1, fft_count, dtype=np.int64).tolist())
    radial_edges = np.linspace(0.0, math.sqrt(0.5), 41)
    fft_bicubic_profile = np.zeros(len(radial_edges) - 1, dtype=np.float64)
    fft_gt_profile = np.zeros(len(radial_edges) - 1, dtype=np.float64)
    fft_bicubic_high = []
    fft_gt_high = []
    fft_bicubic_anisotropy = []
    fft_gt_anisotropy = []

    print(f"Analyzing residuals/noise on the {len(train_records)}-sample training fold only ({device})")
    for start in range(0, len(train_records), args.batch_size):
        batch_records = train_records[start : start + args.batch_size]
        raw_np = np.stack([load_grayscale_npy(record.noisy, mmap_mode="r") for record in batch_records])
        gt_np = np.stack([load_grayscale_npy(record.gt, mmap_mode="r") for record in batch_records])
        raw_tensor = torch.from_numpy(raw_np).unsqueeze(1).to(device)
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(1).to(device)
        with torch.inference_mode():
            bicubic_unclipped_tensor = F.interpolate(
                raw_tensor.clamp(0.0, 1.0), scale_factor=2.0, mode="bicubic", align_corners=False
            )
            bicubic_tensor = bicubic_unclipped_tensor.clamp(0.0, 1.0)
            low_gt_tensor = F.interpolate(gt_tensor, size=raw_tensor.shape[-2:], mode="area")
        bicubic_np = bicubic_tensor.squeeze(1).cpu().numpy()
        bicubic_unclipped_np = bicubic_unclipped_tensor.squeeze(1).cpu().numpy()
        low_gt_np = low_gt_tensor.squeeze(1).cpu().numpy()
        for offset, record in enumerate(batch_records):
            global_index = start + offset
            raw = raw_np[offset]
            gt = gt_np[offset]
            bicubic = bicubic_np[offset]
            bicubic_unclipped = bicubic_unclipped_np[offset]
            low_gt = low_gt_np[offset]
            residual = gt - bicubic
            unclipped_skip_residual = gt - bicubic_unclipped
            lr_noise = raw - low_gt
            hr_residual_stats.update(residual)
            unclipped_skip_residual_stats.update(unclipped_skip_residual)
            lr_noise_stats.update(lr_noise)
            hr_residual_bins.update(np.clip(bicubic, 0.0, 1.0), residual)
            lr_noise_bins.update(low_gt, lr_noise)
            signal_noise_square_correlation.update(low_gt, np.square(lr_noise))
            residual_hist += np.histogram(residual, bins=residual_hist_edges)[0]
            lr_noise_hist += np.histogram(lr_noise, bins=lr_noise_hist_edges)[0]
            spatial_sum += residual
            spatial_sum_sq += np.square(residual)

            below = raw < 0.0
            above = raw > 1.0
            below_map += below
            above_map += above
            if below.any():
                below_magnitudes.append((-raw[below]).astype(np.float32, copy=True))
                oor_gt_correlation.update(raw[below], low_gt[below])
            if above.any():
                above_magnitudes.append((raw[above] - 1.0).astype(np.float32, copy=True))
                oor_gt_correlation.update(raw[above], low_gt[above])
            signal_hist_all += np.histogram(low_gt, bins=signal_hist_edges)[0]
            signal_hist_below += np.histogram(low_gt[below], bins=signal_hist_edges)[0]
            signal_hist_above += np.histogram(low_gt[above], bins=signal_hist_edges)[0]
            pair_rows.append(
                {
                    "key": record.key,
                    "input_height": raw.shape[0],
                    "input_width": raw.shape[1],
                    "residual_mean": float(residual.mean()),
                    "residual_std": float(residual.std()),
                    "residual_rmse": float(np.sqrt(np.mean(np.square(residual)))),
                    "lr_noise_mean": float(lr_noise.mean()),
                    "lr_noise_std": float(lr_noise.std()),
                    "below_zero_fraction": float(below.mean()),
                    "above_one_fraction": float(above.mean()),
                }
            )
            if global_index in fft_indices:
                bic_profile, bic_high, bic_anisotropy = radial_spectrum(bicubic, radial_edges)
                gt_profile, gt_high, gt_anisotropy = radial_spectrum(gt, radial_edges)
                fft_bicubic_profile += bic_profile
                fft_gt_profile += gt_profile
                fft_bicubic_high.append(bic_high)
                fft_gt_high.append(gt_high)
                fft_bicubic_anisotropy.append(bic_anisotropy)
                fft_gt_anisotropy.append(gt_anisotropy)
        if min(start + args.batch_size, len(train_records)) % 320 == 0 or start + args.batch_size >= len(train_records):
            print(f"  restoration analysis: {min(start + args.batch_size, len(train_records))}/{len(train_records)}")
    write_rows(output_dir / "pair_statistics.csv", pair_rows)

    sample_count = len(train_records)
    spatial_mean = spatial_sum / sample_count
    spatial_variance = np.maximum(0.0, spatial_sum_sq / sample_count - np.square(spatial_mean))
    spatial_std = np.sqrt(spatial_variance)
    below_values = np.concatenate(below_magnitudes) if below_magnitudes else np.empty(0, dtype=np.float32)
    above_values = np.concatenate(above_magnitudes) if above_magnitudes else np.empty(0, dtype=np.float32)
    lr_noise_rows = lr_noise_bins.rows()
    hr_residual_rows = hr_residual_bins.rows()
    lr_noise_models = fit_weighted_polynomials(lr_noise_rows)
    residual_models = fit_weighted_polynomials(hr_residual_rows)

    fft_bicubic_profile /= max(1, fft_count)
    fft_gt_profile /= max(1, fft_count)
    radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2
    fft_rows = [
        {
            "frequency_cycles_per_pixel": float(radial_centers[index]),
            "bicubic_log_magnitude": float(fft_bicubic_profile[index]),
            "gt_log_magnitude": float(fft_gt_profile[index]),
        }
        for index in range(len(radial_centers))
    ]
    write_rows(output_dir / "noise_variance_by_signal.csv", lr_noise_rows)
    write_rows(output_dir / "restoration_residual_by_signal.csv", hr_residual_rows)
    write_rows(output_dir / "fft_radial_profiles.csv", fft_rows)

    # Plots use display copies only; no metric or source array is changed.
    plt.figure(figsize=(9, 5))
    residual_centers = (residual_hist_edges[:-1] + residual_hist_edges[1:]) / 2
    noise_centers = (lr_noise_hist_edges[:-1] + lr_noise_hist_edges[1:]) / 2
    plt.semilogy(residual_centers, residual_hist / max(1, residual_hist.sum()), label="GT - bicubic(clipped LR)")
    plt.semilogy(noise_centers, lr_noise_hist / max(1, lr_noise_hist.sum()), label="raw LR - area-downsampled GT")
    plt.xlabel("Residual value")
    plt.ylabel("Pixel fraction (log scale)")
    plt.title("Measured restoration residual and LR discrepancy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "residual_histogram.png", dpi=180)
    plt.close()

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(spatial_mean, cmap="coolwarm")
    axes[0].set_title("Mean GT - bicubic residual")
    axes[0].axis("off")
    figure.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(spatial_std, cmap="magma")
    axes[1].set_title("Residual spatial standard deviation")
    axes[1].axis("off")
    figure.colorbar(im1, ax=axes[1], fraction=0.046)
    figure.tight_layout()
    figure.savefig(output_dir / "residual_spatial_distribution.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, rows, title in (
        (axes[0], lr_noise_rows, "LR discrepancy variance vs downsampled-GT signal"),
        (axes[1], hr_residual_rows, "HR restoration-residual variance vs bicubic signal"),
    ):
        x = np.asarray([row["bin_center"] for row in rows])
        variance = np.asarray([row["variance"] for row in rows])
        axis.plot(x, variance, marker="o", linewidth=1.5)
        axis.set_xlabel("Signal intensity")
        axis.set_ylabel("Within-bin variance")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "noise_variance_vs_signal.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    if below_values.size:
        axes[0, 0].hist(below_values, bins=80, alpha=0.75, label="magnitude below 0")
    if above_values.size:
        axes[0, 0].hist(above_values, bins=80, alpha=0.75, label="magnitude above 1")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Out-of-range excursion magnitude")
    axes[0, 0].set_ylabel("Pixels")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)
    im = axes[0, 1].imshow(below_map / sample_count, cmap="viridis")
    axes[0, 1].set_title("Spatial frequency: raw < 0")
    axes[0, 1].axis("off")
    figure.colorbar(im, ax=axes[0, 1], fraction=0.046)
    im = axes[1, 0].imshow(above_map / sample_count, cmap="magma")
    axes[1, 0].set_title("Spatial frequency: raw > 1")
    axes[1, 0].axis("off")
    figure.colorbar(im, ax=axes[1, 0], fraction=0.046)
    signal_centers = (signal_hist_edges[:-1] + signal_hist_edges[1:]) / 2
    axes[1, 1].plot(signal_centers, signal_hist_all / max(1, signal_hist_all.sum()), label="all LR locations")
    axes[1, 1].plot(signal_centers, signal_hist_below / max(1, signal_hist_below.sum()), label="raw < 0")
    axes[1, 1].plot(signal_centers, signal_hist_above / max(1, signal_hist_above.sum()), label="raw > 1")
    axes[1, 1].set_xlabel("Area-downsampled GT intensity")
    axes[1, 1].set_ylabel("Normalized density")
    axes[1, 1].set_title("GT signal at out-of-range locations")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "out_of_range_analysis.png", dpi=180)
    plt.close(figure)

    plt.figure(figsize=(8, 5))
    plt.plot(radial_centers, fft_bicubic_profile, label="Bicubic(clipped LR)")
    plt.plot(radial_centers, fft_gt_profile, label="GT")
    plt.axvline(0.25, linestyle="--", color="black", alpha=0.4, label="high-frequency threshold")
    plt.xlabel("Radial frequency (cycles/pixel)")
    plt.ylabel("Mean log(1 + FFT magnitude)")
    plt.title(f"Radial frequency profiles ({fft_count} training-fold pairs)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fft_profiles.png", dpi=180)
    plt.close()

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(train_embedding[:, 0], train_embedding[:, 1], s=8, alpha=0.25, label="train fit")
    scatter = axis.scatter(val_embedding[:, 0], val_embedding[:, 1], c=ood_score, cmap="viridis", s=18, label="validation")
    selected_order = np.asarray(sorted(selected_indices), dtype=np.int64)
    axis.scatter(
        val_embedding[selected_order, 0],
        val_embedding[selected_order, 1],
        facecolors="none",
        edgecolors="red",
        s=42,
        linewidths=0.8,
        label="OOD proxy",
    )
    axis.set_xlabel(f"Train-fitted PC1 ({explained[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"Train-fitted PC2 ({explained[1] * 100:.1f}% variance)")
    axis.set_title("Input-only structural diversity and OOD-proxy selection")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="OOD score")
    figure.tight_layout()
    figure.savefig(output_dir / "structural_diversity.png", dpi=180)
    plt.close(figure)

    rmse_sorted = sorted(pair_rows, key=lambda row: row["residual_rmse"])
    representative_positions = np.linspace(0, len(rmse_sorted) - 1, 4, dtype=np.int64)
    representative_keys = [rmse_sorted[int(position)]["key"] for position in representative_positions]
    figure, axes = plt.subplots(len(representative_keys), 4, figsize=(12, 3 * len(representative_keys)))
    for row_index, key in enumerate(representative_keys):
        record = pair_by_key[key]
        raw = load_grayscale_npy(record.noisy).copy()
        gt = load_grayscale_npy(record.gt)
        with torch.inference_mode():
            bicubic = (
                F.interpolate(
                    torch.from_numpy(raw).view(1, 1, *raw.shape).clamp(0.0, 1.0),
                    scale_factor=2.0,
                    mode="bicubic",
                    align_corners=False,
                )
                .clamp(0.0, 1.0)
                .squeeze()
                .numpy()
            )
        residual = gt - bicubic
        panels = ((np.clip(raw, 0, 1), "NoisyLR (display clipped)"), (bicubic, "Bicubic"), (gt, "GT"), (residual, "GT - bicubic"))
        residual_limit = max(0.05, float(np.percentile(np.abs(residual), 99)))
        for column, (image, title) in enumerate(panels):
            cmap = "coolwarm" if column == 3 else "gray"
            kwargs = {"vmin": -residual_limit, "vmax": residual_limit} if column == 3 else {"vmin": 0, "vmax": 1}
            axes[row_index, column].imshow(image, cmap=cmap, **kwargs)
            axes[row_index, column].set_title(f"{key}: {title}" if column == 0 else title)
            axes[row_index, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "representative_training_residuals.png", dpi=180)
    plt.close(figure)

    below_summary = summarize_array(below_values) if below_values.size else {"count": 0}
    above_summary = summarize_array(above_values) if above_values.size else {"count": 0}
    statistics = {
        "format_version": 1,
        "seed": args.seed,
        "official_test_data_used": False,
        "analysis_scope": "saved training fold only for paired degradation; all paired input descriptors with fitting on train fold only for OOD proxy",
        "data": {
            "matched_pairs": len(pairs),
            "train_fold_pairs_analyzed": len(train_records),
            "validation_pairs_described": len(validation_records),
            "input_shape_distribution": dict(Counter(str(load_grayscale_npy(record.noisy, mmap_mode="r").shape) for record in pairs)),
            "gt_shape_distribution": dict(Counter(str(load_grayscale_npy(record.gt, mmap_mode="r").shape) for record in pairs)),
            "pairs_128_to_256": sum(load_grayscale_npy(record.noisy, mmap_mode="r").shape == (128, 128) for record in pairs),
            "pairs_256_to_512": sum(load_grayscale_npy(record.noisy, mmap_mode="r").shape == (256, 256) for record in pairs),
            "dataset_fingerprint": split["dataset_fingerprint"],
            "split_sha256": split_hash,
        },
        "basic_restoration_residual": {
            "definition": "GT - clamp(PyTorch bicubic(clamp(raw LR,0,1)),0,1), align_corners=False; matches benchmark_bicubic.py",
            "global": hr_residual_stats.summary(),
            "unclipped_model_skip_residual": {
                "definition": "GT - PyTorch bicubic(clamp(raw LR,0,1)); this is the internal skip seen during training",
                "global": unclipped_skip_residual_stats.summary(),
            },
            "per_pair_rmse": summarize_array(row["residual_rmse"] for row in pair_rows),
            "spatial_mean_min": float(spatial_mean.min()),
            "spatial_mean_max": float(spatial_mean.max()),
            "spatial_std_min": float(spatial_std.min()),
            "spatial_std_max": float(spatial_std.max()),
        },
        "noise_behavior": {
            "lr_discrepancy_definition": "raw LR - area-downsampled GT; includes noise, blur/downsampling mismatch, and any registration error",
            "global": lr_noise_stats.summary(),
            "correlation_signal_vs_squared_lr_discrepancy": signal_noise_square_correlation.correlation(),
            "variance_models": lr_noise_models,
            "hr_residual_variance_models": residual_models,
            "interpretation_guardrail": "These empirical discrepancies do not by themselves identify a physical noise process.",
        },
        "out_of_range": {
            "train_fold_files_below_zero": sum(row["below_zero_fraction"] > 0 for row in pair_rows),
            "train_fold_files_above_one": sum(row["above_one_fraction"] > 0 for row in pair_rows),
            "below_zero_excursion_magnitude": below_summary,
            "above_one_excursion_magnitude": above_summary,
            "raw_oor_value_correlation_with_downsampled_gt": oor_gt_correlation.correlation(),
            "spatial_below_fraction_min": float((below_map / sample_count).min()),
            "spatial_below_fraction_max": float((below_map / sample_count).max()),
            "spatial_above_fraction_min": float((above_map / sample_count).min()),
            "spatial_above_fraction_max": float((above_map / sample_count).max()),
        },
        "frequency": {
            "sample_count": fft_count,
            "bicubic_high_frequency_energy_ratio": summarize_array(fft_bicubic_high),
            "gt_high_frequency_energy_ratio": summarize_array(fft_gt_high),
            "bicubic_directional_anisotropy": summarize_array(fft_bicubic_anisotropy),
            "gt_directional_anisotropy": summarize_array(fft_gt_anisotropy),
        },
        "structural_diversity": {
            "descriptor_count": len(DESCRIPTOR_NAMES),
            "descriptor_names": DESCRIPTOR_NAMES,
            "train_pca_explained_variance_first_two": explained[:2].tolist(),
            "train_nearest_neighbor_distance": summarize_array(train_nn),
            "validation_nearest_train_distance": summarize_array(val_nn),
        },
        "ood_proxy": {
            "validation_count": len(validation_records),
            "selected_count": len(ood_keys),
            "selected_fraction": args.ood_fraction,
            "score": summarize_array(ood_score),
            "split_file": str(validation_subsets_path.name),
        },
        "representative_training_keys_by_residual_rmse": representative_keys,
    }
    write_json(output_dir / "statistics.json", statistics)

    variance_ratio = lr_noise_models["max_to_min_binned_variance_ratio"]
    linear_r2 = lr_noise_models["models"]["linear"]["weighted_r2_vs_constant"]
    quadratic_r2 = lr_noise_models["models"]["quadratic"]["weighted_r2_vs_constant"]
    residual_summary = hr_residual_stats.summary()
    noise_summary = lr_noise_stats.summary()
    bic_high = statistics["frequency"]["bicubic_high_frequency_energy_ratio"]["mean"]
    gt_high = statistics["frequency"]["gt_high_frequency_energy_ratio"]["mean"]
    summary_text = f"""# Degradation Analysis Summary

This analysis uses only the saved training fold for paired LR/GT measurements. Official test arrays were not opened, described, fitted, or scored. Input-only descriptors were computed for the saved validation fold solely to construct the OOD-proxy subset; all descriptor scaling, covariance, PCA, and nearest-neighbour reference distributions were fitted on the 2,720 training inputs.

## Dataset coverage

- Matched paired samples: {len(pairs):,} ({len(train_records):,} train / {len(validation_records):,} validation).
- Available paired geometry: {statistics['data']['pairs_128_to_256']:,} samples at `128x128 -> 256x256`.
- Paired `256x256 -> 512x512` samples: {statistics['data']['pairs_256_to_512']}. Architecture-level support can be tested, but quality at this resolution cannot be measured from the supplied data.
- Saved split SHA-256: `{split_hash}`.

## Bicubic restoration residual

Using `GT - clamp(bicubic(clamp(raw LR,0,1)),0,1)` to match `benchmark_bicubic.py`, the training-fold pixel residual has mean {residual_summary['mean']:.6f}, standard deviation {residual_summary['std']:.6f}, and range [{residual_summary['min']:.6f}, {residual_summary['max']:.6f}]. Per-image residual RMSE has mean {statistics['basic_restoration_residual']['per_pair_rmse']['mean']:.6f} and spans {statistics['basic_restoration_residual']['per_pair_rmse']['min']:.6f} to {statistics['basic_restoration_residual']['per_pair_rmse']['max']:.6f}. The model's internal bicubic skip is not clamped after interpolation, so its separate residual is also recorded in `statistics.json`. The spatial mean/std maps show whether error is globally distributed or concentrated near recurring structures; they are evidence for retaining an HR residual refinement rather than relying on interpolation alone.

## Empirical noise behavior

The LR discrepancy `raw LR - area-downsampled GT` has mean {noise_summary['mean']:.6f} and standard deviation {noise_summary['std']:.6f}. Its squared magnitude has correlation {signal_noise_square_correlation.correlation():.6f} with the downsampled-GT signal. Across 20 intensity bins, the maximum/minimum positive variance ratio is {variance_ratio:.3f}; a linear variance curve explains {linear_r2:.3%} and a quadratic curve explains {quadratic_r2:.3%} of the between-bin variance relative to a constant model.

These measurements indicate whether the discrepancy is intensity-dependent, but they do **not** uniquely identify Gaussian, speckle, blur, registration, or degradation order. The discrepancy includes all of those possible effects. Synthetic degradation should therefore remain disabled unless a controlled validation/OOD experiment supports it.

## Out-of-range signal

- Training-fold files containing values below zero: {statistics['out_of_range']['train_fold_files_below_zero']:,}.
- Training-fold files containing values above one: {statistics['out_of_range']['train_fold_files_above_one']:,}.
- Below-zero pixels: {below_summary.get('count', 0):,}; mean excursion magnitude {below_summary.get('mean', 0.0):.6f}, maximum {below_summary.get('max', 0.0):.6f}.
- Above-one pixels: {above_summary.get('count', 0):,}; mean excursion magnitude {above_summary.get('mean', 0.0):.6f}, maximum {above_summary.get('max', 0.0):.6f}.
- Correlation between raw out-of-range values and the corresponding downsampled GT signal: {oor_gt_correlation.correlation():.6f}.

The excursions are frequent enough that the raw feature path must be preserved. Their signal and spatial distributions are plotted; clipping remains appropriate only for the conservative bicubic skip and final file contract.

## Frequency behavior

Across {fft_count} deterministic training-fold samples, the mean energy above 0.25 cycles/pixel is {bic_high:.6f} for bicubic LR and {gt_high:.6f} for GT. The GT/bicubic ratio is {gt_high / max(bic_high, 1e-12):.3f}. The saved radial profiles and directional-anisotropy summaries quantify missing high-frequency and oriented/repeating structure, supporting local edge refinement while warning against adversarial hallucination.

## Structural diversity and OOD proxy

The input-only descriptor combines intensity, out-of-range frequency, gradient density/orientation, four frequency bands, spectral anisotropy/concentration, and simple morphology. The first two train-fitted principal components explain {(explained[0] + explained[1]) * 100:.2f}% of standardized descriptor variance. Train nearest-neighbour distances range from {train_nn.min():.4f} to {train_nn.max():.4f}; validation-to-train distances range from {val_nn.min():.4f} to {val_nn.max():.4f}.

`val_random` retains all {len(validation_records)} saved validation samples. `val_ood_proxy` contains the {len(ood_keys)} highest-scoring validation inputs ({args.ood_fraction:.0%}), ranked by the mean of train-empirical shrinkage-Mahalanobis and nearest-train-distance percentiles. The subset is saved in `{validation_subsets_path.name}` and was constructed without official test information.

## Architecture implications

1. Keep the unmodified raw channel available; out-of-range samples are common and correlated with underlying intensity.
2. Keep a clipped bicubic skip for a conservative base image, but learn a residual because bicubic leaves substantial structured error and loses high-frequency energy.
3. Compare raw-only, raw+clipped, and raw+clipped+OOR representations experimentally; frequency alone does not prove that three channels are optimal.
4. Test a lightweight wider-context branch because directional/repeating structures are present, but reject it if the random/OOD Pareto metrics or latency worsen.
5. Avoid strong synthetic degradation and adversarial losses: the measured discrepancy is mixed and a simple physical-noise claim is not supported.

## Generated evidence

- `statistics.json`
- `pair_statistics.csv`
- `structural_descriptors.csv`
- `ood_proxy_scores.csv`
- `noise_variance_by_signal.csv`
- `restoration_residual_by_signal.csv`
- `fft_radial_profiles.csv`
- `residual_histogram.png`
- `residual_spatial_distribution.png`
- `noise_variance_vs_signal.png`
- `out_of_range_analysis.png`
- `fft_profiles.png`
- `structural_diversity.png`
- `representative_training_residuals.png`
"""
    (output_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    print(f"Wrote degradation analysis to {output_dir}")
    print(f"OOD proxy: {len(ood_keys)}/{len(validation_records)} validation samples -> {validation_subsets_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
