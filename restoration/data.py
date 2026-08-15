"""Dataset discovery, deterministic splitting, and paired patch loading."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import inspect_npy_shape, iter_npy_files, load_grayscale_npy


IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
    ".tmp",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "runs",
    "restored_test_outputs",
}


@dataclass(frozen=True)
class PairRecord:
    key: str
    noisy: str
    gt: str


@dataclass(frozen=True)
class DatasetLocations:
    noisy_dir: Path
    gt_dir: Path
    test_dirs: tuple[Path, ...]


def _ignored(path: Path) -> bool:
    return any(part in IGNORED_DIR_NAMES or part == "__MACOSX" for part in path.parts)


def _direct_npy_dirs(root: Path) -> list[Path]:
    directories: set[Path] = set()
    for current, directory_names, file_names in os.walk(root):
        directory_names[:] = [
            name for name in directory_names if name not in IGNORED_DIR_NAMES and name != "__MACOSX"
        ]
        current_path = Path(current)
        for name in file_names:
            if name.lower().endswith(".npy") and not name.startswith("._"):
                directories.add(current_path)
    return sorted(directories)


def _path_tokens(path: Path) -> set[str]:
    text = "/".join(part.lower() for part in path.parts)
    for separator in ("-", "_", "(", ")"):
        text = text.replace(separator, "/")
    return {token for token in text.split("/") if token}


def _is_noisy_dir(path: Path) -> bool:
    compact = path.name.lower().replace("_", "").replace("-", "")
    return "noisylr" in compact or compact in {"input", "inputs", "noisy", "lr"}


def _is_gt_dir(path: Path) -> bool:
    compact = path.name.lower().replace("_", "").replace("-", "")
    return compact in {"gt", "target", "targets", "clean", "groundtruth"} or "groundtruth" in compact


def _relative_stem_map(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in iter_npy_files(directory):
        key = path.relative_to(directory).with_suffix("").as_posix()
        folded = key.casefold()
        if folded in result:
            duplicates.append(key)
        result[folded] = path
    if duplicates:
        raise ValueError(f"duplicate relative stems under {directory}: {duplicates[:5]}")
    return result


def build_pair_index(noisy_dir: str | Path, gt_dir: str | Path, *, verify_shapes: bool = True) -> tuple[list[PairRecord], list[str], list[str]]:
    """Match pairs by exact relative path stem and optionally verify 2x shapes."""

    noisy_dir, gt_dir = Path(noisy_dir).resolve(), Path(gt_dir).resolve()
    noisy_map, gt_map = _relative_stem_map(noisy_dir), _relative_stem_map(gt_dir)
    common = sorted(noisy_map.keys() & gt_map.keys())
    missing_gt = sorted(str(noisy_map[k].relative_to(noisy_dir)) for k in noisy_map.keys() - gt_map.keys())
    missing_noisy = sorted(str(gt_map[k].relative_to(gt_dir)) for k in gt_map.keys() - noisy_map.keys())
    pairs: list[PairRecord] = []
    for key in common:
        noisy_path, gt_path = noisy_map[key], gt_map[key]
        if verify_shapes:
            noisy_hw, gt_hw = inspect_npy_shape(noisy_path), inspect_npy_shape(gt_path)
            if gt_hw != (2 * noisy_hw[0], 2 * noisy_hw[1]):
                raise ValueError(
                    f"pair '{key}' violates 2x geometry: noisy={noisy_hw}, GT={gt_hw}"
                )
        pairs.append(
            PairRecord(
                key=noisy_path.relative_to(noisy_dir).with_suffix("").as_posix(),
                noisy=str(noisy_path),
                gt=str(gt_path),
            )
        )
    return pairs, missing_gt, missing_noisy


def discover_dataset(root: str | Path) -> DatasetLocations:
    """Discover the strongest verified NoisyLR/GT pair and test directories."""

    root = Path(root).resolve()
    directories = _direct_npy_dirs(root)
    noisy_candidates = [path for path in directories if _is_noisy_dir(path)]
    gt_candidates = [path for path in directories if _is_gt_dir(path)]
    ranked: list[tuple[int, Path, Path]] = []
    for noisy in noisy_candidates:
        noisy_tokens = _path_tokens(noisy)
        if "test" in noisy_tokens or any("test" in part.lower() for part in noisy.parts):
            continue
        for gt in gt_candidates:
            try:
                pairs, _, _ = build_pair_index(noisy, gt, verify_shapes=True)
            except (ValueError, OSError):
                continue
            if pairs:
                common_parent_bonus = 1 if noisy.parent == gt.parent else 0
                ranked.append((len(pairs) * 10 + common_parent_bonus, noisy, gt))
    if not ranked:
        searched = "\n  - ".join(str(p) for p in directories) or "<none>"
        raise FileNotFoundError(
            "No valid paired training data found. Searched direct .npy directories:\n"
            f"  - {searched}\nExpected exact relative stems in NoisyLR and GT directories with GT exactly 2x."
        )
    _, noisy_dir, gt_dir = max(ranked, key=lambda row: row[0])
    test_dirs = []
    for path in noisy_candidates:
        if path == noisy_dir:
            continue
        lower = "/".join(part.lower() for part in path.parts)
        if "test" in lower:
            test_dirs.append(path)
    return DatasetLocations(noisy_dir, gt_dir, tuple(sorted(test_dirs)))


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def dataset_fingerprint(pairs: Iterable[PairRecord], project_root: str | Path) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    for pair in sorted(pairs, key=lambda item: item.key.casefold()):
        noisy = Path(pair.noisy)
        gt = Path(pair.gt)
        digest.update(pair.key.encode("utf-8"))
        digest.update(_portable_path(noisy, root).encode("utf-8"))
        digest.update(str(noisy.stat().st_size).encode("ascii"))
        digest.update(_portable_path(gt, root).encode("utf-8"))
        digest.update(str(gt.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def create_or_load_split(
    pairs: list[PairRecord],
    split_path: str | Path,
    project_root: str | Path,
    *,
    seed: int = 42,
    train_fraction: float = 0.85,
) -> dict:
    """Load an immutable split or create one deterministic filename-random split."""

    split_path, root = Path(split_path), Path(project_root).resolve()
    fingerprint = dataset_fingerprint(pairs, root)
    pair_by_key = {pair.key: pair for pair in pairs}
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        if payload.get("seed") != seed:
            raise ValueError(f"existing split seed is {payload.get('seed')}, requested {seed}: {split_path}")
        if payload.get("dataset_fingerprint") != fingerprint:
            raise ValueError(
                f"dataset fingerprint differs from saved split {split_path}; refusing silent regeneration"
            )
        saved_keys = set(payload["train_keys"]) | set(payload["validation_keys"])
        if saved_keys != set(pair_by_key):
            raise ValueError(f"saved split keys do not match the current dataset: {split_path}")
        return payload

    if len(pairs) < 2:
        raise ValueError("at least two matched pairs are required for a train/validation split")
    keys = sorted(pair_by_key, key=str.casefold)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    train_count = min(len(keys) - 1, max(1, int(round(train_fraction * len(keys)))))
    train_keys = sorted((keys[int(i)] for i in order[:train_count]), key=str.casefold)
    validation_keys = sorted((keys[int(i)] for i in order[train_count:]), key=str.casefold)
    payload = {
        "format_version": 1,
        "seed": seed,
        "strategy": "deterministic_filename_random_no_group_metadata_available",
        "train_fraction": train_fraction,
        "dataset_fingerprint": fingerprint,
        "noisy_root": _portable_path(Path(pairs[0].noisy).parent, root),
        "gt_root": _portable_path(Path(pairs[0].gt).parent, root),
        "train_keys": train_keys,
        "validation_keys": validation_keys,
        "train_count": len(train_keys),
        "validation_count": len(validation_keys),
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def records_from_split(
    payload: dict,
    pairs: list[PairRecord],
) -> tuple[list[PairRecord], list[PairRecord]]:
    pair_by_key = {pair.key: pair for pair in pairs}
    try:
        train = [pair_by_key[key] for key in payload["train_keys"]]
        validation = [pair_by_key[key] for key in payload["validation_keys"]]
    except KeyError as exc:
        raise ValueError(f"split references missing pair key {exc}") from exc
    return train, validation


def apply_joint_transform(noisy: np.ndarray, gt: np.ndarray, rotation_k: int, hflip: bool, vflip: bool) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same interpolation-free dihedral transform to an aligned pair."""

    rotation_k %= 4
    if rotation_k:
        noisy, gt = np.rot90(noisy, rotation_k), np.rot90(gt, rotation_k)
    if hflip:
        noisy, gt = np.fliplr(noisy), np.fliplr(gt)
    if vflip:
        noisy, gt = np.flipud(noisy), np.flipud(gt)
    return np.ascontiguousarray(noisy), np.ascontiguousarray(gt)


class PairedPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Mixed-size paired data with deterministic aligned random LR/GT patches."""

    def __init__(
        self,
        records: list[PairRecord],
        *,
        patch_size: int | None = 96,
        augment: bool = True,
        seed: int = 42,
        samples_per_epoch: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("PairedPatchDataset requires at least one record")
        self.records = records
        self.patch_size = patch_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.samples_per_epoch = samples_per_epoch or len(records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index % len(self.records)]
        noisy = load_grayscale_npy(record.noisy, mmap_mode="r")
        gt = load_grayscale_npy(record.gt, mmap_mode="r")
        if gt.shape != (2 * noisy.shape[0], 2 * noisy.shape[1]):
            raise ValueError(f"{record.key}: invalid pair geometry {noisy.shape} -> {gt.shape}")
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
        if self.patch_size:
            patch = min(int(self.patch_size), noisy.shape[0], noisy.shape[1])
            top = int(rng.integers(0, noisy.shape[0] - patch + 1))
            left = int(rng.integers(0, noisy.shape[1] - patch + 1))
            noisy = noisy[top : top + patch, left : left + patch]
            gt = gt[2 * top : 2 * (top + patch), 2 * left : 2 * (left + patch)]
        if self.augment:
            noisy, gt = apply_joint_transform(
                noisy,
                gt,
                rotation_k=int(rng.integers(0, 4)),
                hflip=bool(rng.integers(0, 2)),
                vflip=bool(rng.integers(0, 2)),
            )
        # mmap-backed arrays are read-only. Copy before torch.from_numpy so no
        # tensor ever aliases non-writable organizer data.
        noisy_tensor = torch.from_numpy(np.array(noisy, dtype=np.float32, order="C", copy=True)).unsqueeze(0)
        gt_tensor = torch.from_numpy(np.array(gt, dtype=np.float32, order="C", copy=True)).unsqueeze(0)
        return noisy_tensor, gt_tensor, record.key


class FullImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, records: list[PairRecord], rotation_k: int = 0) -> None:
        self.records = records
        self.rotation_k = rotation_k % 4

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        noisy = load_grayscale_npy(record.noisy, mmap_mode="r")
        gt = load_grayscale_npy(record.gt, mmap_mode="r")
        if gt.shape != (2 * noisy.shape[0], 2 * noisy.shape[1]):
            raise ValueError(f"{record.key}: invalid pair geometry {noisy.shape} -> {gt.shape}")
        if self.rotation_k:
            noisy, gt = apply_joint_transform(noisy, gt, self.rotation_k, False, False)
        return (
            torch.from_numpy(np.array(noisy, dtype=np.float32, order="C", copy=True)).unsqueeze(0),
            torch.from_numpy(np.array(gt, dtype=np.float32, order="C", copy=True)).unsqueeze(0),
            record.key,
        )
