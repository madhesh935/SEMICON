"""Strict NumPy I/O for single-channel restoration arrays."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np


class ArrayValidationError(ValueError):
    """Raised when a NumPy file is not a valid finite grayscale image."""


def normalize_array_shape(array: np.ndarray, source: str | Path = "<array>") -> np.ndarray:
    """Return a 2-D grayscale view from HW, 1HW, or HW1 input.

    This function changes shape only. It deliberately performs no intensity
    normalization or clipping.
    """

    source = str(source)
    if not isinstance(array, np.ndarray):
        raise ArrayValidationError(f"{source}: expected a NumPy array, got {type(array).__name__}")
    if array.dtype.hasobject:
        raise ArrayValidationError(f"{source}: object arrays are not supported")
    if array.size == 0:
        raise ArrayValidationError(f"{source}: empty arrays are not supported")
    if array.ndim == 2:
        result = array
    elif array.ndim == 3 and array.shape[0] == 1:
        result = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        result = array[..., 0]
    else:
        raise ArrayValidationError(
            f"{source}: expected shape (H,W), (1,H,W), or (H,W,1); got {array.shape}"
        )
    if result.shape[0] <= 0 or result.shape[1] <= 0:
        raise ArrayValidationError(f"{source}: invalid spatial shape {result.shape}")
    if not np.issubdtype(result.dtype, np.number) or np.issubdtype(result.dtype, np.complexfloating):
        raise ArrayValidationError(f"{source}: unsupported dtype {result.dtype}")
    return result


def validate_finite(array: np.ndarray, source: str | Path = "<array>") -> None:
    if not np.isfinite(array).all():
        nan_count = int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0
        inf_count = int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0
        raise ArrayValidationError(
            f"{source}: contains non-finite values (NaN={nan_count}, Inf={inf_count})"
        )


def load_grayscale_npy(
    path: str | Path,
    *,
    mmap_mode: str | None = "r",
    require_finite: bool = True,
) -> np.ndarray:
    """Load and validate one grayscale `.npy`, returning float32 HW data."""

    path = Path(path)
    try:
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    except Exception as exc:
        raise ArrayValidationError(f"{path}: unable to load NumPy array: {exc}") from exc
    array = normalize_array_shape(array, path)
    if require_finite:
        validate_finite(array, path)
    return np.asarray(array, dtype=np.float32)


def inspect_npy_shape(path: str | Path) -> tuple[int, int]:
    """Read only the memory-mapped array header and return canonical HW shape."""

    path = Path(path)
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ArrayValidationError(f"{path}: unable to inspect NumPy array: {exc}") from exc
    return tuple(int(v) for v in normalize_array_shape(array, path).shape)


def validate_pair_shapes(noisy: np.ndarray, gt: np.ndarray, source: str = "pair") -> None:
    noisy_hw = normalize_array_shape(noisy, f"{source} noisy").shape
    gt_hw = normalize_array_shape(gt, f"{source} GT").shape
    expected = (2 * noisy_hw[0], 2 * noisy_hw[1])
    if tuple(gt_hw) != expected:
        raise ArrayValidationError(
            f"{source}: GT shape {tuple(gt_hw)} is not exactly 2x noisy shape {tuple(noisy_hw)}; "
            f"expected {expected}"
        )


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    """Atomically save a `.npy` in the destination directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_restored_npy(path: str | Path, array: np.ndarray, expected_hw: tuple[int, int]) -> None:
    """Validate final competition output invariants and save atomically."""

    output = normalize_array_shape(np.asarray(array), path)
    if tuple(output.shape) != tuple(expected_hw):
        raise ArrayValidationError(
            f"{path}: output shape {output.shape} does not match required {expected_hw}"
        )
    output = np.clip(output.astype(np.float32, copy=False), 0.0, 1.0)
    validate_finite(output, path)
    atomic_save_npy(path, output)


def is_metadata_path(path: str | Path) -> bool:
    path = Path(path)
    return "__MACOSX" in path.parts or path.name.startswith("._") or path.name == ".DS_Store"


def iter_npy_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"NumPy image directory does not exist: {root}")
    return sorted(p for p in root.rglob("*.npy") if p.is_file() and not is_metadata_path(p))
