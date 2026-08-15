from __future__ import annotations

import numpy as np
import pytest

from restoration.io import ArrayValidationError, load_grayscale_npy, normalize_array_shape, save_restored_npy


@pytest.mark.parametrize("shape", [(9, 7), (1, 9, 7), (9, 7, 1)])
def test_shape_normalization(shape):
    array = np.zeros(shape, dtype=np.float32)
    assert normalize_array_shape(array).shape == (9, 7)


@pytest.mark.parametrize("shape", [(3, 9, 7), (9, 7, 3), (1, 1, 9, 7), (7,)])
def test_rejects_multichannel_or_unsupported_shapes(shape):
    with pytest.raises(ArrayValidationError):
        normalize_array_shape(np.zeros(shape, dtype=np.float32), "invalid.npy")


def test_rejects_object_empty_and_nonfinite(tmp_path):
    with pytest.raises(ArrayValidationError):
        normalize_array_shape(np.array([object()], dtype=object), "object.npy")
    with pytest.raises(ArrayValidationError):
        normalize_array_shape(np.empty((0, 4), dtype=np.float32), "empty.npy")
    with pytest.raises(ArrayValidationError, match="unsupported dtype"):
        normalize_array_shape(np.ones((2, 2), dtype=np.complex64), "complex.npy")
    path = tmp_path / "nan.npy"
    np.save(path, np.array([[0.0, np.nan], [1.0, np.inf]], dtype=np.float32))
    with pytest.raises(ArrayValidationError, match="non-finite"):
        load_grayscale_npy(path)


def test_load_preserves_raw_range_and_casts_float32(tmp_path):
    path = tmp_path / "raw.npy"
    np.save(path, np.array([[-0.5, 0.25], [1.25, 2.0]], dtype=np.float64))
    result = load_grayscale_npy(path)
    assert result.dtype == np.float32
    assert result.min() == pytest.approx(-0.5)
    assert result.max() == pytest.approx(2.0)


def test_final_save_enforces_dtype_range_and_finiteness(tmp_path):
    path = tmp_path / "nested" / "restored.npy"
    save_restored_npy(path, np.array([[-2.0, 0.5], [1.5, 0.8]], dtype=np.float64), (2, 2))
    restored = np.load(path, allow_pickle=False)
    assert restored.shape == (2, 2)
    assert restored.dtype == np.float32
    assert np.isfinite(restored).all()
    assert float(restored.min()) == 0.0
    assert float(restored.max()) == 1.0
