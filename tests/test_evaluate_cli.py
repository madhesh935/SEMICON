from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from restoration.model import RangeAwareLiteNAFSR


def test_standalone_evaluate_from_different_cwd_preserves_names(tmp_path):
    root = Path(__file__).resolve().parents[1]
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    cwd = tmp_path / "elsewhere"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    cwd.mkdir()
    np.save(input_dir / "a.npy", np.linspace(-0.3, 1.4, 16 * 16, dtype=np.float32).reshape(16, 16))
    np.save(nested / "b.npy", np.ones((1, 12, 10), dtype=np.float32) * 1.2)
    (input_dir / ".DS_Store").write_bytes(b"metadata")
    model = RangeAwareLiteNAFSR(width=8, num_blocks=1, num_hr_blocks=0)
    checkpoint = {
        "format_version": 1,
        "architecture": "RangeAwareLiteNAFSR",
        "model_config": model.model_config(),
        "model_state": model.state_dict(),
        "ema_model_state": None,
        "training_kind": "synthetic_unit_test",
    }
    weights = tmp_path / "tiny.pt"
    torch.save(checkpoint, weights)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "evaluate.py"),
            str(input_dir),
            str(output_dir),
            "--weights",
            str(weights),
            "--device",
            "cpu",
            "--batch-size",
            "2",
        ],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*.npy")} == {"a.npy", "nested/b.npy"}
    for relative, expected_shape in (("a.npy", (32, 32)), ("nested/b.npy", (24, 20))):
        output = np.load(output_dir / relative, allow_pickle=False)
        assert output.shape == expected_shape
        assert output.dtype == np.float32
        assert np.isfinite(output).all()
        assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0

    flag_output_dir = tmp_path / "flag_outputs"
    flag_result = subprocess.run(
        [
            sys.executable,
            str(root / "evaluate.py"),
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(flag_output_dir),
            "--weights",
            str(weights),
            "--device",
            "cpu",
            "--batch-size",
            "2",
        ],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert flag_result.returncode == 0, flag_result.stdout + flag_result.stderr
    assert {path.relative_to(flag_output_dir).as_posix() for path in flag_output_dir.rglob("*.npy")} == {
        "a.npy",
        "nested/b.npy",
    }
