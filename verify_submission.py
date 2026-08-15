#!/usr/bin/env python
"""End-to-end structural and offline-inference submission verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from restoration.io import is_metadata_path, load_grayscale_npy
from restoration.utils import PROJECT_ROOT, contains_absolute_windows_path, load_model_from_checkpoint, select_device


REQUIRED_FILES = (
    ".gitignore",
    ".gitattributes",
    "README.md",
    "requirements.txt",
    "requirements-inference.txt",
    "environment_report.txt",
    "audit_dataset.py",
    "analyze_degradation.py",
    "benchmark_bicubic.py",
    "train.py",
    "validate.py",
    "robustness_test.py",
    "evaluate.py",
    "make_comparison.py",
    "compare_models.py",
    "score_experiment.py",
    "verify_submission.py",
    "restoration/__init__.py",
    "restoration/io.py",
    "restoration/data.py",
    "restoration/model.py",
    "restoration/losses.py",
    "restoration/metrics.py",
    "restoration/utils.py",
    "splits/split_seed42.json",
    "splits/validation_subsets_seed42.json",
    "weights/best.metadata.json",
    "reports/dataset_audit.json",
    "reports/dataset_audit.csv",
    "reports/bicubic_metrics.csv",
    "reports/bicubic_summary.json",
    "reports/validation_metrics.csv",
    "reports/validation_summary.json",
    "reports/test_output_manifest.json",
    "reports/robustness_metrics.csv",
    "reports/robustness_summary.json",
    "reports/degradation_analysis/summary.md",
    "reports/experiments.csv",
    "reports/experiments.md",
    "reports/final_vs_initial.md",
    "docs/model_card.md",
    "docs/presentation_content.md",
    "docs/demo_video_script.md",
    "SUBMISSION_CHECKLIST.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--weights", type=Path, help="Defaults to ROOT/weights/best.pt")
    parser.add_argument(
        "--test-input-dir",
        type=Path,
        help="Optional organizer test-input directory for exact name and 2x-shape comparison",
    )
    parser.add_argument("--skip-official-outputs", action="store_true", help="Use only before official test predictions are generated")
    parser.add_argument("--device", default="cpu", help="CPU by default makes this a portable packaging check")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def check_output(path: Path, input_hw: tuple[int, int]) -> list[str]:
    errors = []
    try:
        output = np.load(path, allow_pickle=False)
    except Exception as exc:
        return [f"{path}: cannot load output: {exc}"]
    expected = (2 * input_hw[0], 2 * input_hw[1])
    if output.shape != expected:
        errors.append(f"{path}: shape {output.shape}, expected {expected}")
    if output.dtype != np.float32:
        errors.append(f"{path}: dtype {output.dtype}, expected float32")
    if not np.isfinite(output).all():
        errors.append(f"{path}: contains NaN/Inf")
    if output.size and (float(output.min()) < 0.0 or float(output.max()) > 1.0):
        errors.append(f"{path}: values outside [0,1]")
    return errors


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    weights = (args.weights if args.weights is not None else root / "weights" / "best.pt").resolve()
    failures: list[str] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            failures.append(f"missing required file: {relative}")
    if not (root / "restored_test_outputs").is_dir():
        failures.append("missing required directory: restored_test_outputs")
    if not weights.is_file():
        failures.append(f"missing final weights: {weights}")
    else:
        try:
            device = select_device(args.device)
            model, checkpoint = load_model_from_checkpoint(weights, device)
            with torch.inference_mode():
                for height, width in ((128, 128), (256, 256)):
                    sample = torch.linspace(-0.2, 1.2, height * width, device=device).reshape(1, 1, height, width)
                    output = model(sample).float().clamp(0, 1)
                    if output.shape != (1, 1, height * 2, width * 2) or not torch.isfinite(output).all():
                        failures.append(f"loaded model failed {height}x{width} forward invariant")
            if checkpoint.get("training_kind") != "full":
                failures.append("weights checkpoint is not marked as a full training run")
            metadata_path = root / "weights" / "best.metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("training_complete") is not True:
                    failures.append("weight metadata does not mark training_complete=true")
                if metadata.get("checkpoint_sha256") != sha256_file(weights):
                    failures.append("weight metadata SHA-256 differs from weights/best.pt")
                if metadata.get("split_identity", {}).get("dataset_fingerprint") != checkpoint.get(
                    "split_identity", {}
                ).get("dataset_fingerprint"):
                    failures.append("weight metadata split fingerprint differs from checkpoint")
        except Exception as exc:
            failures.append(f"final weights failed to load/run: {exc}")
        finally:
            # The cross-working-directory CLI check starts a second process.
            # Release this diagnostic instance first on smaller local GPUs.
            if "model" in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    critical_paths = (
        list((root / "configs").glob("*.json"))
        + list((root / "splits").glob("*.json"))
        + [root / "weights" / "best.metadata.json"]
    )
    for path in critical_paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if contains_absolute_windows_path(payload):
                failures.append(f"absolute Windows path embedded in critical configuration: {path.relative_to(root)}")
        except Exception as exc:
            failures.append(f"invalid JSON {path.relative_to(root)}: {exc}")
    source_pattern = re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]", re.IGNORECASE)
    for path in [root / "evaluate.py", *(root / "restoration").glob("*.py")]:
        if path.is_file() and source_pattern.search(path.read_text(encoding="utf-8")):
            failures.append(f"absolute local path embedded in source: {path.relative_to(root)}")

    if weights.is_file():
        temp_base = root / ".tmp"
        temp_base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_base) as temporary:
            temporary = Path(temporary)
            input_dir = temporary / "synthetic_inputs"
            output_dir = temporary / "synthetic_outputs"
            outside_cwd = temporary / "different_working_directory"
            (input_dir / "nested").mkdir(parents=True)
            outside_cwd.mkdir()
            rng = np.random.default_rng(123)
            np.save(input_dir / "small.npy", rng.normal(0.5, 0.3, (128, 128)).astype(np.float32))
            np.save(input_dir / "nested" / "large.npy", rng.normal(0.5, 0.3, (256, 256)).astype(np.float32))
            command = [
                sys.executable,
                str(root / "evaluate.py"),
                str(input_dir),
                str(output_dir),
                "--weights",
                str(weights),
                "--device",
                args.device,
                "--batch-size",
                "2",
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(root)
            result = subprocess.run(command, cwd=outside_cwd, env=environment, text=True, capture_output=True)
            if result.returncode != 0:
                failures.append(
                    "evaluate.py failed from a different working directory: "
                    f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
                )
            expected = {Path("small.npy"): (128, 128), Path("nested/large.npy"): (256, 256)}
            actual = {path.relative_to(output_dir) for path in output_dir.rglob("*.npy")} if output_dir.exists() else set()
            if actual != set(expected):
                failures.append(f"synthetic output relative names mismatch: got {sorted(map(str, actual))}")
            for relative, shape in expected.items():
                if (output_dir / relative).is_file():
                    failures.extend(check_output(output_dir / relative, shape))

    if not args.skip_official_outputs:
        restored_root = root / "restored_test_outputs"
        manifest_path = root / "reports" / "test_output_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_files = {item["path"]: item for item in manifest["files"]}
            actual_files = sorted(
                path for path in restored_root.rglob("*.npy") if path.is_file() and not is_metadata_path(path)
            )
            actual_rel = {path.relative_to(restored_root).as_posix() for path in actual_files}
            if actual_rel != set(manifest_files):
                failures.append(
                    f"published output names/count differ from manifest: manifest={len(manifest_files)}, "
                    f"actual={len(actual_rel)}"
                )
            for path in actual_files:
                relative = path.relative_to(restored_root).as_posix()
                item = manifest_files.get(relative)
                if item is None:
                    continue
                output = np.load(path, allow_pickle=False)
                if list(output.shape) != item["shape"]:
                    failures.append(f"{relative}: shape differs from manifest")
                if str(output.dtype) != item["dtype"]:
                    failures.append(f"{relative}: dtype differs from manifest")
                if sha256_file(path) != item["sha256"]:
                    failures.append(f"{relative}: SHA-256 differs from manifest")
                if output.dtype != np.float32 or output.ndim != 2 or not np.isfinite(output).all():
                    failures.append(f"{relative}: invalid published output contract")
                elif output.size and (float(output.min()) < 0.0 or float(output.max()) > 1.0):
                    failures.append(f"{relative}: published values outside [0,1]")
            if sha256_file(weights) != manifest["weights"]["sha256"]:
                failures.append("final checkpoint SHA-256 differs from output manifest")
            if sha256_file(root / "evaluate.py") != manifest["evaluator"]["sha256"]:
                failures.append("evaluate.py SHA-256 differs from output manifest")

            if args.test_input_dir is not None:
                test_dir = args.test_input_dir.resolve()
                inputs = sorted(
                    path for path in test_dir.rglob("*.npy") if path.is_file() and not is_metadata_path(path)
                )
                expected_rel = {path.relative_to(test_dir).as_posix() for path in inputs}
                if actual_rel != expected_rel:
                    failures.append(
                        f"official output names/count mismatch: expected {len(expected_rel)}, got {len(actual_rel)}"
                    )
                for path in inputs:
                    relative = path.relative_to(test_dir)
                    output = restored_root / relative
                    if output.is_file():
                        failures.extend(check_output(output, load_grayscale_npy(path).shape))
        except Exception as exc:
            failures.append(f"published output verification failed: {exc}")

    if failures:
        print("SUBMISSION VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("SUBMISSION VERIFICATION PASSED")
    print(f"Weights: {weights}")
    print("Standalone evaluate.py passed from a different working directory for synthetic 128x128 and 256x256 inputs.")
    if not args.skip_official_outputs:
        print("Published output manifest, hashes, shapes, dtype, range, and finiteness passed.")
        if args.test_input_dir is not None:
            print("Official test-input names and exact 2x output shapes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
