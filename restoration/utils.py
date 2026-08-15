"""Shared reproducibility, checkpoint, inference, and reporting utilities."""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .model import RangeAwareLiteNAFSR, create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_deterministic(seed: int = 42, deterministic_algorithms: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic_algorithms
    torch.backends.cudnn.deterministic = deterministic_algorithms
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _mps_is_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            candidate = source[name].detach()
            if value.is_floating_point():
                value.lerp_(candidate, 1.0 - self.decay)
            else:
                value.copy_(candidate)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state, strict=True)


def torch_load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        # Older checkpoints in this project stored torch.__version__ directly;
        # PyTorch represents it as this benign str subclass. Allowlist only
        # that exact class rather than enabling unrestricted pickle loading.
        torch_version_type = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
        if torch_version_type is None:
            raise
        with torch.serialization.safe_globals([torch_version_type]):
            checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"incompatible checkpoint {path}: expected a dictionary")
    return checkpoint


def load_model_from_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    prefer_ema: bool = True,
) -> tuple[RangeAwareLiteNAFSR, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(path, map_location="cpu")
    if checkpoint.get("architecture") not in {"RangeAwareLiteNAFSR", "Range-Aware LiteNAF-SR"}:
        raise RuntimeError(
            f"incompatible checkpoint {path}: architecture={checkpoint.get('architecture')!r}"
        )
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise RuntimeError(f"incompatible checkpoint {path}: missing model_config")
    state = None
    if prefer_ema:
        state = checkpoint.get("ema_model_state")
    state = state or checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"incompatible checkpoint {path}: missing model weights")
    model = create_model(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"incompatible checkpoint tensors in {path}: {exc}") from exc
    model.to(device).eval()
    return model, checkpoint


def inference_with_tta(model: nn.Module, tensor: torch.Tensor, enabled: bool = False) -> torch.Tensor:
    if not enabled:
        return model(tensor)
    restored = []
    for k in range(4):
        rotated = torch.rot90(tensor, k, dims=(-2, -1))
        prediction = model(rotated)
        restored.append(torch.rot90(prediction, -k, dims=(-2, -1)))
    return torch.stack(restored, dim=0).mean(dim=0)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def relative_or_absolute(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def contains_absolute_windows_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(contains_absolute_windows_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_absolute_windows_path(item) for item in value)
    if not isinstance(value, str):
        return False
    return len(value) >= 3 and value[1:3] in {":\\", ":/"}
