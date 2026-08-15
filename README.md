# Range-Aware LiteNAF-SR: Semiconductor Image Restoration

Range-Aware LiteNAF-SR is the completed PyTorch solution for KLA Problem Statement 01. It receives a noisy, low-resolution grayscale semiconductor inspection image, suppresses speckle and Gaussian-like degradation, and reconstructs an image at exactly twice the input height and width. The design emphasizes faithful semiconductor structures and fast residual restoration instead of an unnecessarily heavy generative model.

The standalone evaluator is [`evaluate.py`](evaluate.py). It loads the included [`weights/best.pt`](weights/best.pt), needs no source edits, does not read ground truth, and does not download a model.

## Problem statement

Sharp inspection images matter because small edges, lines, contacts, and texture changes can carry manufacturing evidence. The supplied low-resolution images can combine:

- signal-dependent speckle-like corruption;
- Gaussian-like noise or softness;
- spatial resolution reduction;
- several degradations at once; and
- valid numeric intensities below `0` or above `1`.

A useful solution must handle familiar validation content and previously unseen structures or degradation combinations, preserve fine geometry, and remain efficient enough for benchmark inference. The hidden evaluation may include in-distribution and out-of-distribution cases, so validation quality alone is not a guarantee of deployment performance.

## Solution overview

The final checkpoint uses a 64-feature stem, 24 low-resolution NAF-style blocks, one high-resolution refinement block, expansion factor 2, no dropout, and three learned input channels. The three channels are the raw image, its `[0,1]`-clipped copy, and the signed out-of-range residual `raw - clipped`.

```text
Raw NoisyLR input
-> raw and clipped range-aware representation (3 channels)
-> 3x3 feature stem (width 64)
-> 24 lightweight NAF-style restoration blocks
-> learned PixelShuffle 2x upsampling
-> 1 high-resolution refinement block
-> predicted single-channel residual
+ clipped bicubic 2x skip
-> final evaluator clamp to [0,1]
-> restored float32 .npy
```

The low-resolution trunk also has a learned global residual around its 24-block body. The model contains no BatchNorm, GAN discriminator, diffusion sampler, or mandatory test-time ensemble. The final training objective was `0.70 Charbonnier + 0.15 (1-SSIM) + 0.10 Sobel + 0.05 log-FFT magnitude`.

Rotation tests apply the same rotation to input and ground truth. The output stays in the input's current orientation; the model does not rotate an image back to a canonical orientation.

## Key verified results

All quality metrics below are from the 480-image, seed-42 held-out validation split (`128x128 -> 256x256`). They are not hidden official-test scores.

| Verified item | Result | Conditions |
|---|---:|---|
| Bicubic PSNR | 23.173149 dB | validation mean |
| Bicubic SSIM | 0.539510 | validation mean |
| Model PSNR | **27.869329 dB** | validation mean |
| Model SSIM | **0.748968** | validation mean |
| Model LPIPS | **0.300719** | validation mean; lower is better |
| Improvement | **+4.696180 dB PSNR; +0.209458 SSIM** | model minus bicubic |
| Parameters | 956,609 | final model |
| Checkpoint size | 3,993,611 bytes | `weights/best.pt` |
| Batch-1 model time, `128->256` | 30.163 ms mean; 30.069 median; 31.852 p95 | RTX 4050 Laptop GPU, CUDA AMP, 20 warm-ups, 100 synchronized trials |
| Batch-1 model time, `256->512` | 103.970 ms mean; 103.954 median; 104.635 p95 | same RTX 4050 protocol; functional scale path only |
| 400-file evaluator run | 33.186 ms model time/image; 15.245 s wall | RTX 4050 Laptop GPU, batch 8, CUDA AMP |

The paired data contains no `256x256 -> 512x512` ground truth, so no quality score is claimed for that size. Official test ground truth is unavailable; no official test PSNR, SSIM, or LPIPS is claimed. Detailed measurements are in [`reports/validation_summary.json`](reports/validation_summary.json), [`reports/bicubic_summary.json`](reports/bicubic_summary.json), and [`reports/official_inference_summary.json`](reports/official_inference_summary.json).

## Repository structure

```text
README.md                      project guide and reproducible commands
evaluate.py                    standalone recursive inference CLI
train.py                       training and checkpoint-resume CLI
validate.py                    paired PSNR, SSIM, LPIPS, and latency validation
verify_submission.py           structural, checkpoint, output, and cross-CWD checks
requirements.txt               complete training/validation/test environment
requirements-inference.txt     minimal evaluator environment
restoration/                   model, strict NumPy I/O, data, losses, metrics, utilities
configs/                       recorded selected and diagnostic configurations
splits/                        immutable seed-42 split and validation subset definitions
tests/                         unit and evaluator CLI tests
weights/                       final checkpoint and provenance sidecar
restored_test_outputs/         400 released-test predictions and preserved paths
reports/                       measured summaries, manifests, logs, tables, and figures
docs/                          model card and presentation/demo material
```

Organizer datasets, virtual environments, caches, and archived experiment runs are intentionally excluded.

## Input and output contract

`evaluate.py` recursively discovers `.npy` files. [`restoration/io.py`](restoration/io.py) safely loads them with `allow_pickle=False` and accepts exactly these numeric, finite, grayscale array layouts:

| Stored input shape | Interpretation |
|---|---|
| `(H,W)` | two-dimensional grayscale |
| `(1,H,W)` | channel-first grayscale |
| `(H,W,1)` | channel-last grayscale |

RGB, multichannel, object, complex, empty, and NaN/Inf arrays are rejected. Inputs are converted to `float32` for inference without clipping the learned raw path. Values below `0` and above `1` are therefore supported.

| Input | Output | Verification status |
|---:|---:|---|
| `128x128` | `256x256` | paired validation and functional inference passed |
| `256x256` | `512x512` | functional shape/dtype/range/finiteness inference passed |

The implementation is spatially dynamic and unit-tested on additional compatible rectangular sizes. This is an implementation capability, not an additional official competition requirement or quality claim.

Every saved result is:

- a two-dimensional grayscale `.npy` array;
- exactly twice the input height and width;
- `float32`, finite, and clipped to `[0,1]`; and
- written under the output directory with the same relative path and filename as its input.

## Quick start for a fresh clone

Only Python 3.13.14 is verified in the recorded environment. The dependency files pin the versions used for the completed submission.

### Windows PowerShell

```powershell
git clone <PUBLIC_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-inference.txt
```

### Linux or an H100 benchmark host

```bash
git clone <PUBLIC_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-inference.txt
```

`requirements-inference.txt` contains only NumPy and the pinned PyTorch CUDA build, using PyTorch's official CUDA 13.0 package index. `requirements.txt` is the complete pinned environment for training, validation, LPIPS, figures, and tests:

```powershell
pip install -r requirements.txt
```

No separate PyTorch command is needed when using these files. On a managed H100 system, the local CUDA driver must be compatible with the pinned PyTorch wheel. Verify what the environment sees before benchmarking:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

After dependencies and repository files are present, `evaluate.py` runs offline.

## Model-weight verification

The final checkpoint is committed directly at `weights/best.pt`; this repository is not configured for Git LFS and no manual model download is required.

Windows PowerShell:

```powershell
Get-FileHash .\weights\best.pt -Algorithm SHA256
```

Linux:

```bash
sha256sum weights/best.pt
```

Expected SHA-256:

```text
10FE2F02EAA5ABEAFAA5050F8022D60282C03BF89AB864F871C51C41CE0A20AE
```

## Running inference

Primary positional form:

```powershell
python evaluate.py "path\to\test_inputs" "path\to\outputs"
```

Linux equivalent:

```bash
python evaluate.py /path/to/test_inputs /path/to/outputs
```

Equivalent flag forms use either underscores or hyphens:

```powershell
python evaluate.py --input_dir "path\to\test_inputs" --output_dir "path\to\outputs"
python evaluate.py --input-dir "path\to\test_inputs" --output-dir "path\to\outputs"
```

Supported optional controls are `--weights PATH`, `--batch-size N` (default `8`), `--device auto|cuda|cpu`, `--no-amp`, and `--tta`. TTA is a slower four-rotation self-ensemble and is off by default. Paths containing spaces work when quoted. Inputs of equal shape are batched together; subdirectories and filenames are preserved.

The evaluator prints the selected device and GPU name (when CUDA is used), resolved model path, checkpoint architecture, AMP and TTA status, discovered/processed/failure counts, model load time, total wall time, model inference time, average model time per image, and resolved output directory. Any invalid file is reported by name and makes the command return non-zero.

## Testing a custom `.npy` input

Place one finite grayscale `128x128` or `256x256` array in this layout:

```text
custom_test/
|-- inputs/
|   `-- custom_image.npy
`-- outputs/
```

Create the folders and run inference in PowerShell:

```powershell
New-Item -ItemType Directory -Force ".\custom_test\inputs"
New-Item -ItemType Directory -Force ".\custom_test\outputs"

python evaluate.py `
  ".\custom_test\inputs" `
  ".\custom_test\outputs"
```

The result is `custom_test/outputs/custom_image.npy`. A `128x128` input produces `(256,256)`; a `256x256` input produces `(512,512)`. Check both files safely:

```python
from pathlib import Path

import numpy as np

input_path = Path("custom_test/inputs/custom_image.npy")
output_path = Path("custom_test/outputs/custom_image.npy")
input_array = np.load(input_path, allow_pickle=False)
output_array = np.load(output_path, allow_pickle=False)

print("input shape:", input_array.shape)
print("output shape:", output_array.shape)
print("output dtype:", output_array.dtype)
print("output min/max:", float(output_array.min()), float(output_array.max()))
print("output finite:", bool(np.isfinite(output_array).all()))

assert output_array.shape == (2 * input_array.shape[-2], 2 * input_array.shape[-1])
assert output_array.dtype == np.float32
assert np.isfinite(output_array).all()
assert 0.0 <= float(output_array.min()) <= float(output_array.max()) <= 1.0
```

That final shape assertion assumes the custom file is stored as `(H,W)`. For a `256x256` input its expected output is explicitly `(512,512)`.

## Converting PNG/JPG to model-ready `.npy`

The complete environment includes Pillow. Change `source.png` to a PNG or JPG path and choose `target_size = 128` or `256`:

```python
from pathlib import Path

import numpy as np
from PIL import Image

source = Path("source.png")
destination = Path("custom_test/inputs/custom_image.npy")
target_size = 256

with Image.open(source) as image:
    grayscale = image.convert("L")
    resized = grayscale.resize(
        (target_size, target_size),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)

destination.parent.mkdir(parents=True, exist_ok=True)
np.save(destination, array, allow_pickle=False)
print(destination, array.shape, array.dtype, float(array.min()), float(array.max()))
```

Ordinary photographs and AI-generated circuit images are outside the real KLA training distribution. They are useful for checking that the pipeline runs, but they do not reliably represent expected semiconductor restoration quality.

## Viewing a restored output

Matplotlib is included in `requirements.txt`:

```python
import matplotlib.pyplot as plt
import numpy as np

input_array = np.load("custom_test/inputs/custom_image.npy", allow_pickle=False)
output_array = np.load("custom_test/outputs/custom_image.npy", allow_pickle=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.squeeze(input_array), cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Input {np.squeeze(input_array).shape}")
axes[1].imshow(output_array, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Restored {output_array.shape}")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()
```

Inspect for remaining grain, excessive smoothing, missing or weakened edges, ringing around strong boundaries, checkerboard patterns, and invented false structures. A visually pleasing result is not a substitute for paired quantitative validation.

## Testing with paired ground truth

Real PSNR, SSIM, and LPIPS require both a degraded low-resolution input and its clean, aligned high-resolution ground truth. Without GT, only successful inference, shape, dtype, range, finiteness, and visual quality can be checked.

The organizer data is not redistributed. To reproduce the frozen split without silently creating a different split, keep this relationship:

```text
DATA_ROOT/
|-- kla-semiconductor-restoration-submission/
|-- train (1)/
|   `-- train/
|       |-- NoisyLR/    # 3,200 arrays, 128x128
|       `-- GT/         # 3,200 arrays, 256x256
`-- Test_NoisyLR (2)/   # optional; not used by validation
```

From the repository root, the verified PowerShell validation form is:

```powershell
python validate.py `
  --root ".." `
  --noisy-dir "..\train (1)\train\NoisyLR" `
  --gt-dir "..\train (1)\train\GT" `
  --split ".\splits\split_seed42.json" `
  --weights ".\weights\best.pt" `
  --validation-subsets ".\splits\validation_subsets_seed42.json" `
  --report-dir ".\validation_reproduction" `
  --no-save-outputs
```

Do not omit `--split` or delete the supplied split file. The loader checks its seed, keys, and dataset fingerprint and refuses silent regeneration when the data identity differs. Full LPIPS validation needs the pretrained AlexNet metric resource in the configured cache and may download it once; `--no-lpips` is available only for an offline diagnostic run.

- PSNR: higher is better.
- SSIM: higher is better.
- LPIPS: lower is better.

## Running automated tests

Install the complete environment, then run:

```powershell
python -m pytest -q
python verify_submission.py
```

The verifier defaults to CPU, loads both supported sizes, launches `evaluate.py` from another working directory, and verifies checkpoint/output structure plus the committed 400-file manifest. If the organizer test inputs are locally available, also compare names and exact 2x shapes:

```powershell
python verify_submission.py --test-input-dir "..\Test_NoisyLR (2)\NoisyLR"
```

The current passing test count is recorded in [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) after the latest documentation audit.

## Training reproduction

Training is unnecessary for ordinary inference because the final checkpoint is included. Raw competition data is deliberately absent and must be obtained through the organizer.

The selected settings are recorded in [`configs/base.json`](configs/base.json) and [`reports/full_training_config.json`](reports/full_training_config.json). `train.py` has no `--config` flag, so the reproducible command supplies those values explicitly:

```powershell
python train.py `
  --root ".." `
  --noisy-dir "..\train (1)\train\NoisyLR" `
  --gt-dir "..\train (1)\train\GT" `
  --split ".\splits\split_seed42.json" `
  --run-kind full `
  --experiment "full_litenaf_w64b24_reproduction" `
  --epochs 20 --batch-size 8 --patch-size 64 `
  --width 64 --blocks 24 --hr-blocks 1 --expansion 2 `
  --input-representation raw_clipped_oor `
  --context-kernel 0 --loss composite `
  --lr 0.0002 --weight-decay 0.0001 `
  --warmup-epochs 1 --min-lr-ratio 0.02 `
  --grad-clip 1.0 --ema-decay 0.995 `
  --validation-interval 1 --early-stop-patience 6 `
  --workers 0 --seed 42
```

This uses AdamW with betas `(0.9, 0.99)`, 64-pixel LR patches and aligned 128-pixel GT patches, batch 8, 20 epochs, one-epoch linear warm-up, cosine decay, EMA, horizontal/vertical flips, and rotations by multiples of 90 degrees. Checkpoints are written to `runs/<experiment>/checkpoints/last.pt` and `best_psnr.pt`; the run summary and log stay under `runs/<experiment>/`.

Resume the same experiment from `last.pt` by repeating the identical command and adding `--resume`, or use `--resume PATH` for an explicit checkpoint. Do not add `--finalize` during an ordinary reproduction: that option publishes to `weights/best.pt`, which is already supplied and protected.

## Benchmarking and official evaluation

The expected KLA workflow is:

```text
Clone repository
-> install requirements
-> run evaluate.py as-is
-> pass hidden test-input and output directories
-> generate restored images
-> compare with hidden ground truth
-> calculate quality scores
-> measure inference time on an NVIDIA H100
```

Hidden inputs may contain in-distribution and out-of-distribution structures, simultaneous degradations, out-of-range intensities, and both stated 2x mappings. The exact hidden command, score weighting, batch size, and timing protocol are not published here. Local timing was measured only on an NVIDIA RTX 4050 Laptop GPU; no H100 result is claimed.

## Troubleshooting

### Activation script not found

Create the environment inside the cloned repository before activating it:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### PowerShell execution policy

Use a process-only policy change, which expires when that PowerShell process closes:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Activation is optional; run the environment interpreter directly instead:

```powershell
& ".\.venv\Scripts\python.exe" evaluate.py "inputs" "outputs"
```

### CUDA unavailable

`--device auto` falls back to CPU when CUDA is unavailable, and `--device cpu` selects it explicitly. CPU functional inference is supported, but practical speed is not promised. Check the PyTorch build and host driver if CUDA was expected.

### Out-of-memory errors

Close unnecessary applications, set `--batch-size 1` to process one image at a time, leave TTA disabled, use the inference-only environment, and do not load an entire directory of `.npy` arrays into RAM in a separate wrapper.

### Missing checkpoint

Confirm that `weights/best.pt` exists after cloning and verify its hash with the commands above. Git LFS is not configured for this repository. If the file is absent, obtain a complete clone from the repository owner rather than downloading an unverified checkpoint.

### Incorrect input

Use a finite numeric grayscale array with shape `(H,W)`, `(1,H,W)`, or `(H,W,1)`. RGB/multichannel, empty, object, complex, NaN, and Inf inputs are rejected. PNG/JPG files must first be converted to `.npy` as shown above.

## Limitations

- Official test ground truth is unavailable, so no official test PSNR, SSIM, or LPIPS is claimed.
- `256x256 -> 512x512` is functionally tested for execution, shape, dtype, range, and finiteness but has no paired quality score.
- Synthetic, ordinary photographic, or AI-generated custom images may be outside the training distribution.
- The model restores rotated input in its existing orientation; it does not canonicalize orientation.
- AI restoration cannot guarantee recovery of information that is completely absent from the input.
- The filename-random split cannot guarantee source independence because source/group metadata was unavailable.
- The model is not an independent semiconductor defect-decision system and requires downstream validation before operational use.

## References

- [Official i4C/KLA problem statement and submission requirements](https://i4c.in/hackathon-2026/#ps)
- Chen et al., [Simple Baselines for Image Restoration (NAFNet)](https://arxiv.org/abs/2204.04676), ECCV 2022
- Shi et al., [Efficient Sub-Pixel Convolution](https://arxiv.org/abs/1609.05158), CVPR 2016
- Wang et al., [Structural Similarity](https://doi.org/10.1109/TIP.2003.819861), IEEE TIP 2004
- Zhang et al., [LPIPS](https://arxiv.org/abs/1801.03924), CVPR 2018
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019

Human-only submission items are tracked in [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md): public repository URL, team/contact details, demo video/QR information, and final presentation export.
