# AI-Based Restoration of Degraded Semiconductor Inspection Images

Range-Aware LiteNAF-SR is a lightweight deep-learning system designed to restore noisy, low-resolution grayscale semiconductor inspection images.

The model combines denoising and 2× super-resolution in a single inference pipeline. It accepts degraded NumPy images, suppresses speckle and Gaussian-like noise, preserves semiconductor structures, and reconstructs an output at exactly twice the input resolution.

<p align="left">
<img src="docs/readme_gallery/gallery_01.png" width="220" height="220" alt="128 to 256 and 256 to 512 restoration" />
<img src="docs/readme_gallery/gallery_02.png" width="220" height="220" alt="Noisy input, restored output, and GT reference" />
<img src="docs/readme_gallery/gallery_03.png" width="220" height="220" alt="Training-set noisy, restored, and ground truth" />
<br>
<img src="docs/readme_gallery/gallery_04.png" width="220" height="220" alt="Official test noisy input versus restored output" />
<img src="docs/readme_gallery/gallery_05.png" width="220" height="220" alt="Example NoisyLR and ground truth pair" />
</p>

The evaluator is [`evaluate.py`](evaluate.py). It loads the committed checkpoint [`weights/best.pt`](weights/best.pt), does not need source edits, does not read ground truth, and does not download a model.

```text
Raw NoisyLR
  -> raw + clipped + out-of-range residual (3 channels)
  -> 3x3 stem (width 64)
  -> 24 LiteNAF restoration blocks
  -> PixelShuffle 2x
  -> 1 HR refinement block
  -> residual + clipped bicubic skip
  -> clamp to [0, 1]
  -> float32 .npy
```

Final loss: `0.70 Charbonnier + 0.15 (1-SSIM) + 0.10 Sobel + 0.05 log-FFT magnitude`. 956,609 parameters. No BatchNorm, GAN, or diffusion sampler.

Repository: [https://github.com/madhesh935/SEMICON](https://github.com/madhesh935/SEMICON)

## Accuracy — keep these scores

The numbers below already match the saved reports. They do **not** need to be edited. Retraining is not required for inference.

All quality scores are the **480-image seed-42 validation split** (`128×128 → 256×256`). They are **not** hidden official-test scores. Official test ground truth is not in this repo.

| Metric | Bicubic | Model | Gain |
|---|---:|---:|---:|
| Mean PSNR ↑ | 23.173149 dB | **27.869329 dB** | **+4.696180 dB** |
| Mean SSIM ↑ | 0.539510 | **0.748968** | **+0.209458** |
| Mean LPIPS ↓ | — | **0.300719** | — |

| Extra check | Result |
|---|---|
| Median PSNR | 27.706 dB |
| Images ≥ 25 dB | 349 / 480 |
| Images &lt; 20 dB | 12 / 480 (hard high-noise cases) |
| Best / worst PSNR | 42.269 dB / 10.570 dB |
| OOD-proxy (120 images) | 28.540 dB PSNR, 0.795 SSIM, 0.251 LPIPS |
| Rotation PSNR 0°/90°/180°/270° | 27.869 / 27.871 / 27.872 / 27.871 dB |
| Parameters / checkpoint | 956,609 / 3,993,611 bytes |
| RTX 4050, batch 1, 128→256 | 30.163 ms mean |
| RTX 4050, batch 1, 256→512 | 103.970 ms mean (shape path only; no GT score) |
| 400-file evaluator run | 33.186 ms/image; 15.245 s wall; 0 failures |

Sources: [`reports/validation_summary.json`](reports/validation_summary.json), [`reports/bicubic_summary.json`](reports/bicubic_summary.json), [`reports/official_inference_summary.json`](reports/official_inference_summary.json).

**Verdict:** keep `weights/best.pt`. A smaller faster challenger lost 0.197 dB PSNR and 0.011 SSIM. Mean quality is clearly above bicubic. A small tail of extreme-noise images stays difficult; that is recorded, not hidden.

More panels: [`reports/figures/`](reports/figures/) and [`reports/final_model_test_figures/`](reports/final_model_test_figures/).

## Run

### 1. Clone

```bash
git clone https://github.com/madhesh935/SEMICON.git
cd SEMICON
```

### 2. Install

Recorded training/eval environment is Python 3.13 with the pinned CUDA 13.0 PyTorch wheel. On this Mac, use CPU PyTorch.

**macOS / CPU**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Linux or Windows with NVIDIA GPU** (matches the submitted environment)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-inference.txt
```

Full training / pytest / figures / LPIPS:

```bash
pip install -r requirements.txt
```

Check the device:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Confirm the checkpoint

```bash
shasum -a 256 weights/best.pt
# Windows: Get-FileHash .\weights\best.pt -Algorithm SHA256
```

Expected SHA-256:

```text
10FE2F02EAA5ABEAFAA5050F8022D60282C03BF89AB864F871C51C41CE0A20AE
```

### 4. Run inference

```bash
python evaluate.py /path/to/test_inputs /path/to/outputs
```

Same command with flags:

```bash
python evaluate.py --input-dir /path/to/test_inputs --output-dir /path/to/outputs
```

Useful options: `--weights PATH`, `--batch-size N` (default `8`), `--device auto|cuda|cpu`, `--no-amp`, `--tta`. TTA is a slower 4-rotation ensemble and is off by default.

### 5. Run the bundled demo

The repo already includes a 256×256 input and its 512×512 restoration:

```bash
python evaluate.py custom_test_256/inputs custom_test_256/outputs
```

Input: `custom_test_256/inputs/synthetic_semiconductor_test_256.npy`  
Output: `custom_test_256/outputs/synthetic_semiconductor_test_256.npy`

```python
from pathlib import Path
import numpy as np

inp = np.load(Path("custom_test_256/inputs/synthetic_semiconductor_test_256.npy"), allow_pickle=False)
out = np.load(Path("custom_test_256/outputs/synthetic_semiconductor_test_256.npy"), allow_pickle=False)
print(inp.shape, out.shape, out.dtype, float(out.min()), float(out.max()))
assert out.shape == (2 * inp.shape[-2], 2 * inp.shape[-1])
assert out.dtype == np.float32
assert np.isfinite(out).all()
assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0
```

To view:

```python
import matplotlib.pyplot as plt
import numpy as np

inp = np.load("custom_test_256/inputs/synthetic_semiconductor_test_256.npy", allow_pickle=False)
out = np.load("custom_test_256/outputs/synthetic_semiconductor_test_256.npy", allow_pickle=False)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.squeeze(inp), cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Input {np.squeeze(inp).shape}")
axes[1].imshow(out, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Restored {out.shape}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Input and output contract

`evaluate.py` recursively finds `.npy` files. [`restoration/io.py`](restoration/io.py) loads them with `allow_pickle=False`.

Accepted grayscale layouts: `(H,W)`, `(1,H,W)`, `(H,W,1)`. RGB, object, complex, empty, NaN, and Inf arrays are rejected. Raw values below `0` or above `1` are kept on the learned path.

| Input | Output | Status |
|---|---|---|
| 128×128 | 256×256 | paired validation + inference |
| 256×256 | 512×512 | shape / dtype / range / finiteness only |

Every saved file is 2D grayscale `.npy`, exactly 2× size, `float32`, finite, clipped to `[0,1]`, with the same relative path as the input.

## Repository layout

```text
evaluate.py                 standalone inference
train.py / validate.py      train and paired metrics
verify_submission.py        packaging + checkpoint + output checks
restoration/                model, I/O, data, losses, metrics
weights/best.pt             final checkpoint
custom_test_256/            bundled 256→512 demo
restored_test_outputs/      400 released-test predictions
reports/                    measured summaries and comparison figures
docs/                       model card and demo notes
splits/                     immutable seed-42 split
tests/                      43 unit / CLI tests
```

Organizer train/test arrays are not redistributed.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
python verify_submission.py
```

Passing count from the last full audit: **43 passed**. Details: [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md).

If organizer test inputs are present:

```bash
python verify_submission.py --test-input-dir "../Test_NoisyLR (2)/NoisyLR"
```

## Validation with ground truth

Organizer data must sit next to the clone, not inside it:

```text
DATA_ROOT/
|-- SEMICON/                  # this repository
|-- train (1)/train/NoisyLR   # 3,200 × 128×128
|-- train (1)/train/GT        # 3,200 × 256×256
`-- Test_NoisyLR (2)/NoisyLR  # optional 400 test inputs
```

```bash
python validate.py \
  --root .. \
  --noisy-dir "../train (1)/train/NoisyLR" \
  --gt-dir "../train (1)/train/GT" \
  --split ./splits/split_seed42.json \
  --weights ./weights/best.pt \
  --validation-subsets ./splits/validation_subsets_seed42.json \
  --report-dir ./validation_reproduction \
  --no-save-outputs
```

Do not omit `--split`. The loader checks seed, keys, and dataset fingerprint.

## Training

Not needed for ordinary inference. Selected settings: [`configs/base.json`](configs/base.json).

```bash
python train.py \
  --root .. \
  --noisy-dir "../train (1)/train/NoisyLR" \
  --gt-dir "../train (1)/train/GT" \
  --split ./splits/split_seed42.json \
  --run-kind full \
  --experiment full_litenaf_w64b24_reproduction \
  --epochs 20 --batch-size 8 --patch-size 64 \
  --width 64 --blocks 24 --hr-blocks 1 --expansion 2 \
  --input-representation raw_clipped_oor \
  --context-kernel 0 --loss composite \
  --lr 0.0002 --weight-decay 0.0001 \
  --warmup-epochs 1 --min-lr-ratio 0.02 \
  --grad-clip 1.0 --ema-decay 0.995 \
  --validation-interval 1 --early-stop-patience 6 \
  --workers 0 --seed 42
```

Resume with the same command plus `--resume`. Do not pass `--finalize` unless you intend to overwrite `weights/best.pt`.

## Troubleshooting

- **CUDA missing:** `--device auto` falls back to CPU. `--device cpu` forces it. CPU works; it is slower.
- **Out of memory:** `--batch-size 1`, leave TTA off.
- **Missing checkpoint:** confirm `weights/best.pt` and the SHA-256 above. This repo does not use Git LFS.
- **Bad input:** grayscale `.npy` only. Convert PNG/JPG first (Pillow is in `requirements.txt`).

PNG/JPG conversion:

```python
from pathlib import Path
import numpy as np
from PIL import Image

source = Path("source.png")
destination = Path("custom_test_256/inputs/custom_image.npy")
target_size = 256

with Image.open(source) as image:
    array = np.asarray(
        image.convert("L").resize((target_size, target_size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / np.float32(255.0)

destination.parent.mkdir(parents=True, exist_ok=True)
np.save(destination, array, allow_pickle=False)
```

Ordinary photos and synthetic circuit images are outside the KLA training distribution. They are useful to check that the pipeline runs.

## Limitations

- No official-test PSNR/SSIM/LPIPS is claimed.
- `256×256 → 512×512` is functionally tested only.
- About 12/480 validation images stay below 20 dB PSNR under heavy noise.
- The model restores the input orientation; it does not un-rotate an image.
- Filename-random split cannot guarantee source independence.
- This is not a defect-decision system.

## References

- [i4C / KLA problem statement](https://i4c.in/hackathon-2026/#ps)
- Chen et al., [NAFNet](https://arxiv.org/abs/2204.04676), ECCV 2022
- Shi et al., [Efficient Sub-Pixel Convolution](https://arxiv.org/abs/1609.05158), CVPR 2016
- Wang et al., [SSIM](https://doi.org/10.1109/TIP.2003.819861), IEEE TIP 2004
- Zhang et al., [LPIPS](https://arxiv.org/abs/1801.03924), CVPR 2018
- Loshchilov and Hutter, [AdamW](https://arxiv.org/abs/1711.05101), ICLR 2019

Human submission items (team name, demo video, presentation export) are in [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md).
