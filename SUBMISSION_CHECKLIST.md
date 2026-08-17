# Submission checklist

Last documentation audit: 2026-08-17.

Public repository: https://github.com/madhesh935/SEMICON

## Official i4C / KLA final submission check

Organizer email: *Final Submission Check for KLA Problem Statement* (AI-Based Restoration of Degraded Images), 17 Aug 2026.

Required layout:

```text
SEMICON/
|-- run.py
|-- requirements.txt
|-- README.md
`-- models/
```

Required command:

```bash
python run.py <input-dir> <output-dir>
```

- [x] Team folder contains `run.py`, `requirements.txt`, `README.md`, and `models/`
- [x] Official entry script is `run.py` (not `main.py` / `eval.py` / `evaluate.py` as the evaluator)
- [x] `run.py` reads all `.npy` files from the input directory
- [x] Creates the output directory if it does not exist
- [x] Writes exactly one restored `.npy` per input, with the same filename
- [x] Outputs are grayscale arrays of shape `(H, W)` (organizer also allows `(H, W, 1)`)
- [x] Pixel values are in `[0, 1]`
- [x] No `NaN` or `Inf` values
- [x] Restored images are the correct 2× target resolution
- [x] `requirements.txt` lists dependencies with versions
- [x] `README.md` has setup and execution instructions
- [x] Runs on an NVIDIA GPU
- [x] Fully offline: no internet, API keys, extra model downloads, user interaction, or manual config at run time
- [x] Evaluator I/O is `.npy` (not PNG/JPEG)

## Packaging

- [x] `README.md` has clone URL, install steps, inference command, and output figures
- [x] `run.py` is the official evaluator: `python run.py <input-dir> <output-dir>`
- [x] `models/best.pt` is committed and loaded automatically
- [x] `evaluate.py` remains as a compatibility alias and still loads `weights/best.pt`
- [x] Checkpoint SHA-256 is `7E2016F9BE1CA460366F88D0B1B54D6E026D81D3D1E16FBBCBDE3DAF13B349AC`
- [x] 400 restored test arrays are in `restored_test_outputs/`
- [x] `verify_submission.py` required files are present
- [x] Unit tests: **44 passed** (`python -m pytest -q`)

## Measured quality (do not rewrite)

Held-out 480-image seed-42 validation, `128×128 → 256×256`:

| Metric | Value |
|---|---|
| Bicubic PSNR / SSIM | 23.173149 dB / 0.539510 |
| Model PSNR / SSIM / LPIPS | 28.127191 dB / 0.760388 / 0.266695 |
| Gain vs bicubic | +4.954043 dB PSNR, +0.220878 SSIM |

These match `reports/validation_summary.json` and `reports/bicubic_summary.json`. No official-test quality score is claimed. Checkpoint promoted from a 20-epoch to a 40-epoch full run of the identical architecture/split (see `reports/final_vs_initial.md`).

## Still human-only

Replace these before a live idea-submission upload. They are not model or code blockers.

- [ ] Team name, member names, institution, contact
- [ ] Demo video URL and QR codes
- [ ] Export `TeamName_KLA_PS01.pdf` from the official submission template
