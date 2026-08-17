# Submission checklist

Last documentation audit: 2026-08-16.

Public repository: https://github.com/madhesh935/SEMICON

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
