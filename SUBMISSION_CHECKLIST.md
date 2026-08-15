# Submission checklist

Last documentation audit: 2026-08-15.

Public repository: https://github.com/madhesh935/SEMICON

## Packaging

- [x] `README.md` has clone URL, install steps, inference command, and output figures
- [x] `evaluate.py` is standalone and loads `weights/best.pt`
- [x] Checkpoint SHA-256 is `10FE2F02EAA5ABEAFAA5050F8022D60282C03BF89AB864F871C51C41CE0A20AE`
- [x] 400 restored test arrays are in `restored_test_outputs/`
- [x] `verify_submission.py` required files are present
- [x] Unit tests: **43 passed** (`python -m pytest -q`)

## Measured quality (do not rewrite)

Held-out 480-image seed-42 validation, `128×128 → 256×256`:

| Metric | Value |
|---|---|
| Bicubic PSNR / SSIM | 23.173149 dB / 0.539510 |
| Model PSNR / SSIM / LPIPS | 27.869329 dB / 0.748968 / 0.300719 |
| Gain vs bicubic | +4.696180 dB PSNR, +0.209458 SSIM |

These match `reports/validation_summary.json` and `reports/bicubic_summary.json`. No official-test quality score is claimed.

## Still human-only

Replace these before a live idea-submission upload. They are not model or code blockers.

- [ ] Team name, member names, institution, contact
- [ ] Demo video URL and QR codes
- [ ] Export `TeamName_KLA_PS01.pdf` from the official template using `docs/presentation_content.md`
