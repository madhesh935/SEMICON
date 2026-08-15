# Retraining Experiment Report

## Verdict

No retrained candidate is approved for promotion. The protected final model remains the strongest verified model.

## Baseline versus completed candidate

The fixed split was `splits/split_seed42.json` with 2,720 training records and 480 validation records.

| Model | Parameters | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Submitted `RangeAwareLiteNAFSR` | 956,609 | 27.869329127248292 | 0.7489683585877321 | 0.3007189181749709 |
| Completed archived retraining candidate | 335,521 | 27.67266314747503 | 0.737952285720825 | 0.320541 |

Candidate difference from the submitted model: `-0.196665979773262` dB PSNR, `-0.011015986849907` SSIM, and `+0.019822` LPIPS. Lower LPIPS is better, so this candidate is worse on all three validation measures.

The completed candidate used 20 epochs, the complete training split, seed 42, 64-pixel LR patches, AdamW, and a Charbonnier/SSIM/edge loss. Its training loss decreased from `0.0854201` to `0.0597564`, while validation PSNR improved from `25.058623` to `27.672663` through epoch 20. There was no validation-overfitting reversal in that run, but its final quality remained below the submitted model.

## New same-architecture attempt

A new full-data candidate was started in an external run directory using the submitted architecture and fixed split, with learning rate `1e-4`, 30-epoch budget, seed 42, and early stopping. Batch size 8 failed immediately with Windows/PyTorch `bad allocation`. A batch-size-4 retry failed with CUDA out-of-memory during the first backward pass before producing a validation metric. No candidate checkpoint was promoted.

## Protected artifacts

- `weights/best.pt` was not overwritten.
- The 400 official outputs were not overwritten.
- Raw training and test datasets were not modified.
- The submitted checkpoint remains 3,993,611 bytes with SHA-256 `10FE2F02EAA5ABEAFAA5050F8022D60282C03BF89AB864F871C51C41CE0A20AE`.

The completed candidate and failed-attempt logs remain outside the repository under `<external-retraining-run-root>`. The only repository additions from this analysis are this report and its JSON companion.

## Conclusion

There is no verified accuracy or restoration-quality improvement to adopt. The existing submitted model should remain unchanged. A future higher-capacity retraining should use a machine with sufficient CUDA memory and should be accepted only if it beats `27.869329127248292` PSNR, `0.7489683585877321` SSIM, and `0.3007189181749709` LPIPS on the same 480-image split without a validation-quality decline.
