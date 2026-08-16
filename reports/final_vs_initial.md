# Final Versus Initial Solution

## Outcome

The initial fully trained Range-Aware LiteNAF-SR remains the final selected model. A genuinely completed faster candidate was trained for the same 20 epochs / 6,800 steps, but it lost fidelity on every primary and OOD-proxy metric. Therefore `weights/best.pt` was deliberately preserved byte-for-byte rather than replacing it for a speed-only gain.

| Measurement | Initial incumbent | Full challenger | Final selected |
|---|---:|---:|---:|
| Input representation | raw + clipped + OOR | raw + clipped | raw + clipped + OOR |
| Width / LR blocks | 64 / 24 | 48 / 12 | 64 / 24 |
| Loss | composite incl. FFT | Charbonnier + SSIM + edge | composite incl. FFT |
| Parameters | 956,609 | 335,521 | 956,609 |
| Validation PSNR | **27.869329** | 27.672663 | **27.869329** |
| Validation SSIM | **0.748968** | 0.737952 | **0.748968** |
| Validation LPIPS (lower is better) | **0.300719** | 0.320541 | **0.300719** |
| OOD-proxy PSNR | **28.540335** | 28.327947 | **28.540335** |
| OOD-proxy SSIM | **0.795055** | 0.782696 | **0.795055** |
| OOD-proxy LPIPS | **0.250644** | 0.272179 | **0.250644** |
| RTX 4050 batch-1 latency, 128 input | 30.163 ms | **16.824 ms** | 30.163 ms |
| RTX 4050 batch-1 latency, 256 input | 103.970 ms | **52.227 ms** | 103.970 ms |
| Peak allocated VRAM at 256 input | 279.95 MiB | **211.56 MiB** | 279.95 MiB |
| Published inference checkpoint | 3,993,611 bytes | not exported (resume checkpoint: 5,789,149 bytes) | 3,993,611 bytes |

The challenger was about 65% smaller and 44%/50% faster at the two input sizes, but lost 0.196666 dB PSNR, 0.011016 SSIM, increased LPIPS by 0.019822, and lost 0.212388 dB OOD-proxy PSNR. For semiconductor inspection, that consistent fidelity regression did not justify promotion.

## Improvements retained

- Dataset audit now reports below-zero and above-one file/pixel counts separately.
- Measured degradation analysis, FFT diagnostics, OOR correlations, and a training-fitted structural OOD proxy were added.
- The implementation can now run controlled raw/raw+clipped/raw+clipped+OOR, four-loss, capacity, context, EMA, and patch-size experiments without changing inference numerics of the selected checkpoint.
- Validation now reports the OOD proxy and synchronized mean/median/p95 latency for both supported sizes.
- Training checkpoints created by the revised pipeline record completion, experiment, step, data, and split metadata. A truthful sidecar supplies missing legacy provenance for the unchanged selected bundle.
- Full-image structural diagnostics and difficult-sample panels were added. No systematic PixelShuffle checkerboard justified an upsampler change.

## Negative results retained in the record

- Raw-only input was weaker; OOR gave no decisive controlled-screen advantage over raw+clipped.
- FFT loss gave no meaningful short-screen benefit, but removing it in the full challenger did not beat the incumbent.
- A 7x7 depthwise context branch slightly improved random PSNR while reducing OOD metrics.
- Short-horizon EMA lagged badly.
- Patch sizes 96 and 128 reduced PSNR/OOD PSNR relative to patch 64.
- The faster full challenger did not meet the fidelity selection rule.
- Synthetic degradation was not attempted because the measurements did not support a trustworthy generator.

All values above are measured on the fixed training-data-only validation split; none is an official hidden-test score.

## 2026-08-16 update: 40-epoch promotion

The 20-epoch incumbent above (`full_litenaf_w64b24`) was superseded by a 40-epoch full run of the **identical** architecture, loss, optimizer, and train/validation split (`full_litenaf_w64b24_longer40epoch`). The training-log trajectory at 20 epochs had not plateaued (validation PSNR was still improving epoch-over-epoch, with the LR schedule's cosine decay recomputed over the longer horizon rather than simply continuing training at an already-near-zero learning rate), so doubling the epoch budget was tested as a low-risk, architecture-unchanged lever.

| Measurement | 20-epoch (superseded) | 40-epoch (current `weights/best.pt`) | Change |
|---|---:|---:|---:|
| Validation PSNR | 27.869329 | **28.127191** | **+0.257862 dB** |
| Validation SSIM | 0.748968 | **0.760388** | **+0.011420** |
| Validation LPIPS (lower is better) | 0.300719 | **0.266695** | **−0.034024** |
| OOD-proxy PSNR (120 images) | 28.540335 | **28.846967** | **+0.306632 dB** |
| OOD-proxy SSIM | 0.795055 | **0.806130** | **+0.011075** |
| OOD-proxy LPIPS | 0.250644 | **0.218460** | **−0.032184** |
| Images below 20 dB PSNR (of 480) | 12 | 12 | unchanged (same hard high-noise tail) |
| Parameters | 956,609 | 956,609 | unchanged (same architecture) |
| 400-file evaluator run | 33.186 ms/image | 33.362 ms/image | unchanged within measurement noise |

All three quality metrics improved together on both the random and OOD-proxy validation subsets, with no architecture, parameter-count, or measured sustained-throughput change. The training run needed three resumes from checkpoint after host interruptions (a host-memory allocation error, a CUDA out-of-memory error, and a cuDNN execution failure — all typical of this shared laptop GPU under background load, not code defects); `train.py`'s existing checkpoint/resume mechanism recovered cleanly each time with no lost progress.

The organizer dataset was re-extracted to a different local path (`Dataset/train/train/{NoisyLR,GT}`) for this run rather than the original `../train (1)/train/{NoisyLR,GT}` sibling location, which changes the path-dependent `dataset_fingerprint` recorded in the new checkpoint versus `splits/split_seed42.json`. The actual `train_keys`/`validation_keys` partition was verified identical (same seed=42, same 2,720/480 split, same filenames) before training, so this is a provenance/path detail, not a different split. See `weights/best.metadata.json` for the full note.

The prior 20-epoch checkpoint, its restored test outputs, and its reports were backed up before this promotion and are not redistributed in this repository.
