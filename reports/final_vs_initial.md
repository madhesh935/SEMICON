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
