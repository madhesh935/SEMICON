# Model Card: Range-Aware LiteNAF-SR

## Summary

Range-Aware LiteNAF-SR is a single-channel, exactly 2× image-restoration network for KLA Problem Statement 01. It jointly denoises and super-resolves semiconductor inspection arrays without clipping the learned raw-input path. The selected model has 956,609 parameters; [`weights/best.pt`](../weights/best.pt) is a 3,993,611-byte offline inference bundle produced by a full real-data training run.

## Intended use

- Input: finite numeric grayscale `.npy` in `(H,W)`, `(1,H,W)`, or `(H,W,1)` form.
- Supported competition sizes: 128×128 and 256×256.
- Output: one `(2H,2W)` finite `float32` array in `[0,1]` after save-time clipping.
- Task: faithful removal of mixed speckle/Gaussian degradation plus 2× resolution restoration.
- Not intended for classification, defect decisions, orientation correction, or replacement of calibrated semiconductor metrology.

## Architecture

The stem consumes three channels derived without losing raw signal: `raw`, `clip(raw,0,1)`, and `raw - clip(raw,0,1)`. A 64-channel LR trunk contains 24 activation-free NAF-style residual blocks with per-pixel channel LayerNorm, depthwise convolution, simple gates, and channel attention. Convolution plus PixelShuffle performs learned 2× upsampling, followed by one HR refinement block and a single-channel residual head. The residual is added to a bicubic skip computed from the clipped input copy.

No BatchNorm, adversarial discriminator, diffusion sampling, transformer attention, or mandatory test-time ensemble is used. TTA is optional and disabled by default.

## Training data and split

- 3,200 organizer pairs were audited; all are `float32` 128×128 NoisyLR → 256×256 GT.
- Split: deterministic filename-random 85/15 split, seed 42; 2,720 train and 480 validation.
- Dataset fingerprint: `cf110aa74b939bc359e747ffb526de6482c81abae57c722de2fbd3115a3516ca`.
- No reliable source/group metadata was supplied, so group-aware holdout was not possible.
- No official test images or test GT were used in training.
- No synthetic degradation was added.

Training input values were preserved: the observed NoisyLR range was `[-0.278563, 2.158005]`; 3,182/3,200 files contain out-of-range pixels. GT was exactly in `[0,1]`. Full details are in [`reports/dataset_audit.json`](../reports/dataset_audit.json).

Training-fold degradation analysis found signal-dependent mixed corruption rather than constant-variance noise: signal versus squared LR discrepancy correlation was 0.363646, and a quadratic intensity-bin fit explained 99.923% of between-bin variance. Out-of-range samples remained highly correlated with area-downsampled GT (`r=0.932866`). Bicubic high-frequency energy was 0.037236 versus 0.049281 for GT. These observations supported raw-range preservation and an LR restoration trunk, but do not establish a unique physical noise model.

## Optimization

- 20 complete epochs / 6,800 AdamW steps, batch 8, LR patch 64.
- Base LR `2e-4`, one-epoch linear warm-up, cosine decay to 2% of base LR.
- Composite loss: 0.70 Charbonnier + 0.15 `(1-SSIM)` + 0.10 Sobel + 0.05 log FFT-magnitude.
- CUDA float16 model autocast; SSIM/Sobel/FFT loss calculations remain float32.
- Gradient norm clipped to 1.0 with scalar clipping; EMA decay 0.995.
- Joint H/V flips and rotations by multiples of 90° only.
- Recorded epoch wall time: 1,891.12 seconds on an RTX 4050 Laptop GPU, including validation each epoch.

A four-real-pair overfit test reduced fixed-patch composite loss by 47.68% and reached 31.8331 dB on those same pairs. Controlled 150-step screens covered three input representations, four loss combinations, three capacities, context, EMA, and three patch sizes. A full 20-epoch, 335,521-parameter raw+clipped/Loss-C challenger was also trained. It was faster but lost 0.196666 dB PSNR, 0.011016 SSIM, 0.019822 LPIPS, and both OOD-proxy metrics; the original full model therefore remains selected.

## Held-out results

All 480 validation images are 128→256. Metrics use prediction clipping to `[0,1]`, PSNR/SSIM data range 1.0, and pretrained AlexNet LPIPS with grayscale repeated to three channels and remapped to `[-1,1]`.

| Metric | Bicubic | Model |
|---|---:|---:|
| Mean PSNR ↑ | 23.173149 dB | 27.869329 dB |
| Mean SSIM ↑ | 0.539510 | 0.748968 |
| Mean LPIPS ↓ | not measured | 0.300719 |

At 0°/90°/180°/270°, mean PSNR was 27.8693/27.8708/27.8725/27.8707 dB. These are held-out validation results, not leaderboard or official-test scores.

The train-fitted structural OOD proxy contains 120 unusual held-out inputs. The selected model measured 28.540335 dB PSNR, 0.795055 SSIM, and 0.250644 LPIPS on that subset. This is a diagnostic stress split, not a claim about the unknown official OOD distribution.

## Efficiency

Measured with CUDA AMP, batch 1, 20 warm-ups, 100 synchronized trials on an RTX 4050 Laptop GPU:

- 128→256: 30.163 ms mean, 30.069 median, 31.852 p95, 33.15 images/s, 80.24 MiB peak allocated GPU memory, 14.889 G Conv MACs.
- 256→512: 103.970 ms mean, 103.954 median, 104.635 p95, 9.62 images/s, 279.95 MiB peak allocated GPU memory, 59.555 G Conv MACs.

MACs exclude interpolation, normalization, gating, and other elementwise operations. No H100 latency is claimed.

## Limitations and risks

- The supplied paired validation set contains no 256×256 LR examples. That path is shape-, range-, CLI-, latency-, and save-contract tested, but its PSNR/SSIM/LPIPS are unknown.
- The filename-random split cannot guarantee source independence without source metadata; visually related structures could exist across the split.
- OOD performance cannot be inferred fully from an in-repository split. The model avoids generative/adversarial objectives to reduce hallucination risk, but any learned SR model can still create or suppress fine structure.
- Outputs should be reviewed against calibrated inspection/metrology requirements before production use. This model does not make defect decisions.
- LPIPS is a natural-image perceptual metric and may not perfectly represent semiconductor metrology fidelity; PSNR/SSIM and visual inspection remain important.
- Test outputs are predictions only because organizer test GT was unavailable. No test score is claimed.

## Reproducibility and safety

The model loads through PyTorch's restricted `weights_only=True` path. Official [`evaluate.py`](../evaluate.py) imports only NumPy, PyTorch, and local modules; it never initializes LPIPS, reads GT, contacts a network, or downloads weights. Final saves are atomic where supported. The exact split, environment, training log, metrics, tests, and checkpoint are included in the repository. The unchanged legacy checkpoint omitted epoch/global-step/data fields; [`weights/best.metadata.json`](../weights/best.metadata.json) records them from retained logs and fresh validation without changing checkpoint bytes or output numerics.
