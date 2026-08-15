# 9-Slide Presentation Content

Use the official idea-submission template. Replace every bracketed placeholder, remove its instruction slide, and export the final deck as `TeamName_KLA_PS01.pdf` (8–9 slides maximum).

## Slide 1 — Team details

- Team: **[TEAM NAME]**
- Members and roles: **[NAMES + CV/ML/DEPLOYMENT/PRESENTATION ROLES]**
- Institution: **[COLLEGE/ORGANIZATION]**
- Contact: **[EMAIL/PHONE]**
- Problem: KLA PS01 — AI-Based Restoration of Degraded Images

## Slide 2 — Why restoration matters

- Microscopic inspection depends on sharp edges and faithful fine structures; noise or lost resolution can obscure manufacturing evidence.
- One input may combine speckle excursions outside the nominal range, Gaussian softness/noise, and 2× downsampling in unknown order.
- Required mapping is grayscale 128→256 or 256→512, including OOD structures, with accuracy and H100 inference speed both scored.
- Our audit found 3,182/3,200 training inputs outside `[0,1]`, proving that early clipping would discard real supplied signal.

Suggested visual: one audited NoisyLR/GT example with the raw minimum/maximum annotated.

## Slide 3 — Core idea

**Preserve range information, restore only the missing residual.**

- A range-aware stem sees raw intensity, a clipped copy, and the signed out-of-range residual.
- A compact NAF-style LR trunk removes mixed noise without BatchNorm.
- PixelShuffle learns exact 2× detail reconstruction.
- A bicubic skip anchors low-frequency content; the network predicts only the HR correction.
- No GAN/diffusion objective: semiconductor fidelity is prioritized over plausible hallucination.

## Slide 4 — Architecture and training

Pipeline diagram content:

```text
Raw NoisyLR ─┬─ raw ───────────────┐
             ├─ clip(raw) ─────────┼→ 3x3 stem → 24 LiteNAF blocks → global LR residual
             └─ raw-clip(raw) ─────┘                              ↓
                                                        Conv + PixelShuffle x2
                                                                  ↓
                                                        1 HR refinement block
                                                                  ↓
                                                        predicted HR residual
clip(raw) → bicubic x2 ─────────────────────────────────────────── (+) → restored
```

- 956,609 parameters; 20 epochs, 6,800 AdamW steps; 64×64 LR aligned patches, batch 8.
- Safe joint augmentations: H/V flips and 0°/90°/180°/270° rotations.
- Loss: 0.70 Charbonnier + 0.15 SSIM + 0.10 Sobel + 0.05 FFT magnitude.
- Float16 model AMP, float32 sensitive losses, gradient clipping, EMA, warm-up + cosine.

## Slide 5 — Innovation and evidence-driven choices

- Raw range is never blindly clipped in the feature branch; the auxiliary residual explicitly marks excursion direction and magnitude.
- Bicubic residual learning improves stability and constrains low-frequency behavior.
- Interpolation-free geometric augmentation protects microscopic edges.
- Full audit, immutable fingerprinted split, exact-stem + 2× pairing, canonical train/test duplicate hashes.
- Controlled screens compared three range encodings, four losses, three capacities, context, EMA, and 64/96/128 patches on the same seed-42 split.
- A full 335,521-parameter challenger was 44% faster at 128 input but lost 0.196666 dB PSNR, 0.011016 SSIM, 0.019822 LPIPS, and both OOD-proxy metrics; fidelity won the selection decision.
- Four-real-pair overfit: fixed-patch loss reduced 47.68%, demonstrating learnability before full training.

## Slide 6 — Real measured results

Use only these held-out 480-image validation measurements:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Bicubic | 23.173149 | 0.539510 | — |
| Ours | **27.869329** | **0.748968** | **0.300719** |

- Improvement: **+4.696180 dB PSNR**, **+0.209458 SSIM**.
- Train-fitted structural OOD proxy (120 images): **28.540335 dB**, **0.795055 SSIM**, **0.250644 LPIPS**. Label this as a proxy, not an official OOD score.
- Rotation PSNR at 0°/90°/180°/270°: 27.8693/27.8708/27.8725/27.8707 dB.
- Insert 2–3 images from `reports/figures/comparison_*.png`: NoisyLR | Bicubic | restoration | GT, with detail crops.
- Disclosure: the paired validation set contains only 128→256; no 256→512 quality score is invented.

## Slide 7 — Technology and feasibility

- Stack: Python 3.13, PyTorch 2.11, CUDA 13.0 runtime, NumPy, scikit-image; offline `.npy` inference.
- Training hardware: RTX 4050 Laptop GPU, 6 GB VRAM; 20 epochs in 1,891 recorded seconds including validation.
- Checkpoint: 3.994 MB; 956,609 parameters.
- RTX 4050 batch-1 latency: 30.163 ms mean / 31.852 ms p95 (128→256), 103.970 ms mean / 104.635 ms p95 (256→512).
- Peak allocated inference VRAM: 80.24 MiB / 279.95 MiB.
- `evaluate.py`: recursive, same-shape batching, atomic saves, CUDA/CPU auto-select, no GT/network/LPIPS dependency.
- H100 timing will be measured by KLA; no local H100 result is claimed.

## Slide 8 — Repository and demo

- Public GitHub: **[ADD PUBLIC GITHUB URL]**
- Demo video: **[ADD VIDEO URL]**
- One-command inference:

```text
python evaluate.py TEST_IMAGES OUTPUT_DIR
```

- Repository includes final weights, 400 actual restored test arrays, fixed split/OOD proxy, degradation and experiment reports, comparisons, 43 tests, and submission verifier.
- QR codes: **[ADD GITHUB + VIDEO QR CODES]**

## Slide 9 — References

- i4C, [KLA PS01 official problem statement](https://i4c.in/hackathon-2026/#ps).
- Chen et al., [Simple Baselines for Image Restoration (NAFNet)](https://arxiv.org/abs/2204.04676), ECCV 2022.
- Shi et al., [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN](https://arxiv.org/abs/1609.05158), CVPR 2016.
- Wang et al., [Image Quality Assessment: From Error Visibility to Structural Similarity](https://doi.org/10.1109/TIP.2003.819861), IEEE TIP 2004.
- Zhang et al., [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924), CVPR 2018.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019.

Final reminder: update team/link placeholders, use the official template, remove instruction slides, keep 8–9 slides, and export exactly as `TeamName_KLA_PS01.pdf`.
