# Controlled Experiment Registry

All learned screens use the immutable seed-42 split. Unless marked full, a run used the same 256 training pairs, 64 diagnostic validation pairs, 150 optimizer steps, seed 42, batch 4, and no EMA. Every PSNR/SSIM value in the table was subsequently measured on all 480 held-out images; OOD metrics use the 120-image, input-only proxy fitted from training descriptors. Official test data was never opened by experiment or selection code.

Blank LPIPS cells mean the natural-image perceptual metric was deferred during screening. Latency is CUDA-synchronized RTX 4050 timing, but laptop clock-state variation means it is a supporting rather than sole selection signal.

| ID | Representation | Width/blocks | Loss | Patch | Params | PSNR | SSIM | OOD PSNR | OOD SSIM | 128 ms | 256 ms | Decision |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 00_bicubic | clipped | n/a | none | n/a | 0 | 23.1731 | 0.53951 | 24.1736 | 0.59936 | 0.068 | 0.163 | Baseline |
| 01_current_full | raw+clipped+OOR | 64/24 | D: composite | 64 | 956,609 | **27.8693** | **0.74897** | **28.5403** | **0.79506** | 30.16 | 103.97 | **Selected** |
| 02_rep_raw | raw | 32/6 | D | 64 | 104,417 | 25.9348 | 0.64124 | 26.4483 | 0.67960 | 20.82 | 30.44 | Reject |
| 03_rep_raw_clipped | raw+clipped | 32/6 | D | 64 | 104,705 | 26.0364 | 0.64830 | 26.5500 | 0.68953 | 10.12 | 27.97 | Best screen balance |
| 04_rep_raw_clipped_oor | raw+clipped+OOR | 32/6 | D | 64 | 104,993 | 26.0434 | 0.64756 | 26.5505 | 0.68771 | 18.95 | 31.45 | Reject: SSIM/OOD trade-off |
| 05_loss_A_charbonnier | raw+clipped | 32/6 | A: Charbonnier | 64 | 104,705 | 25.9660 | 0.64374 | 26.5380 | 0.68665 | 11.87 | 28.38 | Reject |
| 06_loss_B_charbonnier_ssim | raw+clipped | 32/6 | B: +SSIM | 64 | 104,705 | 26.0321 | 0.64804 | 26.5501 | 0.68936 | 10.42 | 27.77 | Competitive |
| 07_loss_C_charbonnier_ssim_edge | raw+clipped | 32/6 | C: +edge | 64 | 104,705 | 26.0369 | 0.64828 | 26.5495 | 0.68948 | 10.35 | 27.67 | Best loss screen |
| 08_capacity_small | raw+clipped | 24/4 | C | 64 | 50,833 | 26.0043 | 0.64630 | 26.5479 | 0.68780 | 8.12 | 17.91 | Speed Pareto point |
| 09_capacity_large | raw+clipped | 48/12 | C | 64 | 335,521 | 26.0776 | 0.65022 | 26.5784 | 0.69129 | 17.70 | 59.40 | Full-train candidate |
| 10_context7_large | raw+clipped | 48/12 | C | 64 | 340,417 | 26.0828 | 0.65021 | 26.5744 | 0.69097 | 20.59 | 55.73 | Reject: +0.005 dB, worse OOD |
| 11_ema995_large | raw+clipped | 48/12 | C | 64 | 335,521 | 23.8618 | 0.56240 | 24.8438 | 0.61883 | 17.30 | 54.43 | Reject at 150 steps: EMA lag |
| 12_raw_weights_large | raw+clipped | 48/12 | C | 64 | 335,521 | 26.0776 | 0.65022 | 26.5784 | 0.69129 | 17.74 | 64.56 | Confirms short-EMA lag |
| 13_patch96_large | raw+clipped | 48/12 | C | 96 | 335,521 | 26.0728 | 0.65082 | 26.5461 | 0.69119 | 17.45 | 54.51 | Reject: lower OOD PSNR |
| 14_patch128_large | raw+clipped | 48/12 | C | 128 | 335,521 | 26.0673 | 0.65120 | 26.5100 | 0.69076 | 17.68 | 54.91 | Reject: lower PSNR/OOD |
| 15_candidate_full_w48b12 | raw+clipped | 48/12 | C | 64 | 335,521 | 27.6727 | 0.73795 | 28.3279 | 0.78270 | **16.82** | **52.23** | Faster, rejected on fidelity |

## Conclusions

- Raw-only was consistently weakest. Adding the clipped channel helped. Adding OOR to the short screen gained only 0.0070 dB while losing 0.00074 SSIM and 0.00182 OOD SSIM; it is not automatically superior. The incumbent nevertheless retains OOR because the already-complete full model is the only contender that wins every final fidelity metric.
- Loss C narrowly won the controlled loss screen. Loss D (row 03 under the same raw+clipped configuration) differed by only +0.00002 SSIM and -0.00045 dB. FFT therefore showed no meaningful short-budget gain. The selected incumbent retains D because its full trained checkpoint remains clearly strongest; the full Loss-C candidate did not overtake it.
- The 7x7 depthwise context branch gained 0.0052 dB random-validation PSNR but reduced OOD PSNR/SSIM and added parameters/latency, so it was rejected.
- Larger 96/128 patches slightly moved SSIM but reduced PSNR and OOD PSNR. Patch 64 remains selected.
- EMA at 150 steps lagged severely. This does not invalidate long-horizon EMA: both genuinely completed finalists used decay 0.995 for 6,800 steps and were evaluated from EMA state.
- Mild synthetic corruption was not run. The degradation analysis established signal dependence but did not identify a sufficiently supported physical corruption generator; introducing one would add an uncontrolled assumption.
- Artifact analysis found no compelling repeating PixelShuffle checkerboard. The full candidate had slightly lower phase spread but higher checkerboard projection, greater gradient error, and more missing edges than the incumbent.

Pareto status in [`experiments.csv`](experiments.csv) considers validation PSNR, OOD PSNR, 128-input latency, and parameter count. Final selection additionally uses SSIM, LPIPS, structural diagnostics, and training completeness.
