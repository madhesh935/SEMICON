# Degradation Analysis Summary

This analysis uses only the saved training fold for paired LR/GT measurements. Official test arrays were not opened, described, fitted, or scored. Input-only descriptors were computed for the saved validation fold solely to construct the OOD-proxy subset; all descriptor scaling, covariance, PCA, and nearest-neighbour reference distributions were fitted on the 2,720 training inputs.

## Dataset coverage

- Matched paired samples: 3,200 (2,720 train / 480 validation).
- Available paired geometry: 3,200 samples at `128x128 -> 256x256`.
- Paired `256x256 -> 512x512` samples: 0. Architecture-level support can be tested, but quality at this resolution cannot be measured from the supplied data.
- Saved split SHA-256: `4478fa31fd2b93ace5873c9b6bc421f61ce4d4ddaa15dd368da1ec0b49984402`.

## Bicubic restoration residual

Using `GT - clamp(bicubic(clamp(raw LR,0,1)),0,1)` to match `benchmark_bicubic.py`, the training-fold pixel residual has mean 0.003361, standard deviation 0.082172, and range [-0.999859, 0.999942]. Per-image residual RMSE has mean 0.076642 and spans 0.008753 to 0.310884. The model's internal bicubic skip is not clamped after interpolation, so its separate residual is also recorded in `statistics.json`. The spatial mean/std maps show whether error is globally distributed or concentrated near recurring structures; they are evidence for retaining an HR residual refinement rather than relying on interpolation alone.

## Empirical noise behavior

The LR discrepancy `raw LR - area-downsampled GT` has mean 0.000004 and standard deviation 0.090489. Its squared magnitude has correlation 0.363646 with the downsampled-GT signal. Across 20 intensity bins, the maximum/minimum positive variance ratio is 261.251; a linear variance curve explains 96.856% and a quadratic curve explains 99.923% of the between-bin variance relative to a constant model.

These measurements indicate whether the discrepancy is intensity-dependent, but they do **not** uniquely identify Gaussian, speckle, blur, registration, or degradation order. The discrepancy includes all of those possible effects. Synthetic degradation should therefore remain disabled unless a controlled validation/OOD experiment supports it.

## Out-of-range signal

- Training-fold files containing values below zero: 1,668.
- Training-fold files containing values above one: 2,657.
- Below-zero pixels: 126,865; mean excursion magnitude 0.007228, maximum 0.278563.
- Above-one pixels: 1,396,701; mean excursion magnitude 0.102038, maximum 1.158005.
- Correlation between raw out-of-range values and the corresponding downsampled GT signal: 0.932866.

The excursions are frequent enough that the raw feature path must be preserved. Their signal and spatial distributions are plotted; clipping remains appropriate only for the conservative bicubic skip and final file contract.

## Frequency behavior

Across 256 deterministic training-fold samples, the mean energy above 0.25 cycles/pixel is 0.037236 for bicubic LR and 0.049281 for GT. The GT/bicubic ratio is 1.323. The saved radial profiles and directional-anisotropy summaries quantify missing high-frequency and oriented/repeating structure, supporting local edge refinement while warning against adversarial hallucination.

## Structural diversity and OOD proxy

The input-only descriptor combines intensity, out-of-range frequency, gradient density/orientation, four frequency bands, spectral anisotropy/concentration, and simple morphology. The first two train-fitted principal components explain 43.65% of standardized descriptor variance. Train nearest-neighbour distances range from 0.2115 to 10.0472; validation-to-train distances range from 0.3110 to 4.7844.

`val_random` retains all 480 saved validation samples. `val_ood_proxy` contains the 120 highest-scoring validation inputs (25%), ranked by the mean of train-empirical shrinkage-Mahalanobis and nearest-train-distance percentiles. The subset is saved in `validation_subsets_seed42.json` and was constructed without official test information.

## Architecture implications

1. Keep the unmodified raw channel available; out-of-range samples are common and correlated with underlying intensity.
2. Keep a clipped bicubic skip for a conservative base image, but learn a residual because bicubic leaves substantial structured error and loses high-frequency energy.
3. Compare raw-only, raw+clipped, and raw+clipped+OOR representations experimentally; frequency alone does not prove that three channels are optimal.
4. Test a lightweight wider-context branch because directional/repeating structures are present, but reject it if the random/OOD Pareto metrics or latency worsen.
5. Avoid strong synthetic degradation and adversarial losses: the measured discrepancy is mixed and a simple physical-noise claim is not supported.

## Generated evidence

- Public reproducibility artifacts retained here: `statistics.json` and this summary.
- The optional per-image tables and exploratory plots generated during analysis are preserved in the timestamped recoverable archive; they are not required by the public inference or verification path.
