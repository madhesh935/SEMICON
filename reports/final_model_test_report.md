# KLA PS01 Final Model Validation Report

## Verdict

**READY FOR SUBMISSION**

The final checkpoint, fixed split, unit tests, standalone verifier, functional scale paths, 400 official outputs, quality metrics, rotation robustness, and repository packaging all passed. No source code, checkpoint, organizer dataset, or committed official output was modified.

## PASS/FAIL summary

| Requirement | Result | Evidence |
|---|---|---|
| Dataset integrity | PASS | 3,200 exact pairs; 400 test inputs; zero missing; zero exact train/test noisy-array hash duplicates |
| Checkpoint hash | PASS | 3,993,611 bytes; expected SHA-256 |
| Checkpoint loading | PASS | Strict model loader; no missing/unexpected keys; loaded outside-repository CWD |
| Parameter count | PASS | 956,609 |
| Python compilation | PASS | All project scripts, `restoration/`, and `tests/` compiled |
| Unit tests | PASS | `43 passed` |
| Submission verifier | PASS | `verify_submission.py` passed with real test-input directory |
| 128→256 inference | PASS | Real official 128×128 input produced 256×256 output |
| 256→512 functional inference | PASS | Valid copied 256×256 input produced 512×512 output |
| Out-of-range input | PASS | Below-zero/above-one input processed; raw-vs-clipped prediction max difference 0.0335099 |
| All 400 official outputs | PASS | Both CLI forms: 400 discovered, 400 processed, zero failures |
| Filename preservation | PASS | Relative names matched in all output comparisons |
| Output dtype | PASS | All outputs `float32` |
| Output range/finiteness | PASS | All outputs finite and within `[0,1]` |
| Validation PSNR | PASS | 27.869329127248292 |
| Validation SSIM | PASS | 0.7489683585877321 |
| Validation LPIPS | PASS | 0.3007189181749709 using local cache |
| Bicubic improvement | PASS | +4.696180438455777 dB PSNR; +0.20945803684932096 SSIM |
| Rotation robustness | PASS | All 480 images at 0°, 90°, 180°, 270° |
| Cross-CWD inference | PASS | Flag CLI run from a different working directory |
| Inference speed | PASS | CUDA-synchronized benchmark on RTX 4050 Laptop GPU |
| Repository requirements | PASS | Required files present; `pip check` clean; no private paths/secrets |

## Environment and paths

- Repository: `<repo-root>`
- Training data: `<parent-workspace>/train (1)/train`
- Official test data: `<parent-workspace>/Test_NoisyLR (2)/NoisyLR`
- Git root: repository directory exactly; no parent Git root conflict.
- Python: 3.13.14
- NumPy: 2.4.4
- PyTorch: 2.11.0+cu130
- CUDA runtime: 13.0; cuDNN: 91900
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU; VRAM 6,438,780,928 bytes
- RAM at audit: 25,439,199,232 total / 11,985,469,440 free bytes
- C: disk at audit: 105,287,352,320 free / 329,272,266,752 used bytes
- CUDA matrix multiplication produced finite values.
- `python -m pip check`: `No broken requirements found.`

## Dataset integrity

All checks used per-file memory-mapped NumPy loading and did not modify the datasets.

| Dataset | Count | Shape | dtype | Global min | Global max |
|---|---:|---|---|---:|---:|
| Training NoisyLR | 3,200 | 128×128 | float32 | -0.2785630524 | 2.1580049992 |
| Training GT | 3,200 | 256×256 | float32 | 0.0 | 1.0 |
| Official test NoisyLR | 400 | 128×128 | float32 | -0.2248806655 | 2.1580159664 |

There were zero nonfinite arrays, zero missing training pairs, zero invalid grayscale/dtype files, and zero exact train/test noisy-array hash duplicates. Training GT values were entirely within `[0,1]`; noisy inputs retained legitimate out-of-range values.

## Checkpoint

- Architecture: `RangeAwareLiteNAFSR`
- Configuration: width 64, 24 LR blocks, 1 HR block, expansion 2, scale 2
- Parameters: 956,609
- Training kind: full
- SHA-256: `10FE2F02EAA5ABEAFAA5050F8022D60282C03BF89AB864F871C51C41CE0A20AE`
- Checkpoint loading used PyTorch `weights_only=True`; no network access or download was required.

## Functional inference

Temporary test root: `<temporary-test-root>`

- Standard real input: 128×128 → 256×256, valid `float32`, finite, `[0,1]`.
- Larger supported path: 256×256 → 512×512, valid `float32`, finite, `[0,1]`.
- Out-of-range test: raw input range `[-0.35, 1.3791789]`; output range `[0,1]`; raw and clipped predictions differed by maximum absolute value `0.0335099101`, demonstrating that the raw range-aware branch is active.
- Non-square test: 128×160 → 256×320, valid.
- Inputs and outputs were placed in paths containing spaces and executed from a different current working directory.

The 256→512 and 128×160 tests are functional contract tests only. No quality score is claimed for 256→512.

## Official inference comparison

Both documented CLI forms were run on all 400 official inputs into new temporary directories, never into `restored_test_outputs`.

| Run | Total wall | Model time | Average model time | Comparison with committed outputs |
|---|---:|---:|---:|---|
| Positional CLI | 15.248 s | 13.037 s | 32.592 ms/image | 400/400 hashes equal; max/mean abs diff 0 |
| Flag CLI from different CWD | 15.706 s | 13.487 s | 33.718 ms/image | 400/400 hashes equal; max/mean abs diff 0 |

## Validation quality

Recalculated on all 480 records in `splits/split_seed42.json` using the existing parent-workspace root required by the frozen dataset fingerprint. LPIPS used the local AlexNet cache with no download.

| Method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Bicubic | 23.173148688792516 | 0.5395103217384112 | — |
| RangeAwareLiteNAFSR | 27.869329127248292 | 0.7489683585877321 | 0.3007189181749709 |

Improvements are +4.696180438455777 dB PSNR and +0.20945803684932096 SSIM. These match the saved reference reports within floating-point precision. No official test quality metric and no 256→512 quality metric were calculated.

## Rotation robustness

The existing test applies the same rotation to LR and GT and compares in the rotated orientation; the model preserves that orientation and does not canonicalize it.

| Angle | PSNR | SSIM | Images |
|---:|---:|---:|---:|
| 0° | 27.869329127248 | 0.748968358588 | 480 |
| 90° | 27.870750841024 | 0.748883034111 | 480 |
| 180° | 27.872453317497 | 0.748999837080 | 480 |
| 270° | 27.870725553519 | 0.748898228151 | 480 |

Maximum PSNR spread: 0.003124190248 dB.

## Visual inspection

Five new comparison figures were generated under `reports/final_model_test_figures/`. Each contains enlarged LR, bicubic, model restoration, GT, and absolute error:

- `final_high_noise_000984.png`
- `final_strong_edges_002983.png`
- `final_dark_structure_002225.png`
- `final_bright_structure_001194.png`
- `final_difficult_low_psnr_000355.png`

Visual inspection found no obvious checkerboard artifacts, ringing, abnormal clipping, or hallucinated structures. The high-noise and low-PSNR cases remain visibly difficult, consistent with their quantitative scores; visual inspection is supporting evidence only.

## Speed and memory

CUDA AMP was enabled, TTA was disabled, batch size was 1, and timings used 20 warmups plus 100 synchronized iterations on the RTX 4050 Laptop GPU.

| Path | Mean | Median | P95 | Peak GPU memory |
|---|---:|---:|---:|---:|
| 128→256 | 30.695 ms | 30.159 ms | 32.014 ms | 84,140,544 bytes |
| 256→512 | 105.010 ms | 104.971 ms | 106.160 ms | 293,544,448 bytes |

These are RTX 4050 measurements, not H100 measurements. The 256→512 row is scale-path timing only, not a quality result.

## Hardcoding and standalone audit

- No private absolute Windows paths, secrets, tokens, API keys, or local wheel paths were found in public source/configuration/documentation text.
- `evaluate.py` imports no LPIPS, torchvision, matplotlib, or requests; its import test confirmed those packages were not loaded.
- The default checkpoint path is derived from the evaluator’s own repository path.
- Intentional fixed behavior is limited to the competition’s exact 2× scale contract; the model also passed the non-square 128×160 test.
- CPU fallback passed on small functional inputs.
- Requirements are package-for-package consistent with the active environment; `pip check` passed.

## Commands executed

```powershell
.\.venv\Scripts\python.exe evaluate.py --help
.\.venv\Scripts\python.exe validate.py --help
.\.venv\Scripts\python.exe verify_submission.py --help
.\.venv\Scripts\python.exe robustness_test.py --help
.\.venv\Scripts\python.exe audit_dataset.py --help
.\.venv\Scripts\python.exe train.py --help
.\.venv\Scripts\python.exe -m compileall -q -f <project Python sources>
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe verify_submission.py --test-input-dir "..\Test_NoisyLR (2)\NoisyLR" --device cpu
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe validate.py --root .. --noisy-dir "..\train (1)\train\NoisyLR" --gt-dir "..\train (1)\train\GT" --split splits\split_seed42.json --weights weights\best.pt --validation-subsets splits\validation_subsets_seed42.json --report-dir <temporary validation reports> --lpips-cache <local LPIPS cache> --no-save-outputs
.\.venv\Scripts\python.exe benchmark_bicubic.py --root .. --noisy-dir "..\train (1)\train\NoisyLR" --gt-dir "..\train (1)\train\GT" --split splits\split_seed42.json --report-dir <temporary bicubic reports>
.\.venv\Scripts\python.exe robustness_test.py --root .. --noisy-dir "..\train (1)\train\NoisyLR" --gt-dir "..\train (1)\train\GT" --split splits\split_seed42.json --report-dir <temporary rotation reports> --device cuda
```

The first validation attempt with `--root` set to the repository was rejected by the frozen split fingerprint; rerunning with the intended parent `semi` root passed. This is correct safety behavior, not a model failure.

## Git and modifications

The initial and final Git status contained the same existing untracked top-level submission groups. The only permitted repository additions from this audit are this report, its JSON companion, and five figures under `reports/final_model_test_figures/`. No source file, checkpoint, organizer dataset, or official output was overwritten.

Physical repository content excluding `.git`: 3,430,754,726 bytes (3,271.82 MiB), including the local `.venv`. Prospective Git content from `git ls-files --others --exclude-standard`: 474 files, 117,675,516 bytes (112.22 MiB). No prospective file exceeds 20 MiB or GitHub's 100 MiB limit. The local `.venv` is ignored and contributes 3,312,631,946 physical bytes.

Remaining human actions: team/contact details, demo video URL, and exporting `TeamName_KLA_PS01.pdf`. The public repository is https://github.com/madhesh935/SEMICON.
