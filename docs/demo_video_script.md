# Demo Video Script

Target length: 2.5–3 minutes. Record a clean terminal at the repository root; do not expose private paths, tokens, or unrelated files.

## 0:00–0:20 — Problem and contract

Narration: “KLA PS01 asks one grayscale model to remove mixed speckle and Gaussian degradation while restoring exactly 2× resolution. Out-of-range input is legitimate, so our learned feature path preserves it. Every saved result is a finite float32 single-channel array in `[0,1]`.”

Show the architecture diagram from `docs/presentation_content.md`.

## 0:20–0:45 — Evidence and model

Show `reports/dataset_audit.json`, then `reports/validation_summary.json`.

Narration: “We audited all 6,800 real arrays, matched 3,200 exact 2× pairs, and froze a seed-42 split. The 956,609-parameter Range-Aware LiteNAF-SR uses a raw/clipped/out-of-range stem, a lightweight NAF trunk, PixelShuffle, HR refinement, and a bicubic residual anchor.”

## 0:45–1:25 — Live offline inference

Use a small temporary folder containing copies of two test `.npy` files, then run:

```powershell
.\.venv\Scripts\python.exe evaluate.py "demo_inputs" "demo_outputs"
```

Point out the printed device, model path, input/processed/failure counts, model-load time, total wall time, model inference time, and output path. Disconnect networking before this segment if practical to demonstrate offline behavior.

Inspect outputs without changing them:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import numpy as np; [(lambda a,p: print(p,a.shape,a.dtype,float(a.min()),float(a.max()),bool(np.isfinite(a).all())))(np.load(p,allow_pickle=False),p) for p in Path('demo_outputs').rglob('*.npy')]"
```

Narration: “Filenames and subdirectories are preserved. GT and validation code are not imported by official inference.”

## 1:25–2:00 — Visual comparison

Show two files from `reports/figures/comparison_*.png` at full size and zoom into the highlighted crop.

Narration: “The model suppresses noise while preserving semiconductor edges and fine structures. The visual panels are display-only; metric arrays are never altered.”

## 2:00–2:30 — Results and efficiency

Show the table:

- Bicubic: 23.173149 dB PSNR, 0.539510 SSIM.
- Ours: 27.869329 dB, 0.748968 SSIM, 0.300719 LPIPS.
- RTX 4050 batch-1 latency: 30.163 ms mean / 31.852 ms p95 for 128→256 and 103.970 ms mean / 104.635 ms p95 for 256→512.
- Final checkpoint: 3.994 MB.

Narration: “These are held-out RTX 4050 measurements, not H100 or test-GT claims.”

## 2:30–2:50 — Verification and close

Run:

```powershell
.\.venv\Scripts\python.exe verify_submission.py
```

Narration: “The verifier launches inference from a different working directory, tests both official sizes, and checks output count, names, shape, dtype, range, and finiteness. The project includes reproducible training, weights, reports, and all 400 actual test predictions.”

End on the GitHub URL and team name after replacing placeholders in the presentation.
