# Final model weights

The evaluator loads [`best.pt`](best.pt) by default. It is the selected full-training inference bundle and is committed directly in this repository; Git LFS is not configured.

| Property | Verified value |
|---|---|
| Architecture | `RangeAwareLiteNAFSR` |
| Parameters | 956,609 |
| Size | 4,000,107 bytes |
| SHA-256 | `7E2016F9BE1CA460366F88D0B1B54D6E026D81D3D1E16FBBCBDE3DAF13B349AC` |
| Training kind | `full` (40 epochs, promoted from an initial 20-epoch selection) |
| Inference state | selected trained `model_state` |

The checkpoint contains the architecture identifier, model configuration, selected model state, split identity, training-loop validation metrics, and library versions. Its `ema_model_state` field is `None` because the selected inference state was already exported into `model_state`.

The legacy bundle does not contain epoch, global-step, or complete data-configuration fields. [`best.metadata.json`](best.metadata.json) records that provenance from retained logs and fresh validation without rewriting the checkpoint or changing its output numerics.

Do not replace this file for ordinary inference. Verification, setup, and usage commands are in the [root README](../README.md#model-weight-verification).
