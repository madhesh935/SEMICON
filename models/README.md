# Model weights

This folder is the official checkpoint location used by:

```bash
python run.py <input-dir> <output-dir>
```

`run.py` loads [`best.pt`](best.pt) automatically. No download, API key, or extra configuration is required.

| Property | Verified value |
|---|---|
| Architecture | `RangeAwareLiteNAFSR` |
| Parameters | 956,609 |
| Size | 4,000,107 bytes |
| SHA-256 | `7E2016F9BE1CA460366F88D0B1B54D6E026D81D3D1E16FBBCBDE3DAF13B349AC` |
| Training kind | `full` (40 epochs) |

The same bytes also exist at [`../weights/best.pt`](../weights/best.pt) for older `evaluate.py` commands. Do not replace this file for ordinary inference.
