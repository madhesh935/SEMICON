#!/usr/bin/env python
"""Official KLA PS01 inference entry point.

Reads every `.npy` file under `<input-dir>`, restores it offline from the
committed checkpoint in `models/`, and writes one matching `.npy` under
`<output-dir>`. Creates the output directory if it does not exist.

Usage:
    python run.py <input-dir> <output-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _default_checkpoint() -> Path:
    for candidate in (ROOT / "models" / "best.pt", ROOT / "weights" / "best.pt"):
        if candidate.is_file():
            return candidate
    return ROOT / "models" / "best.pt"


def main() -> int:
    if "--weights" not in sys.argv:
        sys.argv.extend(["--weights", str(_default_checkpoint())])
    from evaluate import main as evaluate_main

    return evaluate_main()


if __name__ == "__main__":
    raise SystemExit(main())
