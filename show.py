#!/usr/bin/env python
"""Show every restored output together as a gallery.

Usage:
    python show.py
    python show.py demo_inputs demo_outputs
    python show.py Test_NoisyLR/NoisyLR restored_test_outputs
"""

from __future__ import annotations

import html
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def collect_npy(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*.npy") if path.is_file() and not path.name.startswith("."))


def load_gray(path: Path):
    import numpy as np

    array = np.squeeze(np.asarray(np.load(path, allow_pickle=False), dtype=np.float32))
    return np.clip(array, 0.0, 1.0)


def to_image(array, size: int):
    import numpy as np
    from PIL import Image

    pixels = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    return image.resize((size, size), Image.Resampling.BILINEAR)


def ensure_outputs(input_dir: Path, output_dir: Path) -> None:
    inputs = collect_npy(input_dir)
    if not inputs:
        raise SystemExit(f"no .npy files in {input_dir}")
    missing = [path for path in inputs if not (output_dir / path.name).is_file()]
    if not missing:
        return
    print(f"Restoring {len(missing)} file(s) with run.py ...")
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), str(input_dir), str(output_dir)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def write_gallery(input_dir: Path, output_dir: Path) -> Path:
    names = [path.name for path in collect_npy(input_dir) if (output_dir / path.name).is_file()]
    if not names:
        raise SystemExit(f"no matching restored .npy files in {output_dir}")

    gallery_dir = ROOT / "gallery"
    thumb_dir = gallery_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    restored_thumbs = []
    print(f"Building gallery for {len(names)} outputs ...")
    for index, name in enumerate(names, start=1):
        stem = Path(name).stem
        noisy = load_gray(input_dir / name)
        restored = load_gray(output_dir / name)
        in_path = thumb_dir / f"{stem}_in.jpg"
        out_path = thumb_dir / f"{stem}_out.jpg"
        to_image(noisy, 256).save(in_path, quality=85)
        out_image = to_image(restored, 256)
        out_image.save(out_path, quality=85)
        restored_thumbs.append(out_image)
        cards.append(
            f"""
            <article class="card">
              <h2>{html.escape(stem)}</h2>
              <div class="pair">
                <figure><img src="thumbs/{html.escape(in_path.name)}" alt="input {html.escape(stem)}" /><figcaption>input {noisy.shape[0]}×{noisy.shape[1]}</figcaption></figure>
                <figure><img src="thumbs/{html.escape(out_path.name)}" alt="restored {html.escape(stem)}" /><figcaption>restored {restored.shape[0]}×{restored.shape[1]}</figcaption></figure>
              </div>
            </article>
            """
        )
        if index % 50 == 0 or index == len(names):
            print(f"  {index}/{len(names)}")

    columns = max(1, math.ceil(math.sqrt(len(restored_thumbs))))
    rows = math.ceil(len(restored_thumbs) / columns)
    cell = 96
    from PIL import Image

    sheet = Image.new("L", (columns * cell, rows * cell), 0)
    for index, thumb in enumerate(restored_thumbs):
        y, x = divmod(index, columns)
        sheet.paste(thumb.resize((cell, cell), Image.Resampling.BILINEAR), (x * cell, y * cell))
    sheet_path = gallery_dir / "all_restored_contact_sheet.jpg"
    sheet.convert("RGB").save(sheet_path, quality=90)

    page = gallery_dir / "all_outputs.html"
    page.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>All restored outputs</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #111; color: #eee; }}
    header {{ position: sticky; top: 0; background: #111; border-bottom: 1px solid #333; padding: 16px 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    p {{ margin: 0; color: #aaa; }}
    main {{ padding: 20px 24px 64px; }}
    .sheet {{ width: min(1100px, 100%); border: 1px solid #333; margin: 0 0 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
    .card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 10px; padding: 10px; }}
    h2 {{ margin: 0 0 8px; font-size: 14px; color: #9ad; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    img {{ width: 100%; height: auto; display: block; background: #000; }}
    figcaption {{ margin: 6px 0 0; font-size: 12px; color: #888; }}
    figure {{ margin: 0; }}
  </style>
</head>
<body>
  <header>
    <h1>All restored outputs</h1>
    <p>{len(names)} files from {html.escape(str(input_dir))} → {html.escape(str(output_dir))}</p>
  </header>
  <main>
    <p>All restored images together</p>
    <img class="sheet" src="all_restored_contact_sheet.jpg" alt="all restored outputs" />
    <div class="grid">
      {''.join(cards)}
    </div>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"Gallery: {page}")
    print(f"Contact sheet: {sheet_path}")
    return page


def main() -> int:
    input_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "Test_NoisyLR" / "NoisyLR")
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "restored_test_outputs")
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    ensure_outputs(input_dir, output_dir)
    page = write_gallery(input_dir, output_dir)
    if sys.platform == "darwin":
        subprocess.run(["open", str(page)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
