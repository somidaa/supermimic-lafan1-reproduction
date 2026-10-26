"""Cut a small GIF (and optionally a still) out of an existing rollout MP4.

Useful for turning a long failure rollout into a README-sized preview without re-simulating it,
and for keeping the GIF's provenance explicit: the GIF is exactly the last (or first) N seconds
of the named MP4.

Usage:
  python scripts/mp4_to_gif.py --mp4 media/fallAndGetUp2_subject3_failure.mp4 \
      --out media/preview_failure.gif --tail-seconds 6.5 --width 320 --fps 10 \
      --still media/still_fallAndGetUp2_subject3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tail-seconds", type=float, default=0.0,
                    help="keep only the last N seconds (0 = the whole video)")
    ap.add_argument("--head-seconds", type=float, default=0.0,
                    help="keep only the first N seconds (0 = no head cut)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--fps", type=float, default=0.0, help="source fps; 0 = read from the file")
    ap.add_argument("--still", type=Path, default=None)
    ap.add_argument("--still-at", type=float, default=0.5, help="fraction of the clip for the still")
    a = ap.parse_args()

    reader = imageio.get_reader(str(a.mp4))
    meta = reader.get_meta_data()
    fps = a.fps or float(meta.get("fps", 10.0))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    if not frames:
        raise SystemExit(f"no frames read from {a.mp4}")

    if a.head_seconds:
        frames = frames[: int(a.head_seconds * fps)]
    if a.tail_seconds:
        frames = frames[-max(1, int(a.tail_seconds * fps)):]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    imgs = []
    for f in frames:
        im = Image.fromarray(f)
        imgs.append(im.resize((a.width, int(im.height * a.width / im.width)), Image.LANCZOS))
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    if a.still:
        idx = min(len(frames) - 1, int(a.still_at * len(frames)))
        a.still.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frames[idx]).save(a.still)
    print(f"[gif] {a.out}: {len(imgs)} frames, {a.width}px wide, {fps:g} fps, "
          f"{len(frames)/fps:.1f} s" + (f", still {a.still}" if a.still else ""))


if __name__ == "__main__":
    main()