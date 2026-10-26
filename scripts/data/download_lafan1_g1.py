"""Fetch the 40 G1-retargeted LAFAN1 clips (30 fps CSV) from HuggingFace.

The dataset is `lvhaidong/LAFAN1_Retargeting_Dataset`; we only need its `g1/` folder.
HuggingFace is not reachable from every network, so the endpoint is configurable and
defaults to the public mirror `https://hf-mirror.com` (override with ``--endpoint`` or the
``HF_ENDPOINT`` environment variable; pass ``https://huggingface.co`` for the canonical host).

The motion data itself is NOT redistributed with this repository -- see docs/DATA_LICENSE.md.
Downloading it means accepting the upstream terms (Ubisoft La Forge LAFAN1, CC BY-NC-ND 4.0).

Usage:
  python scripts/data/download_lafan1_g1.py --out-dir data/raw/lafan1/g1
  python scripts/data/download_lafan1_g1.py --out-dir /tmp/g1 --limit 2     # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
SUBFOLDER = "g1"


def endpoint(arg: str | None) -> str:
    return (arg or os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com").rstrip("/")


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "supermimic-lafan1-repro/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (trusted endpoint)
        return r.read()


def list_clips(ep: str) -> list[dict]:
    url = f"{ep}/api/datasets/{DATASET}/tree/main/{SUBFOLDER}"
    return json.loads(_get(url).decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw/lafan1/g1"))
    ap.add_argument("--endpoint", default=None, help="default: $HF_ENDPOINT or https://hf-mirror.com")
    ap.add_argument("--limit", type=int, default=None, help="download only the first N clips (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    ep = endpoint(a.endpoint)
    print(f"[data] endpoint {ep}")
    entries = list_clips(ep)
    files = [e for e in entries if e.get("type") == "file" and e["path"].endswith(".csv")]
    files.sort(key=lambda e: e["path"])
    if a.limit is not None:
        files = files[: a.limit]
    if not files:
        print("[data] ERROR: no CSV entries listed", file=sys.stderr)
        sys.exit(1)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    done = skipped = 0
    for e in files:
        name = Path(e["path"]).name
        dest = a.out_dir / name
        if dest.exists() and not a.overwrite and dest.stat().st_size > 0:
            skipped += 1
            continue
        url = f"{ep}/datasets/{DATASET}/resolve/main/{e['path']}"
        for attempt in range(3):
            try:
                t0 = time.time()
                blob = _get(url, timeout=180)
                dest.write_bytes(blob)
                total_bytes += len(blob)
                done += 1
                print(f"[data] {name:<28} {len(blob)/1e6:6.2f} MB  {time.time()-t0:5.1f}s")
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == 2:
                    print(f"[data] FAILED {name}: {exc}", file=sys.stderr)
                    sys.exit(1)
                time.sleep(2 * (attempt + 1))

    print(f"[data] downloaded {done}, reused {skipped}, {total_bytes/1e6:.1f} MB -> {a.out_dir}")
    print("[data] next: python scripts/data/csv_to_npz_batch.py --input-dir "
          f"{a.out_dir.as_posix()} --output-dir data/npz/lafan1 --input-fps 30")


if __name__ == "__main__":
    main()