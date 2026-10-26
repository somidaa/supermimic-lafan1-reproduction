"""Collapse an eval-watcher CSV to one row per iteration.

The watcher scores both ``ckpt_<iteration>.pt`` and ``ckpt_latest.pt``; the latter duplicates
the newest iteration. This keeps the numbered checkpoint's row (or the last row seen) and sorts
by iteration, so the published table has exactly one entry per training checkpoint.

Usage:  python scripts/slim_eval_csv.py results/eval_watch.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/eval_watch.csv")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{path}: empty")
        return
    best: dict[int, dict] = {}
    for r in rows:
        it = int(r["iteration"])
        numbered = r["ckpt"].replace("\\", "/").endswith(f"{it:07d}.pt")
        if it not in best or numbered:
            best[it] = r
    out = [best[k] for k in sorted(best)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{path}: {len(rows)} -> {len(out)} rows")


if __name__ == "__main__":
    main()