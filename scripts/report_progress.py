"""Compact progress report for a training run (reads the eval watcher's outputs).

Usage:  python scripts/report_progress.py logs/lafan1_full
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


def family(name: str) -> str:
    return re.sub(r"\d.*$", "", name)


def main() -> None:
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/lafan1_full")
    csv_path = log_dir / "eval_watch.csv"
    rows: dict[int, dict] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows[int(r["iteration"])] = r
                except (KeyError, ValueError):
                    continue
    print(f"run dir: {log_dir}")
    print(f"{'iteration':>9} {'updates':>8} {'failed':>7} {'mean_ret':>9}")
    for it in sorted(rows):
        r = rows[it]
        print(f"{it:>9} {r.get('num_updates',''):>8} {str(r.get('failed','')):>7} {float(r['mean_norm_return']):>9.4f}")

    if not rows:
        return
    best_it = max(rows, key=lambda i: float(rows[i]["mean_norm_return"]))
    print(f"\nbest checkpoint so far: iteration {best_it}, "
          f"mean_ret={float(rows[best_it]['mean_norm_return']):.4f}, failed={rows[best_it]['failed']}")
    js = log_dir / f"eval_ckpt_{best_it:07d}.json"
    if best_it == 0:
        js = log_dir / "eval_ckpt_init.json"
    if not js.exists():
        return
    data = json.loads(js.read_text())
    print(f"\n--- {js.name}: mean={data['all']['mean_norm_return']:.4f} "
          f"step_weighted={data['all']['step_weighted_norm_return']:.4f} failed={data['all']['failed']}/{data['all']['clips']}")
    print(f"{'family':<16} {'n':>3} {'failed':>7} {'mean_ret':>9}")
    fams: dict[str, list] = {}
    for c in data["clips"]:
        fams.setdefault(family(c["name"]), []).append(c)
    for name in sorted(fams):
        cs = fams[name]
        nf = sum(1 for c in cs if c["failed"])
        mr = sum(c["norm_return"] for c in cs) / len(cs)
        print(f"{name:<16} {len(cs):>3} {nf:>7} {mr:>9.3f}")
    print("\n--- reward terms (mean per survived step) ---")
    for k, v in data["all"].get("terms", {}).items():
        print(f"  {k:<24} {v:+.4f}")
    print("\n--- failing clips ---")
    for c in sorted((c for c in data["clips"] if c["failed"]), key=lambda c: c["norm_return"]):
        print(f"  {c['name']:<26} fail_step {c['fail_step']:>6} / {c['denominator_steps']:<6} ret {c['norm_return']:.3f}")


if __name__ == "__main__":
    main()