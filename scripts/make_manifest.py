"""Write a SHA-256 manifest of every published file, and a human-readable inventory.

Usage:
  python scripts/make_manifest.py                  # repo root = parent of scripts/
  python scripts/make_manifest.py --root . --out MANIFEST.sha256
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".ruff_cache", ".pytest_cache"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = a.root.resolve()
    out = a.out or (root / "MANIFEST.sha256")

    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.suffix in SKIP_SUFFIX:
            continue
        if p.resolve() == out.resolve() or rel.name == "FILE_INVENTORY.md":
            continue
        files.append((rel, p))

    lines, rows = [], []
    total = 0
    for rel, p in files:
        size = p.stat().st_size
        total += size
        digest = sha256(p)
        lines.append(f"{digest}  {rel.as_posix()}")
        rows.append((rel.as_posix(), size, digest))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    inv = ["# File inventory", "",
           f"{len(rows)} files, {total/1e6:.2f} MB total. Digests: `MANIFEST.sha256` "
           "(verify with `sha256sum -c MANIFEST.sha256` or `Get-FileHash`).", "",
           "| file | bytes | sha256 (first 16) |", "|---|---:|---|"]
    for rel, size, digest in rows:
        inv.append(f"| `{rel}` | {size} | `{digest[:16]}` |")
    (root / "FILE_INVENTORY.md").write_text("\n".join(inv) + "\n", encoding="utf-8")
    print(f"[manifest] {len(rows)} files, {total/1e6:.2f} MB -> {out}")
    print(f"[manifest] inventory -> {root / 'FILE_INVENTORY.md'}")


if __name__ == "__main__":
    main()