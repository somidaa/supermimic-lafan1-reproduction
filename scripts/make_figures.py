"""Turn the raw training log and the evaluation JSONs into the result tables and figures
used by the README.

Inputs (all produced by the training/evaluation scripts, none re-computed here):
  <run-dir>/../<run>_stdout.log        the training loop's stdout
  <run-dir>/eval_watch.csv             per-checkpoint 40-clip evaluation
  <results-dir>/eval_best_independent_reeval.json   independent re-evaluation of the best checkpoint
  <results-dir>/baseline_transferred/  the transferred checkpoints scored on the same 40 clips

Outputs:
  results/train_metrics.csv            parsed per-log-line training metrics
  results/figures/fig_training_return.png
  results/figures/fig_per_clip.png
  results/figures/fig_world_model_error.png
  results/figures/fig_motion_families.png

Usage:
  python scripts/make_figures.py --run-log logs/lafan1_full_stdout.log \
      --run-dir logs/lafan1_full --results-dir results --figures-dir results/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

PAPER_RETURN = 0.82
LINE_RE = re.compile(
    r"it (?P<iteration>\d+) \| (?P<hours>[\d.]+) h \| (?P<ips>[\d.]+) it/s \| updates (?P<updates>\d+) \| "
    r"train_ret (?P<train_ret>[-\d.nan]+) \(n=(?P<n>\d+)\) \| wm nse1 (?P<nse1>[-\d.nan]+) nse8 (?P<nse8>[-\d.nan]+) \| "
    r"imag_ret (?P<imag_ret>[-\d.nan]+) surv (?P<imag_surv>[-\d.nan]+) \| "
    r"t sim/wm/pi (?P<t_sim>[\d.]+)/(?P<t_wm>[\d.]+)/(?P<t_pi>[\d.]+)s"
)


def read_text_any(path: Path) -> str:
    """Read a log written by either shell: PowerShell redirection emits UTF-16LE, Python UTF-8."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8", errors="replace")


def parse_log(path: Path) -> list[dict]:
    rows = []
    for line in read_text_any(path).splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        row = {"iteration": int(d["iteration"]), "hours": float(d["hours"]), "it_per_s": float(d["ips"]),
               "updates": int(d["updates"]), "n_episodes": int(d["n"])}
        for k in ("train_ret", "nse1", "nse8", "imag_ret", "imag_surv", "t_sim", "t_wm", "t_pi"):
            try:
                row[k] = float(d[k])
            except ValueError:
                row[k] = float("nan")
        rows.append(row)
    return rows


def read_eval_csv(path: Path) -> list[dict]:
    out: dict[int, dict] = {}
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                it = int(r["iteration"])
            except (KeyError, ValueError):
                continue
            prev = out.get(it)
            numbered = r.get("ckpt", "").replace("\\", "/").endswith(f"{it:07d}.pt")
            if prev is None or numbered:
                out[it] = r
    return [out[k] for k in sorted(out)]


def fig_training_return(metrics: list[dict], evals: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    it = [int(e["iteration"]) for e in evals]
    ret = [float(e["mean_norm_return"]) for e in evals]
    wret = [float(e["step_weighted_norm_return"]) for e in evals]
    ax.plot(it, ret, "o-", color="#1f77b4", lw=1.8, ms=4, label="mean normalised return (40 clips)")
    ax.plot(it, wret, "s--", color="#17becf", lw=1.2, ms=3, alpha=0.8, label="step-weighted")
    tr = [(m["iteration"], m["train_ret"]) for m in metrics if m["train_ret"] == m["train_ret"]]
    if tr:
        ax.plot([t[0] for t in tr], [t[1] for t in tr], "-", color="#999999", lw=1.0, alpha=0.85,
                label="training episode return (10 s episodes, random start)")
    ax.axhline(PAPER_RETURN, color="#d62728", ls=":", lw=1.6)
    ax.text(0.99, PAPER_RETURN, " paper Fig. 4a ~ 0.82", color="#d62728",
            transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("normalised return")
    ax.set_title("SuperMimic on LAFAN1 (40 clips, 1x TITAN RTX, 7.93 h)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.savefig(out)
    plt.close(fig)


def fig_per_clip(best: dict, baseline: dict | None, out: Path) -> None:
    clips = sorted(best["clips"], key=lambda c: c["norm_return"])
    names = [c["name"] for c in clips]
    vals = [c["norm_return"] for c in clips]
    failed = [c["failed"] for c in clips]
    colours = ["#d62728" if f else "#2ca02c" for f in failed]
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    y = np.arange(len(clips))
    ax.barh(y, vals, color=colours, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("normalised return (full-clip denominator)")
    ax.set_xlim(0, 1.0)
    ax.axvline(PAPER_RETURN, color="#555555", ls=":", lw=1.2)
    n_fail = sum(failed)
    ax.set_title(f"Per-clip evaluation of ckpt_0048000.pt: {len(clips)} clips, {n_fail} failed\n"
                 f"mean {best['all']['mean_norm_return']:.4f}   "
                 f"step-weighted {best['all']['step_weighted_norm_return']:.4f}")
    if baseline:
        b = {c["name"]: c["norm_return"] for c in baseline["clips"]}
        ax.scatter([b.get(n, 0.0) for n in names], y, marker="|", s=60, color="#1f77b4",
                   label="transferred baseline (s3_core, 798 updates)")
        ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_world_model(metrics: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    it = [m["iteration"] for m in metrics]
    n1 = [m["nse1"] for m in metrics]
    n8 = [m["nse8"] for m in metrics]
    ax.plot(it, n1, "-", color="#1f77b4", lw=1.4, label="1-step open-loop NSE")
    ax.plot(it, n8, "-", color="#ff7f0e", lw=1.4, label="8-step open-loop NSE")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("normalised squared error")
    ax.set_title("World-model open-loop prediction error (online, k = 8)")
    ax.set_ylim(0, max(0.06, max(v for v in n8 if v == v) * 1.15))
    ax.legend(fontsize=9)
    ax.annotate(f"final 1-step {n1[-1]:.4f}\nfinal 8-step {n8[-1]:.4f}",
                xy=(0.62, 0.72), xycoords="axes fraction", fontsize=9,
                bbox={"boxstyle": "round", "fc": "white", "ec": "#cccccc"})
    fig.savefig(out)
    plt.close(fig)


def family(name: str) -> str:
    return re.sub(r"\d.*$", "", name)


def fig_families(best: dict, out: Path) -> None:
    fams: dict[str, list] = {}
    for c in best["clips"]:
        fams.setdefault(family(c["name"]), []).append(c)
    names = sorted(fams)
    mean = [float(np.mean([c["norm_return"] for c in fams[n]])) for n in names]
    nfail = [sum(c["failed"] for c in fams[n]) for n in names]
    nclips = [len(fams[n]) for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(names))
    bars = ax.bar(x, mean, color="#4c72b0", width=0.62)
    for i, f in enumerate(nfail):
        if f:
            bars[i].set_color("#d62728")
    for xi, m, f, n in zip(x, mean, nfail, nclips):
        ax.text(xi, m + 0.02, f"{m:.3f}\n{f}/{n} failed", ha="center", fontsize=8)
    ax.axhline(PAPER_RETURN, color="#555555", ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mean normalised return")
    ax.set_title("Per-motion-family tracking quality (red = at least one failed clip)")
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-log", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    ap.add_argument("--best-json", type=Path, default=None,
                    help="default: results/eval_best_independent_reeval.json")
    a = ap.parse_args()

    a.figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = parse_log(a.run_log)
    if not metrics:
        raise SystemExit(f"no metric lines parsed from {a.run_log}")
    with (a.results_dir / "train_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)
    print(f"[figures] train_metrics.csv: {len(metrics)} rows")

    evals = read_eval_csv(a.run_dir / "eval_watch.csv")
    print(f"[figures] eval points: {len(evals)}")

    best_path = a.best_json or (a.results_dir / "eval_best_independent_reeval.json")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    baseline_path = a.results_dir / "baseline_transferred" / "eval_ckpt_final.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None

    fig_training_return(metrics, evals, a.figures_dir / "fig_training_return.png")
    fig_per_clip(best, baseline, a.figures_dir / "fig_per_clip.png")
    fig_world_model(metrics, a.figures_dir / "fig_world_model_error.png")
    fig_families(best, a.figures_dir / "fig_motion_families.png")
    print("[figures] wrote 4 figures to " + str(a.figures_dir))


if __name__ == "__main__":
    main()