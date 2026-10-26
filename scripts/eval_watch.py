"""Independent 40-clip evaluation watcher.

Evaluates every checkpoint that appears in a run directory with the Appendix-F.1 protocol
(each clip played once from its first frame, no observation/action noise, no reset noise, no
domain randomisation, foot friction 1.0, deterministic physics; failure = the training
termination |pelvis height error| > 0.25 m or tilt error > 0.8).  The normalised return is
sum(reward) / (0.1 * full clip length), so an early termination scores low.

Results are appended to ``<log-dir>/eval_watch.csv`` and written per checkpoint to
``<log-dir>/eval_<stem>.json``.  Use ``--once`` to score the checkpoints present right now.

Usage:
  python scripts/eval_watch.py --log-dir logs/lafan1_full --num-envs 40 --once
  python scripts/eval_watch.py --log-dir logs/lafan1_full --poll-seconds 60 --target-return 0.82
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, summarize

CSV_FIELDS = ["ckpt", "iteration", "num_updates", "clips", "failed", "mean_norm_return",
              "step_weighted_norm_return"]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def evaluate(ckpt: Path, files: list[str], num_envs: int, log_dir: Path) -> dict:
    policy, cfg = load_policy_from_checkpoint(ckpt, "cuda:0")
    results = run_benchmark(
        policy, files, device="cuda:0", num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        anchor_obs=cfg["env"].get("anchor_obs", False),
        verbose=False,
    )
    s = summarize(results)
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        iteration, num_updates = int(ck.get("iteration", -1)), int(ck.get("num_updates", -1))
    except Exception:  # noqa: BLE001
        iteration = num_updates = -1
    out = {
        "ckpt": str(ckpt),
        "iteration": iteration,
        "num_updates": num_updates,
        "protocol": "Appendix F.1: first-frame start, no noise, no DR, friction 1.0, failure = training termination",
        "normalized_return": "sum(reward) / (0.1 * full clip length)",
        "all": s,
        "clips": [
            {"name": r.name, "frames": r.frames, "steps": r.steps, "denominator_steps": r.denominator_steps,
             "norm_return": r.norm_return, "failed": r.failed, "fail_step": r.fail_step,
             "survival_fraction": r.steps / max(r.denominator_steps, 1), "terms": r.terms}
            for r in results
        ],
    }
    (log_dir / f"eval_{ckpt.stem}.json").write_text(json.dumps(out, indent=1))
    del policy
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1"))
    ap.add_argument("--num-envs", type=int, default=40)
    ap.add_argument("--once", action="store_true", help="score current checkpoints and exit")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--target-return", type=float, default=None)
    ap.add_argument("--max-evals", type=int, default=None)
    a = ap.parse_args()

    files = [str(f) for f in discover_npz(a.npz_dir)]
    assert files, f"no clips under {a.npz_dir}"
    csv_path = a.log_dir / "eval_watch.csv"
    done = {r["ckpt"] for r in load_rows(csv_path)}
    print(f"[eval] {len(files)} clips -> {a.log_dir} (already scored: {len(done)})", flush=True)

    n_evals = 0
    while True:
        # ckpt_init first, then numbered checkpoints in ascending order, then final.
        # ``ckpt_latest.pt`` is deliberately skipped: it duplicates the numbered checkpoint written
        # in the same iteration and would double the evaluation load on the training GPU.
        cands: list[Path] = []
        for name in ("ckpt_init.pt", "ckpt_final.pt"):
            p = a.log_dir / name
            if p.exists():
                cands.append(p)
        cands += sorted(a.log_dir.glob("ckpt_[0-9]*.pt"))
        for ckpt in cands:
            if str(ckpt) in done:
                continue
            t0 = time.time()
            out = evaluate(ckpt, files, a.num_envs, a.log_dir)
            s = out["all"]
            rows = load_rows(csv_path)
            rows.append({"ckpt": str(ckpt), "iteration": out["iteration"], "num_updates": out["num_updates"],
                         "clips": s["clips"], "failed": s["failed"],
                         "mean_norm_return": f"{s['mean_norm_return']:.6f}",
                         "step_weighted_norm_return": f"{s['step_weighted_norm_return']:.6f}"})
            write_rows(csv_path, rows)
            done.add(str(ckpt))
            n_evals += 1
            print(f"[eval] {ckpt.name}: it={out['iteration']} upd={out['num_updates']} "
                  f"mean_ret={s['mean_norm_return']:.4f} failed={s['failed']}/{s['clips']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if a.max_evals is not None and n_evals >= a.max_evals:
                print("[eval] max-evals reached", flush=True)
                return
            if a.target_return is not None and s["failed"] == 0 and s["mean_norm_return"] >= a.target_return:
                print(f"[eval] TARGET REACHED: {s['mean_norm_return']:.4f} with 0/{s['clips']} failures", flush=True)
                Path(a.log_dir / "TARGET_REACHED").write_text(
                    json.dumps({"ckpt": str(ckpt), "mean_norm_return": s["mean_norm_return"], "failed": s["failed"]}, indent=1))
                return
        if a.once:
            return
        time.sleep(a.poll_seconds)


if __name__ == "__main__":
    main()