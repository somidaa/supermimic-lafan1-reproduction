"""Stage-3: independent physical-simulation evaluation of a checkpoint.

Runs the Appendix-F protocol (each clip played once from its first frame, no noise / no domain
randomisation, deterministic physics, failure = the training termination) using the repo's
``run_benchmark``, and reports the **corrected** normalised return (stage-3 F1 fix: denominator = a
complete playthrough of the clip, so an early termination scores low).

Reports in-distribution clips and any held-out clip separately, so coverage and transfer are not
conflated.

Usage:
  uv run python scripts/stage3_eval.py --ckpt logs/s3_core/ckpt_init.pt --num-envs 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1"))
    ap.add_argument("--clips", nargs="+", default=["walk1_subject1", "jumps1_subject1",
                                                  "fight1_subject2", "dance1_subject1"])
    ap.add_argument("--train-clips", nargs="+", default=["walk1_subject1", "jumps1_subject1",
                                                        "fight1_subject2"])
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    files = [str(f) for f in discover_npz(a.npz_dir, names=a.clips)]
    assert files, f"no clips found for {a.clips}"
    policy, cfg = load_policy_from_checkpoint(a.ckpt, "cuda:0")
    results = run_benchmark(
        policy, files, device="cuda:0", num_envs=a.num_envs,
        future_frames=cfg["env"]["future_frames"], anchor_obs=cfg["env"].get("anchor_obs", False),
        verbose=False,
    )
    by_name = {r.name: r for r in results}

    def group(names):
        rs = [by_name[n] for n in names if n in by_name]
        return summarize(rs) if rs else None

    train = [n for n in a.clips if n in a.train_clips]
    held = [n for n in a.clips if n not in a.train_clips]
    out = {
        "ckpt": str(a.ckpt),
        "iteration": None,
        "protocol": "Appendix F: first-frame start, no noise, no DR, friction 1.0, failure = training termination",
        "normalized_return": "sum(reward) / (0.1 * full clip length)   [F1 fix: early termination scores low]",
        "all": summarize(results),
        "in_distribution": group(train),
        "held_out": group(held),
        "clips": [
            {"name": r.name, "frames": r.frames, "steps": r.steps, "denominator_steps": r.denominator_steps,
             "norm_return": r.norm_return, "failed": r.failed, "fail_step": r.fail_step,
             "survival_fraction": r.steps / max(r.denominator_steps, 1), "terms": r.terms}
            for r in results
        ],
    }
    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        out["iteration"] = int(ck.get("iteration", -1))
        out["num_updates"] = int(ck.get("num_updates", -1))
    except Exception:  # noqa: BLE001
        pass

    a.out = a.out or a.ckpt.parent / f"eval_{a.ckpt.stem}.json"
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("ckpt", "iteration", "all", "in_distribution", "held_out")}, indent=1))
    print("\nper clip:")
    for c in out["clips"]:
        print(f"  {c['name']:22s} ret {c['norm_return']:.4f}  steps {c['steps']:6d}/{c['denominator_steps']:6d}"
              f"  surv {c['survival_fraction']:.3f}  failed={c['failed']}")
    print("saved", a.out)


if __name__ == "__main__":
    main()