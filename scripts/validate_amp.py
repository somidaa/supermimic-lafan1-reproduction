"""Numeric validation of the optional fp16 AMP path (run with and without --amp, same seed).

Runs a short but real training slice -- collect transitions, then alternate world-model and
policy/critic updates through the *actual* trainers -- and prints the scalar metrics as JSON so
the two runs can be compared. Also reports the GradScaler's final scale: a collapsed scale
means fp16 gradients underflowed and the AMP path is not safe.

Usage:
  python scripts/validate_amp.py --amp
  python scripts/validate_amp.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_train import build_cfg, build_parser  # noqa: E402

from supermimic.training.loop import Trainer  # noqa: E402


def main() -> None:
    ap = build_parser()
    ap.add_argument("--collect-steps", type=int, default=40)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cfg = build_cfg(a)
    tr = Trainer(cfg)
    obs, _ = tr.env.reset()
    for _ in range(a.collect_steps):
        obs, *_ = tr.env.step(tr.act(obs))

    hist: list[dict] = []
    for i in range(a.updates):
        m_wm = tr.wm_trainer.update(tr.buffer)
        m_pi = tr.policy_trainer.update(tr.buffer)
        hist.append({
            "i": i,
            "wm_loss": float(m_wm["loss"]),
            "wm_nse1": float(m_wm["nse_1step"]),
            "wm_nseK": float(m_wm["nse_kstep"]),
            "wm_grad": float(m_wm["grad_norm"]),
            "pi_J": float(m_pi["imag_objective"]),
            "pi_ret": float(m_pi["imag_return_undisc"]),
            "pi_surv": float(m_pi["imag_survival_H"]),
            "pi_gn": float(m_pi["policy_grad_norm"]),
            "pi_absa": float(m_pi["action_abs_mean"]),
            "critic_loss": float(m_pi["critic_loss"]),
        })
    out = {
        "amp": bool(a.amp),
        "seed": a.seed,
        "history": hist,
        "wm_scaler_scale": float(tr.wm_trainer.amp.scaler.get_scale()) if a.amp else None,
        "pi_scaler_scale": float(tr.policy_trainer.amp.scaler.get_scale()) if a.amp else None,
        "all_finite": all(all(v == v and abs(v) != float("inf") for k, v in h.items() if k != "i") for h in hist),
        "final": hist[-1],
    }
    print("AMPVALID " + json.dumps(out), flush=True)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()