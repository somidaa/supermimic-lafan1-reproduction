"""Full LAFAN1 core-method training (paper Table A.5 scale) on the whole clip library.

This is a thin harness around the repo's own ``Trainer`` (Algorithm A.1) -- it adds no
algorithm of its own. It differs from ``scripts/stage3_train.py`` only in that it defaults to
the *paper's* configuration instead of the heavily downscaled stage-3 budget:

  * all npz clips under ``--npz-dir`` (40 LAFAN1 clips, ~2.45 h) unless ``--clips`` is given;
  * world model MLP (2048, 2048, 2048), k = 8 autoregressive windows, 10 minibatches x 2048;
  * policy 2048-1024-512-256 -> z in R^32 -> shared decoder 2048-1024-512-256, critic 256-128;
  * BPTT over H = 32 imagined steps, 4096 start states, GAE(lambda = 0.95), gamma = 0.99;
  * 4096 envs, UTD 0.4, action noise 0.1, domain randomisation on, 250k-iteration cosine LR.

Two schedule knobs from the verified reference reproduction are exposed:
  ``--policy-warmup-updates``  world-model-only phase before the first policy update
  ``--anchor-obs``             add the BeyondMimic reference-anchor-relative pose to the goal

Usage:
  python scripts/full_train.py --run-name lafan1_full --total-iterations 320000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer


def build_cfg(a: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(run_name=a.run_name, log_root=a.log_root, seed=a.seed)
    cfg.data.npz_dir = a.npz_dir
    cfg.data.clips = a.clips
    cfg.data.limit = a.limit
    cfg.env.num_envs = a.num_envs
    cfg.env.episode_length_s = a.episode_length_s
    cfg.env.replay_capacity = a.replay_capacity
    cfg.env.future_frames = a.future_frames
    cfg.env.anchor_obs = a.anchor_obs
    cfg.env.sampling_mode = "return"
    cfg.env.action_noise_std = a.action_noise_std
    cfg.env.obs_noise = not a.no_obs_noise
    cfg.env.reset_noise = not a.no_reset_noise
    cfg.env.domain_randomization = not a.no_dr
    cfg.world_model.hidden = tuple(a.wm_hidden)
    cfg.world_model.term_head = a.term_head
    cfg.world_model.term_temp = a.term_temp
    cfg.wm_trainer.window = a.wm_window
    cfg.wm_trainer.batch_size = a.wm_batch
    cfg.wm_trainer.batches_per_update = a.wm_batches
    cfg.wm_trainer.compile = None if a.compile is None else a.compile
    cfg.wm_trainer.amp = a.amp
    cfg.policy_trainer.amp = a.amp
    cfg.policy.encoder_hidden = tuple(a.pi_hidden)
    cfg.policy.decoder_hidden = tuple(a.pi_hidden)
    cfg.critic.hidden = tuple(a.critic_hidden)
    cfg.policy_trainer.horizon = a.horizon
    cfg.policy_trainer.batch_size = a.pi_batch
    cfg.policy_trainer.critic_epochs = a.critic_epochs
    cfg.policy_trainer.critic_minibatches = a.critic_minibatches
    cfg.policy_trainer.compile = None if a.compile is None else a.compile
    cfg.wm_trainer.lr = a.lr
    cfg.policy_trainer.lr = a.lr
    cfg.policy_trainer.critic_lr = a.lr
    cfg.schedule.total_iterations = a.total_iterations
    cfg.schedule.updates_per_env_step = a.updates_per_env_step
    cfg.schedule.policy_updates_per_wm_update = a.policy_updates_per_wm_update
    cfg.schedule.warmup_iterations = a.warmup
    cfg.schedule.policy_warmup_updates = a.policy_warmup_updates
    cfg.schedule.lr_decay_fraction = a.lr_decay_fraction
    cfg.schedule.log_interval = a.log_interval
    cfg.schedule.checkpoint_interval = a.checkpoint_interval
    cfg.schedule.max_hours = a.max_hours
    cfg.resume = a.resume
    return cfg


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="lafan1_full")
    ap.add_argument("--log-root", type=Path, default=Path("logs"))
    ap.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1"))
    ap.add_argument("--clips", nargs="+", default=None, help="default: every npz in --npz-dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # ---- env (paper Table A.5 / A.3)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--episode-length-s", type=float, default=10.0)
    ap.add_argument("--replay-capacity", type=int, default=1000)
    ap.add_argument("--future-frames", type=int, default=10)
    ap.add_argument("--action-noise-std", type=float, default=0.1)
    ap.add_argument("--anchor-obs", action="store_true",
                    help="append the reference-anchor-relative pose (3 pos + 6D ori) to the goal group")
    ap.add_argument("--no-obs-noise", action="store_true")
    ap.add_argument("--no-reset-noise", action="store_true")
    ap.add_argument("--no-dr", action="store_true")
    # ---- world model (paper Table A.4 / A.5)
    ap.add_argument("--wm-hidden", type=int, nargs="+", default=[2048, 2048, 2048])
    ap.add_argument("--wm-window", type=int, default=8)
    ap.add_argument("--wm-batch", type=int, default=2048)
    ap.add_argument("--wm-batches", type=int, default=10)
    ap.add_argument("--term-head", choices=["learned", "analytic"], default="analytic")
    ap.add_argument("--term-temp", type=float, default=0.05)
    # ---- policy / critic (paper Table A.4)
    ap.add_argument("--pi-hidden", type=int, nargs="+", default=[2048, 1024, 512, 256])
    ap.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--pi-batch", type=int, default=4096)
    ap.add_argument("--critic-epochs", type=int, default=5)
    ap.add_argument("--critic-minibatches", type=int, default=4)
    # ---- optimisation / schedule (paper Table A.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-decay-fraction", type=float, default=250_000 / 320_000)
    ap.add_argument("--total-iterations", type=int, default=320_000)
    ap.add_argument("--updates-per-env-step", type=float, default=0.4)
    ap.add_argument("--policy-updates-per-wm-update", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=64, help="env steps before the first update")
    ap.add_argument("--policy-warmup-updates", type=int, default=500,
                    help="world-model-only updates before the first policy update")
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--checkpoint-interval", type=int, default=2000)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--compile", default=None, help='torch.compile mode: "default", "reduce-overhead", ...')
    ap.add_argument("--amp", action="store_true",
                    help="fp16 tensor-core autocast for the MLP matmuls (Turing has no bf16/TF32)")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--no-init-ckpt", action="store_true")
    return ap


def main() -> None:
    a = build_parser().parse_args()

    cfg = build_cfg(a)
    tr = Trainer(cfg)
    n_clips = tr.lib.num_clips
    print(f"[full] library: {n_clips} clips, {tr.lib.num_frames} frames, {tr.lib.num_bins} bins", flush=True)
    print(f"[full] args: {json.dumps(vars(a), default=str)}", flush=True)
    if not a.no_init_ckpt:
        tr.save("ckpt_init.pt")
        print("[full] saved ckpt_init.pt (before any update)", flush=True)

    t0 = time.time()
    tr.run()
    wall = time.time() - t0
    tr.save("ckpt_final.pt")
    summary = {
        "run_name": a.run_name,
        "seed": a.seed,
        "wall_seconds": wall,
        "wall_hours": wall / 3600.0,
        "gpu_hours": wall / 3600.0,
        "iterations": int(tr.iteration),
        "num_updates": int(tr.num_updates),
        "env_steps": int(tr.iteration) * int(cfg.env.num_envs),
        "env_steps_per_second": (int(tr.iteration) * int(cfg.env.num_envs)) / max(wall, 1e-9),
        "num_envs": cfg.env.num_envs,
        "num_clips": n_clips,
        "library_frames": int(tr.lib.num_frames),
        "horizon": cfg.policy_trainer.horizon,
        "anchor_obs": cfg.env.anchor_obs,
        "policy_warmup_updates": cfg.schedule.policy_warmup_updates,
        "total_iterations_requested": a.total_iterations,
        "config_path": str(tr.log_dir / "config.json"),
        "init_ckpt": str(tr.log_dir / "ckpt_init.pt"),
        "final_ckpt": str(tr.log_dir / "ckpt_final.pt"),
    }
    (tr.log_dir / "run_summary.json").write_text(json.dumps(summary, indent=1))
    print("[full] DONE " + json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()