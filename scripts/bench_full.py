"""Short throughput benchmark -> full-training budget estimate.

Builds the real training env and models (no algorithm change), collects a handful of
transitions so the replay buffer is sampleable, then times the three costs of one training
iteration separately: vectorised simulator step, world-model update, policy (BPTT) update.
Also reports peak GPU memory and the projected wall clock of the requested schedule.

Usage:
  python scripts/bench_full.py --num-envs 4096 --wm-batches 10 --pi-batch 4096 --horizon 32
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_train import build_cfg, build_parser  # noqa: E402

from supermimic.training.loop import Trainer  # noqa: E402


def timed(fn, n: int, label: str) -> tuple[float, list[float]]:
    """Run n calls, returning (mean of all, per-call seconds)."""
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    print(f"[bench] {label} per-call s: " + " ".join(f"{t:.4f}" for t in ts), flush=True)
    return sum(ts) / len(ts), ts


def matmul_probe() -> dict:
    """Confirm whether autocast actually lowers the MLP matmuls to fp16 tensor cores."""
    import supermimic.policy.nets as nets

    mlp = nets.MLP(2048, (2048, 2048, 2048), 70).cuda()
    x = torch.randn(2048, 2048, device="cuda", requires_grad=True)

    def run(amp: bool):
        for _ in range(3):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                y = mlp(x)
            y.float().sum().backward()
            x.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                y = mlp(x)
            y.float().sum().backward()
            x.grad = None
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 10

    t_on, t_off = run(True), run(False)
    print(f"[bench] 2048^3 MLP fwd+bwd: fp16 autocast {t_on*1e3:.2f} ms vs fp32 {t_off*1e3:.2f} ms "
          f"({t_off/t_on:.2f}x)", flush=True)
    return {"mlp_fwd_bwd_fp16_s": t_on, "mlp_fwd_bwd_fp32_s": t_off, "mlp_speedup": t_off / t_on}


def main() -> None:
    ap = build_parser()
    ap.add_argument("--collect-steps", type=int, default=60)
    ap.add_argument("--bench-updates", type=int, default=3)
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cfg = build_cfg(a)
    tr = Trainer(cfg)
    print(f"[bench] library: {tr.lib.num_clips} clips, {tr.lib.num_frames} frames, "
          f"{tr.lib.num_bins} bins", flush=True)
    print(f"[bench] num_envs={cfg.env.num_envs} wm_batches={cfg.wm_trainer.batches_per_update} "
          f"wm_batch={cfg.wm_trainer.batch_size} pi_batch={cfg.policy_trainer.batch_size} "
          f"horizon={cfg.policy_trainer.horizon} anchor_obs={cfg.env.anchor_obs}", flush=True)

    obs, _ = tr.env.reset()
    for _ in range(a.collect_steps):
        obs, *_ = tr.env.step(tr.act(obs))
    print(f"[bench] buffer size after {a.collect_steps} steps: {int(tr.buffer.size)}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t_sim, _ = timed(lambda: tr.env.step(tr.act(obs)), a.bench_iters, "sim")
    mem_sim = torch.cuda.max_memory_allocated() / 1e9
    _, wm_calls = timed(lambda: tr.wm_trainer.update(tr.buffer), a.bench_updates, "wm_update")
    t_wm = sum(wm_calls[1:]) / max(len(wm_calls) - 1, 1) if len(wm_calls) > 1 else wm_calls[0]
    mem_wm = torch.cuda.max_memory_allocated() / 1e9
    _, pi_calls = timed(lambda: tr.policy_trainer.update(tr.buffer), a.bench_updates, "policy_update")
    t_pi = sum(pi_calls[1:]) / max(len(pi_calls) - 1, 1) if len(pi_calls) > 1 else pi_calls[0]
    mem_pi = torch.cuda.max_memory_allocated() / 1e9
    probe = matmul_probe()
    print(f"[bench] steady-state (calls after the first): wm {t_wm:.3f}s  policy {t_pi:.3f}s", flush=True)

    utd = cfg.schedule.updates_per_env_step
    # One "iteration" = one vectorised env step + UTD updates (WM + policy each).
    t_iter = t_sim + utd * (t_wm + t_pi)
    total = cfg.schedule.total_iterations
    est_hours = t_iter * total / 3600.0
    out = {
        "num_envs": cfg.env.num_envs,
        "num_clips": tr.lib.num_clips,
        "library_frames": int(tr.lib.num_frames),
        "sim_seconds_per_step": t_sim,
        "env_steps_per_second": cfg.env.num_envs / t_sim,
        "wm_seconds_per_update": t_wm,
        "policy_seconds_per_update": t_pi,
        "seconds_per_iteration": t_iter,
        "iterations_per_hour": 3600.0 / t_iter,
        "total_iterations": total,
        "estimated_hours": est_hours,
        "peak_gpu_gb_after_sim": mem_sim,
        "peak_gpu_gb_after_wm": mem_wm,
        "peak_gpu_gb_after_policy": mem_pi,
        **probe,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "config": {
            "wm_hidden": cfg.world_model.hidden,
            "wm_window": cfg.wm_trainer.window,
            "wm_batches": cfg.wm_trainer.batches_per_update,
            "wm_batch": cfg.wm_trainer.batch_size,
            "pi_hidden": cfg.policy.encoder_hidden,
            "horizon": cfg.policy_trainer.horizon,
            "pi_batch": cfg.policy_trainer.batch_size,
            "anchor_obs": cfg.env.anchor_obs,
            "term_head": cfg.world_model.term_head,
            "compile": cfg.wm_trainer.compile,
        },
    }
    print("[bench] " + json.dumps(out, indent=1), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        print(f"[bench] saved {a.out}", flush=True)


if __name__ == "__main__":
    main()