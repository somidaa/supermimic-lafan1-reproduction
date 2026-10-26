"""Shared helpers for the Stage-2 verification tests (audit round 2).

These helpers only *build* things (env, world model, policy, critic, buffers) so the
individual stage-2 test modules stay readable. They deliberately contain no assertions.

Platform notes (documented in IMPLEMENTATION_CHANGES.md / TEST_REPORT.md):

* ``torch.compile`` needs Triton, which is unavailable on Windows. Every config built here
  therefore passes ``compile=None`` (the repo exposes this as a first-class cfg field), so
  these tests exercise the identical math in eager mode. No assertion is relaxed by this.
* The GPU available for this audit is a 2 GB MX450, so environment counts are kept small
  (16-64) and network widths are reduced. This changes scale, not semantics.
"""

from __future__ import annotations

from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
NPZ_DIR = REPO / "data/npz/lafan1"

# Clips that were downloaded/converted for this audit (see TEST_REPORT.md §1).
SMOKE_CLIPS = ("walk1_subject1", "dance1_subject1")


def npz_available() -> bool:
    return NPZ_DIR.exists() and len(list(NPZ_DIR.glob("*.npz"))) > 0


def clip_files(names=SMOKE_CLIPS) -> list[str]:
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ_DIR, names=list(names))
    assert files, f"no npz clips found under {NPZ_DIR} for {names}"
    return [str(f) for f in files]


def make_small_env(num_envs: int = 16, clips=SMOKE_CLIPS, record: bool = True, seed: int = 0, **overrides):
    """A small, fully deterministic-by-config SuperMimic env on real LAFAN1 clips."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env

    kwargs = dict(
        npz_files=clip_files(clips),
        num_envs=num_envs,
        replay_capacity=overrides.pop("replay_capacity", 400),
        record_transitions=record,
        seed=seed,
    )
    kwargs.update(overrides)
    return make_env(SuperMimicEnvCfgArgs(**kwargs), device="cuda:0")


def collect(env, steps: int, action_std: float = 0.4, seed: int = 0) -> None:
    """Step the env with a fixed-seed random policy so the replay buffer gets filled."""
    torch.manual_seed(seed)
    for _ in range(steps):
        a = action_std * torch.randn(env.num_envs, 29, device=env.device)
        env.step(a)


def build_heads(env):
    """AnalyticHeads bound to the env's robot asset (same construction as training/loop.py:81-82)."""
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads

    robot = env.scene["robot"]
    fk = G1Kinematics(load_g1_model(), device=env.device)
    return AnalyticHeads(
        fk,
        TRACKED_BODIES,
        ANCHOR_BODY,
        ROOT_BODY,
        robot.data.soft_joint_pos_limits[0].clone(),
        robot.data.default_joint_pos[0].clone(),
    )


def build_stack(env, hidden=(256, 256), enc=(256, 128), dec=(256, 128), critic_hidden=(64, 32), seed: int = 0):
    """World model + policy + critic sized for a 2 GB GPU. All eager (compile=None)."""
    from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    torch.manual_seed(seed)
    heads = build_heads(env)
    wm = WorldModel(WorldModelCfg(hidden=hidden, term_hidden=(32, 32)), heads, env.device)
    pcfg = PolicyCfg(encoder_hidden=enc, decoder_hidden=dec, goal_dim=680)
    policy = Policy(pcfg).to(env.device)
    critic = Critic(CriticCfg(hidden=critic_hidden), pcfg=pcfg).to(env.device)
    lib = env.command_manager.get_term("motion").library
    return wm, policy, critic, lib, heads, pcfg


def param_snapshot(module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def max_param_diff(module, snap: dict) -> float:
    cur = module.state_dict()
    return max((cur[k] - v).abs().max().item() for k, v in snap.items())


def grad_stats(module) -> tuple[float, float, int]:
    """(total abs grad sum, max abs grad, count of params with a grad)."""
    total, mx, n = 0.0, 0.0, 0
    for p in module.parameters():
        if p.grad is None:
            continue
        n += 1
        total += p.grad.detach().abs().sum().item()
        mx = max(mx, p.grad.detach().abs().max().item())
    return total, mx, n


def all_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())