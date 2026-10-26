"""Minimal environment interaction test: imports, G1 model, FK, world model forward/backward.

Cheap by design -- it deliberately does not re-run the repo's 59-test suite. It checks the
pieces the training run depends on: torch/CUDA, mjlab, the G1 MJCF, the differentiable FK, the
world-model step and its k-step loss (forward + backward), and the policy/critic forward.

Usage:  python scripts/check_env.py
"""

from __future__ import annotations

import torch

import mjlab  # noqa: F401
import mujoco
import warp
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def main() -> None:
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"warp {warp.__version__}  mujoco {mujoco.__version__}")
    print(f"tf32 matmul allowed: {torch.backends.cuda.matmul.allow_tf32}  "
          f"cudnn tf32: {torch.backends.cudnn.allow_tf32}")

    model = load_g1_model()
    print(f"G1: nq={model.nq} nv={model.nv} nbody={model.nbody}")

    dev = "cuda:0"
    fk = G1Kinematics(model, device=dev)
    n = 64
    qpos = torch.zeros(n, model.nq, device=dev)
    qpos[:, 3] = 1.0
    qpos[:, 2] = 0.75
    qvel = torch.zeros(n, model.nv, device=dev)
    out = fk(qpos, qvel)
    print(f"FK ok: pos {tuple(out['pos'].shape)} finite={bool(torch.isfinite(out['pos']).all())}")

    robot = None
    try:
        env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(__import__("pathlib").Path("data/npz/lafan1").glob("walk1_subject1.npz").__next__())],
                                            num_envs=8, record_transitions=False, obs_noise=False,
                                            reset_noise=False, domain_randomization=False, seed=0), device=dev)
        robot = env.scene["robot"]
        obs, _ = env.reset()
        print(f"env ok: policy obs {tuple(obs['policy'].shape)} goal {tuple(obs['goal'].shape)}")
        act = torch.zeros(env.num_envs, 29, device=dev)
        obs, rew, term, tout, _ = env.step(act)
        print(f"env step ok: reward mean {float(rew.mean()):.5f} term {int(term.sum())}")
        limits = robot.data.soft_joint_pos_limits[0].clone()
        default_joint_pos = robot.data.default_joint_pos[0].clone()
        env.close()
    finally:
        pass

    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, limits, default_joint_pos)
    wm = WorldModel(WorldModelCfg(), heads, dev)
    print(f"world model params: {sum(p.numel() for p in wm.parameters())/1e6:.2f}M")

    k, b = 8, 256
    s = torch.zeros(k + 1, b, 71, device=dev)
    s[..., 3] = 1.0
    s[..., 2] = 0.75
    s[..., 7:36] = default_joint_pos
    a = torch.zeros(k, b, 29, device=dev)
    xi = torch.zeros(b, 33, device=dev)
    xi[:, 0] = 0.75
    done = torch.zeros(k, b, device=dev)
    from supermimic.world_model.heads import ReferenceFrame

    refs = [ReferenceFrame(None, None, None, None,
                           torch.zeros(b, 3, device=dev) + torch.tensor([0.0, 0.0, 0.75], device=dev),
                           torch.zeros(b, 4, device=dev) + torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev))
            for _ in range(k)]
    loss, m = wm.window_loss(s, a, xi, done, refs)
    loss.backward()
    gn = sum(float(p.grad.norm()) for p in wm.parameters() if p.grad is not None)
    print(f"WM k-step loss {float(loss):.5f} nse1 {float(m['nse_1step']):.5f} nseK {float(m['nse_kstep']):.5f} "
          f"grad_norm_sum {gn:.4f}")

    pcfg = PolicyCfg()
    pol = Policy(pcfg).to(dev)
    cri = Critic(CriticCfg(), pcfg=pcfg).to(dev)
    print(f"policy params {sum(p.numel() for p in pol.parameters())/1e6:.2f}M  "
          f"critic {sum(p.numel() for p in cri.parameters())/1e6:.3f}M")
    o = torch.zeros(4, 930, device=dev)
    g = torch.zeros(4, 680, device=dev)
    act = pol(o, g)
    print(f"policy forward ok: action {tuple(act.shape)} |a|={float(act.abs().mean()):.5f} "
          f"(decoder_last_gain={pcfg.decoder_last_gain})")
    v = cri(torch.zeros(4, 71, device=dev), g, torch.zeros(4, 33, device=dev), torch.zeros(4, 4, device=dev))
    print(f"critic forward ok: {tuple(v.shape)}")

    # fp16 tensor-core check (Turing has fp16 TC but no bf16/TF32)
    a16 = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    b16 = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    for _ in range(3):
        (a16 @ b16)
    torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    for _ in range(50):
        (a16 @ b16)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 50
    print(f"fp16 2048^3 matmul {dt*1e3:.3f} ms ({2*2048**3/dt/1e12:.1f} TFLOPS)")
    a32 = a16.float()
    b32 = b16.float()
    for _ in range(3):
        (a32 @ b32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        (a32 @ b32)
    torch.cuda.synchronize()
    dt32 = (time.perf_counter() - t0) / 20
    print(f"fp32 2048^3 matmul {dt32*1e3:.3f} ms ({2*2048**3/dt32/1e12:.1f} TFLOPS)  speedup fp16/fp32 = {dt32/dt:.1f}x")
    print("ENV CHECK OK")


if __name__ == "__main__":
    main()