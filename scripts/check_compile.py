"""Check whether torch.compile (Inductor) works on this Windows + Turing setup.

Tries a small pointwise+matmul graph first, then the shapes the training loop actually uses:
the world-model k-step loss and the imagined step (which contains the 29-body FK loop that the
repo's own notes call the launch-overhead bottleneck).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


def try_compile(name: str, fn, *args, mode=None):
    try:
        t0 = time.perf_counter()
        cfn = torch.compile(fn, mode=mode, dynamic=False)
        out = cfn(*args)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[compile] {name}: OK  first-call {dt:.1f}s  out={tuple(out.shape) if isinstance(out, torch.Tensor) else type(out)}", flush=True)
        return cfn
    except Exception as e:  # noqa: BLE001
        print(f"[compile] {name}: FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
        return None


def main() -> None:
    print(f"triton import: ", end="")
    try:
        import triton
        print(triton.__version__)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED {e}")
        return
    print(f"cuda {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    def pointwise(x):
        return torch.sin(x) * 2.0 + x.sum(-1, keepdim=True)

    x = torch.randn(4096, 512, device="cuda")
    if try_compile("pointwise", pointwise, x) is None:
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    dev = "cuda:0"
    model = load_g1_model()
    fk = G1Kinematics(model, device=dev)
    q0 = torch.zeros(29, device=dev)
    limits = torch.stack([-torch.ones(29, device=dev) * 3, torch.ones(29, device=dev) * 3], dim=-1)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, limits, q0)
    wm = WorldModel(WorldModelCfg(), heads, dev)

    b, k = 2048, 8
    s = torch.zeros(k + 1, b, 71, device=dev)
    s[..., 3] = 1.0
    s[..., 2] = 0.75
    a = torch.zeros(k, b, 29, device=dev)
    xi = torch.zeros(b, 33, device=dev)
    xi[:, 0] = 0.75
    done = torch.zeros(k, b, device=dev)
    refs = [ReferenceFrame(None, None, None, None, torch.zeros(b, 3, device=dev), torch.tensor([1.0, 0, 0, 0], device=dev).expand(b, 4)) for _ in range(k)]

    wm.enable_compile("default")
    try:
        t0 = time.perf_counter()
        loss, m = wm.window_loss(s, a, xi, done, refs)
        loss.backward()
        torch.cuda.synchronize()
        print(f"[compile] wm window_loss+backward first call {time.perf_counter()-t0:.1f}s", flush=True)
        for _ in range(2):
            wm.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            loss, m = wm.window_loss(s, a, xi, done, refs)
            loss.backward()
            torch.cuda.synchronize()
            print(f"[compile] wm window_loss+backward steady {time.perf_counter()-t0:.3f}s", flush=True)

        def raw_loss():
            return wm._loss_step(s[0], a[0], xi, s[1], done[0], refs[0].root_pos_w, refs[0].root_quat_w)

        def timed(fn, n=3):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n

        print(f"[compile] wm _loss_step eager {timed(raw_loss)*1e3:.2f} ms", flush=True)
        cstep = torch.compile(wm._loss_step, dynamic=False)
        print(f"[compile] compiled _loss_step {timed(cstep)*1e3:.2f} ms", flush=True)

        # imagination step
        from supermimic.imagination.compiled import ImagStep

        pcfg = PolicyCfg()
        pol = Policy(pcfg).to(dev)
        step = ImagStep(wm, q0, compile_mode=None)
        bi = 4096
        si = torch.zeros(bi, 71, device=dev)
        si[..., 3] = 1.0
        si[..., 2] = 0.75
        ai = torch.zeros(bi, 29, device=dev)
        xii = torch.zeros(bi, 33, device=dev)
        xii[:, 0] = 0.75
        ref = ReferenceFrame(torch.zeros(bi, 14, 3, device=dev), torch.zeros(bi, 14, 4, device=dev),
                             torch.zeros(bi, 14, 3, device=dev), torch.zeros(bi, 14, 3, device=dev),
                             torch.zeros(bi, 3, device=dev), torch.tensor([1.0, 0, 0, 0], device=dev).expand(bi, 4))
        ref.body_quat_w[..., 0] = 1.0
        print(f"[compile] imag step eager {timed(lambda: step(si, ai, ai, xii, ref))*1e3:.2f} ms", flush=True)
        step2 = ImagStep(wm, q0, compile_mode="default")
        print(f"[compile] imag step compiled {timed(lambda: step2(si, ai, ai, xii, ref))*1e3:.2f} ms", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[compile] wm/imag compile FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)


if __name__ == "__main__":
    main()