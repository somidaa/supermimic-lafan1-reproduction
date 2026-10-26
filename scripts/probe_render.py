"""Probe whether MuJoCo offscreen rendering works on this host.

Headless rendering needs a working GL backend (WGL on Windows, EGL/OSMesa elsewhere).
This writes one PNG so that a failure is unambiguous.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else "render_probe.png")

print("MUJOCO_GL =", os.environ.get("MUJOCO_GL", "<unset>"))
import mujoco

print("mujoco", mujoco.__version__)

from supermimic.data.motion_convert import load_g1_model

model = load_g1_model()
data = mujoco.MjData(model)
data.qpos[:] = 0.0
data.qpos[2] = 0.79
data.qpos[3] = 1.0

for backend in ("<default>", "osmesa", "egl", "glfw"):
    if backend != "<default>":
        os.environ["MUJOCO_GL"] = backend
    try:
        renderer = mujoco.Renderer(model, height=240, width=320)
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance = 3.0
        cam.elevation = -15.0
        cam.azimuth = 120.0
        cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        pct = np.asarray(img)[..., :3]
        print(f"backend {backend}: OK shape={img.shape} mean={pct.mean():.1f} nonblack={(pct.max(axis=-1) > 10).mean():.3f}")
        try:
            from PIL import Image

            Image.fromarray(img).save(out)
            print(f"  wrote {out}")
        except Exception as e:  # noqa: BLE001
            print(f"  save failed: {e}")
        renderer.close()
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"backend {backend}: FAILED {type(e).__name__}: {str(e)[:200]}")
print("ALL BACKENDS FAILED")
sys.exit(1)