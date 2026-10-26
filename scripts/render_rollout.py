"""Render a deterministic evaluation rollout of a trained policy, with the reference motion
drawn as a translucent ghost.

This reuses the *real evaluation environment* (``supermimic.eval.benchmark`` settings: one clip
from its first frame, no observation/action noise, no reset noise, no domain randomisation, foot
friction 1.0, deterministic physics) and the checkpoint's own configuration -- nothing about the
policy or the physics is altered for the video.

Rendering uses CPU MuJoCo (``mujoco.Renderer``) with the same G1 MJCF the simulation uses, so the
robot and the ghost are drawn from their own joint coordinates; the ghost's collision geoms are
hidden and its visual geoms are tinted blue, exactly like the repository's viewer ghost.

Outputs an MP4 (H.264 via imageio-ffmpeg) and, optionally, a downscaled GIF.

Usage:
  python scripts/render_rollout.py --ckpt logs/lafan1_full/ckpt_0048000.pt \
      --clip run1_subject2 --out media/run1_subject2.mp4 --gif media/run1_subject2.gif \
      --seconds 8 --width 640 --height 480 --fps 30
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch


def quat_to_wxyz(q):
    return q


def collect(cfg, ckpt: Path, clip: str, npz_dir: Path, seconds: float, device: str):
    """Run the eval protocol for one clip and return (robot qpos, ghost qpos, per-step reward)."""
    import mujoco

    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    npz = npz_dir / f"{clip}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found; build the library first (see docs/DATA.md)")

    policy, ck_cfg = load_policy_from_checkpoint(ckpt, device)
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(npz)],
        num_envs=1,
        future_frames=ck_cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=ck_cfg["env"].get("anchor_obs", False),
    )
    env = make_env(args, device=device)
    model = load_g1_model()
    dt = float(env.step_dt)
    n_steps = int(min(seconds / dt, env.command_manager.get_term("motion").library.clips[0].length))

    obs, _ = env.reset()
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("motion")

    def robot_qpos() -> np.ndarray:
        q = np.zeros(model.nq, dtype=np.float64)
        q[0:3] = robot.data.root_link_pos_w[0].cpu().numpy()
        q[3:7] = robot.data.root_link_quat_w[0].cpu().numpy()
        q[7:] = robot.data.joint_pos[0].cpu().numpy()
        return q

    qs_robot, qs_ghost, rewards, failed = [], [], [], None
    qs_robot.append(robot_qpos())
    gq, _ = cmd.library.ref_qpos_qvel(cmd.frame)
    qs_ghost.append(gq[0].cpu().numpy().astype(np.float64))

    with torch.no_grad():
        for step in range(n_steps):
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            qs_robot.append(robot_qpos())
            gq, _ = cmd.library.ref_qpos_qvel(cmd.frame)
            qs_ghost.append(gq[0].cpu().numpy().astype(np.float64))
            rewards.append(float(rew[0]))
            if bool(term[0]) and failed is None:
                failed = step + 1
            if bool(term[0]) or bool(tout[0]):
                break
    env.close()
    meta = {
        "clip": clip,
        "steps": len(qs_robot) - 1,
        "seconds": (len(qs_robot) - 1) * dt,
        "failed": failed is not None,
        "fail_step": failed,
        "sum_reward": float(np.sum(rewards)),
        "checkpoint": str(ckpt),
        "anchor_obs": bool(ck_cfg["env"].get("anchor_obs", False)),
        "protocol": "Appendix F.1 (first frame, no noise, no DR, friction 1.0, deterministic)",
    }
    return qs_robot, qs_ghost, meta, model


def render(qs_robot, qs_ghost, model, width: int, height: int, stride: int, control_hz: float,
           azimuth: float, elevation: float, distance: float, ghost_alpha: float = 0.30):
    """Render the robot and the reference motion as two passes and composite them.

    A single scene cannot express "solid robot behind a translucent ghost": MuJoCo draws every
    geom additively-opaque in note order, so the ghost would hide the robot. We therefore render
    the robot and the (recoloured) reference pose separately and alpha-blend the second over the
    first, which is what the repository's interactive viewer shows.
    """
    import mujoco

    # Robot pass: brighten the visual geoms so the solid robot stays readable under the overlay.
    robot_model = copy.deepcopy(model)
    ghost_model = copy.deepcopy(model)
    for m, colour in ((robot_model, (0.93, 0.91, 0.87, 1.0)), (ghost_model, (0.20, 0.62, 1.00, 1.0))):
        for gi in range(m.ngeom):
            if m.geom_contype[gi] != 0 or m.geom_conaffinity[gi] != 0:
                m.geom_rgba[gi] = (0.0, 0.0, 0.0, 0.0)  # hide collision geoms
            else:
                m.geom_rgba[gi] = colour

    for m in (robot_model, ghost_model):
        m.vis.headlight.ambient[:] = 0.45
        m.vis.headlight.diffuse[:] = 0.90
        m.vis.headlight.specular[:] = 0.20

    d_robot = mujoco.MjData(robot_model)
    d_ghost = mujoco.MjData(ghost_model)
    r_robot = mujoco.Renderer(robot_model, height=height, width=width)
    r_ghost = mujoco.Renderer(ghost_model, height=height, width=width)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    cam.trackbodyid = int(pelvis)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.fixedcamid = -1

    frames = []
    for i in range(0, len(qs_robot), stride):
        d_robot.qpos[:] = qs_robot[i]
        mujoco.mj_forward(robot_model, d_robot)
        r_robot.update_scene(d_robot, camera=cam, scene_option=opt)
        img = r_robot.render().copy()

        d_ghost.qpos[:] = qs_ghost[i]
        mujoco.mj_forward(ghost_model, d_ghost)
        r_ghost.update_scene(d_ghost, camera=cam, scene_option=opt)
        gh = r_ghost.render().copy()

        mask = gh.max(axis=2) > 12
        if mask.any():
            img[mask] = (ghost_alpha * gh[mask] + (1.0 - ghost_alpha) * img[mask]).astype(np.uint8)
        frames.append(img)
    r_robot.close()
    r_ghost.close()
    return frames, control_hz / stride


def write_mp4(frames, out: Path, fps: float) -> None:
    import imageio.v2 as imageio

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7,
                                macro_block_size=1, ffmpeg_log_level="error")
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def write_gif(frames, out: Path, fps: float, width: int = 480, every: int = 2, start_s: float = 0.0) -> None:
    from PIL import Image

    out.parent.mkdir(parents=True, exist_ok=True)
    imgs = []
    begin = int(max(0.0, start_s) * fps / every)
    for f in frames[begin::every]:
        im = Image.fromarray(f)
        h = int(im.height * width / im.width)
        imgs.append(im.resize((width, h), Image.LANCZOS))
    if not imgs:
        return
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 * every / fps), loop=0, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--clip", required=True, help="clip stem, e.g. run1_subject2")
    ap.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1"))
    ap.add_argument("--out", type=Path, required=True, help="output .mp4")
    ap.add_argument("--gif", type=Path, default=None)
    ap.add_argument("--gif-from", type=float, default=0.0, help="start the GIF this many seconds into the rollout")
    ap.add_argument("--gif-every", type=int, default=2)
    ap.add_argument("--gif-width", type=int, default=360)
    ap.add_argument("--meta", type=Path, default=None, help="write rollout metadata JSON here")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=50.0,
                    help="playback frame rate; the control rate is 50 Hz, so the render stride "
                         "becomes round(50/fps) and the file is written at 50/stride fps (real time)")
    ap.add_argument("--stride", type=int, default=None, help="override the render stride")
    ap.add_argument("--still", type=Path, default=None, help="also write one PNG frame of the video")
    ap.add_argument("--azimuth", type=float, default=120.0)
    ap.add_argument("--elevation", type=float, default=-10.0)
    ap.add_argument("--distance", type=float, default=3.0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    stride = a.stride if a.stride else max(1, int(round(50.0 / a.fps)))
    qs_robot, qs_ghost, meta, model = collect(None, a.ckpt, a.clip, a.npz_dir, a.seconds, a.device)
    frames, out_fps = render(qs_robot, qs_ghost, model, a.width, a.height, stride,
                             50.0, a.azimuth, a.elevation, a.distance)
    write_mp4(frames, a.out, out_fps)
    if a.gif:
        write_gif(frames, a.gif, out_fps, width=a.gif_width, every=max(1, a.gif_every), start_s=a.gif_from)
    if a.still:
        from PIL import Image

        a.still.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frames[len(frames) // 2]).save(a.still)
    meta.update({"frames_rendered": len(frames), "render_stride": stride,
                 "output_fps": out_fps, "resolution": [a.width, a.height],
                 "camera": {"azimuth": a.azimuth, "elevation": a.elevation, "distance": a.distance}})
    if a.meta:
        a.meta.parent.mkdir(parents=True, exist_ok=True)
        a.meta.write_text(json.dumps(meta, indent=1))
    print("[render] " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
