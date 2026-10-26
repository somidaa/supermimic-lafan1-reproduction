# Media provenance

Every video and GIF here is a **real rollout of the trained policy in the evaluation environment**.
Nothing is a schematic animation, nothing is drawn from the world model, and each clip's rollout has a
machine-readable record in `results/rollout_*.json`.

## Common setup

| | |
|---|---|
| checkpoint | `ckpt_0048000.pt` (iteration 48,000, 19,174 policy updates) |
| environment | the real evaluation environment (`supermimic.envs.make_env`) with the Appendix-F.1 settings: the clip's first frame, no observation noise, no action noise, no reset noise, no domain randomisation (foot friction 1.0), deterministic policy |
| motion library | `data/npz/lafan1/<clip>.npz`, 50 Hz |
| robot | the same G1 MJCF the simulator uses (`mjlab` asset zoo), rendered offscreen with CPU `mujoco.Renderer` |
| overlay | the reference pose of the same frame drawn as a translucent blue ghost (visual geoms only). **Solid robot = policy, blue ghost = reference** |
| camera | pelvis-tracking, azimuth 120°, elevation −10°, distance 3.0 |
| compositing | two render passes (robot, ghost) alpha-blended in numpy with the ghost at 30 % opacity, then H.264 via `imageio-ffmpeg` |
| tool | `scripts/render_rollout.py` (and `scripts/mp4_to_gif.py` for the GIF cut) |

`<clip>` is the npz stem, i.e. the LAFAN1 clip id (`walk1_subject1`, `jumps1_subject2`, …).

## What is shipped

| file | clip | fps | resolution | rollout | outcome |
|---|---|---|---|---|---|
| `media/walk1_subject1.mp4` | `walk1_subject1` | 50 | 640×480 | 8.0 s (400 steps) | completes, Σ reward 38.82 |
| `media/run1_subject2.mp4` | `run1_subject2` | 50 | 640×480 | 8.0 s (400 steps) | completes, Σ reward 39.01 |
| `media/fallAndGetUp2_subject3_failure.mp4` | `fallAndGetUp2_subject3` | 10 | 480×360 | 98.44 s (**4,922 steps**) | **terminates** at step 4,922, Σ reward 410.13 |
| `media/fallAndGetUp2_subject3.mp4` | `fallAndGetUp2_subject3` | 25 | 480×360 | 163.92 s (8,196 steps) | completes, Σ reward 664.22 |
| `media/jumps1_subject2.mp4` | `jumps1_subject2` | 10 | 480×360 | 244.44 s (12,222 steps) | completes, Σ reward 1,019.80 |
| `media/preview_success.gif` | `walk1_subject1` | 16.7 | 480 px wide | 6 s excerpt starting at 2 s | success preview |
| `media/preview_failure.gif` | `fallAndGetUp2_subject3` | 10 | 320 px wide | **the last 6.5 s** of the terminating MP4 | failure preview |
| `media/still_*.png` | — | — | the video's resolution | one frame of the corresponding MP4 | still |

The GIF is cut from a shipped MP4 rather than re-simulated, so its provenance is exact:

```bash
python scripts/mp4_to_gif.py --mp4 media/fallAndGetUp2_subject3_failure.mp4 \
    --out media/preview_failure.gif --tail-seconds 6.5 --width 320 --fps 10 \
    --still media/still_fallAndGetUp2_subject3.png
python scripts/mp4_to_gif.py --mp4 media/walk1_subject1.mp4 \
    --out media/preview_success.gif --head-seconds 8 --width 480 --fps 50
```

## The failure case, and why it is also a success case

`jumps1_subject2` and `fallAndGetUp2_subject3` are exactly the two clips that the two evaluations of
this checkpoint each flagged as *the* single failure. Repeating the deterministic rollout shows that
**neither is reproducibly a failure**:

| clip | watcher, 40 envs | independent re-evaluation, 40 envs | renders, 1 env |
|---|---|---|---|
| `fallAndGetUp2_subject3` | completes | **terminates, step 4,912** | terminates 4,917 · completes · terminates 4,904 · **terminates 4,922** |
| `jumps1_subject2` | **terminates, step 10,609** | completes | terminates 10,604 · completes · completes · completes |

So `fallAndGetUp2_subject3` terminated in 4 of 6 recorded runs and `jumps1_subject2` in 2 of 5. Both
sit on the termination threshold, and which of them fails in a given run is not stable. This is the
same ±0.006 / "different marginal clip" behaviour documented in
[RESULTS.md](RESULTS.md#4-evaluation-reproducibility--read-this-before-quoting-a-number), made
concrete: the failure preview of this repository is a genuine terminating rollout, while the same clip
also completes in other runs of the same weights.

Consequences for reporting:

* this repository reports a **failure count** (1/40, stable across the runs performed here), not a
  settled list of failing clips;
* the committed evidence for each termination is the JSON record, not the video alone —
  `results/eval_ckpt_0048000.json` (jumps1_subject2, step 10,609),
  `results/eval_best_independent_reeval.json` (fallAndGetUp2_subject3, step 4,912) and
  `results/rollout_fallAndGetUp2_subject3_failure.json` (step 4,922);
* a render uses **one environment** while the reported evaluation uses 40 in parallel. The physics
  model is identical, but the MuJoCo-Warp solver's reduction order is not guaranteed to be, which is
  the most likely source of this spread.

## Reproducing a video

```bash
python scripts/render_rollout.py \
    --ckpt logs/lafan1_full/ckpt_0048000.pt \
    --clip walk1_subject1 --npz-dir data/npz/lafan1 \
    --out media/walk1_subject1.mp4 --gif media/preview_success.gif \
    --still media/still_walk1_subject1.png \
    --meta results/rollout_walk1_subject1.json \
    --seconds 8 --width 640 --height 480 --fps 50 --gif-from 2 --gif-every 3
```

`--fps` is the *playback* rate; the control rate is 50 Hz, so the render stride becomes
`round(50 / fps)` and the file is written at `50 / stride` fps, i.e. real time. `--seconds` caps the
rollout (the run stops early if the clip ends or the termination fires); the failure MP4 was produced
with `--seconds 165`, which is past the clip's own length, so the termination is what ended it.

Rendering needs a working GL context. On this Windows host the default backend works; on another
machine `python scripts/probe_render.py out.png` reports in one command which backend succeeds.

## What the videos do and do not show

* They show the **policy**, not the world model — the robot is simulated with MuJoCo-Warp, never with
  the learned dynamics. Nothing in these videos comes from imagination.
* The blue ghost is the reference motion at the same frame index, so the visible separation is the
  tracking error; when the two coincide the policy is tracking well.
* The two success videos are 8-second excerpts; the reported normalised returns always come from the
  **complete** 40-clip evaluation, never from these excerpts.