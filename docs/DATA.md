# Data

## What is needed

The controlled LAFAN1 comparison of paper Fig. 4a uses **40 G1-retargeted LAFAN1 clips**:

| | |
|---|---|
| clips | 40 (`dance`, `fallAndGetUp`, `fight`, `fightAndSports`, `jumps`, `run`, `sprint`, `walk`) |
| frames | **441,140** at 50 Hz |
| duration | **2.451 h** (the paper states 2.45 h) |
| source | HuggingFace `lvhaidong/LAFAN1_Retargeting_Dataset`, folder `g1/` |
| raw format | 30 fps CSV, no header: `root_pos(3)`, `root_quat xyzw(4)`, 29 joint angles (rad) |
| licence | LAFAN1 is CC BY-NC-ND 4.0 (Ubisoft La Forge) — see [DATA_LICENSE.md](DATA_LICENSE.md) |

The motion data is **not** redistributed in this repository.

## Step 1 — download the CSVs

```bash
python scripts/data/download_lafan1_g1.py --out-dir data/raw/lafan1/g1
```

* defaults to the mirror `https://hf-mirror.com`; pass `--endpoint https://huggingface.co` (or set
  `HF_ENDPOINT`) to use the canonical host;
* re-runs skip files that already exist (`--overwrite` to force);
* `--limit 2` downloads only two clips, which is handy for a smoke test;
* it lists the folder through the HuggingFace tree API, so it needs no `huggingface_hub` login.

## Step 2 — convert to the 50 Hz npz library

```bash
python scripts/data/csv_to_npz_batch.py \
    --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30.0 --workers 12
```

The conversion is vectorised and multiprocess (`--workers`), and takes about **10 s for all 40 clips**
on this machine. It writes one `<name>.npz` per clip plus a `manifest.csv` summarising clip, frame
count, frame rate and duration. Existing npz files are skipped unless `--overwrite` is given, which is
how the four transferred clips were reused byte-for-byte.

Each npz contains, for the reference motion:

| key | shape | meaning |
|---|---|---|
| `joint_pos`, `joint_vel` | (T, 29) | resampled joint angles and finite-difference velocities |
| `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w` | (T, 30, ·) | world-frame link states of all 30 bodies from MuJoCo FK |
| `fps`, `joint_names`, `body_names`, `source`, `input_fps` | — | metadata |

Conversion conventions (all implemented in `src/supermimic/data/motion_convert.py`, mirroring
mjlab's `csv_to_npz.py`):

* 30 → 50 Hz by linear interpolation of position and joints and **slerp** of the root orientation;
* joint and root linear velocities by central differences of the resampled signal;
* root angular velocity from the central SO(3) difference of consecutive quaternions (world frame);
* link velocities from MuJoCo `cvel` with `lin_w = cvel_lin − ω × (subtree_com − xpos)`.

The vectorised slerp here deliberately avoids a float32 early-out that the upstream reference
implementation uses when two consecutive quaternions are nearly equal; the two agree to ~1e-9 on the
vectorised path but the upstream one can lag an input frame in orientation (≈ 1e-3 quaternion error,
0.056 rad/s angular-velocity error on `walk1_subject1`). This repository therefore produces slightly
smoother reference angular velocities than mjlab's own converter.

## Step 3 — verify

```bash
python -c "import csv,pathlib; r=list(csv.DictReader(open('data/npz/lafan1/manifest.csv'))); \
print(len(r),'clips', sum(int(x['frames']) for x in r),'frames', round(sum(float(x['duration_s']) for x in r)/3600,3),'h')"
```

Expected: `40 clips 441140 frames 2.451 h`.

The training script discovers every `*.npz` in `--npz-dir` when `--clips` is not given, so no further
registration is needed. `scripts/data/build_library.py` can additionally pack the npz files into a
single fp16 tensor for very large libraries (the BONES-SEED scale); the 40-clip LAFAN1 library is
loaded from the npz files directly.

## Library shape and memory

The motion library is held on the GPU as flat tensors: per frame the 71-dim reference state, the 14
tracked bodies' world-frame link states, and the precomputed 68-dim goal features of each frame. For
40 clips this is **≈ 0.57 GB**, so the whole library is resident and the goal windows are pure gathers
— no online forward kinematics on the training critical path.