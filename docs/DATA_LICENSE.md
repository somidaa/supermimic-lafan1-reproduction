# Motion-data licence and what may be redistributed

**This repository does not ship motion data.** `*.csv`, `*.npz` and packed motion libraries are
excluded by `.gitignore` deliberately.

## The motion clips

The 40 clips used here are the **G1-retargeted LAFAN1** motions from HuggingFace
[`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset),
folder `g1/`.

| layer | holder | licence |
|---|---|---|
| LAFAN1 motion capture | Ubisoft La Forge | **CC BY-NC-ND 4.0** |
| G1 retargeting + robot description | the dataset authors / Unitree | see the dataset card (robot description BSD-3-Clause) |

Consequences you must respect if you download it:

* **NonCommercial** — research and evaluation only; not for commercial use.
* **NoDerivatives** — do not redistribute modified versions (including retargeted or retimed copies)
  as your own work.
* **Attribution** — credit Ubisoft La Forge and cite the LAFAN1 paper (see the dataset card).

The `scripts/data/download_lafan1_g1.py` script only fetches the files from the original host; it does
not relicense them. If you fork this repository, do not commit the data or derived libraries, and do
not attach them to a release.

## What this repository does ship

| artifact | status |
|---|---|
| source code (`src/`, `scripts/`) | MIT — see `LICENSE` and the attribution section of the README |
| documentation (`docs/`) | MIT |
| evaluation results (`results/*.json`, `results/*.csv`) | numeric measurements, MIT |
| figures (`results/figures/*.png`) | renderings of the above numbers, MIT |
| video / GIF (`media/`) | renderings of the trained policy **and of the reference motion**. The robot meshes come from Unitree's G1 description (BSD-3-Clause); the *motion* shown as the blue ghost derives from LAFAN1 and is therefore covered by the CC BY-NC-ND terms above — treat the media as illustration, not as redistributable data |

## Model weights

The trained checkpoint (`ckpt_0048000.pt`, ≈ 206 MB) is **not** in Git history. It combines the MIT
code with weights fitted to LAFAN1-derived data, so redistributing it is closer to redistributing a
derivative of the data than to shipping code. If you intend to publish weights:

1. check the size (a single checkpoint exceeds GitHub's 100 MB file limit — use a **GitHub Release
   asset**, not a normal commit);
2. re-read the CC BY-NC-ND terms and, if in doubt, publish the *procedure* (which is what this
   repository does — the run is fully reproducible from the documented commands) rather than the
   weights;
3. never present the weights as usable for commercial deployment.

## Third-party code

* `mjlab` (Apache-2.0), MuJoCo and MuJoCo-Warp (Apache-2.0), NVIDIA Warp (Apache-2.0), PyTorch
  (BSD-3-Clause) — depended on, not vendored;
* the upstream SuperMimic re-implementation whose code is reused here is MIT; its `LICENSE` file is
  kept verbatim at the repository root and the attribution is in the README;
* the robot description shipped by the code (G1 MJCF + STL meshes referenced by `mjlab`) is Unitree's,
  BSD-3-Clause.