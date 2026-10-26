# Configuration reference

The configuration is a dataclass tree in `src/supermimic/config.py` and its sub-modules; the CLI of
`scripts/full_train.py` sets the leaves it exposes. `configs/lafan1_full.json` is the file the training
loop wrote for the reported run — that file, not this table, is the authority on what was used.

## The reported run

| group | field | value |
|---|---|---|
| data | `npz_dir` | `data/npz/lafan1` (all 40 clips discovered) |
| env | `num_envs` | 4096 |
| | `episode_length_s` | 10.0 |
| | `action_noise_std` | 0.1 |
| | `obs_noise` / `reset_noise` / `domain_randomization` | true / true / true |
| | `sampling_mode` | `return` (0.5 uniform, 0.4 softmax(−25 G/G_max), 0.1 memory) |
| | `replay_capacity` | 1000 steps per environment |
| | `future_frames` | 10 |
| | `anchor_obs` | **true** (see [PAPER_DIFFERENCES.md](PAPER_DIFFERENCES.md)) |
| world model | `hidden` | (2048, 2048, 2048) |
| | `term_head` / `term_temp` | `analytic` / 0.05 |
| | `height_threshold` / `tilt_threshold` | 0.25 m / 0.8 |
| | `state_spec.dt` | 0.02 |
| | `state_spec.err_scale` | 0.05, 0.1, 0.5, 1.0, 0.1, 2.0 (pos, rot, lin vel, ang vel, joint pos, joint vel) |
| | `state_spec.out_scale` | 0.3, 0.5, 2.0, 0.01, 0.02, 0.02 |
| | `state_spec.in_scale` | 1, 1, 1, 2, 1, 5 |
| WM trainer | `window` | 8 |
| | `batches_per_update` | 10 |
| | `batch_size` | 2048 |
| | `lr` / `grad_clip` / `weight_decay` | 2e-4 / 1.0 / 0.0 |
| | `compile` / `amp` | `default` / true |
| policy | `encoder_hidden` / `decoder_hidden` | (2048, 1024, 512, 256) both |
| | `latent_dim` / `history` | 32 / 10 |
| | `decoder_last_gain` | 0.01 |
| | `goal_dim` | **689** (680 + 9) |
| critic | `hidden` | (256, 128) |
| policy trainer | `horizon` | 32 |
| | `batch_size` | 4096 |
| | `gamma` / `gae_lambda` | 0.99 / 0.95 |
| | `lr` / `betas` / `critic_lr` | 2e-4 / (0.9, 0.95) / 2e-4 |
| | `grad_clip_norm` / `grad_clip_value` | 1.0 / 0.1 |
| | `critic_epochs` / `critic_minibatches` | 5 / 4 |
| | `use_critic` / `obs_noise` | true / true |
| | `compile` / `amp` / `anchor_obs` | `default` / true / true |
| schedule | `total_iterations` | 320000 (paper budget; fixes the LR schedule) |
| | `updates_per_env_step` | 0.4 |
| | `policy_warmup_updates` | 0 |
| | `warmup_iterations` | 64 |
| | `lr_final_ratio` | 0.03 |
| | `lr_decay_fraction` | 0.78125 (= 250000/320000) |
| | `log_interval` / `checkpoint_interval` | 50 / 2000 |
| | `max_hours` | 9.0 |

## Fields that exist but were left at their defaults

| field | default | meaning |
|---|---|---|
| `data.clips` | `None` | restrict training to a named subset; `None` uses every npz found |
| `data.library` | `None` | a packed fp16 library (`.pt`) for very large motion sets; forces compact mode |
| `env.compact_library` | false | store only the 71-dim reference state and compute goal features / FK online |
| `wm_trainer.compile`, `policy_trainer.compile` | `"default"` | passed to `torch.compile`; set `None` with `--compile` unset |
| `world_model.term_head` | `learned` | the code default; both harnesses here pass `analytic` |
| `policy_trainer.use_critic` | true | set false for the critic-free ablated objective |
| `schedule.policy_warmup_updates` | 0 | world-model-only phase before the first policy update |

## CLI mapping

`scripts/full_train.py` exposes the fields above; anything not exposed keeps its dataclass default.
Run `python scripts/full_train.py --help` for the authoritative list. `scripts/bench_full.py`,
`scripts/validate_amp.py` share the same parser, so the same flags work for a shortened run.

## Selecting clips

```bash
# all 40 (default)
python scripts/full_train.py --run-name all40 ... 

# a subset, e.g. for a quick smoke test
python scripts/full_train.py --run-name smoke --clips walk1_subject1 dance1_subject1 \
    --num-envs 512 --wm-batches 2 --wm-batch 512 --pi-batch 512 --total-iterations 2000
```

`--clips` uses the npz file stems; a name that does not exist is silently dropped by the discovery
helper, so check the printed library summary (`[INFO] MotionLibrary(full): N clips ...`) at start-up.