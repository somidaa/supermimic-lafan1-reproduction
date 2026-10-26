# Results directory

Machine-readable outputs of the reported run. Nothing here is a checkpoint or a motion file.

| file | what it is |
|---|---|
| `eval_watch.csv` | one row per training checkpoint (iterations 0 … 48,000): policy updates, mean and step-weighted normalised return over all 40 clips, failure count. Deduped with `scripts/slim_eval_csv.py`. |
| `eval_ckpt_init.json` | full 40-clip evaluation of `ckpt_init.pt` (before any update) from this run |
| `eval_ckpt_0048000.json` | full evaluation of the best checkpoint, as run by the watcher during training |
| `eval_best_independent_reeval.json` | **the primary reported number** (0.8849 / 1 of 40): the same checkpoint re-evaluated in a separate process |
| `baseline_transferred/eval_ckpt_init.json` | the transferred `s3_core` initialisation, scored on the same 40 clips |
| `baseline_transferred/eval_ckpt_final.json` | the transferred `s3_core` 798-update policy, scored on the same 40 clips |
| `baseline_transferred/eval_watch.csv` | the two baseline rows in CSV form |
| `train_metrics.csv` | 978 rows parsed from the training loop's stdout: iteration, wall hours, it/s, updates, training episode return, world-model 1-step and 8-step NSE, imagined return/survival, per-phase timings |
| `train_metrics.csv` (schema) | `iteration, hours, it_per_s, updates, n_episodes, train_ret, nse1, nse8, imag_ret, imag_surv, t_sim, t_wm, t_pi` |
| `rollout_*.json` | provenance of each rendered video: clip, checkpoint, steps, whether it terminated, fail step, reward sum, camera, resolution |
| `figures/*.png` | the four figures used in the README |

The **full training log is not committed** (it is a 330 KB UTF-16 text file of raw stdout); it is
summarised by `train_metrics.csv`. Regenerate the figures from the shipped files with:

```bash
python scripts/make_figures.py --run-log <train_log> --run-dir <run_dir> \
    --results-dir results --figures-dir results/figures
```

`--run-log` is only needed for `train_metrics.csv` and the world-model figure; the other figures read
`eval_watch.csv` and the JSONs, all of which are present here.

## How to read a row of `eval_watch.csv`

```
ckpt,iteration,num_updates,clips,failed,mean_norm_return,step_weighted_norm_return
logs/lafan1_full/ckpt_0048000.pt,48000,19174,40,1,0.890855,0.893494
```

`mean_norm_return` is the unweighted mean over the 40 clips; `step_weighted_norm_return` weights each
clip by its full length. Both live in [0, 1] for this reward definition. `failed` counts clips whose
rollout hit the training termination before the clip ended.

Note that the watcher's row for iteration 48,000 (`0.8909`) and the independent re-evaluation in
`eval_best_independent_reeval.json` (`0.8849`) are two measurements of the *same weights*; see
[../docs/RESULTS.md](../docs/RESULTS.md) section 4 before quoting either.