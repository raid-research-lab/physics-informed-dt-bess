# Input data and checkpoint formats

Default locations are defined in `pidt/paths.py` and can be overridden with the
environment variables `PIDT_ROOT`, `PIDT_DATA_DIR`, `PIDT_CKPT_DIR` and
`PIDT_RESULTS_DIR`, or per run with `--data`, `--val-data`, `--actual-csv`,
`--model`, `--save` and `--output`.

```
data/
  train/trajectories.npz
  val/trajectories.npz
  test/trajectories.npz
  realized_pv_load.csv
checkpoints/pidt/            written by training
results/evaluation/          written by evaluation
```

## `trajectories.npz`

One trajectory is one operating day of one site: 96 steps of 15 minutes.

| Array | Shape, dtype | Description |
|---|---|---|
| `observations` | (N, 96, 298) float32 | raw (not normalized) state vectors, layout below |
| `actions` | (N, 96, 1) float32 | expert action in [-1, 1]; +1 = charge at rated power, -1 = discharge at rated power |
| `rewards` | (N, 96) float32 | minus the cost of the step (EUR) |
| `returns_to_go` | (N, 96) float32 | sum of the rewards from the step to the end of the day |
| `dates` | (N,) str | `YYYY-MM-DD` |
| `site_ids` | (N,) int | site identifier |
| `initial_soc`, `end_soc` | (N,) float | SOC at the start and end of the day (fraction) |
| `capacity_kwh`, `battery_power_kw` | (N,) float | battery capacity and rated power of the site |

`dates` and `site_ids` are needed by the evaluation to chain the days of each
site; the evaluation also uses the `rewards` of the test trajectories as the
expert reference cost. An optional `stats.json` next to the `.npz` is read as
free-form metadata (for example `{"split": "test"}`).

The first time a split is read it is extracted to `mmap_cache/` in the same
folder and memory-mapped afterwards; delete that folder if the `.npz` changes.

### State vector (298 features)

| Index | Feature | Unit |
|---|---|---|
| 0 | PV forecast of the current step | kWh / 15 min |
| 1 | load forecast of the current step | kWh / 15 min |
| 2 | purchase price | EUR/kWh |
| 3 | sell price | EUR/kWh |
| 4 | SOC | fraction |
| 5, 6 | SOC limits (0.20, 1.00 in the paper) | fraction |
| 7, 8 | sin / cos of the time of day | - |
| 9 | battery capacity | kWh |
| 10 | rated battery power | kW |
| 11, 12 | charge / discharge efficiency | - |
| 13-297 | 95 look-ahead steps x (purchase price, PV forecast, load forecast) | as above |

Indices 4-6 and 9-12 are read directly by the physics penalty and the
projection (`pidt/models/physics.py`, `pidt/models/pidt.py`), so they must hold
the real per-site values. Feature names are listed in `pidt/config.py`
(`StateConfig`).

## `realized_pv_load.csv`

Realized PV and load used by the evaluation to score the executed actions (the
controller itself only sees the forecasts in the state). One row per site and
step; columns used:

| Column | Description |
|---|---|
| `datetime` | timestamp of the step; the first 10 characters must be the date `YYYY-MM-DD` |
| `site_id` | site identifier, as in `site_ids` |
| `quarter_hour` | step of the day, 0-95 |
| `pv_kw` | realized PV in the step (kWh / 15 min, despite the name) |
| `load_kw` | realized load in the step (kWh / 15 min, despite the name) |

Every evaluated site-day must have its 96 rows. Prices are taken from the state
vector.

## Checkpoints

`best_model.pt` is a `torch.save` dict:

| Key | Content |
|---|---|
| `model_state_dict` | PI-DT weights |
| `norm` | `state_mean`, `state_std` (298,), `rtg_mean`, `rtg_std` (normalization of states and returns-to-go) |
| `stats` | the `stats.json` of the training split |
| `step`, `loss` | training step and validation loss of the checkpoint |
| `eval_metrics` | validation metrics of the checkpoint |
| `optimizer_state_dict` | optimizer state (only needed to resume training) |

The architecture (hidden size, layers, heads, activation, dropout, context
length, reserve) is read at evaluation from `train_config.json` in the same
folder, so keep both files together.
