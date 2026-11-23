# PI-DT: physics-informed Decision Transformer for battery energy storage control

Reference implementation of the PI-DT controller of

> P. Donate, V. Torres-Sanz, F. J. Martinez, P. Garrido, R. J. Bessa, J. A. Sanguesa.
> **Data-driven predictive control of battery energy storage with physics-informed sequence modeling.**
> *Journal of Energy Storage* 183 (2027) 125523. https://doi.org/10.1016/j.est.2026.125523

PI-DT is a return-to-go conditioned transformer (GPT-2 backbone) that operates
the battery of a site with PV generation under a time-varying tariff. It is
trained offline on expert trajectories and

- receives the physical parameters of each battery (capacity, rated power,
  efficiencies, SOC window) in its state, so one network serves batteries of
  different size without per-site retraining;
- adds a **physics penalty** to the action loss that punishes actions outside
  the per-site feasible band;
- enforces an asymmetric **end-of-day reserve** (soc[T] >= 0.55) at inference
  through a **horizon-aware projection** of every action.

The repository contains the model, the training loop and the closed-loop
evaluation on a nonlinear Thevenin two-branch RC battery plant. It does not
contain data, trained weights or the data-preparation pipeline of the paper:
you train and evaluate on your own trajectories in the format of
[`docs/data.md`](docs/data.md).

## Repository layout

```
pidt/
  models/
    decision_transformer.py   Decision Transformer (GPT-2 backbone, return-to-go conditioning)
    physics.py                per-site action bounds, physics penalty
    pidt.py                   PI-DT: asymmetric horizon-aware bounds and projection, model builder
  training/
    trainer.py                memory-mapped data, normalization, batches, training loop, validation
    train_pidt.py             command-line entry point for training
  battery/
    thevenin.py               nonlinear Thevenin 2RC cell model (OCV, resistances, Peukert)
    plant.py                  battery plant used to execute the actions in closed loop
  evaluation/
    rollouts.py               closed-loop rollout of the model on the plant
    common.py                 data access and metrics
    evaluate.py               command-line entry point for evaluation
  config.py, constants.py     state layout, SOC limits, tariff and degradation constants
  paths.py                    default locations (overridable by environment variables)
docs/data.md                  input data and checkpoint formats
```

## Installation

Python 3.10-3.13. A GPU is recommended for training; evaluation runs on CPU.

```bash
git clone https://github.com/raid-research-lab/physics-informed-dt-bess.git
cd physics-informed-dt-bess
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # versions used for the paper
pip install -e .
```

`transformers` must be **5.17.x**: later releases fail to import `GPT2Model` with
this setup. For a CUDA build of PyTorch install `torch==2.6.0` from the PyTorch
index first.

## Data

Place your data as follows (format in [`docs/data.md`](docs/data.md)):

```
data/
  train/trajectories.npz      training trajectories
  val/trajectories.npz        validation trajectories (early stopping)
  test/trajectories.npz       test trajectories (expert actions and rewards = reference cost)
  realized_pv_load.csv        realized PV and load used to score the actions in evaluation
```

Every trajectory is one day (96 quarter-hour steps) of one site, with a
298-dimensional state. Other locations can be given on the command line or
with the environment variables `PIDT_ROOT`, `PIDT_DATA_DIR`, `PIDT_CKPT_DIR`
and `PIDT_RESULTS_DIR`.

In the paper the training and validation trajectories are MILP expert actions
executed on the nonlinear plant, and the test trajectories are the expert
solutions of the test dates (24,052 / 7,813 / 5,577 site-days of EMSx under the
Spanish PVPC tariff).

## Training

```bash
python -m pidt.training.train_pidt                  # data/train, data/val -> checkpoints/pidt/
```

The defaults are the hyper-parameters of the paper:

| Parameter | Value | Flag |
|---|---|---|
| hidden size / layers / heads | 192 / 4 / 4 | `--hidden-size`, `--n-layer`, `--n-head` |
| activation, dropout | ReLU, 0.1 | `--activation`, `--dropout` |
| context length | 96 steps (one day) | - |
| optimizer | AdamW, lr 3e-4, weight decay 1e-4 | `--learning-rate`, `--weight-decay` |
| batch size | 16 | `--batch-size` |
| steps / warm-up | 30,000 / 2,000 | `--steps`, `--warmup-steps` |
| gradient clipping | 0.25 | `--grad-clip` |
| physics penalty weight | 0.1 | `--lambda-phys` |
| end-of-day reserve | soc[T] >= 0.55 | `--min-terminal-soc` |
| validation / early stopping | every 1,000 steps, patience 8, min. delta 1e-4 | `--eval-interval`, `--early-stopping-*` |
| seed | 42 | `--seed` |

The output folder contains `best_model.pt` (lowest validation loss),
`train_config.json` (architecture and arguments, read back at evaluation) and
`best_metrics.json`. `--lambda-phys 0` trains the variant without physics
penalty. Run `python -m pidt.training.train_pidt --help` for all options.

The first time a split is read, it is extracted to `mmap_cache/` next to its
`trajectories.npz`, so large splits are memory-mapped instead of loaded.

## Evaluation

```bash
python -m pidt.evaluation.evaluate                  # checkpoints/pidt/best_model.pt on data/test
```

For every test site-day the model is rolled out in closed loop on the nonlinear
battery plant: it decides on the forecasts contained in the state, every action
is projected onto the horizon-aware feasible band, and its cost is computed
with the realized PV and load of `realized_pv_load.csv`. Days of the same site
are chained (the final SOC and the internal RC voltages of one day are the
initial state of the next). The reference is the expert cost of the day
(minus the sum of the rewards of the test trajectory).

Outputs in `results/evaluation/`:

- `day_results.csv` - per site-day: expert cost, PI-DT cost, gap (%), final SOC
  and end-of-day reserve deficit;
- `summary.json` - aggregate gap vs. the expert (sum of costs), gap statistics,
  final SOC statistics, days with a deficit above 1 pp and maximum deficit.

Useful options: `--model`, `--data`, `--actual-csv`, `--max-episodes N` (first
N site-days), `--no-projection` (disable the projection), `--rollout-mode
independent` (every day starts from its own initial SOC), `--device`.

## Results reported in the paper

On 5,577 test site-days of EMSx under the PVPC tariff on the nonlinear plant,
PI-DT reached an aggregate gap of 1.39 % vs. the MILP expert, with no day more
than 1 pp below the reserve and a maximum deficit of 0.14 pp (Tables 4-5).

## Citation

```bibtex
@article{donate2027pidt,
  title   = {Data-driven predictive control of battery energy storage with physics-informed sequence modeling},
  author  = {Donate, Pablo and Torres-Sanz, Vicente and Martinez, Francisco J. and Garrido, Piedad and Bessa, Ricardo J. and Sanguesa, Julio A.},
  journal = {Journal of Energy Storage},
  volume  = {183},
  pages   = {125523},
  year    = {2027},
  doi     = {10.1016/j.est.2026.125523}
}
```

## License

MIT, see [`LICENSE`](LICENSE).
