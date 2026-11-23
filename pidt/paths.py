"""Default file locations.

Every script reads its default inputs and outputs from this module; all of them
can also be passed on the command line. The locations can be moved with
environment variables:

    PIDT_ROOT         repository root (default: parent of the ``pidt`` package)
    PIDT_DATA_DIR     trajectories and realized PV / load (default: $PIDT_ROOT/data)
    PIDT_CKPT_DIR     trained models                      (default: $PIDT_ROOT/checkpoints)
    PIDT_RESULTS_DIR  evaluation outputs                  (default: $PIDT_ROOT/results)

The expected content of ``data/`` is described in ``docs/data.md``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _env_path("PIDT_ROOT", PACKAGE_DIR.parent)
DATA_DIR = _env_path("PIDT_DATA_DIR", PROJECT_ROOT / "data")
CHECKPOINTS_DIR = _env_path("PIDT_CKPT_DIR", PROJECT_ROOT / "checkpoints")
RESULTS_DIR = _env_path("PIDT_RESULTS_DIR", PROJECT_ROOT / "results")

# Daily trajectories (one site-day each, see docs/data.md).
TRAIN_SPLIT_DIR = DATA_DIR / "train"    # expert actions executed on the nonlinear plant
VAL_SPLIT_DIR = DATA_DIR / "val"        # idem, later dates
TEST_SPLIT_DIR = DATA_DIR / "test"      # expert trajectories of the test dates
# Realized PV and load per site and quarter hour, used to score executed actions.
REALIZED_PV_LOAD_CSV = DATA_DIR / "realized_pv_load.csv"

PIDT_CHECKPOINT_DIR = CHECKPOINTS_DIR / "pidt"
EVALUATION_RESULTS = RESULTS_DIR / "evaluation"
