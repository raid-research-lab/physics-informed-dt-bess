# -*- coding: utf-8 -*-
"""
Evaluation utilities: data access and metrics.

Conventions of the evaluation:

- The MILP expert cost is the cost of the expert trajectory stored in
  ``actions`` / ``rewards`` of the split; every gap is reported against it.
- Controllers decide with the *forecasts* contained in the state, while the
  executed cost is always computed with the *realized* PV and load of the
  ``realized_pv_load.csv`` (Section 5.6).
- The default rollout is site-chained: the simulated final SOC of a day is the
  initial SOC of the next day of the same site.
- Per-site battery parameters are read from the state vector (indices below),
  never from the global configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pidt.config import apply_emsx_config, cfg
from pidt.constants import STEPS_PER_DAY
from pidt.models.physics import pick_device  # noqa: F401  (re-exported)
from pidt.training.trainer import load_trajectories  # noqa: F401  (re-exported)

apply_emsx_config(cfg)

SOC_INDEX = 4
SOC_MIN_INDEX = 5
SOC_MAX_INDEX = 6
CAPACITY_INDEX = 9
POWER_INDEX = 10
ETA_CHARGE_INDEX = 11
ETA_DISCHARGE_INDEX = 12


def apply_checkpoint_config(model_path: Path) -> Dict:
    """Apply the architecture stored next to a checkpoint (train_config.json) to ``cfg``."""
    train_config_path = model_path.parent / "train_config.json"
    if not train_config_path.exists():
        return {}
    with train_config_path.open("r") as f:
        train_config = json.load(f)

    cfg.dt.hidden_size = int(train_config.get("hidden_size", cfg.dt.hidden_size))
    cfg.dt.n_layer = int(train_config.get("n_layer", cfg.dt.n_layer))
    cfg.dt.n_head = int(train_config.get("n_head", cfg.dt.n_head))
    cfg.dt.n_inner = train_config.get("n_inner", cfg.dt.n_inner)
    cfg.dt.activation_function = train_config.get("activation", cfg.dt.activation_function)
    cfg.dt.resid_pdrop = float(train_config.get("resid_pdrop", cfg.dt.resid_pdrop))
    cfg.dt.attn_pdrop = float(train_config.get("attn_pdrop", cfg.dt.attn_pdrop))
    return train_config


def select_indices(n: int, max_episodes: Optional[int], seed: int) -> np.ndarray:
    if max_episodes is None or max_episodes >= n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_episodes, replace=False))


def load_split_metadata(input_dir: Path, n_trajectories: int) -> Tuple[np.ndarray, np.ndarray]:
    npz_path = Path(input_dir) / "trajectories.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if "dates" not in data or "site_ids" not in data:
            raise ValueError(f"{npz_path} must contain dates and site_ids for site-chained rollouts")
        dates = data["dates"].astype(str)
        site_ids = data["site_ids"].astype(np.int32)

    if len(dates) != n_trajectories or len(site_ids) != n_trajectories:
        raise ValueError(
            "Inconsistent metadata: "
            f"dates={len(dates)}, site_ids={len(site_ids)}, trajectories={n_trajectories}"
        )
    return dates, site_ids


def load_actual_site_days(
    csv_path: Path,
    dates: np.ndarray,
    site_ids: np.ndarray,
    indices: np.ndarray,
) -> Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]]:
    """Load the realized PV / load per site-day, used to score every executed action."""
    selected_dates = set(dates[indices].astype(str).tolist())
    selected_sites = set(site_ids[indices].astype(int).tolist())
    usecols = ["datetime", "site_id", "quarter_hour", "pv_kw", "load_kw"]
    df = pd.read_csv(csv_path, usecols=usecols)
    df["date"] = df["datetime"].astype(str).str.slice(0, 10)
    df = df[df["date"].isin(selected_dates) & df["site_id"].isin(selected_sites)].copy()

    actual_by_key: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
    for (site_id, date), group in df.groupby(["site_id", "date"], sort=False):
        day = group.sort_values("quarter_hour")
        if len(day) != STEPS_PER_DAY:
            continue
        if day["quarter_hour"].astype(int).tolist() != list(range(STEPS_PER_DAY)):
            continue
        actual_by_key[(int(site_id), str(date))] = (
            day["pv_kw"].to_numpy(dtype=np.float64),
            day["load_kw"].to_numpy(dtype=np.float64),
        )

    missing = [
        (int(site_ids[int(idx)]), str(dates[int(idx)]))
        for idx in indices
        if (int(site_ids[int(idx)]), str(dates[int(idx)])) not in actual_by_key
    ]
    if missing:
        preview = ", ".join(f"site_id={site} date={date}" for site, date in missing[:5])
        raise ValueError(f"Missing realized PV/load for {len(missing)} site-days in {csv_path}: {preview}")
    return actual_by_key


def select_eval_indices(
    n: int,
    max_episodes: Optional[int],
    seed: int,
    rollout_mode: str,
    dates: Optional[np.ndarray] = None,
    site_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    max_episodes = None if max_episodes is not None and max_episodes <= 0 else max_episodes
    if rollout_mode == "independent":
        return select_indices(n, max_episodes, seed)

    if dates is None or site_ids is None:
        raise ValueError("rollout_mode=site_chained requires dates and site_ids")
    order = np.lexsort((site_ids, dates))
    if max_episodes is not None and max_episodes < len(order):
        order = order[:max_episodes]
    return order.astype(int)


def raw_params(state_raw: np.ndarray) -> Dict[str, float]:
    return {
        "soc_min": float(state_raw[SOC_MIN_INDEX]),
        "soc_max": float(state_raw[SOC_MAX_INDEX]),
        "capacity": float(state_raw[CAPACITY_INDEX]),
        "power": float(state_raw[POWER_INDEX]),
        "eta_c": float(state_raw[ETA_CHARGE_INDEX]),
        "eta_d": float(state_raw[ETA_DISCHARGE_INDEX]),
    }


def pct_gap(cost: float, reference: float) -> float:
    return float((cost - reference) / max(abs(reference), 1e-9) * 100.0)


def aggregate_gap_pct(costs: pd.Series, references: pd.Series) -> float:
    cost_total = float(costs.sum())
    reference_total = float(references.sum())
    return pct_gap(cost_total, reference_total)


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
