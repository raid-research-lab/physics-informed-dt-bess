# -*- coding: utf-8 -*-
"""
Per-site physics of PI-DT: action bounds, physics penalty and projection.

The physics is not taken from a global ``cfg.battery`` but from the raw state
features of every sample::

  [ ..., soc, soc_min, soc_max, ..., capacity_kwh, battery_power_kw,
    eta_charge, eta_discharge ]

so a single network can be trained on many industrial sites with different
capacities, rated powers and efficiencies. The asymmetric horizon-aware bounds
of the end-of-day reserve are in ``pidt.models.pidt``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from pidt.config import apply_emsx_config, cfg
from pidt.constants import TIMESTEP_HOURS

apply_emsx_config(cfg)

SOC_INDEX = 4
SOC_MIN_INDEX = 5
SOC_MAX_INDEX = 6
CAPACITY_INDEX = 9
POWER_INDEX = 10
ETA_CHARGE_INDEX = 11
ETA_DISCHARGE_INDEX = 12


def pick_device(preferred: str = "auto") -> torch.device:
    preferred = (preferred or "auto").lower()
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    if preferred in {"cuda", "mps", "cpu"}:
        return torch.device(preferred)
    raise ValueError("device must be auto, cuda, mps or cpu")


def compute_action_bounds_from_raw_states(
    states_raw: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-step bounds [lower, upper] on the normalized action of every sample.

    For a positive (charging) action:
      dSOC = eta_charge * action * power_kw * 0.25 / capacity_kwh

    For a negative (discharging) action:
      dSOC = action * power_kw * 0.25 / (eta_discharge * capacity_kwh)

    The bounds keep the next SOC inside [soc_min, soc_max].
    """
    soc = states_raw[..., SOC_INDEX]
    soc_min = states_raw[..., SOC_MIN_INDEX]
    soc_max = states_raw[..., SOC_MAX_INDEX]
    capacity = torch.clamp(states_raw[..., CAPACITY_INDEX], min=eps)
    power = torch.clamp(states_raw[..., POWER_INDEX], min=eps)
    eta_c = torch.clamp(states_raw[..., ETA_CHARGE_INDEX], min=eps)
    eta_d = torch.clamp(states_raw[..., ETA_DISCHARGE_INDEX], min=eps)

    interval_energy = power * float(TIMESTEP_HOURS)
    upper = (soc_max - soc) * capacity / (eta_c * interval_energy)
    lower = -(soc - soc_min) * eta_d * capacity / interval_energy

    upper = torch.clamp(upper, min=0.0, max=1.0)
    lower = torch.clamp(lower, min=-1.0, max=0.0)

    bad = lower > upper
    zero = torch.zeros_like(lower)
    lower = torch.where(bad, zero, lower)
    upper = torch.where(bad, zero, upper)
    return lower, upper


def physics_penalty_from_bounds(
    action_pred: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    action = action_pred.squeeze(-1) if action_pred.shape[-1] == 1 else action_pred
    over = torch.relu(action - upper)
    under = torch.relu(lower - action)
    penalty = over.pow(2) + under.pow(2)
    if mask is not None:
        penalty = penalty * mask
        denom = mask.sum().clamp_min(1.0)
        return penalty.sum() / denom
    return penalty.mean()


def project_actions_from_raw_states(
    action: torch.Tensor,
    states_raw: torch.Tensor,
) -> torch.Tensor:
    lower, upper = compute_action_bounds_from_raw_states(states_raw)
    while lower.ndim < action.ndim:
        lower = lower.unsqueeze(-1)
        upper = upper.unsqueeze(-1)
    return torch.minimum(torch.maximum(action, lower), upper)


