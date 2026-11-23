# -*- coding: utf-8 -*-
"""
PI-DT: physics-informed Decision Transformer with the asymmetric terminal constraint.

The end-of-day reserve is a lower bound, soc[T] >= min_terminal_soc (Section 3.4),
instead of the usual equality soc[T] == target. At every step t the
horizon-aware projection (Eq. 14) therefore restricts the SOC reached at the
end of the step, soc[t+1], to

  - lower bound:  max(soc_min, min_terminal_soc - h * delta_up_per_step),
    with h = total_steps - t - 1 the steps that remain after the current one.
    From soc[t+1] the battery can still be charged up to min_terminal_soc
    before the end of the day at rated power.

  - upper bound:  soc_max (no terminal restriction from above).
    The MILP expert and PI-DT may close the day above min_terminal_soc
    whenever that is cheaper, always within the physical SOC window.

The feasible set is a superset of the one under the equality constraint, so
the optimal MILP cost with `>=` is never higher than with `==`.
"""

from __future__ import annotations

from typing import Tuple

import torch

from pidt.config import apply_emsx_config, cfg
from pidt.constants import MIN_TERMINAL_SOC, STEPS_PER_DAY, TIMESTEP_HOURS
from pidt.models.decision_transformer import DecisionTransformer
from pidt.models.physics import (
    CAPACITY_INDEX,
    ETA_CHARGE_INDEX,
    ETA_DISCHARGE_INDEX,
    pick_device,
    POWER_INDEX,
    project_actions_from_raw_states,
    SOC_INDEX,
    SOC_MAX_INDEX,
    SOC_MIN_INDEX,
)


apply_emsx_config(cfg)

DEFAULT_MIN_TERMINAL_SOC = MIN_TERMINAL_SOC


def compute_min_terminal_bounds_from_raw_states(
    states_raw: torch.Tensor,
    timesteps: torch.Tensor,
    min_terminal_soc: float = DEFAULT_MIN_TERMINAL_SOC,
    total_steps: int = STEPS_PER_DAY,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Asymmetric horizon-aware bounds: soc[T] >= min_terminal_soc.

    For every sample (soc_t, capacity, power, eta_c, eta_d) and t = timestep,
    returns [lower, upper] on the normalized action in [-1, 1] that keeps at
    least min_terminal_soc reachable at the end of the day.

    Lower bound on the SOC after the step:
        soc[t+1] >= max(soc_min, min_terminal_soc - h * delta_up_per_step)
    Upper bound: soc_max only (no terminal restriction from above).
    """
    soc = states_raw[..., SOC_INDEX]
    soc_min = states_raw[..., SOC_MIN_INDEX]
    soc_max = states_raw[..., SOC_MAX_INDEX]
    capacity = torch.clamp(states_raw[..., CAPACITY_INDEX], min=eps)
    power = torch.clamp(states_raw[..., POWER_INDEX], min=eps)
    eta_c = torch.clamp(states_raw[..., ETA_CHARGE_INDEX], min=eps)
    eta_d = torch.clamp(states_raw[..., ETA_DISCHARGE_INDEX], min=eps)

    interval_energy = power * float(TIMESTEP_HOURS)

    if not torch.is_tensor(timesteps):
        timesteps = torch.as_tensor(timesteps, device=soc.device)
    timesteps = timesteps.to(device=soc.device)
    steps_after = (int(total_steps) - 1 - timesteps).to(soc.dtype)
    steps_after = torch.clamp(steps_after, min=0)

    delta_up_per_step = eta_c * interval_energy / capacity
    delta_down_per_step = interval_energy / (eta_d * capacity)

    target_min = torch.full_like(soc, float(min_terminal_soc))
    target_min_clamped = torch.minimum(torch.maximum(target_min, soc_min), soc_max)

    # Lower bound: the SOC at the end of this step must still allow reaching
    # min_terminal_soc by the end of the day.
    min_new_soc = torch.maximum(target_min_clamped - steps_after * delta_up_per_step, soc_min)
    # Upper bound: only the physical limit of the battery.
    max_new_soc = soc_max.clone()

    # Numerical safeguard
    bad = min_new_soc > max_new_soc
    target_clamped = target_min_clamped
    min_new_soc = torch.where(bad, target_clamped, min_new_soc)
    max_new_soc = torch.where(bad, target_clamped, max_new_soc)

    min_delta = min_new_soc - soc
    max_delta = max_new_soc - soc

    denom_c = delta_up_per_step
    denom_d = delta_down_per_step

    lower = torch.where(min_delta >= 0, min_delta / denom_c, min_delta / denom_d)
    upper = torch.where(max_delta >= 0, max_delta / denom_c, max_delta / denom_d)

    lower = torch.clamp(lower, min=-1.0, max=1.0)
    upper = torch.clamp(upper, min=-1.0, max=1.0)

    swap = lower > upper
    closest = torch.where(lower.abs() < upper.abs(), lower, upper)
    lower = torch.where(swap, closest, lower)
    upper = torch.where(swap, closest, upper)
    return lower, upper


def project_min_terminal_from_raw_states(
    action: torch.Tensor,
    states_raw: torch.Tensor,
    timesteps: torch.Tensor,
    min_terminal_soc: float = DEFAULT_MIN_TERMINAL_SOC,
    total_steps: int = STEPS_PER_DAY,
) -> torch.Tensor:
    lower, upper = compute_min_terminal_bounds_from_raw_states(
        states_raw, timesteps,
        min_terminal_soc=min_terminal_soc, total_steps=total_steps,
    )
    while lower.ndim < action.ndim:
        lower = lower.unsqueeze(-1)
        upper = upper.unsqueeze(-1)
    return torch.minimum(torch.maximum(action, lower), upper)


class PIDT(DecisionTransformer):
    """PI-DT with the asymmetric horizon-aware projection."""

    def project_action(self, action: torch.Tensor, state_raw: torch.Tensor) -> torch.Tensor:
        # Per-step bound (no horizon), from the core model.
        return project_actions_from_raw_states(action, state_raw)

    def project_action_min_terminal(
        self,
        action: torch.Tensor,
        state_raw: torch.Tensor,
        timesteps: torch.Tensor,
        min_terminal_soc: float = DEFAULT_MIN_TERMINAL_SOC,
        total_steps: int = STEPS_PER_DAY,
    ) -> torch.Tensor:
        return project_min_terminal_from_raw_states(
            action, state_raw, timesteps,
            min_terminal_soc=min_terminal_soc, total_steps=total_steps,
        )


def build_pidt_model(device: torch.device | None = None) -> PIDT:
    if device is None:
        device = pick_device("auto")
    model = PIDT().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PI-DT created: {total_params:,} params ({trainable:,} trainable)")
    print(
        f"  state_dim={model.state_dim}, act_dim={model.act_dim}, "
        f"K={model.max_length}, terminal_constraint=soc[T]>={DEFAULT_MIN_TERMINAL_SOC}"
    )
    return model
