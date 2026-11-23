# -*- coding: utf-8 -*-
"""
Site-chained rollout of PI-DT on the nonlinear battery plant (Section 5.6).

Internal state carried between the steps of a rollout:
  - SOC (real, evolving with the nonlinear model)
  - vts, vtl (transient voltages of the Thevenin 2RC model)
  - absolute time_step of the site, so that vts/vtl are not reset to 0 at
    every call. The first step of a site is called with time_step=0 to
    initialize the model correctly.

The controller still receives the raw state of the paper (SOC at index 4);
only the evolution of the SOC after applying an action changes.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from pidt.battery.plant import apply_action_step_nonlinear
from pidt.constants import STEPS_PER_DAY
from pidt.evaluation.common import SOC_INDEX


class NLState:
    """State of the nonlinear dynamics along a rollout."""
    __slots__ = ("vts", "vtl", "time_step")

    def __init__(self, vts: float = 0.0, vtl: float = 0.0, time_step: int = 0):
        self.vts = float(vts)
        self.vtl = float(vtl)
        self.time_step = int(time_step)


def _apply(state_raw, soc, action, *, allow_grid_charge, project, pv_kwh, load_kwh,
           nl: NLState):
    step = apply_action_step_nonlinear(
        state_raw=state_raw, soc=soc, action=action,
        allow_grid_charge=allow_grid_charge, project=project,
        pv_kwh=pv_kwh, load_kwh=load_kwh,
        vts=nl.vts, vtl=nl.vtl, time_step=nl.time_step,
    )
    nl.vts = step["vts_next"]
    nl.vtl = step["vtl_next"]
    nl.time_step += 1
    return step


# ---------------------------------------------------------------------------
# PI-DT rollout (asymmetric horizon-aware projection)
# ---------------------------------------------------------------------------

def rollout_dt_nl(
    model, trajectory: Dict[str, np.ndarray], norm: Dict, device: torch.device,
    target_mode: str, allow_grid_charge: bool, project_actions: bool,
    initial_soc: float, actual_pv: np.ndarray, actual_load: np.ndarray,
    min_terminal_soc: float, nl: NLState,
) -> Dict[str, np.ndarray]:
    from pidt.models.pidt import compute_min_terminal_bounds_from_raw_states

    state_mean = norm["state_mean"]
    state_std = norm["state_std"]
    rtg_mean = float(norm["rtg_mean"])
    rtg_std = float(norm["rtg_std"])

    obs = trajectory["observations"]
    state_dim = obs.shape[1]
    act_dim = trajectory["actions"].shape[1]
    soc = float(initial_soc)
    ep_return = float(trajectory["returns_to_go"][0])

    states_t = torch.zeros((1, 0, state_dim), device=device, dtype=torch.float32)
    actions_t = torch.zeros((1, 0, act_dim), device=device, dtype=torch.float32)
    rtg_t = None

    actions: List[float] = []
    raw_actions: List[float] = []
    costs: List[float] = []
    soc_values = [soc]
    violations: List[float] = []

    for t in range(STEPS_PER_DAY):
        state_raw = obs[t].copy()
        state_raw[SOC_INDEX] = soc
        state_norm = ((state_raw - state_mean) / state_std).astype(np.float32)
        cur_state = torch.from_numpy(state_norm).reshape(1, 1, state_dim).to(device)
        states_t = torch.cat([states_t, cur_state], dim=1)
        actions_t = torch.cat([actions_t, torch.zeros((1, 1, act_dim), device=device)], dim=1)

        if target_mode == "expert_rtg":
            desired_rtg = float(trajectory["returns_to_go"][t])
        else:
            desired_rtg = ep_return
        desired_rtg_norm = (desired_rtg - rtg_mean) / rtg_std
        new_rtg = torch.tensor([[[desired_rtg_norm]]], device=device, dtype=torch.float32)
        rtg_t = new_rtg if rtg_t is None else torch.cat([rtg_t, new_rtg], dim=1)
        timesteps = torch.arange(t + 1, device=device, dtype=torch.long).reshape(1, -1)

        with torch.no_grad():
            pred_action = model.get_action(states_t, actions_t, None, rtg_t, timesteps)
        raw = float(pred_action.detach().cpu().numpy().reshape(-1)[0])

        state_raw_t = torch.from_numpy(state_raw.astype(np.float32)).to(device).reshape(1, 1, -1)
        ts_t = torch.tensor([[t]], device=device, dtype=torch.long)
        lower_t, upper_t = compute_min_terminal_bounds_from_raw_states(
            state_raw_t, ts_t,
            min_terminal_soc=min_terminal_soc, total_steps=STEPS_PER_DAY,
        )
        lower = float(lower_t.reshape(-1)[0].item())
        upper = float(upper_t.reshape(-1)[0].item())
        violation = max(0.0, raw - upper, lower - raw)
        applied = raw if not project_actions else float(np.clip(raw, lower, upper))

        step = _apply(
            state_raw, soc, applied,
            allow_grid_charge=allow_grid_charge, project=False,
            pv_kwh=float(actual_pv[t]), load_kwh=float(actual_load[t]),
            nl=nl,
        )
        raw_actions.append(raw)
        actions.append(step["action"])
        costs.append(step["cost"])
        violations.append(violation)
        soc = float(step["soc_next"])
        soc_values.append(soc)
        actions_t[0, -1] = torch.tensor([step["action"]], device=device, dtype=torch.float32)
        if target_mode != "expert_rtg":
            ep_return -= step["reward"]

    return {
        "actions": np.asarray(actions, dtype=np.float32),
        "raw_actions": np.asarray(raw_actions, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "soc": np.asarray(soc_values, dtype=np.float32),
        "violations": np.asarray(violations, dtype=np.float32),
        "total_cost": float(np.sum(costs)),
    }
