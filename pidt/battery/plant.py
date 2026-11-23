# -*- coding: utf-8 -*-
"""
Nonlinear plant step used by every rollout (Section 3.2).

``apply_action_step_nonlinear(...)`` has the same signature and sign
convention as a linear SOC update (action in [-1, 1], > 0 = charge),
so a rollout only has to swap the environment step.

Convention of the controllers:
  - action > 0  -> CHARGE
  - action < 0  -> DISCHARGE

The sign is flipped internally before calling ``pidt.battery.thevenin``,
which uses the opposite convention (action > 0 = discharge).

The transient voltages vts/vtl are carried between calls by the caller (one
state per site); the function keeps no global state.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pidt.battery.thevenin import nonlinear_step
from pidt.constants import CYCLE_COST_EUR_PER_KWH, STEPS_PER_DAY, TIMESTEP_HOURS
from pidt.evaluation.common import raw_params


TIMESTEP_SECONDS = int(round(TIMESTEP_HOURS * 3600))  # 900


def apply_action_step_nonlinear(
    state_raw: np.ndarray,
    soc: float,
    action: float,
    *,
    allow_grid_charge: bool = True,
    project: bool = True,
    pv_kwh: Optional[float] = None,
    load_kwh: Optional[float] = None,
    vts: float = 0.0,
    vtl: float = 0.0,
    time_step: int = 0,
    total_steps: int = STEPS_PER_DAY,
) -> Dict[str, float]:
    """One step under the nonlinear battery. Same signature as the linear
    step, plus ``vts_next`` / ``vtl_next`` to chain the dynamics.
    """
    params = raw_params(state_raw)
    capacity_kwh = float(params["capacity"])
    power_kw = float(params["power"])
    soc_min = float(params["soc_min"])
    soc_max = float(params["soc_max"])

    raw_action = float(np.clip(action, -1.0, 1.0))

    # The nonlinear model uses action_nl > 0 = discharge: flip the sign.
    action_nl = -raw_action
    nl = nonlinear_step(
        action=action_nl,
        current_soc=float(soc),
        capacity_kwh=capacity_kwh,
        power_kw=power_kw,
        soc_min=soc_min,
        soc_max=soc_max,
        vts=float(vts),
        vtl=float(vtl),
        time_step=int(time_step),
        delta_t=TIMESTEP_SECONDS,
    )

    # Bus energy in the convention of the nonlinear model:
    #   energy_kwh > 0  -> discharge (the bus receives energy)
    #   energy_kwh < 0  -> charge    (the bus delivers energy)
    # Map it back to the controller convention (raw_action > 0 = charge):
    energy_bus = float(nl["energy_kwh"])
    if raw_action >= 0:
        # charging -> energy_bus should be <= 0
        charge_kwh = max(0.0, -energy_bus)
        discharge_kwh = 0.0
    else:
        # discharging -> energy_bus should be >= 0
        charge_kwh = 0.0
        discharge_kwh = max(0.0, energy_bus)

    pv = float(state_raw[0] if pv_kwh is None else pv_kwh)
    load = float(state_raw[1] if load_kwh is None else load_kwh)
    price_buy = float(state_raw[2])
    price_sell = float(state_raw[3])

    # Without grid charging, limit the charge to the PV surplus.
    if not allow_grid_charge and charge_kwh > 0.0:
        max_charge_from_pv = max(0.0, pv - load)
        if charge_kwh > max_charge_from_pv:
            charge_kwh = max_charge_from_pv

    net_kwh = pv + discharge_kwh - load - charge_kwh
    grid_buy_kwh = max(0.0, -net_kwh)
    grid_sell_kwh = max(0.0, net_kwh)
    cycle_kwh = 0.5 * (charge_kwh + discharge_kwh)
    cost = (price_buy * grid_buy_kwh
            - price_sell * grid_sell_kwh
            + CYCLE_COST_EUR_PER_KWH * cycle_kwh)

    soc_next = float(np.clip(nl["soc_next"], soc_min, soc_max))

    # Applied action as a fraction of rated power, comparable with the
    # linear plant.
    power_interval = power_kw * TIMESTEP_HOURS
    applied_action = (charge_kwh - discharge_kwh) / power_interval if power_interval > 0 else 0.0
    applied_action = float(np.clip(applied_action, -1.0, 1.0))

    return {
        "raw_action": float(raw_action),
        "action": float(applied_action) if project else float(raw_action),
        "violation": 0.0,
        "cost": float(cost),
        "reward": -float(cost),
        "soc_next": soc_next,
        "charge_kwh": float(charge_kwh),
        "discharge_kwh": float(discharge_kwh),
        "grid_buy_kwh": float(grid_buy_kwh),
        "grid_sell_kwh": float(grid_sell_kwh),
        "vts_next": float(nl["vts_next"]),
        "vtl_next": float(nl["vtl_next"]),
        "vbatt": float(nl["vbatt"]),
        "i_per_cell": float(nl["i_per_cell"]),
        "nl_clipped": int(nl["flag_clipped"]),
    }
