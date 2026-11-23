# -*- coding: utf-8 -*-
"""
Nonlinear battery model (Thevenin two-branch RC) scaled to each site (Section 3.2).

Adapted from a reference ``battery_model(...)`` implementation of the 850 mAh
PL-383562 lithium-ion polymer cell, aggregated into a bank. Differences with
respect to that implementation:

  - The number of cells (``total_nr_cells``) and the C-rate (``crate_h``) are
    scaled to the capacity (E_s, kWh) and rated power (P_s, kW) of each EMSx site.
  - The SOC limits are those of the site (soc_min, soc_max) instead of the
    fixed 0.1 / 0.9 of the reference model.
  - ``delta_t`` is configurable (seconds); the default 900 s is the EMSx step.

Sign convention of the reference model (kept in this module):
  - action > 0  -> DISCHARGE
  - action < 0  -> CHARGE

The controller uses the opposite convention (action > 0 = charge);
``pidt.battery.plant`` flips the sign before calling this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


CELL_CAPACITY_AH = 0.85
CELL_VBATT_MIN = 3.0
CELL_VBATT_MAX = 4.2
CELL_VBATT_NOMINAL = 3.7
ROUNDTRIP_EFFICIENCY = 0.93


@dataclass
class BatteryRunState:
    """State of the nonlinear dynamics carried over between steps."""
    vts: float = 0.0
    vtl: float = 0.0
    step_index: int = 0


def cells_for_capacity(capacity_kwh: float) -> int:
    """Number of cells (bank Ah / cell Ah) giving the bank the nominal capacity
    in kWh, assuming the nominal voltage on every cell.
    """
    if capacity_kwh <= 0:
        return 1
    total_ah_bank = capacity_kwh * 1000.0 / CELL_VBATT_NOMINAL
    return max(1, int(round(total_ah_bank / CELL_CAPACITY_AH)))


def crate_for_power(power_kw: float, capacity_kwh: float) -> float:
    """Bank C-rate in 1/h. P_s / E_s = 0.2 means C/5."""
    if capacity_kwh <= 0:
        return 0.0
    return max(1e-4, power_kw / capacity_kwh)


def nonlinear_step(
    action: float,
    current_soc: float,
    *,
    capacity_kwh: float,
    power_kw: float,
    soc_min: float = 0.20,
    soc_max: float = 1.00,
    vts: float = 0.0,
    vtl: float = 0.0,
    time_step: int = 0,
    delta_t: int = 900,
) -> dict:
    """
    One step of the Thevenin 2RC model scaled to the site. Returns a dict with:
        - energy_kwh: energy at the bus terminals (positive when discharging,
                      negative when charging).
        - soc_next:   SOC after applying the action.
        - vts_next, vtl_next: updated transient voltages.
        - vbatt: terminal voltage of the bank at the end of the step.
        - i_per_cell: current per cell (A).
        - flag_clipped: 1 if the action was truncated by any limit, else 0.

    Convention: action > 0 -> discharge, action < 0 -> charge (that of the
    reference model, not that of PI-DT).
    """
    # ------------------------------------------------------------------
    # Scaling to the site
    # ------------------------------------------------------------------
    total_nr_cells = cells_for_capacity(capacity_kwh)
    crate_h = crate_for_power(power_kw, capacity_kwh)

    cell_capacity = CELL_CAPACITY_AH
    discharge_rate = crate_h
    charge_rate = crate_h

    Cc = 3600.0 * cell_capacity
    Rsd = math.inf
    I_charge_max = -cell_capacity * charge_rate
    I_discharge_max = cell_capacity * discharge_rate
    Vbatt_min = CELL_VBATT_MIN
    Vbatt_max = CELL_VBATT_MAX

    # ------------------------------------------------------------------
    # Initialization (as in the reference model)
    # ------------------------------------------------------------------
    a_clip = max(-1.0, min(1.0, float(action)))
    if a_clip >= 0:
        power = a_clip * I_discharge_max / 3600.0 * Vbatt_max / ROUNDTRIP_EFFICIENCY
    else:
        power = abs(a_clip) * I_charge_max / 3600.0 * Vbatt_max * ROUNDTRIP_EFFICIENCY

    soc_clip = max(0.0, min(1.0, float(current_soc)))
    V_SOC = soc_clip
    if V_SOC < soc_min:
        V_SOC = soc_min
    elif V_SOC > soc_max:
        V_SOC = soc_max

    if time_step == 0:
        vts_ = 0.0
        vtl_ = 0.0
    else:
        vts_ = float(vts)
        vtl_ = float(vtl)

    V_OC = -1.031 * math.exp(-35 * V_SOC) + 3.685 + 0.2156 * V_SOC \
           - 0.1178 * V_SOC**2 + 0.3201 * V_SOC**3
    Rs = 0.1562 * math.exp(-24.37 * V_SOC) + 0.07446
    Rts = 0.3208 * math.exp(-29.14 * V_SOC) + 0.04669
    Cts = -752.9 * math.exp(-13.51 * V_SOC) + 703.6
    Rtl = 6.603 * math.exp(-155.2 * V_SOC) + 0.04984
    Ctl = -6056 * math.exp(-27.12 * V_SOC) + 4475
    # Avoid null denominators in the transient terms
    Cts = max(Cts, 1e-3)
    Ctl = max(Ctl, 1e-3)
    Rts = max(Rts, 1e-6)
    Rtl = max(Rtl, 1e-6)

    # ------------------------------------------------------------------
    # Solver de Vbatt
    # ------------------------------------------------------------------
    a_q = 1.0
    b_q = -(V_OC - vts_ - vtl_)
    c_q = power * Rs
    disc = b_q * b_q - 4 * a_q * c_q
    if disc < 0:
        Vbatt = max(min(V_OC - vts_ - vtl_, Vbatt_max), Vbatt_min)
    else:
        V_pos = (-b_q + math.sqrt(disc)) / (2 * a_q)
        V_neg = (-b_q - math.sqrt(disc)) / (2 * a_q)
        if Vbatt_min <= V_pos <= Vbatt_max:
            Vbatt = V_pos
        elif Vbatt_min <= V_neg <= Vbatt_max:
            Vbatt = V_neg
        else:
            Vbatt = max(abs(V_pos), abs(V_neg))

    if Vbatt < Vbatt_min:
        Vbatt = Vbatt_min
    elif Vbatt > Vbatt_max:
        Vbatt = Vbatt_max

    I = power / Vbatt if Vbatt != 0 else 0.0

    flag_clipped = 0
    if a_clip > 0 and I > I_discharge_max / 3600.0:
        I = I_discharge_max / 3600.0
        Vbatt = V_OC - I * Rs - vts_ - vtl_
        flag_clipped = 1
    elif a_clip < 0 and I < I_charge_max / 3600.0:
        I = I_charge_max / 3600.0
        Vbatt = V_OC - I * Rs - vts_ - vtl_
        flag_clipped = 1
    if Vbatt < Vbatt_min:
        Vbatt = Vbatt_min
    elif Vbatt > Vbatt_max:
        Vbatt = Vbatt_max

    # ------------------------------------------------------------------
    # SOC update
    # ------------------------------------------------------------------
    V_SOC_next = (-I * 3600.0 / Cc - V_SOC / (Rsd * Cc)) * delta_t + V_SOC
    if V_SOC_next < soc_min:
        flag_clipped = 1
        V_SOC_next = soc_min
        I = -((V_SOC_next - V_SOC) / delta_t + V_SOC / (Rsd * Cc)) * Cc / 3600.0
        if a_clip > 0 and I > I_discharge_max / 3600.0:
            I = I_discharge_max / 3600.0
        elif a_clip < 0 and I < I_charge_max / 3600.0:
            I = I_charge_max / 3600.0
        V_SOC_next = (-I * 3600.0 / Cc - V_SOC / (Rsd * Cc)) * delta_t + V_SOC
        Vbatt = V_OC - I * Rs - vts_ - vtl_
        if Vbatt < Vbatt_min:
            Vbatt = Vbatt_min
        elif Vbatt > Vbatt_max:
            Vbatt = Vbatt_max
    elif V_SOC_next > soc_max:
        flag_clipped = 1
        V_SOC_next = soc_max
        I = -((V_SOC_next - V_SOC) / delta_t + V_SOC / (Rsd * Cc)) * Cc / 3600.0
        if a_clip > 0 and I > I_discharge_max / 3600.0:
            I = I_discharge_max / 3600.0
        elif a_clip < 0 and I < I_charge_max / 3600.0:
            I = I_charge_max / 3600.0
        V_SOC_next = (-I * 3600.0 / Cc - V_SOC / (Rsd * Cc)) * delta_t + V_SOC
        Vbatt = V_OC - I * Rs - vts_ - vtl_
        if Vbatt < Vbatt_min:
            Vbatt = Vbatt_min
        elif Vbatt > Vbatt_max:
            Vbatt = Vbatt_max

    # ------------------------------------------------------------------
    # Energy at the bus terminals (including the round-trip efficiency).
    # power > 0  -> discharge (energy to the bus)
    # power < 0  -> charge    (energy from the bus, negative sign)
    # ------------------------------------------------------------------
    bus_power = Vbatt * I
    if a_clip > 0:
        grid_power = bus_power * ROUNDTRIP_EFFICIENCY
    else:
        # Charging: the grid delivers more energy than reaches the cell.
        # bus_power < 0; grid_power is also < 0 but larger in magnitude.
        grid_power = bus_power / ROUNDTRIP_EFFICIENCY if ROUNDTRIP_EFFICIENCY > 0 else 0.0
    energy_kwh = grid_power * delta_t * total_nr_cells / 1000.0

    # Transients for the next step (as in the reference model).
    vts_next = (I / Cts - vts_ / (Rts * Cts)) + vts_
    vtl_next = (I / Ctl - vtl_ / (Rtl * Ctl)) + vtl_

    return {
        "energy_kwh": float(energy_kwh),
        "soc_next":   float(V_SOC_next),
        "vts_next":   float(vts_next),
        "vtl_next":   float(vtl_next),
        "vbatt":      float(Vbatt),
        "i_per_cell": float(I),
        "flag_clipped": int(flag_clipped),
    }
