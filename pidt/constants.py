"""Physical constants, tariff conventions and state layout shared by all stages.

Values match Section 5 of the paper. Trajectories and checkpoints must be built with the same values.
"""

from __future__ import annotations


# Battery operating window and degradation cost (identical for all sites).
SOC_MIN = 0.20
SOC_MAX = 1.00
CYCLE_COST_EUR_PER_KWH = 0.05            # linear degradation cost per kWh cycled

# Asymmetric end-of-day reserve: soc[T] >= MIN_TERMINAL_SOC (Section 3.4).
MIN_TERMINAL_SOC = 0.55

# Time discretization: 96 quarter-hour steps per operating day.
STEPS_PER_DAY = 96
TIMESTEP_HOURS = 0.25
LOOKAHEAD_STEPS = STEPS_PER_DAY - 1

# State vector s_t (Fig. 4): 13 instantaneous / per-site features followed by a
# 95-step look-ahead of 3 features, i.e. 13 + 95 * 3 = 298 dimensions.
BASE_FEATURES = [
    "pv_forecast",
    "load_forecast",
    "price_buy",
    "price_sell",
    "soc",
    "soc_min",
    "soc_max",
    "hour_sin",
    "hour_cos",
    "battery_capacity_kwh",
    "battery_power_kw",
    "eta_charge",
    "eta_discharge",
]
LOOKAHEAD_FEATURES = ["price_buy", "pv_forecast", "load_forecast"]
STATE_DIM = len(BASE_FEATURES) + LOOKAHEAD_STEPS * len(LOOKAHEAD_FEATURES)
