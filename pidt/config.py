"""Global configuration object (``cfg``) used to build and train the models.

The defaults below already correspond to the EMSx + PVPC setting of the paper
(298-dimensional state, 96-step context, SOC window [0.20, 1.00]). Scripts
import the shared instance::

    from pidt.config import cfg, apply_emsx_config
    apply_emsx_config(cfg)   # idempotent; kept so every entry point states it

Model hyper-parameters (hidden size, layers, heads, dropout, ...) are set by
the training scripts from their command-line arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pidt import constants as C, paths as P


# ============================================================
# 1. Physical parameters (fallback values)
# ============================================================
# Per-site capacity, power and efficiencies are read from the state vector
# (EMSx metadata); the values below are only used where no site is involved.

@dataclass
class BatteryConfig:
    """Battery energy storage system (BESS)."""
    capacity_kwh: float = 250.0         # Nominal capacity (kWh)
    max_charge_kw: float = 36.0         # Rated charge power (kW)
    max_discharge_kw: float = 36.0      # Rated discharge power (kW)
    eta_charge: float = 0.95            # Charge efficiency
    eta_discharge: float = 0.95         # Discharge efficiency
    soc_min: float = C.SOC_MIN          # Lower operating SOC (fraction)
    soc_max: float = C.SOC_MAX          # Upper operating SOC (fraction)
    degradation_cost_eur_kwh: float = C.CYCLE_COST_EUR_PER_KWH  # EUR per kWh cycled


@dataclass
class PVConfig:
    """Photovoltaic array."""
    peak_kw: float = 75.0               # Peak power (kWp)


@dataclass
class GridConfig:
    """Grid connection."""
    max_import_kw: float = 70.0         # Maximum import power (kW)
    max_export_kw: float = 70.0         # Maximum export power (kW)
    feed_in_tariff_eur_kwh: float = 0.0  # Remuneration of exported energy (EUR/kWh)


# ============================================================
# 2. State and action spaces
# ============================================================

@dataclass
class StateConfig:
    """Layout of the state vector s_t (see ``pidt.constants``)."""
    base_features: List[str] = field(default_factory=lambda: list(C.BASE_FEATURES))
    # Number of future steps exposed to the model (rest of the operating day).
    lookahead_hours: int = C.LOOKAHEAD_STEPS
    # Features repeated for every look-ahead step.
    lookahead_features: List[str] = field(default_factory=lambda: list(C.LOOKAHEAD_FEATURES))

    @property
    def state_dim(self) -> int:
        """Total dimension of the state vector."""
        return len(self.base_features) + self.lookahead_hours * len(self.lookahead_features)


@dataclass
class ActionConfig:
    """Action space."""
    dim: int = 1
    # a_t in [-1, +1]: +1 = full charge, 0 = idle, -1 = full discharge (rated power).


# ============================================================
# 3. Decision Transformer hyper-parameters
# ============================================================

@dataclass
class DTConfig:
    """Decision Transformer architecture (overridden by the training CLI)."""
    hidden_size: int = 192
    n_layer: int = 4
    n_head: int = 4
    n_inner: Optional[int] = None       # FFN inner size (default: 4 * hidden_size)
    activation_function: str = "relu"
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    context_length: int = C.STEPS_PER_DAY  # K: one full operating day
    max_ep_len: int = C.STEPS_PER_DAY
    action_tanh: bool = True            # tanh on the action head


@dataclass
class TrainConfig:
    """Generic optimizer settings (the training CLI exposes all of them)."""
    num_steps: int = 30_000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 2_000
    grad_clip: float = 0.25
    log_interval: int = 250
    eval_interval: int = 1_000
    seed: int = 42


# ============================================================
# 4. Default data locations
# ============================================================

@dataclass
class PathConfig:
    historical_data_csv: str = str(P.REALIZED_PV_LOAD_CSV)
    trajectories_dir: str = str(P.TRAIN_SPLIT_DIR)
    model_save_dir: str = str(P.PIDT_CHECKPOINT_DIR)
    results_dir: str = str(P.RESULTS_DIR)


# ============================================================
# 5. Global configuration
# ============================================================

@dataclass
class Config:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    pv: PVConfig = field(default_factory=PVConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    state: StateConfig = field(default_factory=StateConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    dt: DTConfig = field(default_factory=DTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PI-DT CONFIGURATION",
            "=" * 60,
            f"  State: {self.state.state_dim} dims "
            f"(base={len(self.state.base_features)}, "
            f"lookahead={self.state.lookahead_hours} steps x "
            f"{len(self.state.lookahead_features)})",
            f"  Action: dim={self.action.dim}, range [-1, +1]",
            f"  SOC window: [{self.battery.soc_min}, {self.battery.soc_max}], "
            f"cycle cost {self.battery.degradation_cost_eur_kwh} EUR/kWh",
            f"  DT: hidden={self.dt.hidden_size}, layers={self.dt.n_layer}, "
            f"heads={self.dt.n_head}, K={self.dt.context_length}",
            "=" * 60,
        ]
        return "\n".join(lines)


def apply_emsx_config(config: Config) -> Config:
    """Set ``config`` to the EMSx + PVPC (no sale) setting of the paper, in place."""
    config.state.base_features = list(C.BASE_FEATURES)
    config.state.lookahead_hours = C.LOOKAHEAD_STEPS
    config.state.lookahead_features = list(C.LOOKAHEAD_FEATURES)
    config.dt.context_length = C.STEPS_PER_DAY
    config.dt.max_ep_len = C.STEPS_PER_DAY
    config.battery.soc_min = C.SOC_MIN
    config.battery.soc_max = C.SOC_MAX
    config.battery.degradation_cost_eur_kwh = C.CYCLE_COST_EUR_PER_KWH
    config.grid.feed_in_tariff_eur_kwh = 0.0
    config.paths.historical_data_csv = str(P.REALIZED_PV_LOAD_CSV)
    config.paths.trajectories_dir = str(P.TRAIN_SPLIT_DIR)
    config.paths.model_save_dir = str(P.PIDT_CHECKPOINT_DIR)
    return config


# Shared instance imported by the rest of the package.
cfg = apply_emsx_config(Config())

if __name__ == "__main__":
    print(cfg.summary())
