# -*- coding: utf-8 -*-
"""
Core training loop of the physics-informed Decision Transformer (Section 4).

The loss is

  MSE(action_pred, expert_action)
  + lambda_phys * violation of the per-site physical action bounds

The bounds are computed for every sample from the de-normalized raw state
(SOC, SOC limits, capacity, rated power and efficiencies), so a single network
learns from sites with heterogeneous batteries. ``pidt.training.train_pidt``
is the command-line entry point.

Trajectories are read through a memory-mapped cache (``mmap_cache/`` next to
each ``trajectories.npz``), created on first use, so the 4 GB training split
does not need to fit in RAM.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pidt.config import apply_emsx_config, cfg
from pidt.constants import MIN_TERMINAL_SOC, STEPS_PER_DAY
from pidt.utils import load_initial_checkpoint, prepare_output_dir  # noqa: F401  (re-exported)

apply_emsx_config(cfg)

from pidt.models.pidt import compute_min_terminal_bounds_from_raw_states  # noqa: E402
from pidt.models.physics import (  # noqa: E402
    compute_action_bounds_from_raw_states,
    physics_penalty_from_bounds,
    pick_device,  # noqa: F401  (re-exported)
)


def compute_physics_bounds(
    states_raw: torch.Tensor,
    timesteps: torch.Tensor,
    horizon_aware: bool,
    target_soc: float,
    total_steps: int,
):
    """Bounds used by the physics penalty.

    With ``horizon_aware`` the asymmetric horizon-aware bands of PI-DT
    (soc[T] >= target_soc, Eq. 14); otherwise only the per-step SOC window.
    """
    if horizon_aware:
        return compute_min_terminal_bounds_from_raw_states(
            states_raw, timesteps, min_terminal_soc=target_soc, total_steps=total_steps
        )
    return compute_action_bounds_from_raw_states(states_raw)


def ensure_mmap_cache(input_dir: Path) -> Path:
    """Extract the compressed .npz to .npy files that can be memory-mapped."""
    input_dir = Path(input_dir)
    npz_path = input_dir / "trajectories.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")

    cache_dir = input_dir / "mmap_cache"
    meta_path = cache_dir / "mmap_meta.json"
    source_meta = {
        "source": str(npz_path),
        "size": npz_path.stat().st_size,
        "mtime_ns": npz_path.stat().st_mtime_ns,
    }

    required = ["observations", "actions", "rewards", "returns_to_go", "dates"]
    if meta_path.exists():
        try:
            with meta_path.open("r") as f:
                meta = json.load(f)
            files_ok = all((cache_dir / f"{key}.npy").exists() for key in meta["keys"])
            source_ok = all(meta.get(key) == value for key, value in source_meta.items())
            if source_ok and files_ok:
                return cache_dir
        except (OSError, KeyError, json.JSONDecodeError):
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building mmap cache from {npz_path}", flush=True)
    with zipfile.ZipFile(npz_path, "r") as zf:
        members = set(zf.namelist())
        keys = [key for key in required if f"{key}.npy" in members]
        if "action_inputs.npy" in members:
            keys.append("action_inputs")

        missing = [key for key in required if key not in keys]
        if missing:
            raise ValueError(f"Missing arrays in {npz_path}: {missing}")

        for key in keys:
            member = f"{key}.npy"
            out_path = cache_dir / member
            tmp_path = cache_dir / f"{member}.tmp"
            print(f"  extracting {member}", flush=True)
            with zf.open(member, "r") as src, tmp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            tmp_path.replace(out_path)

    with meta_path.open("w") as f:
        json.dump({**source_meta, "keys": keys}, f, indent=2)
    return cache_dir


class MMapTrajectories:
    """Trajectory dataset with a list-like API, without loading everything in RAM."""

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        self.cache_dir = ensure_mmap_cache(self.input_dir)
        self.observations = np.load(self.cache_dir / "observations.npy", mmap_mode="r")
        self.actions = np.load(self.cache_dir / "actions.npy", mmap_mode="r")
        self.rewards = np.load(self.cache_dir / "rewards.npy", mmap_mode="r")
        self.returns_to_go = np.load(self.cache_dir / "returns_to_go.npy", mmap_mode="r")
        self.dates = np.load(self.cache_dir / "dates.npy", allow_pickle=True)
        action_inputs_path = self.cache_dir / "action_inputs.npy"
        self.action_inputs = (
            np.load(action_inputs_path, mmap_mode="r")
            if action_inputs_path.exists()
            else None
        )

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        traj = {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "returns_to_go": self.returns_to_go[idx],
            "date": str(self.dates[idx]),
        }
        if self.action_inputs is not None:
            traj["action_inputs"] = self.action_inputs[idx]
        return traj


def load_trajectories(input_dir: Path) -> Tuple[MMapTrajectories, Dict]:
    trajectories = MMapTrajectories(Path(input_dir))
    stats_path = Path(input_dir) / "stats.json"  # optional free-form metadata
    stats = {}
    if stats_path.exists():
        with stats_path.open("r") as f:
            stats = json.load(f)
    print(
        f"Loaded {len(trajectories)} memory-mapped trajectories from "
        f"{trajectories.cache_dir}",
        flush=True,
    )
    return trajectories, stats


def get_batch(
    trajectories,
    batch_size: int,
    norm: Dict,
    max_len: Optional[int],
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    if max_len is None:
        max_len = cfg.dt.context_length

    state_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    state_mean = norm["state_mean"]
    state_std = norm["state_std"]
    rtg_mean = norm["rtg_mean"]
    rtg_std = norm["rtg_std"]

    batch_inds = np.random.choice(len(trajectories), size=batch_size, replace=True)
    s, a, r, rtg, timesteps, mask = [], [], [], [], [], []

    for idx in batch_inds:
        traj = trajectories[idx]
        traj_len = len(traj["rewards"])
        si = np.random.randint(0, traj_len)

        s_slice = traj["observations"][si: si + max_len].copy()
        a_slice = traj["actions"][si: si + max_len]
        r_slice = traj["rewards"][si: si + max_len]
        rtg_slice = traj["returns_to_go"][si: si + max_len].copy()
        actual_len = len(r_slice)

        s_slice = (s_slice - state_mean) / state_std
        rtg_slice = (rtg_slice - rtg_mean) / rtg_std

        s_slice = s_slice.reshape(1, actual_len, state_dim)
        a_slice = a_slice.reshape(1, actual_len, act_dim)
        r_slice = r_slice.reshape(1, actual_len, 1)
        rtg_slice = rtg_slice.reshape(1, actual_len, 1)
        ts = np.arange(si, si + actual_len).reshape(1, actual_len)
        ts = np.clip(ts, 0, cfg.dt.max_ep_len - 1)

        pad_len = max_len - actual_len
        if pad_len > 0:
            s_slice = np.concatenate([np.zeros((1, pad_len, state_dim)), s_slice], axis=1)
            a_slice = np.concatenate([np.ones((1, pad_len, act_dim)) * -10.0, a_slice], axis=1)
            r_slice = np.concatenate([np.zeros((1, pad_len, 1)), r_slice], axis=1)
            rtg_slice = np.concatenate([np.zeros((1, pad_len, 1)), rtg_slice], axis=1)
            ts = np.concatenate([np.zeros((1, pad_len)), ts], axis=1)

        m = (
            np.concatenate([np.zeros((1, pad_len)), np.ones((1, actual_len))], axis=1)
            if pad_len > 0
            else np.ones((1, actual_len))
        )
        s.append(s_slice)
        a.append(a_slice)
        r.append(r_slice)
        rtg.append(rtg_slice)
        timesteps.append(ts)
        mask.append(m)

    return (
        torch.from_numpy(np.concatenate(s, axis=0)).float().to(device),
        torch.from_numpy(np.concatenate(a, axis=0)).float().to(device),
        torch.from_numpy(np.concatenate(r, axis=0)).float().to(device),
        torch.from_numpy(np.concatenate(rtg, axis=0)).float().to(device),
        torch.from_numpy(np.concatenate(timesteps, axis=0)).long().to(device),
        torch.from_numpy(np.concatenate(mask, axis=0)).float().to(device),
    )


def denormalize_states(states: torch.Tensor, norm: Dict, device: torch.device) -> torch.Tensor:
    state_mean = torch.from_numpy(norm["state_mean"]).to(device=device, dtype=states.dtype)
    state_std = torch.from_numpy(norm["state_std"]).to(device=device, dtype=states.dtype)
    return states * state_std + state_mean


def compute_normalization(trajectories) -> Dict[str, np.ndarray]:
    """Streaming normalization statistics, to avoid duplicating the dataset in memory."""
    state_dim = trajectories[0]["observations"].shape[1]
    state_sum = np.zeros(state_dim, dtype=np.float64)
    state_sq_sum = np.zeros(state_dim, dtype=np.float64)
    state_count = 0
    rtg_sum = 0.0
    rtg_sq_sum = 0.0
    rtg_count = 0

    for idx in range(len(trajectories)):
        traj = trajectories[idx]
        states = np.asarray(traj["observations"])
        rtg = np.asarray(traj["returns_to_go"])

        if not np.isfinite(states).all():
            nan_count = int(np.isnan(states).sum())
            inf_count = int(np.isinf(states).sum())
            raise ValueError(
                f"Non-finite states in trajectory {idx}. "
                f"NaN={nan_count}, Inf={inf_count}."
            )
        if not np.isfinite(rtg).all():
            nan_count = int(np.isnan(rtg).sum())
            inf_count = int(np.isinf(rtg).sum())
            raise ValueError(
                f"Non-finite returns_to_go in trajectory {idx}. "
                f"NaN={nan_count}, Inf={inf_count}."
            )

        states64 = states.astype(np.float64, copy=False)
        rtg64 = rtg.astype(np.float64, copy=False)
        state_sum += states64.sum(axis=0)
        state_sq_sum += np.square(states64).sum(axis=0)
        state_count += states64.shape[0]
        rtg_sum += float(rtg64.sum())
        rtg_sq_sum += float(np.square(rtg64).sum())
        rtg_count += int(rtg64.size)

    state_mean64 = state_sum / max(state_count, 1)
    state_var64 = np.maximum(state_sq_sum / max(state_count, 1) - state_mean64 ** 2, 0.0)
    state_mean = state_mean64.astype(np.float32)
    state_std = np.maximum(np.sqrt(state_var64), 1e-3).astype(np.float32)
    rtg_mean = float(rtg_sum / max(rtg_count, 1))
    rtg_var = max(rtg_sq_sum / max(rtg_count, 1) - rtg_mean ** 2, 0.0)
    rtg_std = float(max(np.sqrt(rtg_var), 1e-3))

    print(f"  Streaming normalization computed (state_dim={len(state_mean)}):")
    names = [
        "pv_forecast", "load_forecast", "price_buy", "price_sell", "soc",
        "soc_min", "soc_max", "hour_sin", "hour_cos", "battery_capacity_kwh",
        "battery_power_kw", "eta_charge", "eta_discharge",
    ]
    for i in range(min(len(names), len(state_mean))):
        print(f"    {names[i]:22s}: mu={state_mean[i]:8.3f}, sigma={state_std[i]:8.3f}")
    print(f"    {'rtg':22s}: mu={rtg_mean:8.3f}, sigma={rtg_std:8.3f}")

    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "rtg_mean": rtg_mean,
        "rtg_std": rtg_std,
    }


def maybe_empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def evaluate_loss(
    model,
    trajectories: List[Dict[str, np.ndarray]],
    norm: Dict,
    device: torch.device,
    lambda_phys: float,
    eval_batch_size: int,
    *,
    horizon_aware_physics: bool = False,
    target_soc: float = MIN_TERMINAL_SOC,
    total_steps: int = STEPS_PER_DAY,
    lambda_state_pred: float = 0.0,
    lambda_rtg_pred: float = 0.0,
) -> Dict[str, float]:
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be greater than 0")

    seq_len = trajectories[0]["observations"].shape[0]
    state_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    state_mean = norm["state_mean"]
    state_std = norm["state_std"]
    rtg_mean = float(norm["rtg_mean"])
    rtg_std = float(norm["rtg_std"])

    was_training = model.training
    model.eval()
    mse_sum = 0.0
    mse_count = 0
    phys_sum = 0.0
    phys_count = 0
    state_sum = 0.0
    state_count = 0
    rtg_sum = 0.0
    rtg_count = 0

    with torch.no_grad():
        for start in range(0, len(trajectories), eval_batch_size):
            batch = trajectories[start: start + eval_batch_size]
            batch_size = len(batch)

            states = np.stack([
                (traj["observations"] - state_mean) / state_std for traj in batch
            ]).astype(np.float32)
            actions = np.stack([traj["actions"] for traj in batch]).astype(np.float32)
            rewards = np.stack([
                traj["rewards"].reshape(seq_len, 1) for traj in batch
            ]).astype(np.float32)
            returns_to_go = np.stack([
                ((traj["returns_to_go"] - rtg_mean) / rtg_std).reshape(seq_len, 1)
                for traj in batch
            ]).astype(np.float32)
            timesteps = np.tile(np.arange(seq_len, dtype=np.int64), (batch_size, 1))
            mask = np.ones((batch_size, seq_len), dtype=np.float32)

            states_t = torch.from_numpy(states).to(device=device, dtype=torch.float32)
            actions_t = torch.from_numpy(actions).to(device=device, dtype=torch.float32)
            rewards_t = torch.from_numpy(rewards).to(device=device, dtype=torch.float32)
            returns_t = torch.from_numpy(returns_to_go).to(device=device, dtype=torch.float32)
            timesteps_t = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)
            mask_t = torch.from_numpy(mask).to(device=device, dtype=torch.float32)

            state_preds, action_preds, return_preds = model.forward(
                states_t, actions_t, rewards_t, returns_t, timesteps_t, attention_mask=mask_t
            )
            valid = mask_t.reshape(-1) > 0
            pred = action_preds.reshape(-1, act_dim)[valid]
            target = actions_t.reshape(-1, act_dim)[valid]
            mse_sum += float(torch.sum((pred - target) ** 2).item())
            mse_count += int(pred.numel())

            states_raw = denormalize_states(states_t, norm, device)
            lower, upper = compute_physics_bounds(
                states_raw,
                timesteps_t,
                horizon_aware=horizon_aware_physics,
                target_soc=target_soc,
                total_steps=total_steps,
            )
            phys = physics_penalty_from_bounds(action_preds, lower, upper, mask_t)
            valid_steps = int(mask_t.sum().item())
            phys_sum += float(phys.item()) * valid_steps
            phys_count += valid_steps

            # Auxiliary losses: state_preds[:, t] predicts s_{t+1}; return_preds[:, t]
            # predicts R_{t+1}. Only computed when the corresponding lambdas are
            # > 0, to save memory and FLOPs in the default runs.
            if lambda_state_pred > 0.0 and state_preds.shape[1] > 1:
                next_mask = mask_t[:, 1:].reshape(-1) > 0
                sp = state_preds[:, :-1, :].reshape(-1, state_dim)[next_mask]
                st = states_t[:, 1:, :].reshape(-1, state_dim)[next_mask]
                state_sum += float(torch.sum((sp - st) ** 2).item())
                state_count += int(sp.numel())
            if lambda_rtg_pred > 0.0 and return_preds.shape[1] > 1:
                next_mask = mask_t[:, 1:].reshape(-1) > 0
                rp = return_preds[:, :-1, :].reshape(-1, 1)[next_mask]
                rt = returns_t[:, 1:, :].reshape(-1, 1)[next_mask]
                rtg_sum += float(torch.sum((rp - rt) ** 2).item())
                rtg_count += int(rp.numel())

            del (
                states_t,
                actions_t,
                rewards_t,
                returns_t,
                timesteps_t,
                mask_t,
                state_preds,
                action_preds,
                return_preds,
                states_raw,
                lower,
                upper,
            )
            maybe_empty_device_cache(device)

    if was_training:
        model.train()
    mse = mse_sum / max(mse_count, 1)
    phys = phys_sum / max(phys_count, 1)
    state_mse = state_sum / max(state_count, 1) if state_count > 0 else 0.0
    rtg_mse = rtg_sum / max(rtg_count, 1) if rtg_count > 0 else 0.0
    total = (
        mse
        + lambda_phys * phys
        + lambda_state_pred * state_mse
        + lambda_rtg_pred * rtg_mse
    )
    return {
        "mse": float(mse),
        "phys": float(phys),
        "state_mse": float(state_mse),
        "rtg_mse": float(rtg_mse),
        "total": float(total),
    }


def train_pidt(
    trajectories: List[Dict[str, np.ndarray]],
    stats: Dict,
    val_trajectories: Optional[List[Dict[str, np.ndarray]]],
    model,
    init_checkpoint: Optional[Dict],
    device: torch.device,
    num_steps: Optional[int],
    save_dir: Path,
    lambda_phys: float,
    eval_batch_size: int,
    reuse_norm_from_checkpoint: bool,
    overwrite_save: bool,
    *,
    horizon_aware_physics: bool = False,
    target_soc: float = MIN_TERMINAL_SOC,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    lambda_state_pred: float = 0.0,
    lambda_rtg_pred: float = 0.0,
) -> None:
    tc = cfg.train
    if num_steps is None:
        num_steps = tc.num_steps

    prepare_output_dir(str(save_dir), overwrite=overwrite_save, label="checkpoints")
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    if reuse_norm_from_checkpoint:
        if init_checkpoint is None or "norm" not in init_checkpoint:
            raise ValueError("--reuse-norm-from-checkpoint requires --init-model with stored normalization")
        norm = init_checkpoint["norm"]
    else:
        norm = compute_normalization(trajectories)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / tc.warmup_steps, 1),
    )

    print("\nTraining PI-DT:")
    print(
        f"  steps={num_steps}, batch={tc.batch_size}, "
        f"eval_batch={eval_batch_size}, lambda_phys={lambda_phys}"
    )
    if lambda_state_pred > 0.0 or lambda_rtg_pred > 0.0:
        print(
            f"  aux losses: lambda_state_pred={lambda_state_pred}, "
            f"lambda_rtg_pred={lambda_rtg_pred}"
        )
    print(f"  train={len(trajectories)}, val={len(val_trajectories) if val_trajectories else 0}")
    print(f"  device={device}, state_dim={trajectories[0]['observations'].shape[1]}")
    print(
        f"  horizon_aware_physics={horizon_aware_physics}, "
        f"target_soc={target_soc:.2f}, total_steps={STEPS_PER_DAY}"
    )
    if early_stopping_patience > 0:
        print(
            f"  early_stopping: patience={early_stopping_patience} evals, "
            f"min_delta={early_stopping_min_delta}"
        )

    losses: List[float] = []
    best_metric = float("inf")
    evals_since_best = 0
    early_stopped = False
    last_completed_step = 0
    t_start = time.time()
    model.train()

    for step in range(num_steps):
        states, actions, rewards, returns_to_go, timesteps, attention_mask = get_batch(
            trajectories, tc.batch_size, norm=norm, max_len=cfg.dt.context_length, device=device
        )
        state_preds, action_preds, return_preds = model.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask
        )

        valid = attention_mask.reshape(-1) > 0
        act_dim = action_preds.shape[-1]
        pred = action_preds.reshape(-1, act_dim)[valid]
        target = actions.reshape(-1, act_dim)[valid]
        mse = torch.mean((pred - target) ** 2)

        states_raw = denormalize_states(states, norm, device)
        lower, upper = compute_physics_bounds(
            states_raw,
            timesteps,
            horizon_aware=horizon_aware_physics,
            target_soc=target_soc,
            total_steps=STEPS_PER_DAY,
        )
        phys = physics_penalty_from_bounds(action_preds, lower, upper, attention_mask)
        loss = mse + lambda_phys * phys

        # Auxiliary losses: state_preds[:, t] predicts s_{t+1}, return_preds[:, t] predicts R_{t+1}.
        state_loss_val = 0.0
        rtg_loss_val = 0.0
        if lambda_state_pred > 0.0 and state_preds.shape[1] > 1:
            sdim = state_preds.shape[-1]
            next_mask = attention_mask[:, 1:].reshape(-1) > 0
            sp = state_preds[:, :-1, :].reshape(-1, sdim)[next_mask]
            st = states[:, 1:, :].reshape(-1, sdim)[next_mask]
            state_loss = torch.mean((sp - st) ** 2)
            loss = loss + lambda_state_pred * state_loss
            state_loss_val = float(state_loss.item())
        if lambda_rtg_pred > 0.0 and return_preds.shape[1] > 1:
            next_mask = attention_mask[:, 1:].reshape(-1) > 0
            rp = return_preds[:, :-1, :].reshape(-1, 1)[next_mask]
            rt = returns_to_go[:, 1:, :].reshape(-1, 1)[next_mask]
            rtg_loss = torch.mean((rp - rt) ** 2)
            loss = loss + lambda_rtg_pred * rtg_loss
            rtg_loss_val = float(rtg_loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

        if (step + 1) % tc.log_interval == 0:
            elapsed = time.time() - t_start
            sps = (step + 1) / max(elapsed, 1e-9)
            aux_msg = ""
            if lambda_state_pred > 0.0:
                aux_msg += f" state={state_loss_val:.6f}"
            if lambda_rtg_pred > 0.0:
                aux_msg += f" rtg={rtg_loss_val:.6f}"
            print(
                f"  step {step + 1:>6d}/{num_steps} | "
                f"loss={np.mean(losses[-tc.log_interval:]):.6f} "
                f"mse={mse.item():.6f} phys={phys.item():.6f}{aux_msg} "
                f"lr={scheduler.get_last_lr()[0]:.6f} | {sps:.1f} it/s"
            )

        if (step + 1) % tc.eval_interval == 0:
            selection_metric = float(np.mean(losses[-tc.eval_interval:]))
            eval_metrics = None
            if val_trajectories is not None:
                eval_metrics = evaluate_loss(
                    model,
                    val_trajectories,
                    norm,
                    device,
                    lambda_phys,
                    eval_batch_size=eval_batch_size,
                    horizon_aware_physics=horizon_aware_physics,
                    target_soc=target_soc,
                    total_steps=STEPS_PER_DAY,
                    lambda_state_pred=lambda_state_pred,
                    lambda_rtg_pred=lambda_rtg_pred,
                )
                selection_metric = eval_metrics["total"]
                aux_eval_msg = ""
                if lambda_state_pred > 0.0:
                    aux_eval_msg += f" val_state={eval_metrics['state_mse']:.6f}"
                if lambda_rtg_pred > 0.0:
                    aux_eval_msg += f" val_rtg={eval_metrics['rtg_mse']:.6f}"
                print(
                    f"  eval {step + 1:>6d}: "
                    f"val_total={eval_metrics['total']:.6f} "
                    f"val_mse={eval_metrics['mse']:.6f} "
                    f"val_phys={eval_metrics['phys']:.6f}{aux_eval_msg}"
                )

            improved = selection_metric < (best_metric - float(early_stopping_min_delta))
            if improved:
                best_metric = selection_metric
                evals_since_best = 0
                best_payload = {
                    "step": step + 1,
                    "loss": best_metric,
                    "eval_metrics": eval_metrics,
                    "lambda_phys": lambda_phys,
                    "horizon_aware_physics": bool(horizon_aware_physics),
                    "target_soc": float(target_soc),
                    "batch_size": tc.batch_size,
                    "eval_batch_size": eval_batch_size,
                    "learning_rate": tc.learning_rate,
                    "weight_decay": tc.weight_decay,
                    "warmup_steps": tc.warmup_steps,
                    "seed": tc.seed,
                    "hidden_size": cfg.dt.hidden_size,
                    "n_layer": cfg.dt.n_layer,
                    "n_head": cfg.dt.n_head,
                    "n_inner": cfg.dt.n_inner,
                    "activation": cfg.dt.activation_function,
                    "resid_pdrop": cfg.dt.resid_pdrop,
                    "attn_pdrop": cfg.dt.attn_pdrop,
                }
                torch.save({
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_metric,
                    "stats": {**stats, "pidt_per_site": True, "lambda_phys": lambda_phys},
                    "norm": norm,
                    "eval_metrics": eval_metrics,
                }, save_dir / "best_model.pt")
                with (save_dir / "best_metrics.json").open("w") as f:
                    json.dump(best_payload, f, indent=2)
                print(f"  best_model.pt saved ({best_metric:.6f})")
            else:
                evals_since_best += 1
                if early_stopping_patience > 0 and evals_since_best >= early_stopping_patience:
                    print(
                        f"  [early_stopping] no improvement of val_total for "
                        f"{evals_since_best} evaluations (patience={early_stopping_patience}). "
                        f"best={best_metric:.6f} at an earlier step. Stopping at step {step + 1}."
                    )
                    last_completed_step = step + 1
                    early_stopped = True
                    break

        last_completed_step = step + 1

    final_step = last_completed_step if last_completed_step > 0 else num_steps
    torch.save({
        "step": final_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": float(np.mean(losses[-min(500, len(losses)):])),
        "stats": {**stats, "pidt_per_site": True, "lambda_phys": lambda_phys},
        "norm": norm,
    }, save_dir / "final_model.pt")
    np.save(save_dir / "losses.npy", np.asarray(losses, dtype=np.float32))
    with (save_dir / "train_config.json").open("w") as f:
        json.dump({
            "lambda_phys": lambda_phys,
            "lambda_state_pred": float(lambda_state_pred),
            "lambda_rtg_pred": float(lambda_rtg_pred),
            "horizon_aware_physics": bool(horizon_aware_physics),
            "target_soc": float(target_soc),
            "num_steps": num_steps,
            "final_step": int(final_step),
            "early_stopped": bool(early_stopped),
            "early_stopping_patience": int(early_stopping_patience),
            "early_stopping_min_delta": float(early_stopping_min_delta),
            "batch_size": tc.batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": tc.learning_rate,
            "weight_decay": tc.weight_decay,
            "warmup_steps": tc.warmup_steps,
            "grad_clip": tc.grad_clip,
            "seed": tc.seed,
            "hidden_size": cfg.dt.hidden_size,
            "n_layer": cfg.dt.n_layer,
            "n_head": cfg.dt.n_head,
            "n_inner": cfg.dt.n_inner,
            "activation": cfg.dt.activation_function,
            "resid_pdrop": cfg.dt.resid_pdrop,
            "attn_pdrop": cfg.dt.attn_pdrop,
            "state_dim": int(trajectories[0]["observations"].shape[1]),
            "context_length": int(cfg.dt.context_length),
        }, f, indent=2)


def apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.hidden_size is not None:
        cfg.dt.hidden_size = args.hidden_size
    if args.n_layer is not None:
        cfg.dt.n_layer = args.n_layer
    if args.n_head is not None:
        cfg.dt.n_head = args.n_head
    if args.n_inner is not None:
        cfg.dt.n_inner = args.n_inner
    if args.activation is not None:
        cfg.dt.activation_function = args.activation
    if args.dropout is not None:
        cfg.dt.resid_pdrop = args.dropout
        cfg.dt.attn_pdrop = args.dropout

    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.warmup_steps is not None:
        cfg.train.warmup_steps = args.warmup_steps
    if args.grad_clip is not None:
        cfg.train.grad_clip = args.grad_clip
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.log_interval is not None:
        cfg.train.log_interval = args.log_interval
    if args.eval_interval is not None:
        cfg.train.eval_interval = args.eval_interval

    if cfg.dt.hidden_size % cfg.dt.n_head != 0:
        raise ValueError(
            f"hidden_size ({cfg.dt.hidden_size}) must be divisible by n_head ({cfg.dt.n_head})"
        )
