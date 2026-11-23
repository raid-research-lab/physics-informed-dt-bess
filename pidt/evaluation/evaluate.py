# -*- coding: utf-8 -*-
"""
Evaluate a trained PI-DT on the test split, on the nonlinear battery plant.

Protocol (Section 5.6 of the paper):
  - site-chained rollout: the final SOC and the transient voltages of a day
    carry over to the next day of the same site;
  - the controller decides with the forecasts contained in the state, while
    every executed action is scored with the realized PV and load;
  - the horizon-aware projection enforces soc[T] >= 0.55 at every step;
  - costs are compared with the expert cost stored in the split
    (``-sum(rewards)`` of every trajectory).

Outputs (``--output``, default ``results/evaluation``):
  day_results.csv   one row per site-day (cost, gap, final SOC, reserve deficit)
  summary.json      aggregate gap, mean / median gap, reserve violations

Usage
  python -m pidt.evaluation.evaluate --model checkpoints/pidt/best_model.pt
  python -m pidt.evaluation.evaluate --model ... --max-episodes 100   # quick check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from pidt.config import apply_emsx_config, cfg
from pidt.constants import MIN_TERMINAL_SOC
from pidt.evaluation import common
from pidt.evaluation.common import (
    SOC_INDEX,
    aggregate_gap_pct,
    load_actual_site_days,
    load_split_metadata,
    pct_gap,
    select_eval_indices,
    summarize,
)
from pidt.evaluation.rollouts import NLState, rollout_dt_nl
from pidt.models.pidt import build_pidt_model
from pidt.paths import REALIZED_PV_LOAD_CSV, EVALUATION_RESULTS, TEST_SPLIT_DIR, PIDT_CHECKPOINT_DIR

apply_emsx_config(cfg)


def load_pidt(model_path: Path, device: torch.device):
    """Rebuild PI-DT from ``best_model.pt`` and the ``train_config.json`` next to it."""
    common.apply_checkpoint_config(model_path)
    model = build_pidt_model(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    norm = ckpt.get("norm")
    if norm is None:
        raise ValueError(f"Checkpoint without normalization statistics: {model_path}")
    model.eval()
    return model, norm


def evaluate(args: argparse.Namespace) -> Dict:
    device = common.pick_device(args.device)
    trajectories, stats = common.load_trajectories(args.data)
    dates, site_ids = load_split_metadata(args.data, len(trajectories))
    model, norm = load_pidt(args.model, device)

    indices = select_eval_indices(
        n=len(trajectories), max_episodes=args.max_episodes, seed=args.seed,
        rollout_mode=args.rollout_mode, dates=dates, site_ids=site_ids,
    )
    actual_by_key = load_actual_site_days(
        csv_path=args.actual_csv, dates=dates, site_ids=site_ids, indices=indices,
    )
    print(f"Evaluating {args.model} on {len(indices)} site-days of {args.data}")

    soc_chain: Dict[int, float] = {}
    nl_by_site: Dict[int, NLState] = {}
    rows = []
    for pos, idx in enumerate(indices, start=1):
        traj = trajectories[int(idx)]
        site_id = int(site_ids[int(idx)])
        date = str(dates[int(idx)])
        actual_pv, actual_load = actual_by_key[(site_id, date)]
        expert_initial_soc = float(traj["observations"][0, SOC_INDEX])
        if args.rollout_mode == "site_chained":
            initial_soc = soc_chain.get(site_id, expert_initial_soc)
            nl_state = nl_by_site.setdefault(site_id, NLState())
        else:
            initial_soc = expert_initial_soc
            nl_state = NLState()

        r = rollout_dt_nl(
            model=model, trajectory=traj, norm=norm, device=device,
            target_mode=args.target_mode,
            allow_grid_charge=args.allow_grid_charge,
            project_actions=not args.no_projection,
            initial_soc=initial_soc,
            actual_pv=actual_pv, actual_load=actual_load,
            min_terminal_soc=args.min_terminal_soc,
            nl=nl_state,
        )
        final_soc = float(r["soc"][-1])
        if args.rollout_mode == "site_chained":
            soc_chain[site_id] = final_soc

        expert_cost = float(-np.sum(traj["rewards"]))
        rows.append({
            "site_id": site_id,
            "date": date,
            "initial_soc": float(initial_soc),
            "expert_cost": expert_cost,
            "pidt_cost": float(r["total_cost"]),
            "gap_pct": pct_gap(float(r["total_cost"]), expert_cost),
            "final_soc": final_soc,
            "reserve_deficit": max(0.0, args.min_terminal_soc - final_soc),
        })
        if pos % args.progress_interval == 0 or pos == len(indices):
            print(f"  {pos}/{len(indices)} site-days", flush=True)

    df = pd.DataFrame(rows)
    deficit_pp = 100.0 * df["reserve_deficit"]
    summary = {
        "model": str(args.model),
        "data": str(args.data),
        "split": stats.get("split"),
        "n_site_days": int(len(df)),
        "rollout_mode": args.rollout_mode,
        "min_terminal_soc": float(args.min_terminal_soc),
        "mean_expert_cost_eur_day": float(df["expert_cost"].mean()),
        "mean_pidt_cost_eur_day": float(df["pidt_cost"].mean()),
        "aggregate_gap_pct": aggregate_gap_pct(df["pidt_cost"], df["expert_cost"]),
        "gap_pct": summarize(df["gap_pct"].tolist()),
        "final_soc": summarize(df["final_soc"].tolist()),
        "days_deficit_gt_1pp": int((deficit_pp > 1.0).sum()),
        "max_deficit_pp": float(deficit_pp.max()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "day_results.csv", index=False)
    with (args.output / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSite-days                 : {summary['n_site_days']}")
    print(f"Aggregate gap vs expert   : {summary['aggregate_gap_pct']:.2f} %")
    print(f"Mean / median gap         : {summary['gap_pct']['mean']:.2f} % / {summary['gap_pct']['median']:.2f} %")
    print(f"Days with deficit > 1 pp  : {summary['days_deficit_gt_1pp']}")
    print(f"Max. end-of-day deficit   : {summary['max_deficit_pp']:.2f} pp")
    print(f"Results in {args.output}")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PI-DT on the nonlinear battery plant.")
    p.add_argument("--model", type=Path, default=PIDT_CHECKPOINT_DIR / "best_model.pt")
    p.add_argument("--data", type=Path, default=TEST_SPLIT_DIR,
                   help="Test split (trajectories.npz with the expert actions and rewards).")
    p.add_argument("--actual-csv", type=Path, default=REALIZED_PV_LOAD_CSV,
                   help="Quarter-hourly table with the realized PV and load (docs/data.md).")
    p.add_argument("--output", type=Path, default=EVALUATION_RESULTS)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--max-episodes", type=int, default=10**9,
                   help="Evaluate only the first N site-days of the chained order.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-mode", choices=["expert_return", "expert_rtg"], default="expert_return",
                   help="Return-to-go given to the model (default: the expert return of the day).")
    p.add_argument("--rollout-mode", choices=["site_chained", "independent"], default="site_chained")
    p.set_defaults(allow_grid_charge=True)
    p.add_argument("--no-grid-charge", dest="allow_grid_charge", action="store_false")
    p.add_argument("--no-projection", action="store_true",
                   help="Disable the horizon-aware projection (ablation).")
    p.add_argument("--min-terminal-soc", type=float, default=MIN_TERMINAL_SOC)
    p.add_argument("--progress-interval", type=int, default=200)
    return p.parse_args(argv)


def main(argv=None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
