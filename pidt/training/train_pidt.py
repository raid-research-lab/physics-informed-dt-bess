# -*- coding: utf-8 -*-
"""
Train PI-DT, the model reported in the paper.

Builds the ``PIDT`` model of ``pidt.models.pidt`` and runs the training loop of
``pidt.training.trainer`` (memory-mapped data, streaming normalization, physics
penalty on the asymmetric horizon-aware bands, validation, early stopping).

All defaults reproduce the configuration of the paper (Section 5.5): hidden
size 192, 4 layers, 4 heads, ReLU, dropout 0.1, AdamW with learning rate 3e-4
and weight decay 1e-4, batch 16, 30,000 steps with 2,000 warm-up steps,
gradient clipping 0.25, lambda_phys = 0.1, seed 42, horizon-aware penalty with
soc[T] >= 0.55, validation every 1,000 steps and early stopping with patience 8.

Usage
  python -m pidt.training.train_pidt --device cuda
  # without the physics loss (ablation):
  python -m pidt.training.train_pidt --device cuda --lambda-phys 0 \\
      --save checkpoints/pidt_no_physics_loss
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pidt.config import apply_emsx_config, cfg
from pidt.constants import MIN_TERMINAL_SOC
from pidt.models.pidt import build_pidt_model
from pidt.paths import PIDT_CHECKPOINT_DIR, TRAIN_SPLIT_DIR, VAL_SPLIT_DIR
from pidt.training import trainer

apply_emsx_config(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PI-DT (asymmetric terminal SOC constraint).")
    # ---- data and output ----
    parser.add_argument("--data", type=Path, default=TRAIN_SPLIT_DIR)
    parser.add_argument("--val-data", type=Path, default=VAL_SPLIT_DIR)
    parser.add_argument("--save", type=Path, default=PIDT_CHECKPOINT_DIR)
    parser.add_argument("--overwrite-save", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    # ---- optimization (None = value of pidt.config.TrainConfig) ----
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    # ---- architecture (None = value of pidt.config.DTConfig) ----
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-inner", type=int, default=None)
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None,
                        help="Same dropout for resid_pdrop and attn_pdrop")
    # ---- physics-informed terms ----
    parser.add_argument("--lambda-phys", type=float, default=0.1,
                        help="Weight of the physics penalty (0 = ablation without physics loss).")
    parser.add_argument("--horizon-aware-physics", dest="horizon_aware_physics",
                        action="store_true", default=True,
                        help="Penalize violations of the asymmetric horizon-aware bands (default).")
    parser.add_argument("--no-horizon-aware-physics", dest="horizon_aware_physics",
                        action="store_false",
                        help="Penalize only the per-step SOC window.")
    parser.add_argument("--min-terminal-soc", type=float, default=MIN_TERMINAL_SOC,
                        help="Minimum end-of-day SOC (default 0.55).")
    # ---- optional, not used in the paper ----
    parser.add_argument("--init-model", type=Path, default=None,
                        help="Checkpoint to initialize the weights from (fine-tuning).")
    parser.add_argument("--reuse-norm-from-checkpoint", action="store_true")
    parser.add_argument("--lambda-state-pred", type=float, default=0.0,
                        help="Weight of an auxiliary next-state prediction loss.")
    parser.add_argument("--lambda-rtg-pred", type=float, default=0.0,
                        help="Weight of an auxiliary next return-to-go prediction loss.")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    trainer.apply_cli_overrides(args)
    device = trainer.pick_device(args.device)
    trajectories, stats = trainer.load_trajectories(str(args.data))
    val_trajectories = None
    if args.val_data is not None:
        val_trajectories, _ = trainer.load_trajectories(str(args.val_data))

    model = build_pidt_model(device)
    init_checkpoint = None
    if args.init_model is not None:
        init_checkpoint = trainer.load_initial_checkpoint(str(args.init_model), model, device)

    trainer.train_pidt(
        trajectories=trajectories,
        stats={**stats, "version": "pidt_min_terminal_soc"},
        val_trajectories=val_trajectories,
        model=model,
        init_checkpoint=init_checkpoint,
        device=device,
        num_steps=args.steps,
        save_dir=args.save,
        lambda_phys=args.lambda_phys,
        eval_batch_size=args.eval_batch_size,
        reuse_norm_from_checkpoint=args.reuse_norm_from_checkpoint,
        overwrite_save=args.overwrite_save,
        horizon_aware_physics=bool(args.horizon_aware_physics),
        target_soc=float(args.min_terminal_soc),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        lambda_state_pred=float(args.lambda_state_pred),
        lambda_rtg_pred=float(args.lambda_rtg_pred),
    )


if __name__ == "__main__":
    main()
