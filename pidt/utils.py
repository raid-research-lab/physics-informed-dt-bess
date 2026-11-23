"""Small helpers shared by the training and evaluation scripts."""

from __future__ import annotations

import os
from typing import Dict

import torch


def prepare_output_dir(path: str, overwrite: bool = False, label: str = "output") -> str:
    """Create ``path`` if needed and refuse to overwrite a non-empty directory.

    Raises ``FileExistsError`` when the directory already contains files,
    unless ``overwrite=True``.
    """
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise FileExistsError(f"The {label} path exists and is not a directory: {path}")

        has_entries = any(os.scandir(path))
        if has_entries and not overwrite:
            raise FileExistsError(
                f"The {label} directory is not empty: {path}\n"
                f"Use a new directory or pass --overwrite explicitly."
            )
    else:
        os.makedirs(path, exist_ok=True)

    return path


def load_initial_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device = None) -> Dict:
    """Load weights from a previous checkpoint before fine-tuning."""
    if device is None:
        device = next(model.parameters()).device

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "The initial checkpoint is not compatible with the current architecture "
            f"(state_dim={model.state_dim}, act_dim={model.act_dim}, K={model.max_length}). "
            "This usually happens when reusing a checkpoint trained with a different "
            "look-ahead or context length."
        ) from exc

    loss = checkpoint.get("loss")
    loss_str = f"{float(loss):.6f}" if isinstance(loss, (float, int)) else str(loss)
    print(
        f"Initial checkpoint loaded from {checkpoint_path} "
        f"(step {checkpoint.get('step', '?')}, loss {loss_str})"
    )
    return checkpoint
