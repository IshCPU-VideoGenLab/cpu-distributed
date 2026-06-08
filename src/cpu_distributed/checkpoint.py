"""Checkpoint management for distributed training."""

import json
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    step: int,
    history: list,
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    """Save training checkpoint.

    Args:
        model: Current model.
        step: Current training step.
        history: Training history.
        output_dir: Output directory.
        filename: Optional filename override.

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = filename or f"checkpoint_step{step}.pt"
    path = os.path.join(output_dir, fname)

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "history": history,
    }, path)

    logger.info("Checkpoint saved: %s (step %d)", path, step)
    return path


def load_checkpoint(
    model: nn.Module,
    path: str,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        model: Model to load state into.
        path: Path to checkpoint file.

    Returns:
        Dictionary with step and history.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Checkpoint loaded: %s (step %d)", path, checkpoint["step"])
    return {"step": checkpoint["step"], "history": checkpoint.get("history", [])}
