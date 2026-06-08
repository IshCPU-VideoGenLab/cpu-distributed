"""Zeroth-order gradient estimation using finite differences.

Alternative to ES: directly estimate the gradient by perturbing
each parameter (or random directions) and measuring loss change.
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def zo_gradient_estimate(
    model: nn.Module,
    loss_fn: Callable,
    epsilon: float = 1e-3,
    num_directions: int = 10,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Estimate gradient using random direction finite differences.

    For each random direction d:
        g_estimate += (loss(θ + ε*d) - loss(θ - ε*d)) / (2ε) * d

    Args:
        model: Model to estimate gradient for.
        loss_fn: Loss function taking model as input, returning scalar.
        epsilon: Perturbation size.
        num_directions: Number of random directions to sample.
        seed: Random seed.

    Returns:
        Estimated gradient as a flat tensor.
    """
    if seed is not None:
        torch.manual_seed(seed)

    param_count = sum(p.numel() for p in model.parameters())
    flat_params = torch.cat([p.data.float().flatten() for p in model.parameters()])
    gradient = torch.zeros(param_count)

    for i in range(num_directions):
        direction = torch.randn(param_count)
        direction = direction / direction.norm()

        # Forward difference: f(θ + ε*d)
        _set_params(model, flat_params + epsilon * direction)
        model.eval()
        with torch.no_grad():
            loss_plus = float(loss_fn(model))

        # Backward difference: f(θ - ε*d)
        _set_params(model, flat_params - epsilon * direction)
        with torch.no_grad():
            loss_minus = float(loss_fn(model))

        # Central difference gradient estimate
        grad_scalar = (loss_plus - loss_minus) / (2 * epsilon)
        gradient += grad_scalar * direction

    # Restore original params
    _set_params(model, flat_params)

    gradient /= num_directions
    return gradient


def _set_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Set model parameters from flat vector."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].reshape(p.shape).to(p.dtype))
        offset += numel
