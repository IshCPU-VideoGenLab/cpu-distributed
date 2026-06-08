"""Tests for cpu_distributed.zo_gradient."""
import pytest
import torch
import torch.nn as nn
from cpu_distributed.zo_gradient import zo_gradient_estimate


class TestZOGradient:
    def test_output_shape(self) -> None:
        model = nn.Linear(8, 1, bias=False)
        def loss_fn(m): return m(torch.ones(1, 8)).sum()
        grad = zo_gradient_estimate(model, loss_fn, num_directions=5)
        assert grad.shape == (8,)

    def test_direction_matches_true_gradient(self) -> None:
        """ZO gradient should point in roughly same direction as true gradient."""
        model = nn.Linear(4, 1, bias=False)
        x = torch.ones(1, 4)
        target = torch.tensor([5.0])
        def loss_fn(m): return ((m(x) - target)**2).mean()

        # True gradient
        model.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        true_grad = model.weight.grad.flatten().clone()

        # ZO estimate
        zo_grad = zo_gradient_estimate(model, loss_fn, num_directions=50, epsilon=1e-3)

        # Check cosine similarity > 0 (same general direction)
        cos = torch.dot(true_grad, zo_grad) / (true_grad.norm() * zo_grad.norm() + 1e-8)
        assert cos.item() > 0, f"ZO gradient points wrong direction: cosine={cos.item()}"
