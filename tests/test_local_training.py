"""Integration test: local ES training."""
import pytest
import torch
import torch.nn as nn
from cpu_distributed.config import ESConfig
from cpu_distributed.coordinator import Coordinator


class TestLocalTraining:
    def test_runs_without_error(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        x = torch.randn(2, 4)
        target = torch.ones(2, 1)
        def loss_fn(m): return ((m(x) - target)**2).mean()

        es_config = ESConfig(population_size=10, sigma=0.01, learning_rate=0.001)
        coordinator = Coordinator(model, es_config)
        history = coordinator.start(loss_fn, max_steps=5)
        assert len(history) == 5
        assert all("mean_fitness" in h for h in history)
