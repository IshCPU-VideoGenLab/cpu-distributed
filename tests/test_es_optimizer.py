"""Tests for cpu_distributed.es_optimizer."""
import pytest
import torch
import torch.nn as nn
from cpu_distributed.config import ESConfig
from cpu_distributed.es_optimizer import EvolutionStrategy


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
    def forward(self, x):
        return self.linear(x)


class TestEvolutionStrategy:
    def test_generate_perturbations(self) -> None:
        model = SimpleModel()
        es = EvolutionStrategy(model, ESConfig(population_size=10))
        seeds, noises = es.generate_perturbations()
        assert len(seeds) == 10
        assert len(noises) == 10

    def test_antithetic_sampling(self) -> None:
        model = SimpleModel()
        es = EvolutionStrategy(model, ESConfig(population_size=10, antithetic=True))
        seeds, noises = es.generate_perturbations()
        # With antithetic, noise[i] = -noise[i+1] for pairs
        assert torch.allclose(noises[0], -noises[1])

    def test_step_updates_params(self) -> None:
        model = SimpleModel()
        es = EvolutionStrategy(model, ESConfig(population_size=10, learning_rate=0.1))
        params_before = torch.cat([p.data.flatten() for p in model.parameters()]).clone()

        seeds, noises = es.generate_perturbations()
        fitnesses = [float(i) for i in range(len(seeds))]
        es.step(seeds, fitnesses)

        params_after = torch.cat([p.data.flatten() for p in model.parameters()])
        assert not torch.allclose(params_before, params_after)

    def test_evaluate_perturbation(self) -> None:
        model = SimpleModel()
        es = EvolutionStrategy(model, ESConfig())
        x = torch.randn(1, 4)
        target = torch.tensor([1.0])
        def loss_fn(m): return ((m(x) - target)**2).mean()

        noise = torch.randn(sum(p.numel() for p in model.parameters())) * 0.01
        fitness = es.evaluate_perturbation(noise, loss_fn)
        assert isinstance(fitness, float)

    def test_convergence_on_simple_problem(self) -> None:
        """ES should improve fitness over 50 steps on a simple objective."""
        model = SimpleModel()
        x = torch.ones(1, 4)
        target = torch.tensor([2.0])
        def loss_fn(m): return ((m(x) - target)**2).mean()

        es = EvolutionStrategy(model, ESConfig(population_size=20, sigma=0.05, learning_rate=0.01))
        initial_loss = float(loss_fn(model))

        for _ in range(50):
            seeds, noises = es.generate_perturbations()
            fitnesses = [es.evaluate_perturbation(n, loss_fn) for n in noises]
            es.step(seeds, fitnesses)

        final_loss = float(loss_fn(model))
        assert final_loss < initial_loss, f"Loss didn't improve: {initial_loss} → {final_loss}"
