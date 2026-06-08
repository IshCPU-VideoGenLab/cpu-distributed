"""Evolution Strategies optimizer for gradient-free training.

Implements the OpenAI ES variant: generate parameter perturbations,
evaluate fitness, estimate gradient from the fitness landscape.

Reference: Salimans et al., "Evolution Strategies as a Scalable
Alternative to Reinforcement Learning", 2017.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from cpu_distributed.config import ESConfig

logger = logging.getLogger(__name__)


class EvolutionStrategy:
    """Evolution Strategies optimizer.

    Uses random parameter perturbations and fitness evaluation to
    estimate gradients without backpropagation.

    Args:
        model: The model to optimize.
        config: ES configuration.
    """

    def __init__(self, model: nn.Module, config: Optional[ESConfig] = None) -> None:
        self._model = model
        self._config = config or ESConfig()
        self._param_count = sum(p.numel() for p in model.parameters())
        self._step_count = 0

        # Adam state
        self._m = torch.zeros(self._param_count)
        self._v = torch.zeros(self._param_count)

        logger.info(
            "ES optimizer: %s params, pop=%d, sigma=%.4f, lr=%.4f",
            f"{self._param_count:,}",
            self._config.population_size,
            self._config.sigma,
            self._config.learning_rate,
        )

    def _get_flat_params(self) -> torch.Tensor:
        """Flatten all model parameters into a single vector."""
        return torch.cat([p.data.float().flatten() for p in self._model.parameters()])

    def _set_flat_params(self, flat: torch.Tensor) -> None:
        """Set model parameters from a flat vector."""
        offset = 0
        for p in self._model.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].reshape(p.shape).to(p.dtype))
            offset += numel

    def generate_perturbations(
        self,
        seed_base: Optional[int] = None,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """Generate random perturbations for the population.

        Args:
            seed_base: Base seed for reproducibility.

        Returns:
            Tuple of (seeds, noise_vectors).
        """
        pop_size = self._config.population_size
        if self._config.antithetic:
            # Generate half, mirror the other half
            half = pop_size // 2
        else:
            half = pop_size

        seeds = []
        noises = []

        for i in range(half):
            seed = (seed_base or self._step_count * 1000) + i
            seeds.append(seed)

            rng = torch.Generator()
            rng.manual_seed(seed)
            noise = torch.randn(self._param_count, generator=rng) * self._config.sigma
            noises.append(noise)

            if self._config.antithetic:
                seeds.append(-seed)  # Negative seed = mirror
                noises.append(-noise)

        return seeds, noises

    def evaluate_perturbation(
        self,
        noise: torch.Tensor,
        loss_fn,
        *loss_args,
    ) -> float:
        """Evaluate fitness of a perturbed model.

        Temporarily applies the perturbation, runs forward pass,
        then restores original parameters.

        Args:
            noise: Parameter perturbation vector.
            loss_fn: Function that takes the model and returns a scalar loss.
            *loss_args: Additional arguments to loss_fn.

        Returns:
            Fitness (negative loss) as a float.
        """
        original_params = self._get_flat_params()

        # Apply perturbation
        perturbed = original_params + noise
        self._set_flat_params(perturbed)

        # Evaluate
        self._model.eval()
        with torch.no_grad():
            loss = loss_fn(self._model, *loss_args)

        # Restore
        self._set_flat_params(original_params)

        return -float(loss)  # Fitness = negative loss

    def step(
        self,
        seeds: List[int],
        fitnesses: List[float],
    ) -> Dict[str, float]:
        """Update parameters using fitness-weighted gradient estimate.

        Args:
            seeds: Seeds used to generate perturbations.
            fitnesses: Fitness values for each perturbation.

        Returns:
            Dictionary with step statistics.
        """
        pop_size = len(fitnesses)
        if pop_size == 0:
            return {"step": self._step_count, "error": "no_fitnesses"}

        # Normalize fitnesses (rank-based for robustness)
        fit_array = np.array(fitnesses, dtype=np.float32)
        ranks = np.argsort(np.argsort(fit_array)).astype(np.float32)
        ranks = (ranks / (pop_size - 1)) - 0.5  # Center around 0
        ranks = ranks / (ranks.std() + 1e-8)

        # Reconstruct noise from seeds and compute weighted sum
        gradient = torch.zeros(self._param_count)
        for i, seed in enumerate(seeds):
            rng = torch.Generator()
            actual_seed = abs(seed)
            rng.manual_seed(actual_seed)
            noise = torch.randn(self._param_count, generator=rng) * self._config.sigma
            if seed < 0:
                noise = -noise
            gradient += ranks[i] * noise

        gradient /= (pop_size * self._config.sigma)

        # Apply update (with optional Adam)
        current_params = self._get_flat_params()

        if self._config.use_adam:
            self._step_count += 1
            self._m = self._config.adam_beta1 * self._m + (1 - self._config.adam_beta1) * gradient
            self._v = self._config.adam_beta2 * self._v + (1 - self._config.adam_beta2) * gradient ** 2

            m_hat = self._m / (1 - self._config.adam_beta1 ** self._step_count)
            v_hat = self._v / (1 - self._config.adam_beta2 ** self._step_count)

            update = self._config.learning_rate * m_hat / (torch.sqrt(v_hat) + 1e-8)
        else:
            self._step_count += 1
            update = self._config.learning_rate * gradient

        new_params = current_params + update
        self._set_flat_params(new_params)

        return {
            "step": self._step_count,
            "mean_fitness": float(np.mean(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
            "gradient_norm": float(gradient.norm()),
            "update_norm": float(update.norm()),
        }

    @property
    def step_count(self) -> int:
        return self._step_count
