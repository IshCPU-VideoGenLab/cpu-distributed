"""Configuration for cpu-distributed training."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ESConfig:
    """Evolution strategies configuration.

    Args:
        population_size: Number of perturbations per step.
        sigma: Noise standard deviation for perturbations.
        learning_rate: Step size for parameter updates.
        antithetic: Use mirrored perturbations (halves variance).
        use_adam: Use Adam momentum for updates.
        adam_beta1: Adam beta1.
        adam_beta2: Adam beta2.
    """
    population_size: int = 50
    sigma: float = 0.01
    learning_rate: float = 0.001
    antithetic: bool = True
    use_adam: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999


@dataclass
class DistributedConfig:
    """Configuration for distributed coordinator/worker setup.

    Args:
        host: Coordinator host address.
        port: Coordinator port.
        num_workers: Expected number of workers.
        timeout_seconds: Worker timeout before marking as dead.
        checkpoint_interval: Steps between checkpoints.
        output_dir: Directory for results and checkpoints.
        max_steps: Maximum training steps.
    """
    host: str = "0.0.0.0"
    port: int = 5555
    num_workers: int = 2
    timeout_seconds: float = 30.0
    checkpoint_interval: int = 50
    output_dir: str = "results"
    max_steps: int = 1000

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class TrainingConfig:
    """Combined training configuration."""
    es: ESConfig = None
    distributed: DistributedConfig = None
    model_name: str = "delta_predictor"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.es is None:
            self.es = ESConfig()
        if self.distributed is None:
            self.distributed = DistributedConfig()
