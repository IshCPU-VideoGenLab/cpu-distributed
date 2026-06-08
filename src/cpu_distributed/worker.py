"""Worker node for distributed ES training.

Connects to the coordinator, receives perturbation tasks,
evaluates fitness, and returns results.
"""

import logging
import socket
import time
from typing import Optional

import torch
import torch.nn as nn

from cpu_distributed.protocol import send_message, recv_message, MSG_REGISTER, MSG_RESULT

logger = logging.getLogger(__name__)


class Worker:
    """Distributed training worker.

    Args:
        model: Local copy of the model.
        worker_id: Unique identifier for this worker.
        coordinator_host: Coordinator address.
        coordinator_port: Coordinator port.
    """

    def __init__(
        self,
        model: nn.Module,
        worker_id: str = "worker_0",
        coordinator_host: str = "localhost",
        coordinator_port: int = 5555,
    ) -> None:
        self._model = model
        self._worker_id = worker_id
        self._host = coordinator_host
        self._port = coordinator_port
        self._sock: Optional[socket.socket] = None
        self._running = False

    def connect(self) -> bool:
        """Connect to the coordinator.

        Returns:
            True if connection successful.
        """
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self._host, self._port))
            send_message(self._sock, {
                "type": MSG_REGISTER,
                "worker_id": self._worker_id,
            })
            logger.info("Connected to coordinator at %s:%d", self._host, self._port)
            return True
        except ConnectionError as e:
            logger.error("Failed to connect: %s", e)
            return False

    def evaluate_and_report(
        self,
        seed: int,
        sigma: float,
        loss_fn,
    ) -> float:
        """Evaluate a perturbation and report fitness.

        Args:
            seed: Random seed for the perturbation.
            sigma: Noise scale.
            loss_fn: Loss function.

        Returns:
            Fitness value.
        """
        param_count = sum(p.numel() for p in self._model.parameters())
        flat_params = torch.cat([p.data.float().flatten() for p in self._model.parameters()])

        # Generate noise from seed
        rng = torch.Generator()
        rng.manual_seed(abs(seed))
        noise = torch.randn(param_count, generator=rng) * sigma
        if seed < 0:
            noise = -noise

        # Perturb, evaluate, restore
        perturbed = flat_params + noise
        offset = 0
        for p in self._model.parameters():
            numel = p.numel()
            p.data.copy_(perturbed[offset:offset + numel].reshape(p.shape).to(p.dtype))
            offset += numel

        self._model.eval()
        with torch.no_grad():
            loss = float(loss_fn(self._model))

        # Restore
        offset = 0
        for p in self._model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].reshape(p.shape).to(p.dtype))
            offset += numel

        fitness = -loss
        return fitness

    def disconnect(self) -> None:
        """Disconnect from coordinator."""
        if self._sock:
            self._sock.close()
            self._sock = None
        logger.info("Worker %s disconnected", self._worker_id)
