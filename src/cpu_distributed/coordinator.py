"""Central coordinator for distributed ES training.

Manages worker connections, distributes perturbation tasks,
collects fitness results, and updates parameters.
"""

import logging
import socket
import threading
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from cpu_distributed.config import DistributedConfig, ESConfig
from cpu_distributed.es_optimizer import EvolutionStrategy
from cpu_distributed.protocol import (
    send_message, recv_message,
    MSG_REGISTER, MSG_TASK, MSG_RESULT, MSG_SHUTDOWN,
)

logger = logging.getLogger(__name__)


class Coordinator:
    """Central parameter server and task coordinator.

    Args:
        model: Model to train.
        es_config: Evolution strategies configuration.
        dist_config: Distributed configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        es_config: Optional[ESConfig] = None,
        dist_config: Optional[DistributedConfig] = None,
    ) -> None:
        self._model = model
        self._es = EvolutionStrategy(model, es_config or ESConfig())
        self._config = dist_config or DistributedConfig()
        self._workers: Dict[str, socket.socket] = {}
        self._server: Optional[socket.socket] = None
        self._running = False

    def start(self, loss_fn, max_steps: Optional[int] = None) -> List[Dict]:
        """Start the coordinator and run training.

        For local/testing use. In production, use start_server() and
        handle workers separately.

        Args:
            loss_fn: Loss function for fitness evaluation.
            max_steps: Override max training steps.

        Returns:
            Training history (list of step stats).
        """
        steps = max_steps or self._config.max_steps
        history = []

        for step in range(steps):
            seeds, noises = self._es.generate_perturbations()

            # Evaluate locally (distributed version sends to workers)
            fitnesses = []
            for noise in noises:
                fitness = self._es.evaluate_perturbation(noise, loss_fn)
                fitnesses.append(fitness)

            stats = self._es.step(seeds, fitnesses)
            history.append(stats)

            if step % 10 == 0:
                logger.info(
                    "Step %d: mean_fitness=%.4f, max=%.4f",
                    step, stats["mean_fitness"], stats["max_fitness"],
                )

        return history

    def start_server(self) -> None:
        """Start the TCP server for worker connections."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self._config.host, self._config.port))
        self._server.listen(self._config.num_workers)
        self._running = True
        logger.info("Coordinator listening on %s:%d", self._config.host, self._config.port)

    def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
        for name, sock in self._workers.items():
            try:
                send_message(sock, {"type": MSG_SHUTDOWN})
                sock.close()
            except Exception:
                pass
        if self._server:
            self._server.close()
        logger.info("Coordinator stopped")
