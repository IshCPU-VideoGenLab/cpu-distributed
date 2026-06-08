"""CLI for cpu-distributed training."""
import argparse, logging, sys
from typing import List, Optional

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="cpu-distributed")
    sub = parser.add_subparsers(dest="command")

    local = sub.add_parser("local", help="Run local ES training simulation")
    local.add_argument("--steps", type=int, default=100)
    local.add_argument("--pop-size", type=int, default=50)
    local.add_argument("--sigma", type=float, default=0.01)
    local.add_argument("--lr", type=float, default=0.001)
    local.add_argument("--output", type=str, default="results")
    local.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    if args.command is None:
        print("Usage: cpu-distributed {local|coordinator|worker}")
        return 1

    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    if args.command == "local":
        import torch, torch.nn as nn
        from cpu_distributed.config import ESConfig
        from cpu_distributed.coordinator import Coordinator
        from cpu_distributed.report import save_json, format_training_summary

        # Simple test model
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))
        target = torch.randn(1)
        def loss_fn(m, x=torch.randn(1, 32)):
            return ((m(x) - target) ** 2).mean()

        es_config = ESConfig(population_size=args.pop_size, sigma=args.sigma, learning_rate=args.lr)
        coordinator = Coordinator(model, es_config)
        history = coordinator.start(loss_fn, max_steps=args.steps)

        print(format_training_summary(history))
        save_json(history, args.output, "training_history.json")
        return 0

    return 1

if __name__ == "__main__":
    sys.exit(main())
