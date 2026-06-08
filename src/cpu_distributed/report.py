"""Report generation for cpu-distributed."""
import json, logging, os
from typing import Any
logger = logging.getLogger(__name__)

def save_json(data: Any, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)
    return path

def format_training_summary(history: list) -> str:
    if not history:
        return "No training history."
    lines = ["", "=" * 50, "  ES Training Summary", "=" * 50, ""]
    lines.append(f"  Steps: {len(history)}")
    lines.append(f"  Final mean fitness: {history[-1].get('mean_fitness', 0):.4f}")
    lines.append(f"  Best max fitness: {max(h.get('max_fitness', 0) for h in history):.4f}")
    lines.extend(["", "=" * 50, ""])
    return "\n".join(lines)
