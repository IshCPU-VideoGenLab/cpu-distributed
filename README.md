<p align="center">
  <img src="https://raw.githubusercontent.com/IshCPU-VideoGenLab/.github/main/logo.svg" alt="IshCPU-VideoGenLab" width="80">
</p>

# cpu-distributed

**Distributed training across commodity CPUs using evolution strategies — no GPUs, no InfiniBand, just laptops on WiFi.**

Part of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab) — building the first video generation model that trains and runs entirely on commodity CPUs.

---

## Why Evolution Strategies?

Standard distributed training needs GPU interconnects and synchronized gradient communication. Evolution strategies need only forward passes and scalar fitness values. Communication is seeds + scalars — a 1 Mbps WiFi connection is enough.

---

## Phase 6 of 7

| Phase | Contribution | Status |
|-------|-------------|--------|
| 1-5 | Profiling, Mamba, Codec, BitNet, AVX2 | ✅ |
| **6** | **Distributed CPU training** | **Active** |
| 7 | cpu-video-gen (flagship) | Planned |

---

## How It Works

```
Coordinator (Calyx's laptop)
    │
    ├── Send parameters + noise seeds to workers
    │
    ├── Worker 1 (friend's laptop)
    │   └── Perturb params → forward pass → measure loss → return (seed, loss)
    │
    ├── Worker 2 (another laptop)
    │   └── Same: perturb → evaluate → return
    │
    └── Aggregate fitness → estimate gradient → update parameters
```

**Communication per step:** ~100 bytes per worker (seed + loss scalar). Not megabytes of gradient tensors.

---

## Usage

```bash
# Single-machine simulation (for testing)
python scripts/run_local.py --model delta_predictor --steps 100

# Multi-machine: start coordinator
python scripts/start_coordinator.py --host 0.0.0.0 --port 5555

# On each worker machine:
python scripts/start_worker.py --coordinator 192.168.1.x --port 5555
```

### Python API

```python
from cpu_distributed.es_optimizer import EvolutionStrategy
from cpu_distributed.coordinator import Coordinator

es = EvolutionStrategy(model, population_size=50, sigma=0.01, lr=0.001)

for step in range(1000):
    # Generate perturbations
    seeds, noise = es.generate_perturbations()
    # Evaluate fitness (can be distributed)
    fitnesses = [evaluate(model, seed) for seed in seeds]
    # Update
    es.step(seeds, fitnesses)
```

---

## Citation

```bibtex
@software{kwakye2026cpudistributed,
  author = {Kwakye, Ishmael Affum},
  title = {cpu-distributed: Evolution Strategy Training for CPU-Native Video Generation},
  year = {2026},
  url = {https://github.com/IshCPU-VideoGenLab/cpu-distributed},
  institution = {University of Ghana, Legon}
}
```

## Contributing

See the [Contributing Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/CONTRIBUTING.md)
and [Version Control Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/VERSION_CONTROL_GUIDE.md).

---

## License

MIT License.

---

*Phase 6 of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab). Training without GPUs — across laptops on WiFi.*
