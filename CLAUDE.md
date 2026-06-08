# CLAUDE.md — cpu-distributed

> Read by Claude Code at the start of every session.

---

## Project Identity

- **Org:** IshCPU-VideoGenLab
- **Repo:** cpu-distributed (Phase 6 of 7)
- **Author:** Ishmael Affum Kwakye (Calyx)
- **GitHub:** calyxish
- **Institution:** University of Ghana, Legon

---

## What This Project Is

This is the **distributed training** phase. We implement training/fine-tuning
of the CPU-native video generation model across multiple commodity CPUs using
gradient-free optimization methods.

Traditional distributed training (data-parallel SGD) requires:
- GPU interconnects (NVLink, InfiniBand)
- Synchronized gradient computation
- High-bandwidth communication

We can't use any of that. Instead we use **zeroth-order optimization** and
**evolution strategies** — methods that only need forward passes and scalar
loss values, communicated over regular network connections.

**cpu-distributed** implements:

1. **Evolution Strategies (ES)** — population-based optimization using
   parameter perturbations and fitness evaluation
2. **Zeroth-Order Gradient Estimation** — approximate gradients from
   finite differences of forward passes
3. **Worker coordination** — simple TCP-based coordinator/worker protocol
4. **Parameter server** — centralized parameter distribution and aggregation
5. **Fault tolerance** — handles worker disconnects gracefully

---

## Why Not Standard Distributed Training?

Standard SGD needs backpropagation → needs autograd → needs float weights
→ needs GPU memory for activation checkpoints. Our model has 1-bit weights
and no attention. Backprop through sign() with STE is possible but noisy.

**Evolution strategies** bypass all of this:
1. Coordinator sends current parameters to workers
2. Each worker adds random noise, runs forward pass, measures loss
3. Workers send (noise_seed, loss) back to coordinator
4. Coordinator estimates gradient from the loss landscape
5. Update parameters. Repeat.

**Communication: only seeds + scalars.** No gradient tensors. A 1 Mbps
connection is sufficient.

---

## Hardware Context

- **Target: 2-4 commodity laptops** connected over WiFi or LAN
- Each machine: 2-4 cores, 8-16 GB RAM, no GPU — **x86 or ARM** (mixed architectures are fine;
  the portable SIMD kernels from Phase 5 run on both)
- Network: regular WiFi (10-100 Mbps) or ethernet
- Coordinator: a MacBook Air M4 (ARM64); Pentium Gold remains the benchmark reference
- Workers: 1-2 friend laptops borrowed for the demo (any CPU architecture)

This is NOT trying to match GPU cluster throughput. It's proving the
concept that distributed CPU training is FEASIBLE for 1-bit models.

---

## Previous Phases

| Phase | What It Provides |
|-------|-----------------|
| 2 | Mamba model (cheap forward pass) |
| 3 | Delta predictor (tiny model, fast to evaluate) |
| 4 | 1-bit weights (minimal communication) |
| 5 | Portable SIMD kernels — AVX2 + NEON (fast forward pass on x86 and ARM) |
| **6** | **Distributed training across CPUs** |

---

## Code Conventions

- Python 3.9, type hints, Google docstrings, logging
- `socket` and `threading` for networking (no heavy deps)
- JSON messages over TCP for simplicity
- No external dependencies beyond torch and numpy

---

## File Structure

```
cpu-distributed/
├── CLAUDE.md
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── lessons.md
├── tasks/todo.md
├── .claude/{settings.json, commands/, rules/}
├── configs/default.json
├── src/cpu_distributed/
│   ├── __init__.py
│   ├── config.py
│   ├── es_optimizer.py      ← Evolution strategies optimizer
│   ├── zo_gradient.py       ← Zeroth-order gradient estimation
│   ├── coordinator.py       ← Central parameter server
│   ├── worker.py            ← Worker node
│   ├── protocol.py          ← TCP message protocol
│   ├── aggregator.py        ← Gradient/fitness aggregation
│   ├── checkpoint.py        ← Save/resume training state
│   ├── report.py
│   └── cli.py
├── scripts/
│   ├── start_coordinator.py
│   ├── start_worker.py
│   └── run_local.py          ← Single-machine simulation
├── tests/
│   ├── __init__.py
│   ├── test_es_optimizer.py
│   ├── test_zo_gradient.py
│   ├── test_protocol.py
│   └── test_local_training.py
├── results/.gitkeep
└── docs/training_plan.md
```

---

## Task & Lessons

Check `tasks/todo.md` before starting. Check `lessons.md` before writing code.
