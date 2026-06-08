# Distributed Training Plan

## Objective

Fine-tune the CPU-native video generation model across multiple commodity
laptops using gradient-free optimization. No GPUs, no InfiniBand.

## Evolution Strategies (ES)

### Algorithm

```
1. Coordinator holds current parameters θ
2. For each step:
   a. Generate N random seeds s_1, ..., s_N
   b. Send seeds to workers (or evaluate locally)
   c. Each worker i:
      - Generate noise ε_i from seed s_i
      - Evaluate fitness F(θ + σ·ε_i)
      - Return (s_i, F_i)
   d. Coordinator estimates gradient:
      g ≈ (1/Nσ) Σ F_i · ε_i
   e. Update: θ ← θ + α · Adam(g)
```

### Why ES Works for 1-Bit Models

- **No backprop needed** — only forward passes
- **Communication is tiny** — seeds + scalars, not gradient tensors
- **Embarrassingly parallel** — workers are independent
- **Works with non-differentiable operations** — sign(), quantization
- **Scales linearly with workers** — add laptops = faster training

### Antithetic Sampling

For each perturbation ε, also evaluate -ε. This halves variance:
- Without: Var ∝ 1/N
- With antithetic: Var ∝ 1/(2N) (same cost, half the noise)

## Communication Protocol

```
Coordinator ←→ Worker messages (JSON over TCP):

REGISTER:  Worker → Coordinator  {"type":"register", "worker_id":"w1"}
TASK:      Coordinator → Worker  {"type":"task", "seeds":[1,2,3], "sigma":0.01}
RESULT:    Worker → Coordinator  {"type":"result", "fitnesses":[-0.5,-0.3,-0.7]}
PARAMS:    Coordinator → Worker  {"type":"params", "state_dict_url":"..."}
SHUTDOWN:  Coordinator → Worker  {"type":"shutdown"}
```

Total bytes per step per worker: ~200 bytes. A 56kbps modem could handle it.

## Target Setup

- Coordinator: Calyx's Pentium Gold laptop
- Workers: 1-2 borrowed laptops
- Network: campus WiFi
- Model: Phase 3 delta predictor (~50-100M params)
- Training objective: MSE on synthetic frame deltas

## Expected Results

- **Not trying to match GPU training speed.** This is a proof of concept.
- Expected: 10-100 steps/minute with 2 workers
- Convergence: 1000-5000 steps for delta predictor fine-tuning
- Wall-clock: hours to days, not minutes

The point is: it WORKS. On laptops. Over WiFi. No cloud bill.
