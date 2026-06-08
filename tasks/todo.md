# Phase 6 — cpu-distributed Task Roadmap

---

## Milestone 1: Evolution Strategies Core
- [ ] Implement `EvolutionStrategy` optimizer (OpenAI ES variant)
- [ ] Generate parameter perturbations from shared random seeds
- [ ] Compute fitness-weighted gradient estimate
- [ ] Antithetic sampling (mirrored perturbations for variance reduction)
- [ ] Adam-style momentum for ES updates
- [ ] Unit test: ES on simple quadratic objective
- [ ] Unit test: convergence on small MLP training

## Milestone 2: Zeroth-Order Gradient Estimation
- [ ] Implement forward-difference gradient estimate
- [ ] Implement central-difference (2-point) estimate
- [ ] Implement multi-point estimate for better accuracy
- [ ] Compare: ES vs ZO gradient on same objective
- [ ] Unit test: gradient estimate vs true gradient (autograd)

## Milestone 3: TCP Protocol
- [ ] Define JSON message format: {type, payload, worker_id, step}
- [ ] Implement `send_message()` / `recv_message()` over TCP
- [ ] Handle message framing (length-prefixed)
- [ ] Unit test: send/receive roundtrip on localhost

## Milestone 4: Coordinator
- [ ] Implement coordinator server (accepts worker connections)
- [ ] Distribute parameters + noise seeds to workers
- [ ] Collect fitness results from workers
- [ ] Aggregate and update parameters
- [ ] Handle worker timeouts and disconnects
- [ ] Checkpoint saving (periodic state dump)

## Milestone 5: Worker
- [ ] Implement worker client (connects to coordinator)
- [ ] Receive parameters and seeds
- [ ] Evaluate perturbed model (forward pass + loss)
- [ ] Return (seed, fitness) to coordinator
- [ ] Auto-reconnect on connection loss

## Milestone 6: Local Simulation
- [ ] `run_local.py` — simulate distributed training on one machine
- [ ] Use threading to emulate multiple workers
- [ ] Train delta predictor on synthetic data
- [ ] Plot training loss curve
- [ ] Measure: steps/sec, convergence rate

## Milestone 7: Documentation & Polish
- [ ] Write docs/training_plan.md
- [ ] Update README with results
- [ ] All tests pass
- [ ] Tag v0.1.0
- [ ] Write Phase 7 handoff
