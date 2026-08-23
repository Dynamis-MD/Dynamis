# Experiment 2: Scaling Analysis

## 2a. Strong Scaling (real data, fixed problem size)

**Goal**: Fix the system size, increase GPU count, observe how communication fraction grows and where scaling breaks.

### Configuration

| System | Atoms | GPU counts | Nodes |
|--------|------:|-----------|-------|
| 1AKI | 25,846 |  | |
| ApoA1 | 92,224 |  | |
| 2RIK | 112,727 |  | |
| 9GMO | 1,536,442 |  | |

- Partition: SFC Morton (default)
- Steps: 200 (skip episode 1, use steady-state)
- Profiler: enabled, measure_bw=True

### Metrics

| Metric | Definition |
|--------|-----------|
| Steps/s | Overall throughput |
| Speedup | Steps/s(N) / Steps/s(1) |
| Parallel efficiency | Speedup / N |
| Comm fraction (%) | (halo_exchange + feature_exchange + grad_exchange + force_reduction) / total |
| Compute fraction (%) | (model_forward + model_backward + graph_construction) / total |

### Output

- Steps/s vs. GPU count curves (one line per system, with ideal linear reference)
- Parallel efficiency vs. GPU count
- Stacked area chart: compute / communication / other vs. GPU count
- Identify the scaling knee-point per system

---

## 2b. Weak Scaling (synthetic data, fixed 1K atoms/GPU)

**Goal**: Isolate communication overhead from partition quality effects by using uniform-density synthetic data with perfect load balance.

### Configuration

Generated via `baseline/generate_xyz.py` — uniform CH4, 1K atoms/GPU:

1k to 128k atoms

- Partition: SFC Morton
- Steps: 200 (steady-state)

### Metrics

| Metric | Definition |
|--------|-----------|
| Steps/s | Should remain constant under ideal scaling |
| Weak scaling efficiency | T₁ / Tₙ (1.0 = perfect) |
| Per-phase overhead | Time increase in each phase vs. single GPU |

### Overhead Decomposition

For each GPU count, record per-phase times and compute overhead vs. 1-GPU baseline:

| Phase | 1 GPU | 4 GPU | 8 GPU | 16 GPU | 32 GPU |
|-------|-------|-------|-------|--------|--------|
| Partition | — | Δ | Δ | Δ | Δ |
| Halo map build | — | Δ | Δ | Δ | Δ |
| Halo exchange | — | Δ | Δ | Δ | Δ |
| Feature exchange | — | Δ | Δ | Δ | Δ |
| Force reduction | — | Δ | Δ | Δ | Δ |
| Compute | ref | Δ | Δ | Δ | Δ |

### Output

- Steps/s vs. GPU count (with ideal horizontal reference line)
- Weak scaling efficiency curve
- Stacked bar: overhead breakdown per GPU count
- Key insight: which communication phase dominates overhead at scale

