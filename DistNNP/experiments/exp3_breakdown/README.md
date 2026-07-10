# Experiment 3: Per-Step Timing Breakdown

## Goal

Detailed decomposition of each MD step into individual phases, with bandwidth utilization analysis for all communication phases.

## Configuration

- **GPU configs**: 4 GPU (1× gpuA100x4), 8 GPU (2× gpuA100x4)
- **Representative systems**:
  - 1AKI 26K (spherical protein + water) (default)
  - 2RIK 113K (rod protein + water)
  - Pt+water 14K (slab interface)
  - DPPC 28K (lipid bilayer)
- Steps: 200, steady-state (episode 2+)
- Profiler: `cfg.profiler.enabled=True`, `cfg.profiler.measure_bw=True`

## Phase Decomposition

| # | Phase | Profiler Category | Communication Op | Notes |
|---|-------|-------------------|-----------------|-------|
| 1 | Partition | `sfc_partition` | — | Amortized over rebalance_interval |
| 2 | Halo map build | `build_halo_maps` | `all_gather` | Amortized |
| 3 | Halo exchange | `halo_exchange` | `all_to_all_single` | Ghost atoms: [N_ghost, 7] |
| 4 | Graph construction | `atom2graph` | — | `radius_graph_pbc` neighbor list |
| 5 | Feature exchange ×L | `feature_exchange` | `all_to_all_single` | Per GNN layer, feat_dim=1152 |
| 6 | Model forward | `predict` − comm time | — | GNN forward (pure compute) |
| 7 | Model backward | `backward` − comm time | — | Autograd backward (pure compute) |
| 8 | Gradient exchange | `grad_exchange` | `all_to_all_single` | DifferentiableExchange backward |
| 9 | Force reduction | `force_reduction` | `all_to_all_single` | Async, overlapped with energy |
| 10 | Integration | `integration` | — | Langevin / Euler step |

### Profiler Hook Requirements

Existing `CUDAEventProfiler` already covers phases 1–5, 9–10. May need to add:
- **Phase 7 (backward)**: Split `predict` into forward vs. backward components
- **Phase 8 (grad_exchange)**: Separate from backward if not already instrumented

## Bandwidth Analysis

For each communication phase, record:

| Field | Source |
|-------|--------|
| Data volume (bytes) | send_bytes + recv_bytes from profiler |
| Elapsed time (ms) | CUDA event timing |
| Achieved bandwidth (GB/s) | volume / elapsed |
| Peak bandwidth (GB/s) | NV4 intra-node: ~100 GB/s; CXI inter-node: ~12.5 GB/s |
| Bandwidth utilization (%) | achieved / peak × 100 |

### Per-phase bandwidth expectations

| Phase | Message pattern | Expected BW utilization |
|-------|----------------|------------------------|
| Halo exchange | All-to-all, small messages (7 floats/atom) | Low–medium |
| Feature exchange | All-to-all, large messages (1152 floats/atom) | Medium–high |
| Gradient exchange | All-to-all, same size as feature exchange | Medium–high |
| Force reduction | All-to-all, medium messages (3 floats/atom) | Medium |

## Output

### Tables
- Per-system × per-GPU-count timing table (ms per phase)
- Bandwidth utilization table per communication phase

### Figures
- Stacked bar chart: time per phase, one bar per system (4 GPU and 8 GPU side by side)
- Pie chart: compute vs. communication vs. other per system
- Optional: `nsys` CUDA timeline screenshot showing kernel/NCCL overlap

## Execution

```bash
# 4 GPU, 1AKI
torchrun --nproc_per_node=4 baseline/dist_md.py \
    md.xyz_file=experiments/exp_datasets/protein_solvated/1aki_density.xyz \
    md.total_steps=200 \
    dist.partition_algorithm=sfc_morton \
    profiler.enabled=True \
    profiler.measure_bw=True \
    profiler.log_dir=experiments/exp3_breakdown/results/1aki_4gpu

# 8 GPU (2 nodes), 2RIK
# (sbatch script with 2× gpuA100x4 nodes)
srun torchrun --nproc_per_node=4 --nnodes=2 baseline/dist_md.py \
    md.xyz_file=experiments/exp_datasets/protein_solvated/2rik_density.xyz \
    md.total_steps=200 \
    dist.partition_algorithm=sfc_morton \
    profiler.enabled=True \
    profiler.measure_bw=True \
    profiler.log_dir=experiments/exp3_breakdown/results/2rik_8gpu
```
