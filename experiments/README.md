# Dynamis Performance Experiments

## Directory Structure

```
experiments/
├── README.md                  # This file — experiment plan overview
├── exp_datasets/              # All benchmark systems (XYZ files)
├── exp1_partition/            # Experiment 1: Partition analysis
├── exp2_scaling/              # Experiment 2: Strong & weak scaling
├── exp3_breakdown/            # Experiment 3: Per-step timing breakdown
└── exp4_comm_backend/         # Experiment 4: Communication backend comparison
```

## Cluster Configuration

- **Single node**: gpuA100x4 (4× A100-SXM4-40GB, NV4 NVLink)
- **Multi-node**: N × gpuA100x4, interconnected via CXI fabric
- GPU count sequence: 1, 2, 4 (single node) → 8, 12, 16, 32 (multi-node)

## Datasets

### Real Data (strong scaling, breakdown, partition analysis)

| Category | System | Atoms | Geometry | File |
|----------|--------|------:|----------|------|
| Baseline | CH4 uniform | 8,001 | Uniform gas | `baselines/ch4_8k_splited.xyz` |
| Baseline | CH4 clustered | 8,001 | Artificial cluster | `baselines/ch4_8k_cluster.xyz` |
| Protein+water | DHFR | 23,558 | Uniform (equilibrated) | `protein_solvated/dhfr_23k_openmm.xyz` |
| Protein+water | 1AKI (lysozyme) | 25,846 | Spherical protein + water | `protein_solvated/1aki_density.xyz` |
| Protein+water | 2RIK (titin) | 112,727 | Rod-shaped protein + water | `protein_solvated/2rik_density.xyz` |
| Protein+water | 9GMO (ribosome) | 1,536,442 | Large complex + water | `protein_solvated/9gmo_density.xyz` |
| Electrochemical | Pt(111)+water 12K | 14,236 | Slab interface | `electrochemical/pt_water_12k_density.xyz` |
| Electrochemical | Pt(111)+water 17K | 21,180 | Slab interface | `electrochemical/pt_water_17k_density.xyz` |
| Membrane | DPPC | 28,160 | Lipid bilayer | `membrane/dppc_membrane_28k.xyz` |
| Membrane | ApoA1 | 92,224 | Protein + POPC + water | `membrane/apoa1_92k_openmm.xyz` |
| Vacuum protein | GroEL | 56,700 | Ring, 7-fold symmetry | `protein_vacuum/groel_vacuum_openmm.xyz` |
| Vacuum protein | 2W49 (filament) | 153,456 | Filament, sparse | `protein_vacuum/2w49_vacuum_openmm.xyz` |

### Synthetic Data (weak scaling)

Generated via `baseline/generate_xyz.py`. Fixed **1K atoms/GPU**, uniform CH4 density:

| GPUs | Total atoms | Nodes |
|-----:|------------:|------:|
| 1 | 1K | 1 |
| 2 | 2K | 1 |
| 4 | 4K | 1 |
| 8 | 8K | 2 |
| 16 | 16K | 4 |
| 32 | 32K | 8 |

## Experiment Summary

| # | Experiment | Priority | Key Question |
|---|-----------|----------|-------------|
| 1 | [Partition Analysis](exp1_partition/README.md) | P1 | How do partition algorithms perform across different geometries? |
| 2 | [Scaling Analysis](exp2_scaling/README.md) | P0 | How does communication overhead scale with GPU count? |
| 3 | [Per-Step Breakdown](exp3_breakdown/README.md) | P0 | Where does time go in each MD step? |
| 4 | [Comm Backend Comparison](exp4_comm_backend/README.md) | P1 | NCCL vs IPC push vs sparse P2P — which wins when? |
