# Experiment 1: Partition Analysis

## 1a. Partition Balance Quality

**Goal**: Evaluate partition quality across all benchmark systems and geometries.

### Configuration

- **Systems**: All real datasets (12 systems, 8K–1.5M atoms)
- **Algorithms**: Simple, SFC Morton, SFC Hilbert, KDTree. (weighted or unweighted)
- **Partition count**: k = 4, 8, 16, 32, 64, 128, 256, 512, 1024

### Metrics

| Metric | Definition | Importance |
|--------|-----------|------------|
| I²_atom | max(atoms_per_rank) / avg(atoms_per_rank) | Memory balance |
| I²_edge | max(edges_per_rank) / avg(edges_per_rank) | Compute balance (primary) |
| Edge cut % | cut_edges / total_edges | Communication volume proxy |
| Halo atoms | Total ghost atoms across all ranks | Direct communication cost |

### Focus

Different geometries stress partition algorithms differently:

| Geometry | System | Expected Challenge |
|----------|--------|--------------------|
| Uniform | CH4 8K | Baseline (easy) |
| Spherical + water | 1AKI 26K | Density gradient: protein core vs. solvent |
| Rod + water | 2RIK 113K | High aspect ratio |
| Slab interface | Pt+water 14K/21K | 2D layered, 1D uniform |
| Bilayer membrane | DPPC 28K | Double-layer structure |
| Ring | GroEL 57K | 7-fold symmetry, no solvent |
| Filament | 2W49 153K | Extreme aspect ratio, sparse |

### Output

- Table: per-system × per-algorithm × per-k metrics
- Bar chart: I²_edge comparison across geometries
- Recommendation: best algorithm per geometry class

---

## 1b. Partition Computation Performance

**Goal**: Measure the partition algorithm overhead itself.

### Configuration

- **Systems**: all real datasets (12 systems, 8K–1.5M atoms)
- **Algorithms**: SFC Morton, SFC Hilbert, KDTree (weighted or unweighted)
- **k**: 4, 8, 16, 32, 64, 128, 256, 512, 1024

### Breakdown

| Phase | What it measures |
|-------|-----------------|
| SFC key generation + sort | Morton/Hilbert curve computation |
| KDTree construction | Spatial tree build |
| Halo map build | all_gather positions + BFS graph traversal |

### Existing benchmarks

- `benchmarks/bench_sfc_cuda.py`
- `benchmarks/bench_kdtree_cuda.py`

Extend to larger systems (113K, 1.5M).

### Output

- Time (ms) vs. atom count (log-log plot)
- Scaling exponent per algorithm

---

## 1c. Communication Matrix Structure

**Goal**: Characterize the communication pattern for each system to motivate sparse P2P optimization.

### Tools

- `scripts/analyze_comm_matrix.py` — offline analysis
- `scripts/plot_comm_matrix.py` — NxN heatmap visualization

### Configuration

- **Systems**: All real datasets
- **Algorithms**: SFC Morton, KDTree
- **Ranks**: 4, 8, 16, 32, 64, 128, 256, 512, 1024

### Metrics

| Metric | Definition |
|--------|-----------|
| Sparsity | Fraction of zero-sized messages in NxN matrix |
| Max message size | Largest single rank-to-rank transfer (bytes) |
| Avg message size | Mean of non-zero messages (bytes) |
| Total volume | Sum of all messages × feat_dim × 4 bytes |

### Output

- NxN heatmaps per system (selected configurations)
- Sparsity vs. rank count trend (SFC vs. KDTree lines)
- Message size distribution histograms
