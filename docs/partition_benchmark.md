# Partition Algorithm Benchmark: Edge-Based Workload Imbalance

## Motivation

In distributed GNN inference, each GPU computes edges where `dst` is a local atom. The edge count per partition (inner + boundary edges where dst is local) determines GPU compute workload. Imbalanced edge counts cause stragglers that limit throughput.

## Methods

| Method               | Description                                                                                                                                                                  |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Simple**           | Sequential block partition: atoms `[0..N/k)` go to partition 0, `[N/k..2N/k)` to partition 1, etc. No spatial awareness; quality depends entirely on atom ordering in file. |
| **Random**           | Each atom assigned uniformly to `randint(0, k)`. Gives near-perfect balance but maximum edge cut (most edges cross boundaries).                                             |
| **SFC Morton**       | Sort atoms by Morton (Z-curve) index, then split into k equal-count blocks. Preserves spatial locality along the SFC.                                                       |
| **SFC Hilbert**      | Same as Morton but using Hilbert curve. Hilbert has better locality in theory but our implementation uses order=7.                                                           |
| **SFC Morton Wt.**   | Sort by Morton index, split by cumulative edge count (instead of atom count) so each partition gets equal edge workload.                                                     |
| **SFC Hilbert Wt.**  | Same weighted approach with Hilbert curve.                                                                                                                                   |
| **KD-Tree Spatial**  | Recursive bisection: at each level, sort along the longest axis and split at the median. Produces geometrically compact partitions.                                          |
| **METIS**            | Graph-based multilevel partitioning. Coarsens graph, partitions, then refines with Kernighan-Lin. Minimizes edge cut with atom balance constraint.                          |
| **METIS Edge-Wt.**   | METIS with `vweights=per_atom_edge_count`. Balances total edge count per partition instead of atom count. Best of both worlds.                                              |

## Metrics

| Metric       | Definition                                            | Goal                     |
|--------------|-------------------------------------------------------|--------------------------|
| **EdgeImb**  | `max(edge_counts) / min(edge_counts)` per partition   | Close to 1.0 (primary)   |
| **AtomImb**  | `max(atom_counts) / min(atom_counts)`                 | Close to 1.0             |
| **EdgeCut**  | Number of edges crossing partition boundaries         | Lower = less comm        |
| **EdgeCut%** | `EdgeCut / total_edges * 100`                         | Lower is better          |
| **HaloNodes** | Total unique ghost (halo) atoms across all partitions | Lower = less mem/bw      |

## Systems

| System              |  Atoms |    Edges  | Description                                          |
|---------------------|-------:|----------:|------------------------------------------------------|
| `ch4_8k_splited`    |  8,001 |   300,312 | CH4+O2 gas mixture, uniform density                  |
| `1aki_8k_solvated`  |  8,000 |   664,154 | Lysozyme protein + TIP3P water, non-uniform density  |
| `1aki_16k_solvated` | 15,998 | 1,405,950 | Lysozyme protein + TIP3P water, larger box           |

Note: Solvated protein has ~2x more edges per atom than the gas mixture due to higher water density (~83 edges/atom vs ~37 edges/atom).

## Results

All runs on CPU (login node). SFC timing is inflated; focus on quality metrics.
Tables sorted by EdgeImb descending (worst at top, best at bottom).

### CH4 8k (8,001 atoms, 300k edges)

#### k = 2

| Method               | AtomImb | EdgeImb   | EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|--------:|---------:|--------:|
| METIS                |   1.000 |     1.219 |   3,984 |     1.3% |     569 |
| Simple               |   1.000 |     1.218 |  30,266 |    10.1% |   3,352 |
| SFC Hilbert          |   1.000 |     1.108 |  17,384 |     5.8% |   1,794 |
| Random               |   1.024 |     1.033 | 150,404 |    50.1% |   8,001 |
| KD-Tree Spatial      |   1.000 |     1.011 |  13,686 |     4.6% |   1,526 |
| SFC Morton           |   1.000 |     1.011 |  13,916 |     4.6% |   1,549 |
| SFC Morton Wt.       |   1.009 | **1.000** |  13,942 |     4.6% |   1,549 |
| SFC Hilbert Wt.      |   1.102 | **1.000** |  17,176 |     5.7% |   1,845 |
| METIS Edge-Wt.       |   1.192 | **1.000** |   5,576 |     1.9% |     707 |

#### k = 4

| Method               | AtomImb | EdgeImb   | EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|--------:|---------:|--------:|
| Simple               |   1.000 |     1.266 | 152,746 |    50.9% |  11,352 |
| METIS                |   1.000 |     1.255 |  14,880 |     5.0% |   1,986 |
| SFC Hilbert          |   1.000 |     1.170 |  46,736 |    15.6% |   5,238 |
| Random               |   1.040 |     1.052 | 225,116 |    75.0% |  24,000 |
| SFC Morton           |   1.000 |     1.017 |  28,836 |     9.6% |   3,456 |
| KD-Tree Spatial      |   1.000 |     1.017 |  27,428 |     9.1% |   3,221 |
| SFC Morton Wt.       |   1.015 | **1.001** |  29,566 |     9.8% |   3,508 |
| SFC Hilbert Wt.      |   1.168 | **1.001** |  47,800 |    15.9% |   5,380 |
| METIS Edge-Wt.       |   1.242 | **1.001** |  15,542 |     5.2% |   1,971 |

#### k = 8

| Method               | AtomImb | EdgeImb   | EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|--------:|---------:|--------:|
| KD-Tree Spatial      |   1.001 |     1.337 |  41,342 |    13.8% |   5,102 |
| METIS                |   1.002 |     1.301 |  26,906 |     9.0% |   3,470 |
| Simple               |   1.001 |     1.283 | 214,222 |    71.3% |  26,161 |
| SFC Morton           |   1.001 |     1.276 |  37,386 |    12.4% |   4,868 |
| SFC Hilbert          |   1.001 |     1.256 |  81,468 |    27.1% |  10,504 |
| Random               |   1.086 |     1.098 | 262,682 |    87.5% |  55,293 |
| METIS Edge-Wt.       |   1.287 | **1.002** |  26,402 |     8.8% |   3,385 |
| SFC Morton Wt.       |   1.245 | **1.001** |  42,450 |    14.1% |   5,388 |
| SFC Hilbert Wt.      |   1.249 | **1.001** |  87,224 |    29.0% |  10,781 |

#### k = 16

| Method               | AtomImb | EdgeImb   | EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|--------:|---------:|--------:|
| METIS                |   1.062 |     1.483 |  42,466 |    14.1% |   5,669 |
| SFC Hilbert          |   1.002 |     1.429 | 122,428 |    40.8% |  18,476 |
| KD-Tree Spatial      |   1.002 |     1.382 |  65,338 |    21.8% |   8,658 |
| SFC Morton           |   1.002 |     1.333 |  64,186 |    21.4% |   8,219 |
| Simple               |   1.002 |     1.317 | 244,870 |    81.5% |  46,760 |
| Random               |   1.114 |     1.120 | 281,590 |    93.8% | 107,831 |
| METIS Edge-Wt.       |   1.449 |     1.060 |  43,554 |    14.5% |   5,874 |
| SFC Morton Wt.       |   1.352 | **1.004** |  70,222 |    23.4% |   8,981 |
| SFC Hilbert Wt.      |   1.390 | **1.004** | 126,830 |    42.2% |  18,436 |

### Lysozyme 8k (8,000 atoms, 664k edges)

#### k = 2

| Method               | AtomImb | EdgeImb   | EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|--------:|---------:|--------:|
| Simple               |   1.000 |     1.152 |  65,168 |     9.8% |   4,216 |
| METIS                |   1.000 |     1.118 |  43,126 |     6.5% |   2,567 |
| SFC Hilbert          |   1.000 |     1.063 |  96,614 |    14.5% |   5,268 |
| Random               |   1.024 |     1.029 | 331,480 |    49.9% |   8,000 |
| KD-Tree Spatial      |   1.000 |     1.009 |  59,294 |     8.9% |   3,505 |
| SFC Morton           |   1.000 |     1.003 |  62,916 |     9.5% |   3,673 |
| METIS Edge-Wt.       |   1.112 | **1.001** |  43,792 |     6.6% |   2,561 |
| SFC Morton Wt.       |   1.003 | **1.000** |  62,764 |     9.5% |   3,670 |
| SFC Hilbert Wt.      |   1.061 | **1.000** | 101,038 |    15.2% |   5,419 |

#### k = 4

| Method               | AtomImb | EdgeImb   | EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|--------:|---------:|--------:|
| Simple               |   1.000 |     1.605 | 162,728 |    24.5% |   9,535 |
| METIS                |   1.001 |     1.312 |  94,560 |    14.2% |   6,510 |
| SFC Hilbert          |   1.000 |     1.126 | 214,558 |    32.3% |  13,323 |
| SFC Morton           |   1.000 |     1.113 | 125,646 |    18.9% |   8,574 |
| KD-Tree Spatial      |   1.000 |     1.112 | 115,908 |    17.5% |   7,943 |
| Random               |   1.040 |     1.045 | 497,662 |    74.9% |  24,000 |
| SFC Morton Wt.       |   1.170 | **1.001** | 128,986 |    19.4% |   8,697 |
| METIS Edge-Wt.       |   1.238 | **1.001** |  95,854 |    14.4% |   6,556 |
| SFC Hilbert Wt.      |   1.119 | **1.000** | 218,858 |    33.0% |  13,441 |

#### k = 8

| Method               | AtomImb | EdgeImb   | EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|--------:|---------:|--------:|
| Simple               |   1.000 |     1.907 | 287,956 |    43.4% |  18,868 |
| METIS                |   1.002 |     1.797 | 139,670 |    21.0% |  10,846 |
| SFC Morton           |   1.000 |     1.276 | 193,902 |    29.2% |  14,412 |
| SFC Hilbert          |   1.000 |     1.269 | 303,568 |    45.7% |  24,099 |
| KD-Tree Spatial      |   1.000 |     1.207 | 153,240 |    23.1% |  11,805 |
| Random               |   1.086 |     1.083 | 580,646 |    87.4% |  55,970 |
| METIS Edge-Wt.       |   1.490 | **1.003** | 139,314 |    21.0% |  10,827 |
| SFC Morton Wt.       |   1.297 | **1.002** | 199,274 |    30.0% |  14,736 |
| SFC Hilbert Wt.      |   1.256 | **1.002** | 306,616 |    46.2% |  24,060 |

#### k = 16

| Method               | AtomImb | EdgeImb   | EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|--------:|---------:|--------:|
| Simple               |   1.000 |     1.941 | 415,886 |    62.6% |  32,301 |
| METIS                |   1.062 |     1.821 | 180,530 |    27.2% |  14,860 |
| KD-Tree Spatial      |   1.000 |     1.441 | 206,644 |    31.1% |  16,982 |
| SFC Hilbert          |   1.000 |     1.415 | 376,542 |    56.7% |  39,741 |
| SFC Morton           |   1.000 |     1.320 | 247,450 |    37.3% |  21,081 |
| Random               |   1.114 |     1.142 | 622,658 |    93.8% | 118,581 |
| METIS Edge-Wt.       |   1.348 |     1.061 | 188,634 |    28.4% |  15,648 |
| SFC Morton Wt.       |   1.339 | **1.004** | 246,726 |    37.1% |  21,003 |
| SFC Hilbert Wt.      |   1.365 | **1.003** | 382,650 |    57.6% |  40,659 |

### Lysozyme 16k (15,998 atoms, 1.4M edges)

#### k = 2

| Method               | AtomImb | EdgeImb   |  EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|---------:|---------:|--------:|
| Simple               |   1.000 |     1.052 |   70,560 |     5.0% |   4,653 |
| Random               |   1.025 |     1.026 |  702,994 |    50.0% |  15,998 |
| SFC Hilbert          |   1.000 |     1.014 |  143,068 |    10.2% |   7,795 |
| METIS                |   1.000 |     1.011 |   55,224 |     3.9% |   3,215 |
| KD-Tree Spatial      |   1.000 |     1.003 |  108,596 |     7.7% |   6,150 |
| SFC Morton           |   1.000 |     1.002 |  112,166 |     8.0% |   6,343 |
| SFC Morton Wt.       |   1.002 | **1.000** |  112,226 |     8.0% |   6,346 |
| SFC Hilbert Wt.      |   1.015 | **1.000** |  144,426 |    10.3% |   7,895 |
| METIS Edge-Wt.       |   1.016 | **1.000** |   55,144 |     3.9% |   3,215 |

#### k = 4

| Method               | AtomImb | EdgeImb   |   EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|----------:|---------:|--------:|
| Simple               |   1.000 |     1.234 |   216,020 |    15.4% |  13,128 |
| METIS                |   1.001 |     1.082 |   159,176 |    11.3% |  10,027 |
| SFC Hilbert          |   1.000 |     1.060 |   323,138 |    23.0% |  19,568 |
| SFC Morton           |   1.000 |     1.053 |   221,480 |    15.8% |  14,507 |
| KD-Tree Spatial      |   1.000 |     1.046 |   210,500 |    15.0% |  13,690 |
| Random               |   1.036 |     1.038 | 1,053,734 |    74.9% |  47,994 |
| SFC Morton Wt.       |   1.081 | **1.000** |   223,952 |    15.9% |  14,664 |
| SFC Hilbert Wt.      |   1.060 | **1.000** |   328,404 |    23.4% |  19,702 |
| METIS Edge-Wt.       |   1.070 | **1.000** |   159,404 |    11.3% |  10,050 |

#### k = 8

| Method               | AtomImb | EdgeImb   |   EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|----------:|---------:|--------:|
| Simple               |   1.001 |     1.584 |   460,330 |    32.7% |  26,542 |
| METIS                |   1.001 |     1.165 |   251,016 |    17.9% |  17,732 |
| SFC Morton           |   1.001 |     1.109 |   294,574 |    21.0% |  21,235 |
| SFC Hilbert          |   1.001 |     1.093 |   513,322 |    36.5% |  35,732 |
| KD-Tree Spatial      |   1.001 |     1.086 |   260,002 |    18.5% |  18,388 |
| Random               |   1.056 |     1.066 | 1,229,700 |    87.5% | 111,976 |
| SFC Morton Wt.       |   1.139 | **1.001** |   300,884 |    21.4% |  21,611 |
| SFC Hilbert Wt.      |   1.098 | **1.001** |   516,768 |    36.8% |  36,311 |
| METIS Edge-Wt.       |   1.071 | **1.001** |   247,476 |    17.6% |  17,719 |

#### k = 16

| Method               | AtomImb | EdgeImb   |   EdgeCut | EdgeCut% | HaloNodes |
|----------------------|--------:|----------:|----------:|---------:|--------:|
| Simple               |   1.001 |     1.810 |   804,224 |    57.2% |  51,938 |
| METIS                |   1.061 |     1.418 |   329,714 |    23.5% |  24,368 |
| SFC Morton           |   1.001 |     1.205 |   421,298 |    30.0% |  32,325 |
| KD-Tree Spatial      |   1.001 |     1.172 |   365,652 |    26.0% |  26,804 |
| SFC Hilbert          |   1.001 |     1.130 |   709,686 |    50.5% |  61,364 |
| Random               |   1.086 |     1.097 | 1,318,166 |    93.8% | 238,323 |
| METIS Edge-Wt.       |   1.277 |     1.061 |   332,610 |    23.7% |  24,633 |
| SFC Morton Wt.       |   1.179 | **1.002** |   429,904 |    30.6% |  32,871 |
| SFC Hilbert Wt.      |   1.137 | **1.002** |   713,500 |    50.7% |  62,425 |

## Analysis

### Simple = File-Order Block Partitioning

Simple partitions atoms by **their index in the file**, ignoring coordinates. For the CH4 data, molecules are written to the file on a grid, so adjacent indices are spatially close and k=2 works reasonably well; but as k increases, index order becomes uncorrelated with spatial position and edge imbalance degrades to 1.94x. For protein data, the file writes protein atoms first followed by water molecules, so k=2 happens to split into protein + water blocks (both spatially compact), but k>=4 arbitrarily slices through the water shell.

### KD-Tree = Recursive Spatial Bisection

KD-Tree is a purely spatial method: sort along the longest axis, split at the median, and recurse until k partitions are formed. It is essentially spatial sorting with adaptive axis selection. The difference from SFC:
- **SFC**: Fixed 3D-to-1D mapping (Morton uses bitwise interleaving), then linear splitting
- **KD-Tree**: Recursively bisects along the longest dimension; the split position adapts to data distribution

KD-Tree always cuts perpendicular to the longest axis, minimizing the cut surface area, which in turn minimizes edge cut and HaloNodes.

### METIS vs METIS Edge-Wt.

METIS uses no spatial coordinates at all; it operates solely on the **graph adjacency structure**:
1. Coarsening: Repeatedly merge adjacent nodes, shrinking the graph from N to a few hundred
2. Initial partitioning: Partition the coarsest graph
3. Uncoarsening: Restore to the original graph, refining boundaries at each level using Kernighan-Lin

**Default METIS** constrains atom balance but ignores edge balance, so for non-uniform density systems, EdgeImb can reach 1.82x.

**METIS Edge-Wt.** uses `vweights=per_atom_edge_count` to make METIS constrain edge balance, **achieving both low edge cut and low edge imbalance**. For lysozyme 8k at k=8:

| Method               | AtomImb | EdgeImb | EdgeCut% | HaloNodes |
|----------------------|--------:|--------:|---------:|--------:|
| METIS                |   1.002 |   1.797 |    21.0% |  10,846 |
| METIS Edge-Wt.       |   1.490 |   1.003 |    21.0% |  10,827 |
| SFC Morton Wt.       |   1.297 |   1.002 |    30.0% |  14,736 |

METIS Edge-Wt. and SFC Morton Wt. have comparable EdgeImb (~1.00), but METIS Edge-Wt. achieves **30% lower EdgeCut and 27% lower HaloNodes**.

### Key Findings

1. **METIS Edge-Wt. is the best overall method**: EdgeImb ~= 1.00 with the lowest EdgeCut/HaloNodes. The cost is requiring the full adjacency list and longer computation time.
2. **SFC Morton Wt.**: EdgeImb ~= 1.00, requires only coordinates (no graph structure), fast, suitable for real-time repartitioning.
3. **KD-Tree Spatial**: Excellent EdgeCut/HaloNodes, moderate EdgeImb (1.01-1.44), simplest implementation.
4. **Default METIS**: Lowest EdgeCut but poor EdgeImb; not recommended for non-uniform density systems.
5. **Morton consistently outperforms Hilbert** (edge cut ~2x lower), because the Hilbert implementation has limited resolution (order=7 vs 21).
6. **Edge imbalance degrades for all methods at k=16**; even METIS Edge-Wt. rises to 1.06.

### Method Selection Guide

| Scenario                           | Recommended Method  | Reason                                                    |
|------------------------------------|---------------------|-----------------------------------------------------------|
| Best overall performance           | METIS Edge-Wt.      | EdgeImb ~= 1.00 + lowest EdgeCut/HaloNodes               |
| No graph available / real-time repartitioning | SFC Morton Wt.      | Only needs coordinates, EdgeImb ~= 1.00, fast             |
| Communication bottleneck, edge balance not critical | KD-Tree Spatial     | Lowest EdgeCut and HaloNodes (among spatial methods)      |
| Small k (2-4 GPUs)                | METIS Edge-Wt.      | All metrics are optimal                                   |
| Large k (16+ GPUs)                | SFC Morton Wt.      | METIS Edge-Wt. EdgeImb also starts degrading to 1.06     |

## Data and Reproduction

```bash
# Prepare solvated protein systems
python scripts/prepare_solvated_protein.py --target-atoms 8000 16000

# Run benchmarks (requires pip install pymetis to enable METIS)
python scripts/partition_benchmark.py --input data/1aki_8k_solvated.xyz --k 2 4 8 16
python scripts/partition_benchmark.py --input data/1aki_16k_solvated.xyz --k 2 4 8 16
python scripts/partition_benchmark.py --input data/ch4_8k_splited.xyz --k 2 4 8 16
```

CSV results: `results/partition_bench_*.csv`
