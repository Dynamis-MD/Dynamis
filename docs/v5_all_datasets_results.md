# v5 Distributed build_graph — Multi-Dataset Benchmark

Per-dataset timing of `DistGraphBuilder` (graph-aware distributed build_graph,
no NCCL halo exchange) across ch4, solvated proteins, membrane, and water.

**Date:** 2026-06-13 (re-run after the `distnnp_state` csrc consolidation: `graph/`+`partition/`+`io/` layout, `DistGraphBuilderV5` → `DistGraphBuilder`)
**Hardware:** 4× NVIDIA A100-SXM4-40GB on Delta `gpua060` (single node, gpuA100x4, NVLink NV4)
**Parameters:** `cutoff = 6.0 Å`, `ghost = 0.5 Å` (r+skin = 6.5), `refresh_interval K = 100`
**Methodology:** CUDA-event timing, median of timed iters (2 warmup + 3 for Mode A,
2 warmup + 10 for Mode B); distributed time is the **max-over-ranks** median
(true wall = slowest rank). Baseline = single-GPU `radius_graph_pbc` on rank 0.
**Script:** `DistNNP/benchmarks/dist_build_graph_benchmark.py --mode timing`
**Implementation:** Mode A steps 7 (halo extraction) and 8 (edge filter) run as
fused CUDA ops (`_impl.extract_halo_info` / `_impl.filter_edges_to_radius`).
**Correctness:** the distributed edge multiset is bit-exact vs the single-GPU
serial build — **9/9 datasets PASS at W=4, 0 duplicate edges** (this run; e.g.
9gmo_density = 132,316,209 edges identical to serial). The SFC weighted partition
uses an exact int64 prefix sum so boundary-atom ownership is deterministic across
ranks (`dist_build_graph_benchmark.py --mode verify`).

Columns:
- `A_ms` — Mode A (refresh): full pipeline incl. `all_reduce(degrees)` + SFC weighted partition.
- `B_ms` — Mode B (steady): zero-NCCL rebuild reusing the cached partition/halo plan.
- `amort` — amortized over K=100: `(A + 99·B) / 100`.
- `base_ms` — single-GPU `radius_graph_pbc` (whole graph on 1 GPU).
- `speedup` — `base_ms / amort` (distributed build vs single GPU).
- `int%` — interior fraction of rank-0's outer atoms (atoms whose full neighborhood
  is local → no halo dependency). Higher = better halo efficiency.
- `peakGB` — peak GPU memory of the **distributed build**, max over ranks,
  captured *before* the rank-0 single-GPU baseline runs (so it is the v5
  partition's true per-GPU footprint, not the full-graph baseline's).

---

## 1. ch4 Lattice Sweep (8k → 1024k), W=4

Uniform methane lattice, near-cubic boxes. `splited` partitioning.

```
   N        box(Å)   A_ms    B_ms   amort   base_ms  speedup   r0_outer  r0_halo  int%   peakGB
   8001       60     4.46    1.47    1.50     1.26     0.84x       1981     1018   52%    0.01
  16002       76     4.34    1.48    1.51     1.40     0.93x       3979     1192   71%    0.02
  32004       95     4.29    1.55    1.58     1.72     1.09x       8042     2400   75%    0.03
  63999      120     4.55    1.72    1.75     2.23     1.28x      16054     4238   76%    0.06
 127998      151     4.82    2.05    2.08     3.22     1.55x      31930     6490   81%    0.10
 255996      190     5.76    2.60    2.63     5.37     2.04x      63918    10068   85%    0.20
 512001      240     9.06    4.38    4.43    11.92     2.69x     127988    15764   88%    0.38
1024002      302    11.82    6.00    6.06    18.01     2.97x     255965    24929   90%    0.75
```

**Trends**
- **Speedup grows monotonically with N:** <1× below ~32k → **2.97× (1M)** on 4 GPUs
  (0.84× at 8k, 0.93× at 16k, 1.09× at 32k). Crossover (speedup > 1) is at ~32k.
- **interior% climbs 52% → 90%** as the box grows. Larger box ⇒ smaller
  surface-to-volume ratio ⇒ relatively fewer halo atoms ⇒ higher distributed
  efficiency. This is the root cause of the size-dependent speedup.
- **Mode B is ~2–3× faster than Mode A** (the steady-state path skips the
  `all_reduce` + SFC partition that Mode A pays on refresh).
- **Memory is linear in N**, only 0.75 GB/rank at 1M (W=4) — the distributed
  build's per-GPU footprint is tiny, far from the 40 GB ceiling, leaving huge
  headroom for larger systems or fewer GPUs.

---

## 2. Mixed Datasets (proteins / membrane / water + ch4), W=4

Realistically packed solvated systems span a range of densities and box shapes.

```
system                    N       box(Å)   A_ms    B_ms   amort   base_ms  speedup  int%   peakGB
ch4_8k_splited            8001     60       4.21    1.46    1.48     1.46    0.99x    52%    0.01
1aki_density             25846     62       4.61    1.63    1.66     1.80    1.09x    36%    0.05
pt_water_17k_density     21180     67       4.32    1.59    1.62     1.70    1.05x    35%    0.05
dppc_membrane_28k        28160     56       4.47    1.63    1.66     1.86    1.12x    42%    0.06
2rik_density            112727     73       5.31    2.26    2.29     3.72    1.62x    53%    0.19
ch4_128k_splited        127998    151       5.57    2.40    2.43     3.97    1.64x    81%    0.10
ch4_512k_splited        512001    240       9.10    4.41    4.45    11.87    2.67x    88%    0.38
9gmo_512k_solvated      511998    283       9.15    4.45    4.50    11.42    2.54x    84%    0.69
9gmo_density           1536442    229      25.37   13.81   13.92    36.60    2.63x    80%    2.32
```

**Topology matters as much as N**
- At comparable N, **ch4 (regular lattice) beats solvated proteins** on
  interior%: ch4_128k = 81% vs 2rik_density (113k) = 53%. Denser, irregular
  protein/water packing pulls more atoms into the halo region.
- Small dense systems (1aki, pt_water, dppc at 20–28k) sit at speedup ≈ 1×:
  the fixed `all_reduce(degrees)` + SFC partition cost dominates at low N
  regardless of topology.
- The largest system, **9gmo_density (1.54M), reaches 2.63×** at just 2.32 GB/rank.
  The regular ch4 1M lattice does better (2.97×): irregular solvated packing caps
  interior% at 80% and makes Mode B costlier per edge, so the dense 1.5M box lands
  below the uniform 1M lattice despite its larger N.

---

## 3. Takeaways

1. **v5 distributed build_graph pays off at scale, not at small N.** The
   crossover is ~32k for uniform ch4 and higher for dense proteins. Below that,
   the per-refresh collective + partition overhead exceeds the work saved by
   splitting.
2. **interior% is the single best predictor of distributed efficiency** — it
   captures the surface-to-volume effect (grows with box size) and the topology
   effect (regular lattice > dense solvated). Speedup tracks interior% closely.
3. **Mode B dominates the amortized cost** (99 of every 100 steps). Its ~2–3×
   edge over Mode A is why the dual-mode design matters for MD, where positions
   drift slowly and a partition stays valid for ~K steps.
4. **Memory headroom is large** — the distributed build's per-GPU footprint is
   small (1.5M atoms in ~2.3 GB/rank, 1M ch4 in 0.75 GB/rank at W=4), leaving
   ample room for bigger systems or fewer GPUs. (Earlier drafts conflated this
   with the single-GPU baseline's full-graph memory, which is ~4× larger.)

---

## 4. Reproduction

`benchmarks/dist_build_graph_benchmark.py` is the single entry point (modes: `timing` [this doc],
`verify`, `sizes`, `diag`); it auto-detects torchrun vs SLURM srun.

```bash
# Interactive 4-GPU node (NCCL modules already loaded):
cd DistNNP
torchrun --standalone --nproc_per_node=4 benchmarks/dist_build_graph_benchmark.py --mode timing

# Full ch4 ladder only:
torchrun --standalone --nproc_per_node=4 benchmarks/dist_build_graph_benchmark.py --mode timing \
  --systems ch4_8k_splited,ch4_16k_splited,ch4_32k_splited,ch4_64k_splited,\
ch4_128k_splited,ch4_256k_splited,ch4_512k_splited,ch4_1024k_splited

# Any subset:
torchrun --standalone --nproc_per_node=4 benchmarks/dist_build_graph_benchmark.py --mode timing \
  --systems 9gmo_density,2rik_density

# Correctness (distributed == serial) / size sweep / one-dataset edge diff:
torchrun --standalone --nproc_per_node=4 benchmarks/dist_build_graph_benchmark.py --mode verify
torchrun --standalone --nproc_per_node=2 benchmarks/dist_build_graph_benchmark.py --mode sizes
torchrun --standalone --nproc_per_node=4 benchmarks/dist_build_graph_benchmark.py --mode diag --system 9gmo_512k_solvated
```
