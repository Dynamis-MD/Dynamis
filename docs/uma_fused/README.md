# Fused UMA-s path (`dynamis_model.nnp.uma_fused`)

Fused CUDA / CuTe-DSL kernels for the edge-wise blocks of fairchem's UMA-s (`uma-s-1p2`) and the integration that runs
the whole UMA call (forward + forces) as one CUDA graph. Inference only, Blackwell (sm_100: B200 / GB200) only.

```python
from dynamis_model.nnp import UMAModel
model = UMAModel("uma-s-1p2", dataset="omol", device="cuda", inference_settings=settings,
                 fused=True, fused_graph=True)        # or env DYNAMIS_UMA_FUSED=1
```

`UMAModel(fused=False)` (the stock fairchem path) is the fallback on every other GPU and the reference for debugging
the physics. `enable_fused` raises a clear error on a GPU that is not sm_100.

## 1. What is replaced

| fairchem module (per UMA-s backbone) | replaced by | file |
|---|---|---|
| `Edgewise` (4 blocks): rotate, SO(2) conv1, gate, conv2, rotate back, scatter | one custom op `dynamis::uma_edge_block` (+ `_bwd`) per block | `edge_block.py` + kernels below |
| `UnifiedRadialMLP` last Linear (fc3, 128 -> 1536) | folded into the edge op (K0r computes the radial in SMEM / TMEM) | `fairchem_patches.patch_unified_radial_hidden` |
| `EdgeDegreeEmbedding` scatter | fused Triton gather / scatter (no [E, 9, C] intermediate) | `ede_fused.py` |
| l = 2 Wigner einsum (a TF32 GEMM under `allow_tf32`) | exact fp32 Triton fwd / bwd custom op `dynamis_fused::wigner_l2` | `wigner_l2.py` |
| RMS-norm SH, SpectralAtomwise gate | broadcast form / 2-D GEMM | `fairchem_patches.py` |
| radial MLP fc1 / fc2, EDE rad_func, node FFN GEMMs | FP16 operands, fp32 accumulation | `fp16_patches.py` |
| gate expansion | deterministic (no atomics in the backward) | `nnp/det_ops.py` |
| whole `UMAModel.__call__` predict | one CUDA graph, edges padded to a capacity | `graphed_predict.py`, `dynamis_fused.graphed_uma_call` |

## 2. The edge-block op

Contract (`FusedEdgeBlock.op(x, edge_index, wigner, wigner_inv_env, h)`):
* `x` [N, 9, 128] fp32 (N may include ghost rows), `edge_index` [2, E] grouped by target (contiguous segments, any
  segment order; the target is the LOCAL row of x), `wigner` / `wigner_inv_env` [E, 9, 9] fp32 block diagonal
  (`inv_env` = envelope * D^T), `h` [E, 128] = the radial MLP's last hidden layer.
* returns [N, 9, 128] fp32 messages (exactly zero rows for nodes that are no edge's target);
  backward: dx, d wigner, d wigner_inv_env (block-diagonal entries, others exactly zero), dh.
* UMA-s shapes only (lmax = mmax = 2, C = H = 128), frozen weights, a target may have at most 128 edges.

Math per edge (fp64 ground truth: `reference.edge_block_reference`): rotate `y = D x[src|tgt]`, permute L -> M rows,
scale by the radial (`h W_last^T + b_last`), conv1 (three m-block GEMMs + bias), gate (SiLU on M0, sigmoid gates on
M1..M8), conv2, permute back, rotate with `Dinv_env`, sum into the target.

Tiles: edges are packed into 128-edge tiles holding whole target segments of at most 7 targets (`MAX_SLOTS`), filled
greedily from a look-ahead window of 32 segments (94-97 % fill). Packing runs on the GPU without host sync
(`gpu_pack.py`, chunked greedy, 1024 segments per warp). Padding slots carry `Dinv = 0` and contribute nothing in the
forward and backward.

Kernels (one CUDA graph per block for the forward, one for the backward):

| stage | kernel | what |
|---|---|---|
| K0r | `k0r_cute.py` (CuTe, tcgen05) | radial = h W_last^T + b (TMEM), `A = radial * D x[src|tgt]` in M-row order, FP16 [Ep, 2304]; also writes the FP16 packed radial for S3 |
| K1 | `k1_cute.py` (CuTe, paired-CTA tcgen05) | conv1 GEMMs + bias + sigmoid epilogue -> saved `zp` [Ep, 1152], `g` [Ep, 256] FP16, tile-interleaved layout |
| K2 | `k2_cute.py` (CuTe, tcgen05 TS MMAs) | `z = act(zp) * g`, conv2, inverse rotation + node reduction as an MMA (a target lives in one tile: no atomics) |
| S0 + S1 | `s1_mma.py` (Triton prep + cuBLAS + generated CUDA, mma.sync) | per node `dOut . V`; per edge dz -> dh (dgate, dzp) and dDinv |
| S2 | cuBLAS FP16 | `da = dh . W` (3 GEMMs, reduced-precision reduction off) |
| S3 | `node_geo_cu.py` (generated CUDA) | dx, dD, d radial (FP16, packed) node-parallel over a CSR; `dh = d radial . W_last` (Triton, `edge_block.dh_gemm`) |

Generated CUDA (`gpu_pack`, S1, S3) is built with nvcc on first use (`common.nvcc_build`: cache key = source + nvcc
command line + nvcc version, private temp files + atomic rename, so concurrent ranks may build at once). `nvcc` must be
on `PATH`; CUDA headers are taken from `$CUDA_HOME/include` (or `$CUDA_PATH/include`) and, in a conda environment, from
`$CONDA_PREFIX/targets/<sbsa|x86_64>-linux/include`. Builds are cached in `<tempdir>/uma_fused_<uid>/` (`$TMPDIR`).
CuTe kernels are compiled per shape (`common.KERNELS`, shared by the 4 blocks; first compile ~10-40 s each).

## 3. Capacity mode and CUDA graphs

All kernel shapes and buffer addresses depend on a capacity, not on the current graph: E_cap = 1.15 E, N_cap = 1.15 N
(`DYNAMIS_UMA_CAP_HEADROOM`, `DYNAMIS_UMA_CAP_N_HEADROOM`; uma.py raises the node headroom to 1.5 for multi-GPU
padded halos), tile capacity = 1.15 x the measured tile count. A new neighbour list (`dynamis_fused.new_topology()`
bumps an epoch; otherwise detected by tensor identity) is re-packed in place on the GPU; the block graphs are replayed
unchanged. E > E_cap or N > N_cap re-allocates (new generation: re-compile + re-capture). Tile-capacity overflow and
too-high degrees are flagged on the device and checked lazily at the next rebuild (`OverflowError` / `ValueError`).
The packed Wigner matrices are shared by the 4 blocks of a predict (`_cap_geo`).
Inside an outer capture (`graphed_predict`, whole-call graph) the blocks record their kernels directly.
Large systems (> 1M packed edges) share one graph memory pool (`DYNAMIS_UMA_SHARED_POOL`).

## 4. Precision

FP16 operands with fp32 accumulation everywhere in the edge block (same mantissa width as TF32);
sigmoid / SiLU, rotations, reductions and the radial bias in fp32.
* Op vs fp64 reference (synthetic weights, `tests/uma_fused/test_uma_fused_edge_block.py`): output 6.3e-4,
  dx 7.3e-4, dD 7.4e-4, dDinv 7.4e-4, dh 7.1e-4 relative.
* Whole model, CH4/O2 mixed (W = 1, fused graph, FP64 energy head) vs an FP32 reference: forces 1.48e-3 relative
  (2k and 16k), energy -0.14 eV (2k) / -1.04 eV (16k, -6.5e-5 eV/atom). Stock TF32 path: forces 1.4-1.5e-3.
* Wigner l = 2: exact fp32 (max abs 1e-7 vs fp64); EDE: fp32 FMAs (1e-7).

## 5. Determinism

No atomics and fixed reduction orders in every kernel -> the op is bitwise reproducible (tested). The whole model is
bitwise reproducible run to run and across processes when the torch.compile / inductor cache is shared
(`dynamis_model/compile_cache.py`, default `~/.cache/dynamis/inductor`): inductor picks Triton reduction configs by
timing, and two processes that autotune independently differ by ~5e-4 in the forces. Do not let processes that
compile a different structure of the same size class populate the cache concurrently with a production run.
`configure_compile_cache()` also turns off inductor's `dynamic_scale_rblock`: it picks between two reduction block
sizes by timing at the first launch of each process, uncached, so processes could sum in different orders (forces
2.5e-4 apart at 2k atoms); `DYNAMIS_INDUCTOR_SCALE_RBLOCK=1` restores torch's default.

## 6. Switches (defaults in brackets)

| env | effect |
|---|---|
| `DYNAMIS_UMA_FUSED` [0] | `UMAModel(fused=...)` default |
| `FP16_PARTS` [`all,wig32`] | FP16 non-edge GEMMs + exact Wigner; `off` = fairchem precision outside the edge op |
| `DYNAMIS_DET_GATE` [1] | deterministic gate expansion |
| `DYNAMIS_UMA_CAP_HEADROOM` / `_CAP_N_HEADROOM` [1.15] | edge / tile / node capacity headroom |
| `DYNAMIS_UMA_SHARED_POOL` [auto] | one graph pool for all blocks above `DYNAMIS_UMA_SHARED_POOL_MIN_EP` [1e6] packed edges |
| `DYNAMIS_UMA_GRAPH_PAD` [1.01] | whole-call graph: edge padding factor |

`enable_fused` warns when an environment variable of a removed kernel variant is set (`RAD_FOLD`, `DYNAMIS_UMA_K1`,
`S1_IMPL`, `DYNAMIS_UMA_PACK_CHUNK`, ...); these have no effect.

## 7. Performance (GB200, one GPU, CH4/O2 mixed, `benchmarks/uma_fused/dist_bench.py --fused --graph`)

| system | E | ms / call (fused graph) | stock UMAModel (turbo) |
|---|---|---|---|
| 2k atoms | 69k | 12.1-13.1 | ~23 |
| 16k atoms | 558k | 50.1-52.6 | - |
Per-call time includes the neighbour list and host work (median of 20 calls; +-1 ms with other jobs on the node);
the graph replay alone is ~7.3 ms (2k) / ~44.6 ms (16k). K2 alone (16k-like synthetic graph, 4,877 tiles): 0.99 ms.

## 8. Tests

`dynamis_model/tests/uma_fused/` (skipped without an sm_100 GPU):
* `test_uma_fused_edge_block.py`: op fwd + bwd vs fp64 autograd on synthetic weights / graphs, bitwise repeat,
  neighbour-list change within capacity (epoch and identity), capacity growth, degree > 128 error.
* `test_uma_fused_small_kernels.py`: Wigner l = 2 and EDE vs fairchem in fp64, GPU packer validity and flags.
* `test_uma_fused_model.py`: `UMAModel` fused graph vs stock on `ch4_2k_splited.xyz` over neighbour-list changes,
  bitwise repeat (`DYNAMIS_DATA_DIR`, default `experiments/exp_datasets/baselines`).

## 9. Files (`dynamis_model/src/dynamis_model/nnp/uma_fused/`, lines)

| file | lines | role |
|---|---:|---|
| `common.py` | 103 | constants, M-row tables, CuTe compile cache, `set_regs`, `ilv_alloc`, `nvcc_build` |
| `reference.py` | 143 | fp64-capable reference of the edge block (`edge_block_reference`, `EdgeBlockWeights`) |
| `gpu_pack.py` | 251 | GPU tile packer (chunked greedy) + CSR, `GPUTopology` |
| `ede_fused.py` | 148 | fused edge-degree-embedding scatter (Triton) |
| `wigner_l2.py` | 302 | exact fp32 l = 2 Wigner (Triton fwd / bwd, custom op) |
| `k0r_cute.py` | 296 | K0r (CuTe): radial GEMM + rotation -> A, packed radial |
| `k1_cute.py` | 346 | K1 (CuTe, 2-CTA): conv1 GEMMs + bias + gate -> zp, g |
| `k2_cute.py` | 588 | K2 (CuTe): gate, conv2, inverse rotation, node reduction |
| `s1_mma.py` | 456 | S0 (Triton + cuBLAS) + S1 (generated CUDA, mma.sync) |
| `node_geo_cu.py` | 251 | S3 (generated CUDA): dx, dD, d radial |
| `edge_block.py` | 399 | `FusedEdgeBlock`: capacity state, CUDA graphs, custom ops, backward glue |
| `dynamis_fused.py` | 393 | `enable_fused`, UMA adaptor, skin neighbour list, whole-call graph entry |
| `fairchem_patches.py` | 153 | radial fold, RMS-norm / gate / EDE patches of the fairchem backbone |
| `fp16_patches.py` | 151 | FP16 GEMMs outside the edge op, exact Wigner switch |
| `graphed_predict.py` | 343 | whole-predict CUDA graph with padded edges |

