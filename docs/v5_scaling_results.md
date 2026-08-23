# v5 Distributed build_graph — Graph-aware halo (9GMO 1.5M)

**Branch**: `dist_build_graph` @ commit `9252cb0`
**Date**: 2026-05-22
**Test system**: 9GMO density, N = 1,536,442 atoms, box ≈ 229 × 233 × 291 Å
**Hardware**: Delta gpuA100x4 (4× A100-SXM4-40GB / node, NV4 NVLink)
**Cutoff**: 6.0 Å, ghost_buffer (skin): 0.5 Å, refresh_interval K = 100, SFC = Morton

---

## 1. What's new in v5

v5 is a new distributed graph builder that produces a **graph-aware halo** —
each halo atom is guaranteed to have a real edge to the local partition,
matching the semantics of `main::distnnp_state/src/partition/build_halo_maps.cpp`
but **distributed** (per-rank work, no full-N replicated graph).

### Design differences vs v4

| | v4 | v5 |
|---|---|---|
| Inner partition | RCB (recursive coordinate bisection) | **cell-block** (blockwise n_cells/W per rank) |
| Inner halo | explicit `dist_build_inner_source_halo_map` | **none** (count-only on full pos, no halo gather) |
| Outer halo @ r+skin | **geometric cell-list** (over-provisioned 3-10×) | **graph-aware** (exact, from edges) |
| Mode A edge build | inline B at r on (my_outer ∪ geom_halo) | **single edge build at r+skin on (full pos)** |
| Halo send/recv plan | from geometric halo + custom CUDA pipeline | **local symmetry of edges** (no NCCL exchange) |
| `interior_mask` | not produced | **produced** (drop-in for `main`'s `PartitionMessagePassingInfo`) |
| NCCL collectives in Mode A | 1× `all_reduce(deg[N])` | **1× `all_reduce(deg[N])` only** (same) |

### Files added (commit 9252cb0)

```
distnnp_state/src/distnnp_state/dist_build_graph_v5.py    (414 lines)
distnnp_state/tests/test_dist_v5_correctness.py           (~530 lines)
benchmarks/bench_v5_w2_9gmo.py                            (W=2 local PoC)
benchmarks/bench_v5_scaling.py                            (W=2/4/8/16 v4 vs v5)
benchmarks/run_v5_scaling_w{2,4,8,16}.sh                  (sbatch + bad-node exclude)
```

### Algorithm (Mode A)

```
1. PBC wrap                                                   ~0.1 ms
2. cell-block partition (n_cells/W per rank, deterministic)   ~0.05 ms
3. count-only @ r on full pos, target = atoms in my cells     ~2 ms
4. all_reduce(degrees) → full_degrees[N]                      0.2 / 2.5 ms (intra/inter)
5. SFC weighted partition → pid_outer, my_outer_atoms         ~1 ms
6. edges_at_r_skin = radius_graph_pbc_ex(
       pos, cell, pbc, r+skin, target=my_outer)               ~6-50 ms (W-dependent)
7. local extract (NO NCCL — geometric symmetry):
   - src,dst = edges_at_r_skin  (dst ∈ my_outer guaranteed)
   - boundary = pid_outer[src] != my_rank
   - RECV: unique(src[boundary]) grouped by peer, sorted by atom-id
   - SEND: unique(dst[boundary]) grouped by peer, sorted by atom-id
   - interior_mask = ~scatter(my atoms with any remote src)
                                                              ~0.5 ms
8. Output edges_at_r (filter r+skin → r by distance)          ~0.5 ms
```

### Why no NCCL exchange in step 7

`radius_graph_pbc_ex` is geometrically symmetric: `dist(a, b) < r+skin ⟺
dist(b, a) < r+skin`. For each boundary edge `(src, dst)` with `dst ∈
my_outer`, the edge encodes both:
- **RECV side**: I need `src` from `pid_outer[src]`
- **SEND side**: I send `dst` to `pid_outer[src]`

Rank A's local edge `(b, a)` is the mirror of rank B's local edge `(a, b)`.
Both ranks arrive at the same global send/recv plan from their own local
edges, with no rank-to-rank communication.

**Crucial fix (Fix 1)**: both sides must sort send/recv lists by global atom
ID within each peer segment so sender and receiver agree on order. v5
implements this via composite key `peer * N + atom_id` then sort+unique.

---

## 2. Correctness verification

All tests passing on local 2× A100 (gpua064/gpua079).

### Test 1 — W=1 single-rank smoke
- All atoms ∈ my_outer
- halo empty, send/recv CSR empty
- interior_mask all True

### Test 2 — W=2 vs single-GPU oracle
The oracle mirrors `main::build_halo_maps.cpp` line-by-line. v5's per-rank
output is **bit-equal** to oracle's per-rank slice:
- `send_offsets`, `recv_offsets`: bit-equal
- `send_local_indices`: bit-equal (Fix 1 + Fix 2 verified)
- `recv_ghost_indices`: bit-equal
- `halo_recv_global_ids`: bit-equal
- `interior_mask`: bit-equal

### Test 3 — W=2 mirror-symmetry
Rank 0's `send_to_1[k]` ≡ rank 1's `recv_from_0[k]` as global atom IDs.
This is the empirical proof that Fix 1's global-ID sort makes the no-NCCL
send-plan derivation correct.

### Test 4 — Mode B consistency + drift watchdog
- Mode A and Mode B on same pos produce identical edges
- Small drift (uniform |d| ≤ 0.1 Å < skin/2) stays in Mode B
- Large drift (1 Å > skin/2) triggers Mode A refresh

Run with: `pytest distnnp_state/tests/test_dist_v5_correctness.py -v`

---

## 3. Performance results — 9GMO 1.5M

### 3.1 Full mode breakdown (ms)

| W | v4 Mode A | v5 Mode A | v4 Mode B | v5 Mode B | v4 Mode A − B | v5 Mode A − B | v4 Amort@K100 | v5 Amort@K100 | speedup |
|---|---|---|---|---|---|---|---|---|---|
| 2  | 34.85 | 62.83 | 22.94 | 22.46 | 11.91 | **40.37** | 23.06 | 22.86 | 1.01× |
| 4  | 23.47 | 37.57 | 11.91 | 11.81 | 11.56 | **25.76** | 12.03 | 12.07 | 1.00× |
| 8  | 19.72 | 20.62 |  8.25 |  **6.63** | 11.47 | **13.99** |  8.36 |  **6.77** | **1.23×** |
| 16 | 18.20 | **17.55** |  4.88 |  4.74 | 13.32 | **12.81** |  5.01 |  4.87 | 1.03× |

W=2 from local test on gpua079 (torchrun); W=4/8 from jobs 18354714/18354715;
W=16 from job 18400733. W=2/4/8 used the previous (Hilbert default) v4
baseline; W=16 uses Morton-default v4. All v5 runs use Morton.

### 3.2 Halo size comparison (rank 0)

| W | v4 N_my_halo (geometric r+skin) | v5 N_my_halo (graph-aware) | **reduction** | v5 interior atoms |
|---|---|---|---|---|
| 2  |   181,421 |  67,373 | **2.69×** |  702,257 / 769,910  (**91%**) |
| 4  |   447,919 |  74,159 | **6.04×** |  307,356 / 383,826  (80%) |
| 8  |   638,266 |  59,733 | **10.69×** |  136,694 / 191,162  (72%) |
| 16 |   361,927 |  39,109 | **9.25×** |   60,843 /  94,461  (64%) |

Graph-aware halo is **2.7-10.7× smaller** than v4's geometric r+skin halo,
with the ratio growing through W=8 then plateauing.

### 3.3 v5 Mode A − Mode B (pure refresh overhead)

This is the work v5 does **beyond** producing Mode B's edges:
- Builds halo + send/recv CSR + interior_mask
- Replaces v4's RCB + inner halo + outer halo + count-only chain

| W | v4 refresh overhead | v5 refresh overhead | v5 − v4 (ms) | v5 / v4 |
|---|---|---|---|---|
| 2  | 11.91 | 40.37 | +28.46 | 3.39× |
| 4  | 11.56 | 25.76 | +14.20 | 2.23× |
| 8  | 11.47 | 13.99 |  +2.52 | 1.22× |
| 16 | 13.32 | **12.81** |  −0.51 | **0.96×** ✓ |

**v5 refresh overhead drops below v4 at W=16**, because v4's geometric-halo
construction scales with W (more peer ranks to track) while v5's single
edge-build on full pos is cheap when my_outer is small.

### 3.4 Why W=2 v5 Mode A is slow (+28 ms)

Two compounding factors at small W:

1. **r+skin (6.5 Å) vs r (6.0 Å)** — neighbor candidate volume +27%.
   v5 must build at r+skin to support K-step amortization (the cached halo
   must cover all atoms that could become neighbors within K-1 drift steps).
2. **Cell-list built on full N pos (1.5M)** vs v4's inline-B subset
   (~951K = my_outer ∪ geom_halo). +58% atoms in cell-list construction.

At W=2 my_outer = 770K (half the system), so the kernel scans many targets
× many candidates. At W≥8, my_outer is small and the absolute kernel cost
is dominated by overhead, masking the difference.

K=100 amortized impact: W=2 +28 ms / 100 = **+0.28 ms/step** — vs Mode B's
22.5 ms/step (1.2%). Negligible in practice.

### 3.5 Why Mode B speedup peaks at W=8

| W | v4 my+halo (kernel input) | v5 my+halo | input reduction | Mode B speedup |
|---|---|---|---|---|
| 2  |   951K |   837K | 1.14× | 1.02× |
| 4  |   832K |   458K | 1.82× | 1.01× |
| 8  |   829K |   251K | **3.30×** | **1.24×** ✓ |
| 16 |   457K |   133K | 3.43× | 1.03× |

W=8 hits the sweet spot — input reduction is large AND kernel work is still
big enough to benefit. By W=16, the kernel is small enough (~4.8 ms) that
fixed overhead (cell-list build, launch, sync) dominates, masking the 3.4×
input reduction.

---

## 4. Real value of v5 (beyond build_graph timing)

The Mode-A/Mode-B kernel timings only tell part of the story. v5's
graph-aware halo unlocks several downstream wins that will surface once
v5 is integrated into the GNN forward pass:

### 4.1 `interior_mask` exposes massive feature-exchange savings

| W | interior atoms | feature exchange skipped |
|---|---|---|
| 2  | 702K / 770K | **91%** |
| 4  | 307K / 384K | 80% |
| 8  | 137K / 191K | 72% |
| 16 |  61K /  94K | 64% |

For ESCN-class NNPs with ~200 float embeddings per atom, this is the
difference between exchanging all my_outer features each layer vs only
boundary atoms. At W=16: 94K × 200 × 4B = 75 MB per layer (v4) → 34K ×
200 × 4B = 27 MB (v5), a **2.8× cut** in cross-layer exchange.

### 4.2 Halo recv volume reduction

The other half of feature exchange — atoms received as ghosts. v4 receives
the entire geometric halo each layer; v5 receives only the graph-aware
subset (3-10× fewer).

W=16: 362K × 200 × 4B = 290 MB ghost exchange (v4) → 39K × 200 × 4B = 31
MB (v5), a **9.4× reduction**.

### 4.3 Alignment with `main::build_halo_maps`

v5's `halo_info` (PartitionMessagePassingInfo) fields are bit-equal to
the serial reference at correctness-test scale (N=1000-1500, W=2). This
means v5 is a **drop-in replacement** for `main`'s halo construction in
any downstream code expecting that struct, with the distributed
implementation underneath.

### 4.4 Scales to 100M (no cell-mask OOM)

v4's `rank_cells_mask[n_cells × W]` is a dense bool tensor used in the
geometric halo CUDA pipeline. At 100M @ W=128 this is ~256 MB per rank
before any timing matters. v5 uses no such per-rank-W-channel tensor —
its memory footprint is per-rank atoms (N/W) plus edges (~N/W × ~50).

---

## 5. Reproduction

```bash
cd /projects/bcpt/hqin1/git/DistNNP/DistNNP

# Correctness tests (single node, ≥ 2 GPUs)
pytest distnnp_state/tests/test_dist_v5_correctness.py -v

# Scaling benchmark (compares v4 + v5)
sbatch benchmarks/run_v5_scaling_w2.sh    # 1N × 2 GPU
sbatch benchmarks/run_v5_scaling_w4.sh    # 1N × 4 GPU
sbatch benchmarks/run_v5_scaling_w8.sh    # 2N × 4 GPU
sbatch benchmarks/run_v5_scaling_w16.sh   # 4N × 4 GPU

# Collect results
grep '\[bench\]' benchmarks/logs/v5_w*_*.out
```

### Bad-node exclude list

`gpua010`, `gpua012`, `gpua022` consistently hang silently after the SLURM
banner (no NCCL/Python output, killed by time limit). Pre-emptively
excluded via `#SBATCH --exclude=gpua010,gpua012,gpua022` in all v5 sbatch
scripts.

W=16 first attempt (job 18354716) hit a TCPStore rendezvous timeout
(`12/16 clients joined`); resubmit usually works (job 18400733 ran
cleanly).

---

## 6. Open work

1. **Integration into `dist_md.py` / GNN forward pass** — unlock §4 wins.
   Requires routing `interior_mask` into the feature-exchange path and
   using `halo_info`'s graph-aware send/recv CSR.
2. **W=2 Mode A optimization** — add a quick geometric r+skin halo
   bootstrap to limit step-6 edge build to a subset; saves ~20 ms at W=2.
   K=100 amortized impact is small (~0.2 ms/step), so deferred.
3. **W=32 / W=64 runs** — find the asymptotic ceiling.
4. **100M dataset testing** — validate cell-mask-OOM hypothesis.
