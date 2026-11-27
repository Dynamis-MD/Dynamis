# Dynamis Multi-GPU Scaling Experiment Plan

This document specifies **how to configure and run the shared Zenodo benchmark
datasets** ([record 24487955](https://zenodo.org/records/24487955)) across all
Dynamis NNP backends (**ORB**, **UMA**, **MACE**, and future models), what has
already been completed, and what remains to be collected.

The goal is **apples-to-apples comparability** across models while accounting
for real per-model memory differences (e.g., conservative/autograd models like
MACE and UMA hold fewer atoms per GPU than Orb-Direct).

---

## 1. General Scaling Strategy (Model-Agnostic)

Every model/backend should follow the **same four-stage methodology** on each
dataset, adapting only the per-GPU atom budget to the model's measured 1-GPU
HBM ceiling $N_{\max}^{(1)}$:

1. **Stage 0 — Preflight Gate (Parity + 1-GPU HBM Ceiling + MNNVL Check)**
   - Verify distributed force/energy parity ($W=2$) against single-GPU execution.
   - Sweep ascending structure sizes on $W=1$ GPU to find the model's HBM ceiling
     $N_{\max}^{(1)}$ (recording the first OOM).
   - On 2 nodes ($W=8$), verify `GPUDirectExchanger` activates MNNVL CUDA Fabric
     handles (`use_fabric=True`).
2. **Stage 1 — Weak Scaling (Fixed Atoms/GPU Ladder, $W = 1 \to 32$)**
   - Select the largest ladder density $N_0 \le N_{\max}^{(1)}$ with ~20–30% HBM
     headroom for multi-GPU halo growth, and scale $N(W) \approx W \cdot N_0$
     across $W \in \{1, 2, 4, 8, 16, 32\}$ GPUs.
   - If the smallest file in a fixed ladder (e.g., `li3po4_32032.xyz`) exceeds a
     model's 1-GPU ceiling, anchor the weak ladder at the smallest feasible GPU
     count $W_{\min}$ (e.g., $W=2 \to 32$ at ~16K atoms/GPU).
   - Report both raw efficiency $T_{W_{\min}} / T_W$ and **atom-normalized
     efficiency** (accounting for slight supercell count variation in crystal
     ladders like Li₃PO₄).
3. **Stage 2 — Two Strong Scaling Regimes ($W_{\min} \to 32$)**
   - **Compute-bound strong scaling (million-atom regime):** Fix the largest
     standard structure (`1.02M` CH₄:O₂, `1.01M` Li₃PO₄, or `1.15M` solvated
     2W49) and scale from the **smallest GPU count $W_{\min}$ that fits in HBM**
     up to $W=32$. Compute speedup and parallel efficiency relative to $W_{\min}$:
     $$\text{Speedup}(W) = \frac{\text{sps}(W)}{\text{sps}(W_{\min})}, \qquad
       \text{Eff}(W) = \frac{\text{Speedup}(W)}{W / W_{\min}}.$$
     *(Example: edge-chunked BF16 Orb-Direct fits `1.01M` Li₃PO₄ at $W_{\min}=1$,
     unchunked FP32 Orb-Direct fits `1.02M` CH₄:O₂ at $W_{\min}=4$, and heavier
     UMA/MACE configurations may start at $W_{\min}=4$ or $8$.)*
   - **Latency-bound strong scaling (small-system regime):** Fix a smaller
     structure that fits on $W=1$ GPU (target `64K` on CH₄:O₂ or `61K` on Li₃PO₄;
     drop to `32K` if `61K`/`64K` OOMs on 1 GPU) and scale $W = 1 \to 32$ to
     deliberately expose the halo-ratio and kernel-launch latency knee.
4. **Stage 3 — Hero Scale ($W = 32$ GPUs / 8 Nodes)**
   - Run the largest Zenodo structure that fits within 32-GPU HBM (e.g., `10.0M`
     atoms on `ch4_o2` for Orb-Direct `k=20`, or `2.05M`/`4.10M` CH₄:O₂ for
     higher-memory models).

---

## 2. Shared Control Variables (Hold Constant Across All Models)

Keep the following simulation, cluster, and instrumentation settings identical
across models unless a deliberate ablation is being tested:

| Parameter | Standard Value | Notes |
|-----------|----------------|-------|
| Hardware | NVIDIA GB200 NVL72 (4× B200 / node, 184 GiB HBM3e) | Always allocate all 4 GPUs/node (`--gpus-per-node=4 --mem=0 --cpus-per-task=32`) |
| Topology | `--segment=<nodes>` (single NVL72 rack block, $N \le 18$) | Enforced by `scripts/sbatch_submit.sh` |
| GPU ladder | $W \in \{1, 2, 4, 8, 16, 32\}$ (1, 1, 1, 2, 4, 8 nodes) | Run $W \in \{1, 2, 4\}$ sequentially in one 1-node allocation |
| Exchanger | `gpudirect` (primary); optional `nccl` comparison | Verify `use_fabric=True` for all $W \ge 2$ `gpudirect` runs |
| Partitioner | `sfc_morton`, `edge_weighted_partition=True`, `scalable_halo=True` | Uses `bench_common.wrap_scalable_halo` ($O(N/W)$ halo build) |
| Integrator | `Langevin`, $\Delta t = 0.5\,\text{fs}$, $T = 300\,\text{K}$, $\gamma = 0.01\,\text{fs}^{-1}$ | Always call `bench_common.thermalize` at 300 K after `sim.setup()` |
| Cutoff & ghost buffer | Model native cutoff (`6.0 Å` for ORB / UMA / MACE-MP), ghost buffer `0.5 Å` | `rebalance_interval=1000`, `drift_watchdog=False` during timing |
| Warmup / timed steps | `10` warmup + `50` timed steps (`20` timed for $\ge 4\text{M}$ hero runs) | `Timer.set_level(Level.Off)` during warmup; `Level.Basic` during timed window |
| Outputs | `BenchRecord` JSON + `timings/<stem>/timing_rank*.jsonl` | Summarized by `analysis/summarize_scaling.py` |

> **Velocity initialization gotcha:** `TrajectoryReader.read()` converts XYZ
> momenta via `kAseFs`, which can under-scale files whose momenta are already in
> `amu·Å/fs` (such as `li3po4_*.xyz`, which load at ~2.9 K). Always call
> `bench_common.thermalize(sim, temperature_k=300.0, seed=1234 + rank)` before
> warmup and verify finite `mean_temperature_k` (~200–350 K) in the output JSON.

---

## 3. Datasets & Recommended Run Matrices

Download datasets once into the shared directory:

```bash
bash experiments/exp_datasets/scripts/download_zenodo.sh --scaling   # ch4_o2 + li3po4
bash experiments/exp_datasets/scripts/download_zenodo.sh 2w49 apoa1  # biomolecular systems
bash experiments/exp_datasets/scripts/download_zenodo.sh --list      # all 8 Zenodo archives
```

### 3.1 `ch4_o2` — Molecular Fluid Scaling Ladder (`ch4_o2.zip`)
- **Physics:** Uniform CH₄:O₂ (1:2) fluid, PBC, $\rho \approx 0.037\,\text{atoms/\AA}^3$, uncapped avg degree $\approx 33.6$ at $r_c = 6\,\text{\AA}$.
- **Available structures:** `ch4_{8001,16002,32004,50004,63999,127998,255996,512001,1024002,2048004,4095999,9999999}_splited.xyz`.
- **Standard studies:**

| Study | Tag | Structure(s) | GPU Ladder ($W$) | Adaptation for Lower-$N_{\max}^{(1)}$ Models (UMA / MACE) |
|-------|-----|--------------|------------------|-----------------------------------------------------------|
| Latency-bound strong | `strong64k` (or `strong32k`) | `ch4_63999_splited.xyz` (64K) | `1, 2, 4, 8, 16, 32` | Use `ch4_32004_splited.xyz` (32K) if 64K OOMs on 1 GPU |
| Mid-scale strong | `strong256k` | `ch4_255996_splited.xyz` (256K) | `$W_{\min} \to 32$` | Start at $W_{\min} \in \{1, 2, 4\}$ based on ceiling probe |
| Compute-bound strong | `strong1m` | `ch4_1024002_splited.xyz` (1.02M) | `$W_{\min} \to 32$` | Start at $W_{\min} \in \{1, 4, 8\}$ based on ceiling probe |
| Weak scaling | `weak{16,32,64,128}k` | $W \times N_0$ from ladder | `1, 2, 4, 8, 16, 32` | Pick highest $N_0 \in \{16\text{K}, 32\text{K}, 64\text{K}, 128\text{K}\}$ fitting 1 GPU |
| Hero run | `hero{2m,4m,10m}` | Largest fitting at $W=32$ | `32` (8 nodes) | `ch4_9999999` (10M), `ch4_4095999` (4.10M), or `ch4_2048004` (2.05M) |

### 3.2 `li3po4` — Dense Crystalline Supercells (`li3po4.zip`)
- **Physics:** Crystalline Li₃PO₄ (`mp-13725`, orthorhombic PBC), $\rho \approx 0.105\,\text{atoms/\AA}^3$ ($2.8\times$ denser than CH₄:O₂), uncapped avg degree $\approx 93.2$ at $r_c = 6\,\text{\AA}$ (~2.1–2.4× higher halo fraction).
- **Available structures (exact 6 Zenodo supercells):**
  `li3po4_32032.xyz` (32K), `li3po4_60928.xyz` (61K), `li3po4_126720.xyz` (127K),
  `li3po4_256256.xyz` (256K), `li3po4_501760.xyz` (502K), `li3po4_1010240.xyz` (1.01M).
- **Standard studies:**

| Study | Tag | Structure(s) | GPU Ladder ($W$) | Adaptation for Lower-$N_{\max}^{(1)}$ Models (UMA / MACE) |
|-------|-----|--------------|------------------|-----------------------------------------------------------|
| Weak scaling | `weak32k` | `32K → 61K → 127K → 256K → 502K → 1.01M` | `1, 2, 4, 8, 16, 32` | If 32K OOMs on 1 GPU, shift ladder to `weak16k` starting at $W=2$ (`32K@2 → 502K@32`) |
| Compute-bound strong | `strong1m` | `li3po4_1010240.xyz` (1.01M) | `$W_{\min} \to 32$` | Start at smallest $W_{\min} \in \{1, 2, 4, 8\}$ that fits HBM |
| Latency-bound strong | `strong61k` (or `strong32k`) | `li3po4_60928.xyz` (61K) | `1, 2, 4, 8, 16, 32` | Use `li3po4_32032.xyz` (32K) if 61K OOMs on 1 GPU |

### 3.3 `2w49` — Biomolecular Protein (`2w49.zip`)
- **Physics:** Real non-uniform biological system (PDB 2W49 filament) testing both solvated PBC (`2w49_solvated_openmm.xyz`, **1,148,835 atoms**) and sparse non-periodic geometry (`2w49_vacuum_openmm.xyz`, **138,136 atoms**, `--pbc auto`).
- **Do NOT use** `2w49_vacuum_openmm_strained.xyz` for production scaling runs (unphysically strained).
- **Standard studies:**
  - `strong_solvated` (`2w49_solvated_openmm.xyz`, 1.15M): $W_{\min} \to 32$ GPUs.
  - `strong_vacuum` (`2w49_vacuum_openmm.xyz`, 138K): $W_{\min} \to 32$ GPUs (evaluates partition load balance on sparse filament geometry).

### 3.4 `apoa1` — ApoA1 Biomolecular Benchmark (`apoa1.zip`)
- **Physics:** OpenMM-prepared ApoA1 biomolecular benchmark (`apoa1_vacuum_openmm.xyz`, **92,224 atoms**, cubic PBC box `108.861 Å`, `pbc="T T T"`). Note: the file contains positions only (`species`, `pos`) with no momenta column, so `bench_common.thermalize` is required to initialize 300 K velocities.
- **Standard study:**
  - `strong_apoa1` (`apoa1_vacuum_openmm.xyz`, 92K): strong scaling across $W \in \{1, 2, 4, 8, 16, 32\}$ GPUs (or $W_{\min} \to 32$ if 92K exceeds a model's 1-GPU HBM ceiling).

### 3.5 Auxiliary Zenodo Archives
`CH4_{450,11250,1125000}.zip` (pure methane) and `RP3_1320.zip` (RP-3 jet fuel) are available via `download_zenodo.sh` for smaller domain checks, accuracy/conservation validation, or single-GPU profiling.

---

## 4. Model-Specific Guidance

| Model Family | Dynamis Class | Recommended Variants / Heads | Precision & Kernel Configs | Notes |
|--------------|---------------|------------------------------|----------------------------|-------|
| **ORB** | `ORBModel` | `orb_v3_direct_20_omat` ($k=20$), `orb_v3_direct_inf_omat` ($k=\infty$) | (a) Baseline unchunked `float32-high`; (b) Edge-chunked (`edge_chunk_size=524288`) + Blackwell fused `bf16` (PRs #156 & #158) | `omat` covers CH₄:O₂, Li₃PO₄, 2W49, and ApoA1 |
| **UMA** | `UMAModel` | `uma-s-1p1` / `uma-s-1` (`dataset="omat"` for Li₃PO₄, `dataset="omol"` for CH₄:O₂, 2W49 & ApoA1) | Baseline vs. Blackwell `sm_100` fused edge-block kernels (PR #165) | Check 1-GPU ceiling first; expect smaller $N_{\max}^{(1)}$ than Orb-Direct |
| **MACE** | `MACEModel` | `mace_mp` (`medium-mpa-0`) for Li₃PO₄ / CH₄:O₂; `mace_off` / `mace_omol` for 2W49 & ApoA1 | `float32` + `enable_cueq=True` (CuEquivariance) where installed | Conservative autograd forces increase per-atom activation HBM; start 1M strong scaling from $W_{\min}$ found in ceiling probe |

---

## 5. Experiment Status Tracker (Done vs. To Do)

### Completed / In Progress

| Model & Configuration | Dataset | Studies | GPU Range | Status |
|-----------------------|---------|---------|-----------|--------|
| **ORB-v3 Direct** (`k=20` & `k=∞`, unchunked `float32-high`, `gpudirect`) | `ch4_o2` | Strong (`64K`, `256K`, `1.02M`), Weak (`64K/GPU → 2.05M`, `128K/GPU → 4.10M`), Hero (`10.0M`) | `1 → 32` GPUs (1–8 nodes) | **Done** (logged in `Dynamis-experiments-log` & manuscript) |
| **ORB-v3 Direct** (`k=20`, edge-chunked `524,288` + Blackwell fused `bf16`, `gpudirect`) | `li3po4` | `weak32k`, `strong1m` (1.01M), `strong61k` (61K) | `1 → 16` GPUs (1–4 nodes) | **Done** (logged in `Dynamis-experiments-log` & manuscript) |
| **ORB-v3 Direct** (`k=20`, edge-chunked + Blackwell fused `bf16`, `gpudirect`) | `li3po4` | `weak32k`, `strong1m`, `strong61k` | `32` GPUs (8 nodes) | **Queued** on `beta` (appends to existing campaign once complete) |

### To Do (Remaining Campaigns)

- [ ] **ORB — `li3po4` 32-GPU completion:** Ingest 8-node results once the queued job finishes and refresh figures/tables.
- [ ] **ORB — `apoa1` (92K) & `2w49` (solvated 1.15M, vacuum 138K):** Run strong scaling ($W = 1 \to 32$) with edge-chunked fused BF16 `orb_v3_direct_20_omat`.
- [ ] **ORB — `ch4_o2` fused BF16 ablation (optional):** Re-run `1.02M` strong ($1 \to 32$) and `10M` hero with chunked fused BF16 for direct kernel-speedup comparison on the same fluid dataset.
- [ ] **UMA (`uma-s-1p1` / `uma-s-1`, baseline + `sm_100` fused kernels):**
  - [ ] Implement `drivers/run_uma_scaling.py` using `bench_common.py` + run parity & 1-GPU HBM ceiling probe.
  - [ ] `li3po4` (`omat`): weak ladder + `strong1m` ($W_{\min} \to 32$) + latency-bound strong (`61K` or `32K`, $1 \to 32$).
  - [ ] `ch4_o2` (`omol`): strong (`64K`/`32K`, `256K`, `1.02M`), weak ladder at max feasible $N_0$, and 32-GPU hero.
  - [ ] `apoa1` (`omol`): `apoa1_vacuum_openmm.xyz` (92K) strong scaling ($W_{\min} \to 32$).
  - [ ] `2w49` (`omol`): `2w49_solvated_openmm.xyz` (1.15M) and `2w49_vacuum_openmm.xyz` (138K) strong scaling ($W_{\min} \to 32$).
- [ ] **MACE (`mace_mp` `medium-mpa-0` / `mace_off` / `mace_omol`, CuEq enabled):**
  - [ ] Implement `drivers/run_mace_scaling.py` using `bench_common.py` + run parity & 1-GPU HBM ceiling probe.
  - [ ] `li3po4`: ceiling-matched weak ladder + `strong1m` ($W_{\min} \to 32$) + latency-bound strong (`61K` or `32K`, $1 \to 32$).
  - [ ] `ch4_o2`: strong + ceiling-matched weak scaling ($W_{\min} \to 32$) + 32-GPU hero.
  - [ ] `apoa1`: `apoa1_vacuum_openmm.xyz` (92K) strong scaling ($W_{\min} \to 32$).
  - [ ] `2w49`: `2w49_solvated_openmm.xyz` and `2w49_vacuum_openmm.xyz` strong scaling ($W_{\min} \to 32$).

---

## 6. Verification & Logging Checklist (Before Closing Any Campaign)

1. Every completed cell has `status == "ok"`, `use_fabric == True` (for $W \ge 2$ `gpudirect`), and physical `mean_temperature_k` (no NaNs/explosions).
2. Run `analysis/summarize_scaling.py` on the campaign results directory and inspect both wall-clock mean `steps_per_sec` and steady-state `med_steps_per_sec` (especially for small latency-bound systems where early-step allocator growth can add jitter).
3. Ingest normalized `result.json` + gzipped `timing_rank*.jsonl.gz` records into `Dynamis-experiments-log` and pass `dynamis-log validate records/ --check-timings --strict` and `pytest`.
