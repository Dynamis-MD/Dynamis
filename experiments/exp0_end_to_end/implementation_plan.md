# Implementation Plan — Synthetic End-to-End MD Benchmark: DistNNP-UMA vs FairChem-UMA

> Self-contained plan for coding agents that do **not** have the originating
> session's context. Everything needed to build the benchmark is below.

> **⚠ Superseded in places — read `README.md` for the current state.** Since this
> plan was written: (1) the `EV_TO_AMU_A2_PER_FS2` constant in §5 is **inverted** —
> the correct value is `0.0096485332` (`ase.units.fs**2`), not its reciprocal, and
> the unit conversion now lives in `distnnp_model/nnp/uma.py` (the model owns its
> units), not in `bench_common.py`; (2) the FairChem driver uses
> `inference_settings_default()` (not `turbo()`) to match `UMAModel`; (3) both
> drivers now re-thermalize velocities (Maxwell-Boltzmann) before timing;
> (4) the matrix was re-scoped to `strong` / `fill` / `ceiling` (see README §4)
> after the isotropic "gas box" was found to be a dense fluid and the wrong
> regime to show DistNNP's decomposition advantage; (5) all scripts are now
> cluster-agnostic via `scripts/cluster.env` + `scripts/sbatch_exp0.sh`.

---

## 1. Context & goal

We are producing end-to-end MD performance results for the DistNNP paper. The
headline claim: DistNNP's spatial-decomposition + halo-exchange strategy runs
**larger** NNP-MD systems (no OOM) and runs them **faster** than the available
baselines on the same GPUs.

**This deliverable — the synthetic ablation only:** UMA-based MD on synthetic
CH₄:O₂ gas boxes scaling from ~1k to ~1M atoms, on a **single `gpuA100x4`
node** (1/2/4 GPUs), comparing two engines:

- **DistNNP-UMA** — `distnnp_model.nnp.UMAModel` + `distnnp.Simulator`, one rank
  per GPU via `torchrun`; atoms spatially partitioned (SFC-Morton), only ghost
  atoms/features exchanged per message-passing layer.
- **FairChem-UMA** (primary baseline) — one ASE MD driver whose force/energy
  call is graph-parallel across the node's GPUs via
  `fairchem` `get_predict_unit(..., workers=N)` → Ray-spawned
  `ParallelMLIPPredictUnit` (mode = ALLGATHER + INDEX_SPLIT: contiguous atom
  slices, node embeddings all-gathered every layer, energy all-reduced). This is
  the fairchem "multi-GPU inference" path.

### Locked decisions

| Topic | Decision |
|---|---|
| Environment | Build a **fresh** conda env (fairchem-core 2.x + ray). OK to **drop `compute_70`/Volta** and build extensions for `80;90` only. |
| Model + integrator (both engines) | **`uma-s-1p2`**, task **`omol`**, **Langevin** thermostat. |
| Synthetic system | **CH₄:O₂ 1:2 `splited`** from the existing, unmodified `exp_datasets/scripts/03_generate_ch4.py`. The mix / spin questions are **out of scope** — wire in the code as-is. If that generator is later changed, regenerate datasets and re-run the matrix. |
| Run scope | **Strong scaling + weak scaling + OOM ceiling.** |
| Execution | **Everything runs via `sbatch`** (shared HPC cluster). No long interactive sessions. Validation/smoke jobs must finish in **≤ ~3 minutes**; matrix jobs kept as short as correctness allows (small step counts, see §7). |

### Cluster facts (NCSA Delta)

- Account `bcpt-delta-gpu`, partition `gpuA100x4` (4× A100-SXM4-40GB, NVLink).
- Build/runtime modules: `cudatoolkit/26.5_12.9-unsupported` (CUDA 12.9, pairs
  with torch `cu128`), `gcc-native/14`, `aws-ofi-nccl`.
- `conda` base is at `/sw/external/python/anaconda3` (on PATH after the shell
  init). SLURM scripts must `conda activate` by explicit env name and **not**
  rely on `.venv` activation (observed to silently no-op under `sbatch` here).
- Repo layout: git root `/u/szaman5/DistNNP/`; the Python monorepo is the
  **inner** `/u/szaman5/DistNNP/DistNNP/` (4 editable sub-packages:
  `distnnp_state`, `distnnp_model`, `distnnp_integrator`, `distnnp`);
  experiments live in `/u/szaman5/DistNNP/experiments/`.

### Reference code to copy idioms from

- DistNNP-UMA MD loop: `DistNNP/distnnp_model/tests/run_distributed_trajectory.py`
  + `DistNNP/distnnp_model/tests/md_test_common.py` (`make_config`, `run_md_loop`,
  `extract_frame`, `real_singleton_torch_dist_env`).
- DistNNP-UMA parity test: `DistNNP/distnnp_model/tests/test_uma_md.py`.
- LJ throughput harness (Timer + JSON + peak-mem pattern):
  `DistNNP/distnnp/experiments/synthetic/main.py` + `setup_system.py` (`Timer`).
- FairChem calculator setup: `experiments/run_md.py` (`get_predict_unit`,
  `inference_settings_turbo`, `FAIRChemCalculator(task_name="omol")`).
- Existing SLURM template: `DistNNP/DistNNP/scripts/run_synthetic_baseline.sbatch`.

---

## 2. Deliverable directory layout

All new files under `experiments/exp0_end_to_end/`:

```
exp0_end_to_end/
├── implementation_plan.md        # this file
├── README.md                     # NEW — goal, matrix, how-to-run, resolved env versions, results table
├── exp_datasets/                 # EXISTS — CH4 generator + dataset catalog
├── drivers/
│   ├── bench_common.py           # NEW — args, warmup+timed loop, JSON schema, OOM capture, unit constants
│   ├── run_distnnp_uma.py        # NEW — DistNNP-UMA driver (torchrun, 1 rank/GPU)
│   └── run_fairchem_uma.py       # NEW — FairChem-UMA driver (1 proc + Ray workers)
├── scripts/
│   ├── env_setup.sh              # NEW — build the fresh conda env + rebuild the 4 extensions
│   ├── gen_datasets.sh           # NEW — thin wrapper over exp_datasets/scripts/03_generate_ch4.py
│   ├── smoke.sbatch              # NEW — ≤3 min: 1 tiny cell per engine on 2 GPUs, asserts JSON is well-formed
│   ├── run_cell.sbatch           # NEW — run ONE (engine,xyz,ngpu) cell; on failure write an oom/error stub
│   └── submit_matrix.sh          # NEW — enumerate the matrix, sbatch one job per cell (with throttling)
├── results/                      # NEW — one JSON per cell (add to .gitignore; keep .gitkeep)
└── analysis/
    ├── summarize.py              # NEW — merge results/*.json → CSV + matplotlib figures
    └── figures/                  # NEW (gitignored)
```

Also, **only if dropping Volta** (recommended): two one-token edits —
`DistNNP/distnnp_state/CMakeLists.txt:99` and
`DistNNP/distnnp_model/CMakeLists.txt:108`,
`CUDA_ARCHITECTURES "70;80;90"` → `"80;90"`. These are per-target
`set_target_properties`, so `-DCMAKE_CUDA_ARCHITECTURES` on the command line does
**not** override them — the file must be edited. `distnnp_integrator` has no
hardcoded value and honours `CMAKE_ARGS`.

Add to `experiments/exp0_end_to_end/.gitignore` (new): `results/*.json`,
`analysis/figures/`, `*.log`.

---

## 3. Step 1 — Build the UMA environment  (`scripts/env_setup.sh`)

Goal: a fresh conda env `distnnp-uma` with fairchem-core 2.x + ray + the 4
distnnp extensions rebuilt into it. Model on `DistNNP/environment.yml` (the
known-good spec: py3.12, torch 2.8.0, `fairchem-core 2.16.1.dev10`, `ray 2.53.0`,
`e3nn 0.5.9`, `numpy 2.2.6`).

`env_setup.sh` (run once, on a compute node via a short `sbatch` or a brief
`salloc` — it compiles CUDA, so it needs a GPU node and ~15–20 min):

1. `module load cudatoolkit/26.5_12.9-unsupported gcc-native/14 aws-ofi-nccl`.
2. `conda create -y -n distnnp-uma python=3.12 && conda activate distnnp-uma`.
3. `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`.
4. `pip install "fairchem-core>=2.16,<2.23" ray huggingface-hub ase hydra-core omegaconf`.
   Expected resolve: fairchem-core **2.22.0** (also pulls `e3nn`, `torchtnt`,
   `triton`, `numba`, `numpy>=2`).
5. **HF-gated weights.** User action, once: accept the `facebook/UMA` license on
   huggingface.co, then `huggingface-cli login` (or export `HF_TOKEN`).
   Pre-warm the checkpoint on a login node so compute nodes need no network:
   `python -c "from fairchem.core.calculate.pretrained_mlip import get_predict_unit; get_predict_unit('uma-s-1p2', device='cpu', workers=1)"`
   (downloads into `~/.cache/fairchem`).
6. (chosen) Drop Volta: apply the two CMakeLists edits from §2.
7. Rebuild extensions in dependency order with torch's CMake config on the path:
   ```bash
   export CMAKE_PREFIX_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake"))'):$CMAKE_PREFIX_PATH"
   export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;90"     # covers distnnp_integrator
   cd /u/szaman5/DistNNP/DistNNP
   pip install -e ./distnnp_state -e ./distnnp_model -e ./distnnp_integrator -e ./distnnp
   ```
8. Smoke-check imports:
   ```bash
   python -c "import distnnp_state, distnnp_integrator, distnnp_model, distnnp; from distnnp_model.nnp import UMAModel; print('distnnp ok')"
   python -c "import fairchem.core, ray; from fairchem.core.calculate.pretrained_mlip import get_predict_unit; print('fairchem ok')"
   ```
9. Write the resolved versions (`pip freeze | grep -E 'fairchem|torch|ray|e3nn|numpy|ase'`)
   into `exp0_end_to_end/README.md`.

**Known risk — fairchem 2.22 vs 2.16 API drift.**
`DistNNP/distnnp_model/src/distnnp_model/nnp/uma.py` imports several private
fairchem internals: `fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone`,
`fairchem.core.models.uma.nn.mole_utils.{model_search_and_replace,
recursive_replace_so2_MOLE, replace_MOLE_with_linear}`,
`fairchem.core.units.mlip_unit.{InferenceSettings, guess_inference_settings}`,
`fairchem.core.units.mlip_unit.utils.tf32_context_manager`,
`fairchem.core.units.mlip_unit.single_atom_patch.single_atom_prediction_from_lookup`.
If any moved in 2.22, either (a) pin `fairchem-core==2.16.*` (may need
`--pre` / a git ref matching `2.16.1.dev10+g0d7cb7ee6`), or (b) patch the import
sites. If imports break, the fix is here (pin / patch); there is no exp0-side
parity gate — model correctness is validated in `distnnp_model/tests/`.

---

## 4. Step 2 — Datasets  (`scripts/gen_datasets.sh`)

Thin wrapper; **does not modify** `exp_datasets/scripts/03_generate_ch4.py`
(numpy-only, bit-reproducible, writes extended-XYZ with
`Lattice="L 0 0 0 L 0 0 0 L"`, `Properties=species:S:1:pos:R:3:momenta:R:3`,
`pbc="T T T"`, `L = round(60·(k/8)^(1/3))` Å at constant density).

```bash
python exp_datasets/scripts/03_generate_ch4.py --sizes 1 2 16 32 64 128 256 512 1024
# → exp_datasets/baselines/ch4_<k>k_splited.xyz   (variant default = splited)
```

Sizes needed:
- smoke: `1`, `2` (≈1k/2k atoms)
- weak scaling (~16k atoms/GPU): `16` (1 GPU), `32` (2 GPU), `64` (4 GPU)
- strong scaling (fixed size, 1/2/4 GPU): `32` (≈32k; fits on 1 GPU for UMA,
  real work for 4 GPU). Keep the fixed size a `submit_matrix.sh` variable.
- OOM ceiling (4 GPU, push to failure): `64, 128, 256, 512, 1024`.

Both drivers must honour each file's own box:
- FairChem: `ase.io.read` picks up the cell automatically.
- DistNNP: `Simulator` builds its cell from
  `SimulatorConfig.side_length_angstrom` (`torch.eye(3)*side_length`), **ignoring
  the file's `Lattice=`**. So `run_distnnp_uma.py` must read `L` via
  `ase.io.read(path).cell.lengths()[0]` and pass it as `side_length_angstrom`.
  (The C++ reader `distnnp_state/src/csrc/trajectory_writer.cpp` does parse
  `Lattice=`/`Properties=`/`pbc=`, so the generator output loads directly for
  positions/species/velocities.)

Verify positions are reproduced: `sha256sum -c exp_datasets/scripts/checksums.txt`
filtered to `baselines/` lines.

---

## 5. Step 3 — Shared harness  (`drivers/bench_common.py`)

One module both drivers import.

- **`parse_bench_args()`** → argparse with:
  `--input` (xyz path), `--engine {distnnp-uma,fairchem-uma}`,
  `--model uma-s-1p2`, `--task omol`,
  `--warmup-steps` (default 10), `--timed-steps` (default 50),
  `--timestep-fs 0.5`, `--temperature-k 300`, `--friction-inv-fs 0.01`,
  `--rebalance-interval 50` (DistNNP only), `--workers` (FairChem only,
  = #GPUs), `--output-dir`, `--tag` (optional).
- **Unit constants** (documented in-module):
  `BOLTZMANN_EV_PER_K = 8.617333262e-5`.
  DistNNP's `Langevin(dt, gamma, kT)` BBK kernel uses
  `c2 = sqrt(gamma * kT * dt / m)` with `m` in amu, velocities in Å/fs, `dt` in
  fs, `gamma` in 1/fs ⇒ **`kT` must be in amu·Å²/fs²**, not eV.
  `EV_TO_AMU_A2_PER_FS2 = 1 / 0.0096485332` (≈103.6427).
  `kT_native = BOLTZMANN_EV_PER_K * T_kelvin * EV_TO_AMU_A2_PER_FS2`.
  This constant is validated in `distnnp_integrator/tests/test_langevin.py`; the
  per-cell `mean_temperature_k` is recorded but is not an exp0 pass/fail gate.
- **`TimedLoop(step_fn, warmup, timed)`** — runs `warmup` untimed steps, then
  `torch.cuda.synchronize()`, start CUDA event / `time.perf_counter()`, run
  `timed` steps, `torch.cuda.synchronize()`, stop. Returns
  `elapsed_s`, `steps_per_sec = timed/elapsed_s`,
  `ns_per_day = timestep_fs * 1e-6 * steps_per_sec * 86400`.
  (May reuse the `Timer` class from
  `DistNNP/distnnp/experiments/synthetic/setup_system.py`.)
- **`capture_oom(fn) -> record`** — runs `fn`; on
  `torch.cuda.OutOfMemoryError` returns/writes a record with `status="oom"`
  (never re-raises, so a sweep continues). Other exceptions → `status="error"`
  with `repr(e)`.
- **`peak_mem_mb()`** = `torch.cuda.max_memory_allocated() / (1024**2)`, read
  after a `torch.cuda.reset_peak_memory_stats()` at the start of the timed
  section.
- **`write_result(output_dir, record)`** — rank-0-only JSON write.
  Filename: `results/<engine>_<num_atoms>atoms_<world_size>gpu[_<tag>].json`.
  **Schema:**
  ```json
  {
    "engine": "distnnp-uma",
    "model": "uma-s-1p2", "task": "omol", "integrator": "langevin",
    "num_atoms": 32004, "box_angstrom": 60.0,
    "world_size": 4, "gpus_per_node": 4, "nodes": 1,
    "warmup_steps": 10, "timed_steps": 50,
    "timestep_fs": 0.5, "temperature_k": 300, "friction_inv_fs": 0.01,
    "status": "ok",                         // "ok" | "oom" | "error"
    "elapsed_s": 12.3, "steps_per_sec": 4.07, "ns_per_day": 0.176,
    "peak_mem_mb_per_rank": [18234.0, 18110.0, 18190.0, 18205.0],
    "peak_mem_mb_max": 18234.0,
    "error": null,
    "env": {"fairchem_core": "2.22.0", "torch": "2.8.0+cu128", "ray": "2.53.0"},
    "git_commit": "abc1234", "timestamp": "2026-08-27T12:00:00Z"
  }
  ```

---

## 6. Step 4 — Drivers

### 6a. `drivers/run_distnnp_uma.py` (torchrun, 1 rank/GPU)

Idiom from `run_distributed_trajectory.py`:

```python
env = TorchDistEnv(is_distributed=True)                       # distnnp_state
L = ase.io.read(args.input).cell.lengths()[0]
model = UMAModel(args.model, dataset=args.task, device=env.device.type)
kT = BOLTZMANN_EV_PER_K * args.temperature_k * EV_TO_AMU_A2_PER_FS2
integ = Langevin(dt=args.timestep_fs, gamma=args.friction_inv_fs, kT=kT)
cfg = SimulatorConfig(
    timestep_fs=args.timestep_fs,
    total_steps=args.warmup_steps + args.timed_steps,
    side_length_angstrom=L, cutoff_angstrom=6.0, ghost_buffer_angstrom=0.5,
    partition_algorithm="sfc_morton", edge_weighted_partition=True,
    rebalance_interval=args.rebalance_interval,
    traj_interval=0, log_interval=0, output_dir=args.output_dir, backend="nccl",
)
os.makedirs(args.output_dir, exist_ok=True)      # every rank — TrajectoryWriter mkdir race
sim = Simulator(model, integ, env.dist_env, cfg, env.device)
sim.setup(args.input)                            # reads xyz → global MolecularInfo → partitions
# manual loop so warmup vs timed is explicit (mirror md_test_common.run_md_loop):
def one_step(step):
    if sim.should_rebalance(step): sim.sync_and_repartition()
    sim.step(cfg.timestep_fs)
TimedLoop(one_step, args.warmup_steps, args.timed_steps)
# per-rank peak mem: each rank records peak_mem_mb(); rank 0 dist.all_gather_object the list
# rank 0: write_result(...)
```

- Wrap `sim.setup` + the loop in `capture_oom`.
- Verify `Langevin` units against `DistNNP/distnnp_integrator/tests/test_langevin.py`
  and the `.cu` kernel (that test is the acceptance gate — exp0 does not re-check it).
- `dist.destroy_process_group()` at the end.

### 6b. `drivers/run_fairchem_uma.py` (single process + Ray workers)

Reuses `experiments/run_md.py`'s calculator setup, **strips all per-step I/O**
(no `trajectory.xyz`, no `traj_step_*.npz` — these dominate at ≥256k atoms):

```python
settings = inference_settings_turbo()
settings.merge_mole = True
settings.compile = False           # match run_md.py defaults; compile adds minutes of warmup
predictor = get_predict_unit(args.model, device="cuda",
                             inference_settings=settings, workers=args.workers)
calc = FAIRChemCalculator(predictor, task_name=args.task)
atoms = ase.io.read(args.input); atoms.set_pbc(True)
atoms.info["charge"] = 0; atoms.info["spin"] = 1
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_k)
atoms.calc = calc
dyn = ase.md.langevin.Langevin(atoms, args.timestep_fs*units.fs,
                               temperature_K=args.temperature_k,
                               friction=args.friction_inv_fs/units.fs)
capture_oom(lambda: (dyn.run(args.warmup_steps),
                     TimedLoop(lambda _: dyn.run(1), 0, args.timed_steps)))
```

- `--workers 1` → `MLIPPredictUnit` (no Ray, no graph parallel) — the 1-GPU
  point. `--workers {2,4}` → `ParallelMLIPPredictUnit` (Ray + graph parallel).
  `run_cell.sbatch` sets `--workers` = `--gpus-per-node`.
- **Peak mem across Ray workers:** rank 0 is in-process
  (`predictor.local_rank0`), ranks ≥1 are Ray actors (`predictor.workers`).
  After the timed section, add a tiny `.remote()` call returning each actor's
  `torch.cuda.max_memory_allocated()`; assemble `peak_mem_mb_per_rank`. If that
  proves brittle in v1, record the rank-0 peak into `peak_mem_mb_max` and set
  `peak_mem_mb_per_rank: null` with a note.
- **OOM in a Ray worker** kills the actor; the next `broadcast` from rank 0
  hangs. This is why every cell is its **own SLURM job with a wall-clock cap**
  (§7): a job that dies / times out without writing JSON is recorded by
  `run_cell.sbatch` as `status="oom"`. `capture_oom` still catches the clean
  `--workers 1` single-process OOM.

---

## 7. Step 5 — SLURM job matrix

**All runs via `sbatch`.** One node per job.

### `scripts/run_cell.sbatch <engine> <xyz> <ngpu> [--timed-steps N ...]`
```bash
#SBATCH --account=bcpt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200g
#SBATCH --time=00:25:00           # cap; big cells override to 00:40:00
#SBATCH --output=exp0_end_to_end/results/logs/%x_%j.out
```
- `module load cudatoolkit/26.5_12.9-unsupported gcc-native/14 aws-ofi-nccl`
- `source $(conda info --base)/etc/profile.d/conda.sh && conda activate distnnp-uma`
- `export NCCL_SOCKET_IFNAME=hsn`
- dispatch:
  - `distnnp-uma` → `torchrun --standalone --nproc_per_node=<ngpu> exp0_end_to_end/drivers/run_distnnp_uma.py --input <xyz> --output-dir exp0_end_to_end/results ...`
  - `fairchem-uma` → `python exp0_end_to_end/drivers/run_fairchem_uma.py --input <xyz> --workers <ngpu> --output-dir exp0_end_to_end/results ...`
- **trap**: on non-zero exit or missing `results/<cell>.json`, write a stub JSON
  `{status: "oom" or "error", exit_code, ...}` so the matrix has no holes.

### `scripts/submit_matrix.sh`
Enumerates and `sbatch`es one job per cell, with `--gpus-per-node` set per cell,
and light throttling (`sbatch --dependency=singleton` or a small
`MAX_INFLIGHT` loop) so we're polite on the shared queue.

| Sub-experiment | engine × (atoms @ GPUs) | timed steps |
|---|---|---|
| **Strong scaling** | both × `32k @ {1,2,4}` | 50 |
| **Weak scaling** | both × `{16k@1, 32k@2, 64k@4}` | 50 |
| **OOM ceiling (4 GPU)** | both × `{64k, 128k, 256k, 512k, 1024k} @ 4` | 30 for ≤256k, **15** for 512k/1024k |

≈ 21 jobs. `warmup=10` everywhere. Rationale for small step counts: UMA MD is
~7 steps/s at 4k on 4×A100 (`DistNNP/CLAUDE.md`); throughput drops steeply with
size, so 30–50 timed steps is enough for a stable steps/s while keeping each job
well under 25 min. Model load + Ray bootstrap is a fixed ~1–3 min per job,
excluded from timing by the warmup boundary.

---

## 8. Verification (all via `sbatch` or a ≤3-min interactive check)

**No correctness/parity gates.** exp0 is a throughput + memory-ceiling
benchmark; model correctness (distributed MD == single-process baseline,
thermostat unit constants) is owned by `distnnp_model/tests/` and
`distnnp_integrator/tests/` and is validated there once, not re-run here. The
`mean_temperature_k` field is still recorded per cell as an observability
signal, but it is not a pass/fail gate.

1. **`scripts/smoke.sbatch`** (≤3 min, 2 GPUs): run `run_fairchem_uma.py`
   (`--workers 2`) and `run_distnnp_uma.py` on `ch4_1k_splited.xyz` with
   `--warmup-steps 3 --timed-steps 10`. Assert each wrote a schema-valid
   `results/*.json` with `status=="ok"` and `0 < steps_per_sec < 1e4`. This is
   a "does the driver launch + emit JSON" check, not a correctness check.
2. **OOM path** — one `run_cell.sbatch fairchem-uma ch4_1024k_splited.xyz 1`
   job: must terminate (crash or `capture_oom`) and leave a
   `status in {oom,error}` JSON, **not** hang past the wall cap.
3. **Full matrix** — `scripts/submit_matrix.sh`; when the queue drains,
   `python analysis/summarize.py`; eyeball the figures + headline table.

---

## 9. Step 6 — Analysis  (`analysis/summarize.py`)

Merge `results/*.json` → `analysis/summary.csv` + PNGs in `analysis/figures/`
(matplotlib, no seaborn needed):

1. **Strong scaling** — steps/s & ns/day vs GPU count, one line per engine,
   ideal-linear reference; parallel efficiency `T₁/(N·Tₙ)`.
2. **Weak scaling** — steps/s vs total atoms at fixed ~16k/GPU (flat = ideal);
   efficiency `T₁/Tₙ`.
3. **OOM ceiling** — max atoms completed per engine on 4 GPUs (bar), plus
   `peak_mem_mb_max` vs atoms (expect FairChem's all-gather memory to blow up
   while DistNNP's per-rank curve stays near-flat).
4. **Headline table** — per matched (atoms, GPU) cell: DistNNP steps/s,
   FairChem steps/s, speedup, OOM markers → also pasted into
   `exp0_end_to_end/README.md`.

---

## 10. Explicitly out of scope *for the isotropic matrix above*

Multi-node (8–32 GPU), the per-phase profiler breakdown, and the
partition-algorithm ablation. The `03_generate_ch4.py` CH₄:O₂ composition / O₂
spin question is **not** to be fixed here — use the generator as-is; if it
changes, `scripts/gen_datasets.sh` + `scripts/submit_matrix.sh` re-run cleanly.
The harness (`bench_common.py` schema, `run_cell.sbatch`, `submit_matrix.sh`) is
built so new sizes / input files / GPU counts are one-line additions.

Two items previously listed here — **heterogeneous-density systems** and the
**LAMMPS/SevenNet baseline + extra model families** — have been promoted to
active parallel workstreams; see §11 and §12. They share this directory's
harness and drivers and can proceed independently of each other and of the
isotropic matrix.

---

## 11. Workstream A — heterogeneous-density comparison (real systems)

> Self-contained spec. Depends on the §1–§9 harness (`drivers/bench_common.py`,
> `drivers/run_distnnp_uma.py`, `drivers/run_fairchem_uma.py`, `scripts/*`,
> `analysis/summarize.py`) and the cluster-agnostic layer
> (`scripts/cluster.env` + `scripts/sbatch_exp0.sh`). Independent of §12.

### 11.1 Claim under test

FairChem graph-parallel partitions atoms by **contiguous index**:
`node_partition = torch.tensor_split(torch.arange(N), gp_world_size)[gp_rank]`
(`.venv-uma/.../fairchem/core/models/uma/escn_md.py:591`; no spatial reorder, no
degree balancing, no shuffle). On a system whose density is not spatially
uniform **and** whose atoms are listed in any spatially-correlated order (PDB =
protein-then-water; our own `ch4_*_splited` = all CH₄ then all O₂), a contiguous
index slice is a contiguous spatial/chemical region. Then:

- per-rank **edge work** ∝ Σ(degree over the slice) becomes imbalanced — the
  step is gated by the busiest rank (throughput → ~ideal/`imbalance`);
- FairChem shards the edge tensors, so **peak memory per rank** and the **OOM
  ceiling** are set by the *densest* rank, not the mean;
- a one-time input shuffle fixes mean degree but not the memory variance, and
  cannot help once the system *evolves* (interface drifts, phases separate).

DistNNP: `partition_algorithm="kdtree"` + `edge_weighted_partition=True`
balances edge count spatially, and `rebalance_interval` re-partitions as the
system evolves. This is the regime §7 deliberately excludes.

### 11.2 Use `exp_datasets/`, do NOT build a synthetic generator

`exp_datasets/` exists **for exactly this** (its README: "Experiment Datasets
for Partition Imbalance Analysis"). Get the files with
`bash exp_datasets/scripts/setup_data.sh` (generates CH₄, downloads PDBs, fetches
the SHA256-verified archive; on Delta served from a shared cache). Each
sub-README carries a pre-computed imbalance ratio `I2` = max/mean per-rank work
at a partition count `k`.

| system | file(s) | atoms | anisotropy | `I2` @ k=64 |
|---|---|---:|---|---:|
| **electrochemical** | `electrochemical/pt_water_{12,17}k_density.xyz` | 14k / 21k | 1-D sharp layering (Pt slab z 0–10 Å, water z 11–49 Å); Pt degree 54 vs water 103 | **2.4–2.6** (worst) |
| **membrane** | `membrane/dppc_membrane_28k.xyz`, `membrane/apoa1_92k_openmm.xyz` | 28k / 92k | 1-D bilayer + water; apoa1 adds protein | ~1.8–2.2 |
| **protein_solvated** | `protein_solvated/{dhfr_23k_openmm,2rik_density,9gmo_density}.xyz` | 24k / 113k / 1.5M | 3-D localised solute in water | 1.15–1.63 |
| **protein_vacuum** | `protein_vacuum/{6p9x,2w49,groel,9gmo}_*.xyz` | 22k–229k | extreme — filament / multi-chain, box mostly empty (ρ 4e-5–0.02) | ≫ (measure) |
| **capsid** | `capsid/3j3q_capsid_sub{150,350}.xyz`, `capsid/3j3q_capsid.xyz` | 33k / 175k / 2.4M | dense shell, hollow interior (3-D, large scale) | (measure) |
| *isotropic control* | `baselines/ch4_{32,64,128}k_splited.xyz` | — | none | ≈ 1.0 |

Real systems beat a synthetic slab: reviewer-proof (a hand-tuned profile invites
"you engineered it"); `2w49` / `6p9x` are already in the **DistMLIP** benchmark;
native ordering *is* the sorted-input worst case. A continuous anisotropy sweep
is the only thing a slab adds and the discrete `I2` points already span
`1.0 → ≫`. If review demands a sweep: carve slabs from `ch4_32k` (fill fraction
f ∈ {1, ½, ¼, ⅛}, vacuum-pad the rest), otherwise skip.

Caveats to state in the writeup: `capsid` and vacuum PDBs are **heavy-atoms-only**
→ UMA energies are unphysical; this is a throughput/partition benchmark, not a
production trajectory. `*_density.xyz` use grid-placed (not MD-equilibrated)
water — both drivers already re-thermalise (§ fixes), so fine.

### 11.3 Prerequisite code work (do first, in order)

**(A) Non-cubic / non-PBC cells — the blocker.**
`distnnp/src/distnnp/simulator.py:150-157` (`Simulator._get_simulation_config`)
**hardcodes** `sim_config.cell = torch.eye(3) * self.config.side_length_angstrom`
and `sim_config.pbc = [True,True,True]`, discarding whatever the file's
`Lattice=` header said (the C++ reader parses it into `global_mol_info`, then
this overrides it for partitioning + the radius graph).

- Add `cell: Optional[torch.Tensor] = None` (3×3) and `pbc: Optional[...] = None`
  to `SimulatorConfig` (`simulator.py:45`). In `_get_simulation_config`, use them
  when set, else fall back to the cubic `side_length_angstrom` path (keeps §7
  working unchanged).
- `run_distnnp_uma.py`: replace `box = float(_atoms0.cell.lengths()[0])` with the
  full `_atoms0.cell.array` (3×3) and `_atoms0.pbc`; pass both to
  `SimulatorConfig`. For vacuum systems (`pbc` all False or a tight box):
  centre the atoms and set a cubic cell = `bbox + 2×cutoff`, `pbc=False`, and
  do the **identical** transform in `run_fairchem_uma.py` (`atoms.set_pbc(...)`,
  `atoms.set_cell(...)`, `atoms.center()`) so both engines see the same box.
- Verify `radius_graph_pbc_cuda` and the partitioners accept a non-cubic
  `sim_config.cell` (the KD-tree partitioner should; the SFC one may assume a
  cube — use `kdtree` here regardless, per §11.1).

**(B) `--shuffle-atoms {none,random}` flag** in `bench_common.parse_bench_args`
(default `none`), applied in both drivers immediately after `ase.io.read`, seeded
(`--shuffle-seed`, default 0). Permute atoms (and momenta) consistently. `none`
= native order (FairChem worst case). DistNNP is the order-invariant control.

**(C) Imbalance instrumentation** — extend `bench_common.BenchRecord`
(`bench_common.py:299`) with:
`edges_per_rank: list[int] | None`, `atoms_per_rank: list[int] | None`,
`step_ms_per_rank: list[float] | None`, and computed
`work_imbalance` (= max/mean of `edges_per_rank`),
`mem_imbalance` (= `peak_mem_mb_max / mean(peak_mem_mb_per_rank)`).
- DistNNP: per-rank owned-atom count is
  `sim.state.local_partition_info.atom_counts`; per-rank edge count = sum of
  `mol.edge_counts` (`distnnp_state/.../dist_graph_builder.py:139`, per-atom
  degree) over `partition_indices[:n_local]`, or count `edge_index[0] < n_local`
  inside the model call and `all_gather_object` it. Time each timed step per
  rank (`torch.cuda.synchronize` + `perf_counter`), gather the mean.
- FairChem: rank-0 is in-process; ranks ≥1 are Ray actors. Extend the existing
  `_worker_peaks` remote probe (`run_fairchem_uma.py:72`) to also return the
  actor's owned-node count and local edge count from the last `generate_graph`
  (`escn_md.py` `node_partition` slice + `graph_dict["edge_index"].shape[1]`).

**(D) `analysis/summarize.py`** — add `imbalance_scaling(rows, out_dir)`:
x = `work_imbalance` (or system name ordered by it), two panels
(`steps/s` normalised to the 1-system-per-engine baseline; `mem_imbalance`),
one series per engine × order. Add a per-system table to `headline_table.md`.

### 11.4 Matrix

`scripts/submit_matrix.sh` gains `--only hetero`. Cells:
`{system} × {distnnp-uma, fairchem-uma} × {--shuffle-atoms none, random} × {ngpu}`.

| tier | systems | ngpu | notes |
|---|---|---|---|
| small | `pt_water_17k`, `dhfr_23k_openmm`, `6p9x_vacuum` | 2, 4 | fast; the `I2` spread 1.2 → large |
| mid | `2rik_density` (113k), `2w49_vacuum` (153k), `3j3q_capsid_sub350` (175k) | 4 | |
| large / multi-node | `3j3q_capsid` (2.4M), `9gmo_density` (1.5M) | 4, then 8–32 (§10 multi-node) | separate follow-up |

Two passes: **headline** with `rebalance_interval` huge (isolate the initial
partition); **rebalance** with `rebalance_interval=25` on 2–3 evolving systems
to show DistNNP recovering balance over the run. `--timed-steps 20`,
`--warmup-steps 8`.

### 11.5 Acceptance gates & expected result

- Non-cubic parity: covered by `distnnp_model/tests/` (distributed energy ==
  single-process UMA reference), not re-verified in exp0.
- Order-invariance: DistNNP `steps/s` and `work_imbalance` change < 3 % between
  `--shuffle-atoms none` and `random`.
- Headline: FairChem `steps/s` drops and `mem_imbalance` climbs monotonically
  with `I2` (steeply for `none`, less for `random`); DistNNP flat on both and
  clears a higher OOM ceiling on the vacuum/filament systems. This is the
  inverse of §-Results and the core "runs larger" evidence.

---

## 12. Workstream B — LAMMPS/SevenNet baseline + additional model family

> Self-contained spec. Two coupled pieces: **(1) a `DistNNPModel` wrapper for a
> second model family** so the model is held constant, and **(2) a LAMMPS
> baseline engine** in the harness. Independent of §11. Start with §12.4.

### 12.1 Why LAMMPS is the baseline that matters more than FairChem

`pair_mace` / `pair_nequip` / `pair_allegro` / `pair_e3gnn` are thin C++ wrappers
that dispatch to the model's own (TorchScript) forward — the equivariant kernels
(e3nn / cuEquivariance) are **identical** to what DistNNP runs. LAMMPS supplies
neighbor lists, spatial domain decomposition, MPI comm, the integrator, `fix
balance` (RCB). It does **not** build the NNP kernels. A fair comparison holds
the model fixed and measures only decomposition + halo strategy — and there,
DistNNP's per-layer 1-hop exchange is *structurally* competitive:

| LAMMPS path | mechanism | expected vs DistNNP |
|---|---|---|
| `pair_nequip`, non-parallel `pair_allegro` | one forward call per rank; `comm cutoff = n_layers × r_max` so each rank runs the whole net locally | ghost-shell volume ∝ (n·r)³ and redundant ghost message recompute → **DistNNP wins on memory / max size** (headline) |
| `pair_e3gnn_parallel` (SevenNet) | communicates **per message-passing layer** — same as DistNNP `feature_exchangers` | **target parity**; gap is only buffer packing / comm-overlap / neighbor-rebuild quality |
| `pair_mace` MPI (Kokkos) | check: extended-cutoff or per-layer? | classify after measuring |

### 12.2 `DistNNPModel` contract (what the wrapper must implement)

ABC: `distnnp_model/src/distnnp_model/potential.py` — `class DistNNPModel(ABC)`,
two `@abstractmethod`s (both must be explicitly defined; `__getattr__` forwarding
does **not** satisfy `abc`):

```python
def __call__(self, distributed_state: DistNNPState, *a, **kw) -> tuple[Tensor, Tensor]
        # returns (energy_eV_scalar_detached, local_forces_detached)
        # local_forces MUST be in the integrator's native units amu·Å/fs²:
        #   multiply the (scalar) energy by _EV_PER_A_TO_AMU_A_PER_FS2 = 0.0096485332
        #   INSIDE the autograd.grad / backward call (one multiply, not per-component).
        #   Energy is returned unscaled (eV) for logging/drift.
def initialize_buffers(self, partition_info: PartitionMessagePassingInfo, device) -> None
```

**Reference implementations to mirror** (read both):

- `distnnp_model/src/distnnp_model/empirical/lj_potential_model.py` — minimal:
  `FeaturePushToExchange.apply(local_feat, distributed_state, exchanger)` to build
  the haloed position/type tensors, `radius_graph_pbc_cuda(...)` for the local+ghost
  graph, `energy.backward()` → `forces = -local_positions.grad`, mask edges by
  `idx_i < n_local`.
- `distnnp_model/src/distnnp_model/nnp/uma.py:660` `UMAModel` — the real MPNN
  pattern:
  - `self.feature_exchangers = [GPUDirectExchanger() for _ in range(num_layers + 1)]`
  - `initialize_buffers` calls `.setup_exchanger(feature_dim=<per-node feat width>,
    partition_info=..., device=...)` on each, then hands the list to the backbone:
    `self.main_model.backbone._feature_exchangers = self.feature_exchangers`
  - the **backbone** invokes exchanger *i* after interaction layer *i* (UMA gets a
    purpose-built `distributed_escnmd_backbone` from fairchem's registry via
    `overrides={"backbone": {"model": "distributed_escnmd_backbone"}}`).
  - `MaskedHeadWrapper.forward` sums energy over the `data["num_local"]` local
    atoms and does
    `torch.autograd.grad(local_energy * _EV_PER_A_TO_AMU_A_PER_FS2, data.pos)[0]`.
  - `__call__` reduces per-rank forces through the halo via a dummy backward:
    `dummy_full = FeaturePushToExchange.apply(dummy_local, state, self.position_exchanger);
     dummy_full.backward(forces); local_forces = dummy_local.grad`.

### 12.3 The wrapper — target MACE (not NequIP)

MACE is the better second family: high body-order equivariant foundation model
(closer to UMA), `mace-torch` is a clean PyTorch package (`mace.modules.MACE`),
`pair_mace` has a mature Kokkos parallel path, MACE-MP-0 / MACE-OFF are widely
used. NequIP/SevenNet only if a literally-same-arch-as-`pair_e3gnn` comparison
is later wanted.

The hard part: **MACE has no `distributed_*` backbone** like UMA's. `mace.modules.MACE`
runs `for interaction, product, readout in zip(self.interactions, self.products,
self.readouts): ... node_feats = interaction(...)`. The wrapper must inject a
halo exchange on `node_feats` between iterations. Options, cheapest first:
  1. Subclass `mace.modules.MACE`, override `forward` to call
     `self._feature_exchangers[i](node_feats, distributed_state)` after each
     interaction+product (mirrors what `distributed_escnmd_backbone` does
     internally).
  2. If `forward` is monolithic, monkeypatch the module list with wrapper
     `nn.Module`s that each own an exchanger.
`node_feats` is a plain `[N_local+ghost, n_channels * (l_max+1)²]`-ish irreps
tensor → `feature_dim` for `setup_exchanger` is its trailing size. MACE readout
sums per-node energies → apply the masked-local-sum + scaled-`autograd.grad`
exactly as `MaskedHeadWrapper`. MACE default is 2 interaction layers → 3
exchangers.

Files: `distnnp_model/src/distnnp_model/nnp/mace.py` (new, `class MACEModel(DistNNPModel)`),
export from `distnnp_model/src/distnnp_model/nnp/__init__.py`.
Parity gate: `distnnp_model/tests/test_mace_model.py` — distributed (2-rank)
energy+forces vs single-process `mace` on a 200-atom random box, within
`rtol=1e-4` (forces scaled by `_EV_PER_A_TO_AMU_A_PER_FS2`); model via a small
MACE-OFF/MP checkpoint.

### 12.4 De-risk spike FIRST (1–2 days, before the wrapper)

1. Build LAMMPS + `pair_e3gnn` / `pair_e3gnn_parallel` (SevenNet) OR `pair_mace`
   (MACE) on Delta against the `.venv-uma` libtorch. This is a real yak-shave —
   budget it, keep the recipe in `scripts/build_lammps.sh`.
2. Convert `exp_datasets/baselines/ch4_{32,64,128}k_splited.xyz` → LAMMPS data
   files (ASE `write(..., format="lammps-data")`).
3. Run the same Langevin NVT (`fix langevin` + `fix nve`, dt 0.5 fs, T 300 K,
   `damp` = 1/friction) for `warmup 8 + timed 20` steps on 1/2/4 GPU. Record
   `steps/s` (from LAMMPS `Loop time`) and per-rank peak memory
   (`nvidia-smi` sampling or LAMMPS `info`).
4. Run the **naive** path (`pair_nequip`, or `pair_mace` without domain
   decomposition) on `ch4_128k`, `ch4_192k`, `ch4_256k` and confirm it OOMs
   earlier than `pair_e3gnn_parallel` — this is the memory-ceiling headline.
5. **Decide from data:** DistNNP within ~20 % of `pair_e3gnn_parallel` on
   `steps/s` ⇒ build the wrapper + full comparison. Large gap ⇒ reframe the
   paper around the memory ceiling + generality, cite the throughput gap
   honestly, still do the wrapper (it proves model-agnosticism).

### 12.5 LAMMPS baseline engine in the harness

**IMPLEMENTED** (branch `end-to-end-exps`; user-facing docs in `README.md`
"LAMMPS-MACE baseline"). Decisions taken: model = MACE-MP `medium-mpa-0`
(matches the `MACE-Adaptation` adapter); scope = CH₄:O₂ baselines only; the
`pair_style mliap unified` route (per-layer ML-IAP-Kokkos hooks), **not** the
older libtorch `pair_mace`.

- `drivers/run_lammps.py` + `drivers/templates/lammps_nvt.in` — reads an `.xyz`,
  `atoms.wrap()`s the origin-centred CH₄ box, writes a `lammps-data` file
  (`specorder` = element→type map), renders the Langevin-NVT deck (`fix nve` +
  `fix langevin`, `damp = 1/friction` in ps), `subprocess`-spawns
  `srun … lmp -k on g <N> -sf kk …`, parses the **second** `Loop time` (after
  `reset_timers`) + the `Temp` thermo column, emits the same `BenchRecord` JSON
  (`engine="lammps-mace"`). Peak mem = background `nvidia-smi` sampler minus a
  pre-run baseline (needs `--exclusive`). OOM (log markers) → `status="oom"`,
  exit 0; un-attributable crash → exit 1 so `run_cell.sbatch` writes the stub.
- `scripts/build_lammps_env.sh` — the dedicated `.venv-lammps-mace` (torch
  cu126 + `mace-torch` + `cuequivariance*<0.11` + `numpy<2`), pre-fetches the
  model.
- `scripts/build_lammps.sh` — LAMMPS from source: `-C kokkos-cuda.cmake` +
  `PKG_ML-IAP MLIAP_ENABLE_PYTHON PKG_ML-SNAP PKG_PYTHON PKG_KOKKOS
  Kokkos_ARCH_AMPERE80`, linked against the venv's libtorch, `make
  install-python`. Forces g++ as the Kokkos host compiler (nvhpc-hpcx puts nvc++
  on PATH). Verifies the per-layer `forward_exchange`/`reverse_exchange` hooks.
- `scripts/spike_lammps_mace.sbatch` — §12.4 gate: convert model → single-GPU
  (must pass) → 2/4-GPU (**mace#1294** gate) → naive-cutoff OOM check → writes
  `results/spike/VERDICT.txt` (MULTI-GPU OK | SINGLE-GPU ONLY | BLOCKED). No
  energy/force parity step — exp0 is a throughput benchmark, and the mliap path
  was validated once already (see RESULTS below).
- `scripts/run_cell.sbatch`: `lammps-mace)` branch in the engine `case`; the
  `lammps-*` engines use `exp0_check_lammps` + `exp0_load_lammps_modules`
  (separate venv + module set) instead of the UMA ones.
- `scripts/submit_matrix.sh`: `lammps-mace` added to the engine loop **gated**
  on `EXP0_LAMMPS_BIN` existing (or `--with-lammps`); new `--only` values
  `lammps` / `lammps-1gpu` (single-GPU OOM sweep — the fallback headline) /
  `lammps-naive`.
- `scripts/cluster.env` + `.example`: `EXP0_LAMMPS_{ROOT,VENV,BIN,BUILD_DIR,
  MODULES,PYTHON,TORCH,TORCH_INDEX,REF}`, `EXP0_MACE_{MODEL,MODEL_NAME,ELEMENTS}`.
- `analysis/summarize.py`: `ENGINES` gains `lammps-mace`; `headline_table()`
  rewritten to N engines (one steps/s col per engine + a speedup col per
  non-reference engine).
- `scripts/smoke_lammps.sbatch`: tiny `lammps-mace` cell + schema validator.

**Open at build time** (see the session plan file): the exact LAMMPS commit
(`EXP0_LAMMPS_REF`); whether `mliap/kk unified` accepts `-sf kk` in that commit;
`cuequivariance-ops-torch-cu12` ↔ torch-minor compatibility; `--exclusive`
grantability for 1–2 GPU cells.

**RESULTS (2026-08-27, spike job 21497033 + matrix).** Built LAMMPS release
"4 Jul 2026" (`EXP0_LAMMPS_REF=release`), torch 2.7.1+cu126, `mace-torch` 0.3.16,
**`cuequivariance*` pinned `>=0.4,<0.9`** — 0.9/0.10 call
`torch.fx._symbolic_trace.is_fx_symbolic_tracing()` (torch ≥ 2.8) at model
forward; 0.8.1 uses `is_fx_tracing()`. Integration fixes folded into the drivers:
- LAMMPS "4 Jul 2026" **has no `reset_timers` command** — the deck now does a
  plain warmup `run` + timed `run` (parser already takes the last `Loop time`).
- 1-GPU → `lmp` singleton; multi-GPU → HPC-X `mpirun` with
  `--host <node>:16 --map-by slot:OVERSUBSCRIBE -x <embedded-interp vars>`
  (the sbatch job is `--ntasks=1`, so PRRTE otherwise counts 1 slot).
- `_lmp_env()` sets `PYTHONHOME`/`PYTHONPATH`/`LD_LIBRARY_PATH`(+pyhome/lib)/
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`/`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**`mace#1294` does NOT manifest** on this build — 2- and 4-rank MD ran to
completion, no `pack_exchange` crash. A one-off `run 0` energy/force check
(since retired with `spike_energy_check.py`) agreed with the `mace_mp` ASE calc
to |ΔE|/atom 1.1e-8 eV, force MAE 4.1e-7 eV/Å. The spike's auto-gate
(`MULTI_OK`) was too strict (keyed only on 2gpu/32k, which OOM'd); reworked to
accept any multi-rank `status=ok`, and `VERDICT.txt` carries a manual CORRECTED
VERDICT = **MULTI-GPU OK**.

**float64 memory envelope ≈ 8–12k atoms/GPU** (MACE-MPA-medium cuEq TP kernel
needs one ~22 GB intermediate at 16k atoms):

| box | 1 GPU | 2 GPU | 4 GPU |
|---|---|---|---|
| 2k  | 21.1 st/s | 23.2 | — |
| 8k  | 6.5 | 11.3 | 18.1 (2.8× strong scaling; mem 27→14→7.6 GB/rank) |
| 16k | OOM | 6.4 | 11.5 |
| 32k | OOM | OOM | 6.1 |
| ≥64k, `fill`, `lammps-naive` | all OOM | | |

Matrix run as `submit_matrix.sh --with-lammps --engine lammps-mace` (`--only lammps`
matches no sub-experiment). `headline_table()` now **suppresses cross-engine
steps/s ratios when the models differ** (UMA `uma-s-1p2` vs MACE `mace-mpa-0-medium`)
— the real MACE-vs-MACE comparison waits on the DistNNP-MACE adapter.
**Next lever:** re-convert the mliap model with `--dtype float32` (≈½ the
footprint) once the DistNNP-MACE side settles on a dtype.

### 12.6 Framing

The durable contribution is **distributed equivariant-NNP MD for arbitrary
PyTorch models** — LAMMPS pair styles exist for ~3 model families; DistNNP runs
anything that loads in torch, no TorchScript export, no C++ pair style. The
benchmark shows DistNNP (a) matches the sophisticated per-layer LAMMPS path
(`pair_e3gnn_parallel`), (b) beats the naive n-hop pair styles on memory / max
size, (c) generalises across model families (UMA + MACE).
