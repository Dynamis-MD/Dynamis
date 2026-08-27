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
sites. **Gate:** the env is only "done" when the parity test in §8.1 passes.

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
  This is calibrated/confirmed by the NVT check in §8.4 — if temperature comes
  out wrong, fix the constant here only.
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
  and the `.cu` kernel; the NVT check in §8.4 is the acceptance gate.
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

1. **Env / parity gate** — after `env_setup.sh`, run the existing UMA test on
   2 GPUs (tiny, ~2 min):
   ```bash
   torchrun --nproc_per_node=2 DistNNP/distnnp_model/tests/run_distributed_trajectory.py --path $SCRATCH/uma_check
   python DistNNP/distnnp_model/tests/test_uma_md.py --path $SCRATCH/uma_check   # must print PASSED
   ```
   If this fails on imports → fairchem API drift (§3 risk) — pin closer to 2.16.
2. **`scripts/smoke.sbatch`** (≤3 min, 2 GPUs): run `run_fairchem_uma.py`
   (`--workers 2`) and `run_distnnp_uma.py` on `ch4_1k_splited.xyz` with
   `--warmup-steps 3 --timed-steps 10`. Assert each wrote a schema-valid
   `results/*.json` with `status=="ok"` and `0 < steps_per_sec < 1e4`.
3. **OOM path** — one `run_cell.sbatch fairchem-uma ch4_1024k_splited.xyz 1`
   job: must terminate (crash or `capture_oom`) and leave a
   `status in {oom,error}` JSON, **not** hang past the wall cap.
4. **Thermostat parity** — 1 job per engine: 100 Langevin steps at 300 K on
   `ch4_2k_splited.xyz`, print mean kinetic temperature over the last 50 steps.
   Both must land at **300 ± ~30 K**. If DistNNP is off, fix
   `EV_TO_AMU_A2_PER_FS2` / `kT_native` in `bench_common.py` only.
5. **Full matrix** — `scripts/submit_matrix.sh`; when the queue drains,
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

## 10. Explicitly out of scope (future work)

Real systems (proteins / Pt-water / membrane / capsid), the SevenNet+LAMMPS
baseline, multi-node (8–32 GPU), the per-phase profiler breakdown, and the
partition-algorithm ablation. The `03_generate_ch4.py` CH₄:O₂ composition / O₂
spin question is **not** to be fixed here — use the generator as-is; if it
changes, `scripts/gen_datasets.sh` + `scripts/submit_matrix.sh` re-run cleanly.
The harness (`bench_common.py` schema, `run_cell.sbatch`, `submit_matrix.sh`) is
built so new sizes / input files / GPU counts are one-line additions.
