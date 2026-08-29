"""Shared benchmark harness for the synthetic end-to-end MD ablation.

Both drivers (`run_distnnp_uma.py`, `run_fairchem_uma.py`) import this module so
that argument parsing, the warmup/timed loop, OOM capture, peak-memory
accounting and the on-disk JSON schema are defined exactly once.

See `../implementation_plan.md` section 5 for the design rationale.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import subprocess
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Unit constants
# ---------------------------------------------------------------------------
# Boltzmann constant in eV / K.
BOLTZMANN_EV_PER_K = 8.617333262e-5

# DistNNP's `Langevin(dt, gamma, kT)` BBK kernel
# (`distnnp_integrator/src/csrc/langevin_integrator.cu`) computes the thermal
# kick amplitude as `c2 = sqrt(gamma * kT * dt / m)` with
#   * m  in amu               (from `distnnp_integrator.masses.TORCH_MASSES`)
#   * dt in fs
#   * gamma in 1/fs
#   * velocities in Å/fs
# For that to satisfy equipartition <1/2 m v^2> = 1/2 kT per DOF, `kT` must be
# expressed in amu * Å^2 / fs^2, NOT in eV. Dimensionally
#   1 eV = 1.602176634e-19 J,  1 amu*Å^2/fs^2 = 1.66053907e-17 J
#   => 1 eV = 0.0096485332 amu*Å^2/fs^2   (this is exactly `ase.units.fs**2`)
# NOTE: this constant was previously inverted (1 / 0.0096485332 ≈ 103.64), which
# made `kT_native` ~1e4x too hot and `_kinetic_temperature` ~1e4x too cold.
EV_TO_AMU_A2_PER_FS2 = 0.0096485332

# ASE-side Langevin uses eV / Å / amu / (Å sqrt(amu/eV)) internally and takes
# temperature_K directly, so no manual constant is needed there.


def kT_native(temperature_k: float) -> float:
    """kT in DistNNP-native units (amu * Å^2 / fs^2)."""
    return BOLTZMANN_EV_PER_K * temperature_k * EV_TO_AMU_A2_PER_FS2


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_bench_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic end-to-end UMA-MD benchmark cell",
    )
    p.add_argument("--input", required=True, help="path to an extended-XYZ system")
    p.add_argument(
        "--engine",
        required=True,
        choices=["distnnp-uma", "distnnp-uma-opt", "fairchem-uma",
                 "lammps-mace", "lammps-sevennet"],
        help="distnnp-uma / fairchem-uma run the MATCHED kernel config (exp0's "
        "apples-to-apples axis); distnnp-uma-opt is the same DistNNP driver at "
        "its fastest safe config (turbo: tf32 + no-ackpt + merge_mole + compile).",
    )
    p.add_argument("--model", default="uma-s-1p2")
    p.add_argument("--task", default="omol")

    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--timed-steps", type=int, default=20)

    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--temperature-k", type=float, default=300.0)
    p.add_argument("--friction-inv-fs", type=float, default=0.01)

    # DistNNP only
    p.add_argument("--rebalance-interval", type=int, default=50)
    p.add_argument("--cutoff-angstrom", type=float, default=6.0)
    p.add_argument("--ghost-buffer-angstrom", type=float, default=0.5)
    p.add_argument(
        "--partition-algorithm",
        default="sfc_morton",
        choices=["sfc_morton", "sfc_hilbert", "kdtree", "simple"],
        help="DistNNP spatial partitioner. Use 'kdtree' for anisotropic / "
        "real systems (Workstream A); 'sfc_morton' for the isotropic matrix.",
    )

    # Real-system cells (Workstream A). The synthetic CH4:O2 boxes are cubic +
    # fully periodic; real inputs may be orthorhombic, triclinic, or have no
    # cell at all (heavy-atom PDB dumps).
    p.add_argument(
        "--pbc",
        default="auto",
        choices=["auto", "true", "false"],
        help="auto: use the file's Lattice/pbc; if the file has no usable cell "
        "OR its pbc flags are all false, wrap it in a non-periodic padded box. "
        "true: force the file cell + full PBC. false: always build a "
        "non-periodic padded box (centre atoms, pbc off).",
    )
    p.add_argument(
        "--nonperiodic-pad",
        type=float,
        default=None,
        help="padding (Å) added on every side when a non-periodic box is "
        "synthesised (default: 2*(cutoff + ghost buffer)).",
    )

    # FairChem only (= number of GPUs on the node)
    p.add_argument("--workers", type=int, default=1)

    # ---- UMA InferenceSettings (BOTH engines; see build_inference_settings) --
    # `default` is fairchem's *most general, not speed-optimised* preset:
    # tf32=False, activation_checkpointing=True, merge_mole=False, compile=False.
    # With merge_mole=False / activation_checkpointing=True the predictor also
    # cannot select the `umas_fast_gpu` (Triton) execution backend and silently
    # falls back to `general`. `turbo` flips all four.
    p.add_argument("--inference-settings", default="default",
                   choices=["default", "turbo", "traineval"],
                   help="fairchem InferenceSettings preset applied to BOTH the "
                        "DistNNP UMAModel and the FairChem baseline. The "
                        "individual --tf32/--compile/... flags override fields "
                        "of the chosen preset.")
    p.add_argument("--tf32", type=int, default=-1, choices=[-1, 0, 1],
                   help="InferenceSettings.tf32; -1 = keep the preset's value. "
                        "Large speedup on Ampere+ at a small accuracy cost.")
    p.add_argument("--activation-checkpointing", type=int, default=-1,
                   choices=[-1, 0, 1],
                   help="InferenceSettings.activation_checkpointing; -1 = preset. "
                        "1 trades speed for a much smaller memory footprint; it "
                        "also disqualifies the umas_fast_* execution backends.")
    p.add_argument("--compile", type=int, default=-1, choices=[-1, 0, 1],
                   help="InferenceSettings.compile; -1 = preset. torch.compile "
                        "costs minutes of warmup and recompiles on every new "
                        "shape -- with --warmup-steps in the single digits it is "
                        "usually a net loss.")
    p.add_argument("--merge-mole", type=int, default=-1, choices=[-1, 0, 1],
                   help="InferenceSettings.merge_mole; -1 = preset. 0 = stable "
                        "re-dispatch path; 1 = merged experts (required by the "
                        "umas_fast_gpu backend, but on the FairChem side it "
                        "keeps a persistent NCCL broadcast loop alive in the Ray "
                        "workers and deadlocks on this cluster)")
    p.add_argument("--execution-mode", default=None,
                   choices=["general", "umas_fast_pytorch", "umas_fast_gpu"],
                   help="InferenceSettings.execution_mode; unset = let fairchem "
                        "auto-select (umas_fast_gpu when the preset allows it).")

    # LAMMPS only. The LAMMPS driver is not launched under torchrun, so it
    # cannot read WORLD_SIZE from the env -- the rank count is passed explicitly.
    p.add_argument("--world-size", type=int, default=1,
                   help="LAMMPS: number of MPI ranks / GPUs (srun -n)")
    p.add_argument("--lammps-bin", default=None,
                   help="LAMMPS: path to the `lmp` binary (else EXP0_LAMMPS_BIN)")
    p.add_argument("--mace-model", default=None,
                   help="LAMMPS: path to the MACE .model checkpoint")
    p.add_argument("--mliap-pt", default=None,
                   help="LAMMPS: path to the pre-converted <model>-mliap_lammps.pt "
                        "(else derived as <mace-model>-mliap_lammps.pt)")
    p.add_argument("--elements", default="C H O",
                   help="LAMMPS: element -> type order for pair_coeff (subset of "
                        "the model's trained elements)")
    p.add_argument("--pair-style", default="mliap",
                   choices=["mliap"],
                   help="LAMMPS: MACE coupling style (only `mliap unified` for now)")
    p.add_argument("--comm-cutoff", type=float, default=0.0,
                   help="LAMMPS: `comm_modify cutoff <A>` for the ghost shell. "
                        "0 = let LAMMPS/ML-IAP choose (per-layer hooks); set to "
                        "~n_layers*r_max for the naive extended-cutoff series")
    p.add_argument("--kokkos-comm", default="device",
                   choices=["device", "host"],
                   help="LAMMPS: `-pk kokkos comm <device|host>` (host is the "
                        "fallback if GPU-aware halo exchange crashes -- mace#1294)")
    p.add_argument("--lammps-workdir", default=None,
                   help="LAMMPS: scratch dir for the generated .data / decks / logs")

    # Watchdogs -- an OOM here surfaces as a Ray/NCCL hang, not an exception,
    # so we bound the feasibility phase and each step and call a timeout OOM.
    p.add_argument("--build-timeout-s", type=float, default=420.0,
                   help="wall budget for model load + system setup + the single "
                        "feasibility step; exceeding it is recorded as OOM")
    p.add_argument("--step-timeout-s", type=float, default=90.0,
                   help="wall budget for one timed MD step")

    p.add_argument("--output-dir", default="results")
    p.add_argument("--tag", default=None, help="optional filename suffix")
    p.add_argument(
        "--nvt-check",
        action="store_true",
        help="print mean kinetic temperature over the timed window and exit "
        "(thermostat parity gate, plan section 8.4)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# UMA InferenceSettings  (shared by both engines)
# ---------------------------------------------------------------------------
def build_inference_settings(args):
    """Resolve `args` into a fairchem `InferenceSettings` object.

    Both drivers call this so the two engines are provably running the same
    kernel configuration -- the only thing exp0 wants to vary is the
    *distribution strategy*, not the precision / checkpointing / compile policy.

    `NAME_TO_INFERENCE_SETTING` hands out shared module-level singletons, so we
    deep-ish copy via `dataclasses.replace` before touching any field; mutating
    the preset in place would leak into every later `guess_inference_settings`
    call in the process (including the one inside `UMAModel.__init__`).
    """
    import dataclasses

    from fairchem.core.units.mlip_unit.api.inference import (
        guess_inference_settings,
    )

    settings = dataclasses.replace(
        guess_inference_settings(args.inference_settings)
    )

    overrides: dict[str, Any] = {}
    if args.tf32 >= 0:
        overrides["tf32"] = bool(args.tf32)
    if args.activation_checkpointing >= 0:
        overrides["activation_checkpointing"] = bool(args.activation_checkpointing)
    if args.compile >= 0:
        overrides["compile"] = bool(args.compile)
    if args.merge_mole >= 0:
        overrides["merge_mole"] = bool(args.merge_mole)
    if args.execution_mode is not None:
        overrides["execution_mode"] = args.execution_mode

    if overrides:
        settings = dataclasses.replace(settings, **overrides)
    return settings


def inference_settings_summary(settings) -> dict[str, Any]:
    """The speed/memory-relevant subset of `settings`, for the JSON record.

    `execution_mode=None` means fairchem auto-selects at load time; that
    selection needs `merge_mole=True` *and* `activation_checkpointing=False`
    (see `UMASFastGPUBackend.validate`), so we note what it will resolve to
    rather than recording an ambiguous null.
    """
    auto_fast = (
        settings.execution_mode is None
        and settings.merge_mole
        and not settings.activation_checkpointing
    )
    return {
        "tf32": settings.tf32,
        "activation_checkpointing": settings.activation_checkpointing,
        "merge_mole": settings.merge_mole,
        "compile": settings.compile,
        "execution_mode": settings.execution_mode,
        "execution_mode_auto_eligible_for_fast_gpu": bool(auto_fast),
        "base_precision_dtype": str(settings.base_precision_dtype),
        "internal_graph_gen_version": settings.internal_graph_gen_version,
    }


# ---------------------------------------------------------------------------
# System / cell resolution  (Workstream A -- real, non-cubic inputs)
# ---------------------------------------------------------------------------
@dataclass
class SystemSpec:
    """Simulation cell + PBC resolved from an input file, parsed through the
    SAME reader the DistNNP simulator uses (`distnnp_state.TrajectoryWriter`), so
    both engines get one parse -- element table, velocity-column detection and
    `Lattice=` handling all identical.

    `info` is a `distnnp_state.MolecularInfo` -- pass it straight to
    `Simulator.setup()`; no on-disk round trip. For a non-periodic system its
    `atom_positions` are already shifted into the synthesised box. `to_ase()`
    rebuilds the equivalent `ase.Atoms` for the FairChem driver.
    """
    info: Any                       # distnnp_state.MolecularInfo
    cell: list                      # 3x3 row-vector matrix
    pbc: list                       # [bool, bool, bool]
    box_angstrom: float             # representative size for the JSON record
    num_atoms: int
    periodic: bool

    def to_ase(self):
        import ase
        import numpy as np

        numbers = np.asarray(self.info.atom_types.detach().cpu(), dtype=int).reshape(-1)
        positions = np.asarray(self.info.atom_positions.detach().cpu(), dtype=float)
        return ase.Atoms(numbers=numbers, positions=positions,
                         cell=self.cell, pbc=self.pbc)


def _read_molecular_info(input_path: str):
    """Parse an .xyz/.distnnp file through `distnnp_state.TrajectoryWriter` --
    the exact path `Simulator.setup()` takes -- and also recover the cell/pbc it
    caches from the `Lattice=` / `pbc=` header via `get_cell()` / `get_pbc()`
    (which `MolecularInfoReader.read_struct` drops).

    Returns `(MolecularInfo, cell_3x3_tensor_or_None, pbc_tensor_or_None)`; the
    cell is None when the header carried no usable `Lattice=` (degenerate/zero).
    Imported lazily so the module still imports in the LAMMPS venv.
    """
    import tempfile

    import torch
    from distnnp_state import TrajectoryWriter

    assert os.path.exists(input_path), f"{input_path} could not be found"
    with tempfile.TemporaryDirectory() as scratch:
        # TrajectoryWriter truncates its output on construction, so the read
        # goes through a throwaway scratch file, never input_path.
        with TrajectoryWriter(0, bytes(16), "exp0-load-system", scratch,
                              "_scratch.distnnp") as w:
            info = w.initialize(os.fspath(input_path), 1.0, 1)
            try:
                cell = w.get_cell().detach().clone().to(torch.float32)
                pbc = w.get_pbc().detach().clone()
            except Exception:
                cell, pbc = None, None
    if cell is not None and float(torch.det(cell).abs()) <= 1e-6:
        cell = None
    return info, cell, pbc


def load_system(
    input_path: str,
    *,
    pbc_mode: str = "auto",
    cutoff_angstrom: float = 6.0,
    ghost_buffer_angstrom: float = 0.5,
    nonperiodic_pad: Optional[float] = None,
) -> SystemSpec:
    """Resolve the simulation cell for `input_path`.

    pbc_mode:
      "auto"  -- use the file's `Lattice` + `pbc`. If the file has no usable
                 cell, or its pbc flags are all False, fall back to a
                 non-periodic padded box.
      "true"  -- require a usable file cell; use it with full PBC.
      "false" -- always synthesise a non-periodic padded box (centre atoms).

    A non-periodic box is orthorhombic: side = atom_extent + 2*pad, with
    pad = nonperiodic_pad or 2*(cutoff + ghost_buffer). `info.atom_positions`
    are translated so the system sits at [pad, pad+extent] on every axis (the
    radius-graph kernel wraps fractional coords into [0,1), which is only
    correct if the atoms actually lie inside the box).
    """
    import numpy as np
    import torch

    info, file_cell, file_pbc = _read_molecular_info(input_path)
    n = int(info.num_atoms)
    pos = info.atom_positions.detach().cpu().to(torch.float32)
    extent = (pos.max(0).values - pos.min(0).values).numpy()

    have_cell = file_cell is not None
    have_pbc = bool(file_pbc is not None and file_pbc.to(torch.bool).any())

    pad = nonperiodic_pad
    if pad is None:
        pad = 2.0 * (cutoff_angstrom + ghost_buffer_angstrom)

    if pbc_mode == "true" and not have_cell:
        raise ValueError(f"--pbc true but {input_path} has no usable Lattice= header")

    make_nonperiodic = (
        pbc_mode == "false"
        or (pbc_mode == "auto" and (not have_cell or not have_pbc))
    )

    if make_nonperiodic:
        side = extent + 2.0 * pad
        shifted = pos - pos.min(0).values + pad          # into [pad, pad+extent]
        info.atom_positions = shifted.to(info.atom_positions.dtype)
        cell = np.diag(side)
        pbc = [False, False, False]
        periodic = False
    else:
        cell = file_cell.numpy().astype(float)
        pbc = [bool(x) for x in (
            file_pbc.to(torch.bool).tolist() if have_pbc else [True, True, True]
        )]
        periodic = True

    box_angstrom = float(abs(np.linalg.det(cell)) ** (1.0 / 3.0))
    cell_list = [[float(v) for v in row] for row in np.asarray(cell)]
    return SystemSpec(
        info=info,
        cell=cell_list,
        pbc=[bool(x) for x in pbc],
        box_angstrom=box_angstrom,
        num_atoms=n,
        periodic=periodic,
    )


# ---------------------------------------------------------------------------
# Timed loop
# ---------------------------------------------------------------------------
@dataclass
class LoopResult:
    elapsed_s: float
    steps_per_sec: float
    ns_per_day: float
    timed_steps: int
    warmup_steps: int


class StepTimeout(Exception):
    """A single build/step ran past its wall budget -- in this benchmark that
    almost always means an OOM-induced Ray/NCCL deadlock, so we treat it as OOM
    rather than sit idle burning 4 GPUs until the SLURM walltime."""


@contextmanager
def deadline(seconds: float, label: str):
    """SIGALRM watchdog. Raises StepTimeout(label) if the block runs longer than
    `seconds`. No-op off the main thread or where SIGALRM is unavailable."""
    secs = max(1, int(seconds))

    def _fire(signum, frame):
        raise StepTimeout(f"{label} exceeded {secs}s (treating as OOM-induced hang)")

    try:
        prev = signal.signal(signal.SIGALRM, _fire)
    except (ValueError, OSError, AttributeError):
        yield  # not the main thread / unsupported -- run unguarded
        return
    signal.setitimer(signal.ITIMER_REAL, secs)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def TimedLoop(
    step_fn: Callable[[int], Any],
    warmup: int,
    timed: int,
    timestep_fs: float,
    step_timeout_s: float = 0.0,
) -> LoopResult:
    """Run `warmup` untimed steps, then time `timed` steps.

    `step_fn(i)` runs exactly one MD step. Wall-clock timing brackets a
    `torch.cuda.synchronize()` on both sides so async kernel launches are fully
    accounted. If `step_timeout_s > 0`, each step is guarded by a watchdog that
    raises `StepTimeout` (-> OOM) rather than hanging on a dead worker.
    """
    def _step(idx: int) -> None:
        if step_timeout_s > 0:
            with deadline(step_timeout_s, f"MD step {idx}"):
                step_fn(idx)
        else:
            step_fn(idx)

    for i in range(warmup):
        _step(i)

    _cuda_sync()
    t0 = time.perf_counter()
    for i in range(timed):
        _step(warmup + i)
    _cuda_sync()
    elapsed = time.perf_counter() - t0

    sps = timed / elapsed if elapsed > 0 else float("nan")
    ns_per_day = timestep_fs * 1e-6 * sps * 86400.0
    return LoopResult(
        elapsed_s=elapsed,
        steps_per_sec=sps,
        ns_per_day=ns_per_day,
        timed_steps=timed,
        warmup_steps=warmup,
    )


# ---------------------------------------------------------------------------
# OOM / error capture
# ---------------------------------------------------------------------------
@dataclass
class RunOutcome:
    status: str  # "ok" | "oom" | "error"
    error: Optional[str] = None
    payload: Any = None


def capture_oom(fn: Callable[[], Any]) -> RunOutcome:
    """Run `fn`; never re-raise, so a sweep continues.

    A `torch.cuda.OutOfMemoryError` (or a CUDA OOM surfaced as a generic
    RuntimeError with "out of memory" in the message) becomes
    `status="oom"`; anything else becomes `status="error"` with a repr.
    """
    try:
        import torch

        oom_types: tuple = (torch.cuda.OutOfMemoryError,)
    except Exception:
        oom_types = tuple()

    try:
        payload = fn()
        return RunOutcome(status="ok", payload=payload)
    except StepTimeout as e:
        return RunOutcome(status="oom", error=repr(e))
    except oom_types as e:  # type: ignore[misc]
        return RunOutcome(status="oom", error=repr(e))
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda oom" in msg:
            return RunOutcome(status="oom", error=repr(e))
        return RunOutcome(status="error", error=repr(e) + "\n" + traceback.format_exc())
    except Exception as e:  # noqa: BLE001
        return RunOutcome(status="error", error=repr(e) + "\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Peak memory
# ---------------------------------------------------------------------------
def reset_peak_mem() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_mem_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024.0**2)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Result record + JSON writer
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _env_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for mod, key in (
        ("torch", "torch"),
        ("fairchem.core", "fairchem_core"),
        ("ray", "ray"),
        ("e3nn", "e3nn"),
        ("numpy", "numpy"),
        ("ase", "ase"),
        ("mace", "mace"),
        ("cuequivariance_torch", "cuequivariance_torch"),
        ("lammps", "lammps"),
    ):
        try:
            m = __import__(mod, fromlist=["__version__"])
            ver = getattr(m, "__version__", None)
            if ver is None and hasattr(m, "version"):
                # the `lammps` python module exposes version() -> int (YYYYMMDD)
                try:
                    ver = m.version()
                except Exception:
                    ver = None
            out[key] = str(ver) if ver is not None else "unknown"
        except Exception:
            out[key] = "not-installed"
    return out


@dataclass
class BenchRecord:
    engine: str
    model: str
    task: str
    integrator: str
    num_atoms: int
    box_angstrom: float
    world_size: int
    gpus_per_node: int
    nodes: int
    warmup_steps: int
    timed_steps: int
    timestep_fs: float
    temperature_k: float
    friction_inv_fs: float
    status: str
    input_file: str = ""
    tag: Optional[str] = None
    # resolved UMA InferenceSettings (see build_inference_settings); None for
    # engines that do not go through fairchem, e.g. lammps-mace.
    inference_settings: Optional[dict] = None
    elapsed_s: Optional[float] = None
    steps_per_sec: Optional[float] = None
    ns_per_day: Optional[float] = None
    peak_mem_mb_per_rank: Optional[list[float]] = None
    peak_mem_mb_max: Optional[float] = None
    mean_temperature_k: Optional[float] = None
    error: Optional[str] = None
    env: dict[str, str] = field(default_factory=_env_versions)
    git_commit: str = field(default_factory=_git_commit)
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def result_filename(record: BenchRecord) -> str:
    stem = f"{record.engine}_{record.num_atoms}atoms_{record.world_size}gpu"
    if record.tag:
        stem += f"_{record.tag}"
    return stem + ".json"


def write_result(output_dir: str, record: BenchRecord) -> str:
    """rank-0-only JSON write. Returns the path written."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, result_filename(record))
    with open(path, "w") as fh:
        fh.write(record.to_json())
        fh.write("\n")
    return path
