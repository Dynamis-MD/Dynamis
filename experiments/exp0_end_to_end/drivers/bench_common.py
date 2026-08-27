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
        choices=["distnnp-uma", "fairchem-uma"],
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

    # FairChem only (= number of GPUs on the node)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--merge-mole", type=int, default=0,
                   help="fairchem InferenceSettings.merge_mole (0/1). 0 = stable "
                        "re-dispatch path; 1 = persistent NCCL loop (faster, "
                        "deadlocks on this cluster)")

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
    ):
        try:
            m = __import__(mod, fromlist=["__version__"])
            out[key] = getattr(m, "__version__", "unknown")
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
