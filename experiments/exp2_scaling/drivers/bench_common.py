"""Model-agnostic benchmark harness for exp2_scaling (Dynamis distributed MD).

Shared across all NNP scaling drivers (ORB, UMA, MACE, ...). Provides:
  - Base CLI argument parser (`build_base_parser`) that model drivers extend
  - TrajectoryReader-based cell/PBC loading (`load_system`)
  - Per-rank O(N/W) scalable halo wrapper (`ScalableHaloPartitioner`, `wrap_scalable_halo`)
  - Cell/PBC injection (`inject_cell`), Maxwell-Boltzmann thermalization (`thermalize`),
    and distributed kinetic temperature (`kinetic_temperature`)
  - Wall-clock `TimedLoop`, SIGALRM `deadline`, OOM/error capture (`capture_oom`),
    and per-rank GPU memory / atom / ghost gathering (`gather_rank_metrics`)
  - Self-describing JSON result record (`BenchRecord`, `write_result`)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Unit constants (Dynamis native units: amu, Å, fs)
# ---------------------------------------------------------------------------
BOLTZMANN_EV_PER_K = 8.617333262e-5
EV_TO_AMU_A2_PER_FS2 = 0.0096485332  # exact ase.units.fs**2


def kT_native(temperature_k: float) -> float:
    """kT in Dynamis-native units (amu * Å^2 / fs^2)."""
    return BOLTZMANN_EV_PER_K * temperature_k * EV_TO_AMU_A2_PER_FS2


# ---------------------------------------------------------------------------
# Base argument parser (model drivers extend this with model-specific flags)
# ---------------------------------------------------------------------------
def build_base_parser(
    description: str = "Dynamis distributed MD scaling benchmark cell",
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--input", required=True, help="Path to extended-XYZ input structure")
    p.add_argument(
        "--exchanger",
        default="gpudirect",
        choices=["gpudirect", "nccl"],
        help="gpudirect = MNNVL Fabric / CUDA IPC; nccl = torch.distributed all_to_all",
    )
    p.add_argument(
        "--partition-algorithm",
        default="sfc_morton",
        choices=["sfc_morton", "sfc_hilbert", "kdtree", "simple"],
    )
    p.add_argument(
        "--scalable-halo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-rank O(N/W) halo builder (radius_graph_pbc_ex + extract_halo_info)",
    )
    p.add_argument(
        "--drift-watchdog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Simulator drift watchdog during benchmark loop (default off for steady-state timing)",
    )

    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--timed-steps", type=int, default=50)
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--temperature-k", type=float, default=300.0)
    p.add_argument("--friction-inv-fs", type=float, default=0.01)
    p.add_argument("--rebalance-interval", type=int, default=1000)
    p.add_argument("--cutoff-angstrom", type=float, default=6.0)
    p.add_argument("--ghost-buffer-angstrom", type=float, default=0.5)

    p.add_argument(
        "--pbc",
        default="auto",
        choices=["auto", "true", "false"],
    )
    p.add_argument("--nonperiodic-pad", type=float, default=None)

    p.add_argument("--build-timeout-s", type=float, default=480.0)
    p.add_argument("--step-timeout-s", type=float, default=120.0)

    p.add_argument(
        "--study",
        default="strong",
        choices=["strong", "weak", "hero", "probe"],
    )
    p.add_argument("--tag", default=None, help="Optional filename/grouping tag")
    p.add_argument("--output-dir", default="results")
    return p


# ---------------------------------------------------------------------------
# System loading via Dynamis TrajectoryReader
# ---------------------------------------------------------------------------
@dataclass
class SystemSpec:
    info: Any  # dynamis_state.MolecularInfo
    cell: list[list[float]]
    pbc: list[bool]
    box_angstrom: float
    num_atoms: int
    periodic: bool


def load_system(
    input_path: str,
    *,
    pbc_mode: str = "auto",
    cutoff_angstrom: float = 6.0,
    ghost_buffer_angstrom: float = 0.5,
    nonperiodic_pad: Optional[float] = None,
) -> SystemSpec:
    import numpy as np
    import torch
    from dynamis_state import TrajectoryReader

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    info, file_cell, file_pbc = TrajectoryReader.read(os.fspath(input_path))
    n = int(info.num_atoms)
    pos = info.atom_positions.detach().cpu().to(torch.float32)
    extent = (pos.max(0).values - pos.min(0).values).numpy()

    have_cell = (
        file_cell is not None
        and float(torch.det(file_cell.to(torch.float32)).abs()) > 1e-6
    )
    have_pbc = bool(file_pbc is not None and file_pbc.to(torch.bool).any())

    pad = (
        nonperiodic_pad
        if nonperiodic_pad is not None
        else 2.0 * (cutoff_angstrom + ghost_buffer_angstrom)
    )
    if pbc_mode == "true" and not have_cell:
        raise ValueError(f"--pbc true but {input_path} has no usable Lattice=")

    make_nonperiodic = pbc_mode == "false" or (
        pbc_mode == "auto" and (not have_cell or not have_pbc)
    )

    if make_nonperiodic:
        side = extent + 2.0 * pad
        shifted = pos - pos.min(0).values + pad
        info.atom_positions = shifted.to(info.atom_positions.dtype)
        cell = np.diag(side)
        pbc = [False, False, False]
        periodic = False
    else:
        cell = file_cell.detach().cpu().numpy().astype(float)
        pbc = [
            bool(x)
            for x in (
                file_pbc.to(torch.bool).tolist() if have_pbc else [True, True, True]
            )
        ]
        periodic = True

    box_angstrom = float(abs(np.linalg.det(cell)) ** (1.0 / 3.0))
    cell_list = [[float(v) for v in row] for row in np.asarray(cell)]
    return SystemSpec(
        info=info,
        cell=cell_list,
        pbc=pbc,
        box_angstrom=box_angstrom,
        num_atoms=n,
        periodic=periodic,
    )


# ---------------------------------------------------------------------------
# Model-agnostic Dynamis Simulator helpers
# ---------------------------------------------------------------------------
class ScalableHaloPartitioner:
    """Per-rank O(N/W) halo builder using dynamis_state_impl.radius_graph_pbc_ex
    and extract_halo_info.

    Avoids AbstractPartitioner::build_halo_maps's all-atom radius graph and [N, W]
    boolean matrix on every rank, saving several GiB/GPU on million-atom systems.
    """

    def __init__(self, inner, rank: int, world_size: int):
        self._inner = inner
        self._rank = int(rank)
        self._world_size = int(world_size)

    def partition(self, molecular_system, dist_env, *args, **kwargs):
        return self._inner.partition(molecular_system, dist_env, *args, **kwargs)

    def build_halo_maps(self, info, global_state, dist_env, config) -> None:
        import torch
        import dynamis_state.dynamis_state_impl as _impl

        pos = global_state.atom_positions
        dev = pos.device
        n_atoms = int(global_state.num_atoms)
        r = float(config.cutoff) + float(config.ghost_cutoff_buffer)
        cell = config.cell.to(dev).to(torch.float32)
        pbc = config.pbc.to(dev).to(torch.bool)

        pid = info.partition_id.to(device=dev, dtype=torch.long).contiguous()
        my_outer = (
            torch.nonzero(pid.eq(self._rank), as_tuple=False).squeeze(-1).contiguous()
        )
        num_local = int(my_outer.numel())

        # Halo maps never cap neighbors (max_num_neighbors=0)
        edges, _cell_shifts, _nbi, _nc = _impl.radius_graph_pbc_ex(
            pos.to(torch.float32),
            cell,
            pbc,
            r,
            0,
            False,
            my_outer,
            None,
        )
        (
            send_off,
            send_local_idx,
            recv_off,
            recv_ghost_idx,
            interior_mask,
            _recv_global,
        ) = _impl.extract_halo_info(
            edges, pid, my_outer, self._rank, self._world_size, n_atoms
        )

        info.send_offsets = send_off
        info.send_local_indices = send_local_idx
        info.recv_offsets = recv_off
        info.recv_ghost_indices = recv_ghost_idx
        info.interior_mask = interior_mask
        info.edge_index_halo = edges

        m = info.global_send_recv_matrix
        counts = (send_off[1:] - send_off[:-1]).to(m.dtype)
        m[self._rank].copy_(counts)
        m[self._rank, self._rank] = num_local
        info.global_send_recv_matrix = m


def inject_cell(sim, cell, pbc) -> None:
    """Inject resolved (cell, pbc) from TrajectoryReader into SimulatorConfig."""
    import torch

    cell_t = torch.as_tensor(cell, dtype=torch.float32)
    pbc_t = torch.as_tensor(pbc, dtype=torch.bool)
    _orig = sim._get_simulation_config

    def _patched():
        sc = _orig()
        sc.cell = cell_t
        sc.pbc = pbc_t
        return sc

    sim._get_simulation_config = _patched


def wrap_scalable_halo(sim, rank: int, world_size: int) -> None:
    """Wrap Simulator's partitioner with ScalableHaloPartitioner."""
    _orig = sim._get_partitioner

    def _patched():
        return ScalableHaloPartitioner(_orig(), rank, world_size)

    sim._get_partitioner = _patched


def thermalize(sim, temperature_k: float, seed: int) -> None:
    """Initialize local atom velocities from a Maxwell-Boltzmann distribution."""
    import torch
    from dynamis_state import ATOMIC_MASSES

    local = sim.state.local_state
    n_local = int(local.num_atoms)
    dev = local.atom_velocities.device
    masses = ATOMIC_MASSES.to(dev)[
        local.atom_types[:n_local].reshape(-1).long()
    ].unsqueeze(-1)
    kb_native = BOLTZMANN_EV_PER_K * EV_TO_AMU_A2_PER_FS2
    sigma = torch.sqrt(
        torch.as_tensor(kb_native * temperature_k, device=dev) / masses
    )
    g = torch.Generator(device=dev).manual_seed(seed)
    local.atom_velocities[:n_local] = (
        torch.randn((n_local, 3), generator=g, device=dev) * sigma
    )


def kinetic_temperature(sim) -> float:
    """Compute global kinetic temperature (K) across all distributed ranks."""
    import torch
    import torch.distributed as dist
    from dynamis_state import ATOMIC_MASSES

    local = sim.state.local_state
    n_local = int(local.num_atoms)
    vel = local.atom_velocities[:n_local]
    masses = ATOMIC_MASSES.to(vel.device)[
        local.atom_types[:n_local].reshape(-1).long()
    ]
    ke = (0.5 * torch.sum(masses.unsqueeze(-1) * vel * vel)).detach().clone()
    n = torch.tensor(float(n_local), device=vel.device)
    if dist.is_initialized():
        dist.all_reduce(ke, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
    kb_native = BOLTZMANN_EV_PER_K * EV_TO_AMU_A2_PER_FS2
    return float((2.0 * ke.item()) / (3.0 * float(n.item()) * kb_native))


def resolve_exchanger_cls(exchanger_name: str):
    """Map CLI exchanger name ('gpudirect' | 'nccl') to Dynamis exchanger class."""
    from dynamis_model.comm.exchanger import GPUDirectExchanger, PythonNCCLExchanger

    mapping = {
        "gpudirect": GPUDirectExchanger,
        "nccl": PythonNCCLExchanger,
    }
    if exchanger_name not in mapping:
        raise ValueError(f"Unsupported exchanger: {exchanger_name!r}")
    return mapping[exchanger_name]


def detect_use_fabric(model, exchanger_name: str) -> Optional[bool]:
    """Query whether a GPUDirectExchanger is using MNNVL CUDA Fabric handles."""
    if exchanger_name != "gpudirect":
        return None
    try:
        exch = getattr(model, "exchanger", None) or getattr(
            model, "_feature_exchanger", None
        )
        return bool(exch.exchanger.use_fabric()) if exch is not None else None
    except Exception:
        return None


def gather_rank_metrics(
    sim, my_peak_mb: float, world_size: int
) -> tuple[list[float], list[int], list[int], list[str]]:
    """Gather peak memory, local/ghost atom counts, and unique hostnames across ranks."""
    import torch.distributed as dist

    my_local = int(sim.state.local_state.num_atoms)
    my_ghost = int(sim.state.local_partition_info.recv_offsets[-1].item())
    peaks = [my_peak_mb]
    local_counts = [my_local]
    ghost_counts = [my_ghost]
    hostnames = [socket.gethostname()]

    if dist.is_initialized() and world_size > 1:
        with deadline(60, "gather rank metrics"):
            p_list: list[Any] = [None] * world_size
            l_list: list[Any] = [None] * world_size
            g_list: list[Any] = [None] * world_size
            h_list: list[Any] = [None] * world_size
            dist.all_gather_object(p_list, my_peak_mb)
            dist.all_gather_object(l_list, my_local)
            dist.all_gather_object(g_list, my_ghost)
            dist.all_gather_object(h_list, socket.gethostname())
        peaks = [float(x) for x in p_list if x is not None]
        local_counts = [int(x) for x in l_list if x is not None]
        ghost_counts = [int(x) for x in g_list if x is not None]
        hostnames = sorted({str(x) for x in h_list if x is not None})

    return peaks, local_counts, ghost_counts, hostnames


# ---------------------------------------------------------------------------
# Timed loop & watchdogs
# ---------------------------------------------------------------------------
@dataclass
class LoopResult:
    elapsed_s: float
    steps_per_sec: float
    ns_per_day: float
    timed_steps: int
    warmup_steps: int


class StepTimeout(Exception):
    """Raised when a build or step exceeds its wall budget (typically OOM hang)."""


@contextmanager
def deadline(seconds: float, label: str):
    secs = max(1, int(seconds))

    def _fire(signum, frame):
        raise StepTimeout(f"{label} exceeded {secs}s (treating as OOM/hang)")

    try:
        prev = signal.signal(signal.SIGALRM, _fire)
    except (ValueError, OSError, AttributeError):
        yield
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
    on_timed_start: Optional[Callable[[], None]] = None,
) -> LoopResult:
    def _step(idx: int) -> None:
        if step_timeout_s > 0:
            with deadline(step_timeout_s, f"MD step {idx}"):
                step_fn(idx)
        else:
            step_fn(idx)

    for i in range(warmup):
        _step(i)

    _cuda_sync()
    if on_timed_start is not None:
        on_timed_start()
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
# OOM / error capture & peak memory
# ---------------------------------------------------------------------------
@dataclass
class RunOutcome:
    status: str  # "ok" | "oom" | "error"
    error: Optional[str] = None
    payload: Any = None


def capture_oom(fn: Callable[[], Any]) -> RunOutcome:
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


def env_versions(extra_modules: Iterable[str] = ()) -> dict[str, str]:
    out: dict[str, str] = {}
    mods = ["torch", "numpy", "scipy", *extra_modules]
    for mod in dict.fromkeys(mods):
        try:
            m = __import__(mod, fromlist=["__version__"])
            out[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            out[mod] = "not-installed"
    return out


def short_model_name(model_name: str) -> str:
    """Compact filename-friendly model slug."""
    if model_name.startswith("orb_v3_direct_"):
        return "orb_direct_" + model_name[len("orb_v3_direct_") :]
    if model_name.startswith("orb_v3_conservative_"):
        return "orb_cons_" + model_name[len("orb_v3_conservative_") :]
    return model_name.replace("/", "_").replace(" ", "_")


@dataclass
class BenchRecord:
    study: str
    model: str
    model_family: str
    precision: str
    max_num_neighbors: int
    exchanger: str
    use_fabric: Optional[bool]
    partition_algorithm: str
    scalable_halo: bool
    integrator: str
    num_atoms: int
    atoms_per_gpu: float
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
    local_atoms_per_rank: Optional[list[int]] = None
    ghost_atoms_per_rank: Optional[list[int]] = None
    timing_dir: Optional[str] = None
    hostnames: Optional[list[str]] = None
    slurm_job_id: Optional[str] = None
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=env_versions)
    git_commit: str = field(default_factory=_git_commit)
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def result_stem(
    model: str,
    exchanger: str,
    num_atoms: int,
    world_size: int,
    study: str,
    tag: Optional[str] = None,
) -> str:
    m = short_model_name(model)
    stem = f"{m}_{exchanger}_{study}_{num_atoms}atoms_{world_size}gpu"
    if tag:
        stem += f"_{tag}"
    return stem


def result_filename(record: BenchRecord) -> str:
    return (
        result_stem(
            record.model,
            record.exchanger,
            record.num_atoms,
            record.world_size,
            record.study,
            record.tag,
        )
        + ".json"
    )


def write_result(output_dir: str, record: BenchRecord) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, result_filename(record))
    with open(path, "w") as fh:
        fh.write(record.to_json())
        fh.write("\n")
    return path
