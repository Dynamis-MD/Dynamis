"""
Shared constants and helpers for the distributed-vs-global trajectory
equivalence check.

Two-phase comparison
---------------------
Two independent single-GPU reference runs of the SAME simulation diverge
from each other by ~1e-5 at step 0 (irreducible index_reduce_cuda reduction-
order noise) and by ~1e-3-4e-3 by step 6 — i.e. chaotic (Lyapunov-style)
amplification of that noise, not incremental drift. Past a handful of steps,
NO fixed per-atom tolerance can distinguish "expected chaos" from "an actual
distributed bug", because two correct runs look exactly as different as one
correct run and one buggy run.

So the comparison is split:
  - Phase 1 (steps 0..EARLY_PHASE_STEPS): tight per-atom position/velocity/
    energy checks. Early enough that chaotic amplification hasn't kicked in
    yet; this phase actually validates forward/backward correctness.
  - Phase 2 (steps EARLY_PHASE_STEPS+1..N_STEPS-1): per-atom equality is
    dropped in favor of three aggregate, chaos-tolerant checks:
      1. RMSD-of-difference between the two runs (washes out single-atom
         chaotic outliers; a broken run moves the whole aggregate, not just 1-2
         atoms).
      2. Per-run energy drift (a velocity-Verlet/NVE run should show bounded
         O(dt^2) oscillation around E(0), no secular drift — this needs no
         cross-run comparison at all, and specifically catches non-
         conservative-force bugs like a broken ghost-gradient path). The
         distributed run's drift is compared against the REFERENCE run's own
         observed drift (not a hardcoded eV value we can't justify a priori),
         since that's the real noise floor for this system/timestep.
      3. RDF similarity over the last few frames — a coarse structural
         sanity check, loose by design (64 atoms / 1-5 frames is not enough
         for a precise g(r)), reported but only failing on gross mismatch.

Calibration
-----------
The Phase-2 numeric bounds (_pos_rmsd_tolerance, _vel_rmsd_tolerance,
ENERGY_DRIFT_MARGIN, RDF_TOL) below are reasoned starting points, NOT
empirically measured. Before trusting this test, run
`reference_vs_reference_check()` (bottom of this file) — it runs the
world_size=1 reference twice and reports the RMSD-of-difference between two
runs that have zero distributed code involved. If your chosen Phase-2
schedule doesn't comfortably exceed that self-comparison at every step, it
will produce false failures no matter what the distributed code does.
"""
import os
import socket

import numpy as np

N_STEPS = 100  # long enough for statistical read on energy evolution
REBALANCE_INTERVAL = 7  # ensures multiple repartitions occur
SEED = 1234
N_ATOMS = 64
SIDE_LENGTH_ANGSTROM = 20.0
ELEMENT_SYMBOL = "He"  # noble gas
MIN_ATOM_DISTANCE_ANGSTROM = 3.0  # He-He equilibrium separation is ~3 A

# ---- Phase 1 (tight, per-atom) ----
EARLY_PHASE_STEPS = 4  # inclusive upper bound for phase 1
POS_VEL_TOL = 2e-5
ENERGY_RTOL = 1e-3

# ---- Phase 2 (aggregate, chaos-tolerant) ----
# Calibrated by floor+slope least-squares fit combining two independent full
# N_STEPS=100 reference_vs_reference_check() runs (see conversation history).
_RESOLUTION_FLOOR = 1e-6
_POS_NOISE_RMSD_FLOOR = 1e-6
_POS_NOISE_RMSD_SLOPE_PER_STEP = 5.6e-7
_VEL_NOISE_RMSD_FLOOR = _RESOLUTION_FLOOR
_VEL_NOISE_RMSD_SLOPE_PER_STEP = 3.7e-8
RMSD_MARGIN = 30.0


def _pos_rmsd_tolerance(step: int) -> float:
    noise = _POS_NOISE_RMSD_FLOOR + max(0, step - 5) * _POS_NOISE_RMSD_SLOPE_PER_STEP
    return RMSD_MARGIN * noise


def _vel_rmsd_tolerance(step: int) -> float:
    noise = _VEL_NOISE_RMSD_FLOOR + max(0, step - 5) * _VEL_NOISE_RMSD_SLOPE_PER_STEP
    return RMSD_MARGIN * noise


# Per-run energy-drift statistics (see energy_drift_slope/energy_oscillation_std
# below) are compared against the reference run's own observed values, same
# margin-over-reference-noise-floor approach as the RMSD checks above, with a
# floor for when the reference's own slope/std happens to be near zero.
ENERGY_DRIFT_MARGIN = 3.0
ENERGY_SLOPE_FLOOR_EV_PER_STEP = 1e-3
ENERGY_STD_FLOOR_EV = 1e-2

RDF_LAST_K_FRAMES = 20
RDF_N_BINS = 50
RDF_R_MAX = 8.0
RDF_TOL = 0.3


def _generate_positions(
    n_atoms: int,
    cell_length: float,
    min_distance: float,
    seed: int,
    max_attempts_per_atom: int = 200,
) -> np.ndarray:
    """Rejection-sample positions under periodic minimum-image distance.
    """
    rng = np.random.default_rng(seed)
    positions = np.empty((n_atoms, 3))
    for i in range(n_atoms):
        placed = False
        for _ in range(max_attempts_per_atom):
            candidate = rng.uniform(0, cell_length, size=3)
            if i == 0:
                positions[i] = candidate
                placed = True
                break
            diffs = positions[:i] - candidate
            diffs -= cell_length * np.round(diffs / cell_length)  # minimum image
            dists = np.sqrt(np.sum(diffs**2, axis=-1))
            if np.all(dists >= min_distance):
                positions[i] = candidate
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Could not place atom {i}/{n_atoms} with min_distance={min_distance}"
                f" after {max_attempts_per_atom} attempts each"
            )
    return positions


def ensure_test_structure(
    path: str,
    n_atoms: int = N_ATOMS,
    seed: int = SEED,
    min_distance: float = MIN_ATOM_DISTANCE_ANGSTROM,
) -> None:
    """Writes a small ASE-readable .xyz if `path` doesn't already exist.
    Idempotent and shared by path, not regenerated independently per script.
    """
    if os.path.exists(path):
        return
    from ase import Atoms
    from ase.io import write

    positions = _generate_positions(n_atoms, SIDE_LENGTH_ANGSTROM, min_distance, seed)
    atoms = Atoms(
        ELEMENT_SYMBOL * n_atoms, positions=positions, cell=[SIDE_LENGTH_ANGSTROM] * 3, pbc=True
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write(path, atoms)


def free_port() -> int:
    """Bind to port 0 to get an OS-assigned free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def real_singleton_torch_dist_env():
    """A genuine (not skipped) world_size=1 process group.

    Exchanger._is_single_rank() requires dist.is_initialized() == True to
    take its identity fast path — TorchDistEnv(is_distributed=False), which
    never calls init_process_group, does NOT satisfy that and will instead
    crash on `assert dist.is_initialized()` inside setup_exchanger. A real,
    trivial single-rank group is required instead.

    rank/world_size are passed explicitly rather than read from RANK/
    WORLD_SIZE env vars, and given its own isolated rendezvous port — not to
    avoid colliding with anything else (this is the only process group this
    process ever opens), just because dist.init_process_group needs some
    rendezvous address and none is provided when launched standalone
    (no torchrun).
    """
    from dynamis_state import TorchDistEnv

    port = free_port()
    return TorchDistEnv(
        is_distributed=True,
        rank=0,
        world_size=1,
        init_method=f"tcp://127.0.0.1:{port}",
    )


def teardown_singleton_dist_env():
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def make_config(output_dir: str):
    from dynamis import SimulatorConfig

    os.makedirs(output_dir, exist_ok=True)
    return SimulatorConfig(
        timestep_fs=0.5,
        total_steps=N_STEPS,
        side_length_angstrom=SIDE_LENGTH_ANGSTROM,
        cutoff_angstrom=6.0,
        partition_algorithm="sfc_morton",
        edge_weighted_partition=True,
        rebalance_interval=REBALANCE_INTERVAL,
        traj_interval=0,
        log_interval=1,
        output_dir=output_dir,
        backend="nccl",
    )


def extract_frame(sim, energy) -> dict:
    partition_info, mol_info = sim.state.local_partition_info, sim.state.local_state
    vid = partition_info.partition_indices[: mol_info.num_atoms].cpu().numpy()
    order = np.argsort(vid)
    return {
        "energy": float(energy),
        "pos": mol_info.atom_positions.detach().cpu().numpy()[order],
        "vel": mol_info.atom_velocities.detach().cpu().numpy()[order],
        "vid": vid[order],
    }


def run_md_loop(sim, config):
    """Shared step loop: identical code path for the reference (world_size=1)
    and distributed runs, so a repartition-with-one-rank bug would show up
    here too rather than being silently skipped on the reference side.
    """
    frames = []
    for step in range(N_STEPS):
        if sim.should_rebalance(step):
            sim.sync_and_repartition()
        energy, _ = sim.step(config.timestep_fs)
        frames.append(extract_frame(sim, energy))
    return frames


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def rmsd_between_trajectories(traj_a, traj_b) -> dict:
    """Per-step RMSD-of-difference, for both use in the real comparison and
    for the reference_vs_reference_check() calibration helper below.
    """
    assert len(traj_a) == len(traj_b)
    return {
        "pos": [rmsd(a["pos"], b["pos"]) for a, b in zip(traj_a, traj_b)],
        "vel": [rmsd(a["vel"], b["vel"]) for a, b in zip(traj_a, traj_b)],
    }


def compute_energy_drift(traj) -> np.ndarray:
    """|E(t) - E(0)| for a single trajectory. Kept as a simple diagnostic
    print alongside the slope-based check below, which is the actual gating
    metric.
    """
    e0 = traj[0]["energy"]
    return np.array([abs(f["energy"] - e0) for f in traj])


def energy_drift_slope(traj) -> float:
    """Linear-regression slope of E(t) vs. step index, in eV/step.

    A conservative-force integrator should show ~0 slope (small oscillation). A
    nonzero, run-independent slope points at systematic non-conservative forces.
    """
    energies = np.array([f["energy"] for f in traj])
    steps = np.arange(len(energies))
    slope, _ = np.polyfit(steps, energies, 1)
    return float(slope)


def energy_oscillation_std(traj) -> float:
    """Std of E(t) about its own linear trend -- the bounded-oscillation
    amplitude.
    """
    energies = np.array([f["energy"] for f in traj])
    steps = np.arange(len(energies))
    slope, intercept = np.polyfit(steps, energies, 1)
    residual = energies - (slope * steps + intercept)
    return float(np.std(residual))


def compute_rdf(positions: np.ndarray, cell_length: float, r_max=RDF_R_MAX, n_bins=RDF_N_BINS):
    """g(r) under periodic minimum-image convention, cubic cell."""
    n = positions.shape[0]
    diffs = positions[:, None, :] - positions[None, :, :]
    diffs -= cell_length * np.round(diffs / cell_length)
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_dists = pair_dists[pair_dists < r_max]

    hist, edges = np.histogram(pair_dists, bins=n_bins, range=(0, r_max))
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    shell_vol = 4 * np.pi * r**2 * dr
    density = n / (cell_length**3)
    n_pairs_ideal = np.maximum(0.5 * n * density * shell_vol, 1e-12)
    g_r = hist / n_pairs_ideal
    return r, g_r


def averaged_rdf(traj, last_k: int, cell_length: float):
    frames = traj[-last_k:]
    g_accum = None
    r = None
    for f in frames:
        r, g = compute_rdf(f["pos"], cell_length)
        g_accum = g if g_accum is None else g_accum + g
    return r, g_accum / len(frames)


def build_comparison_report(ref_traj, dist_traj) -> list:
    """Row per step, full diagnostics computed for EVERY step regardless of
    pass/fail — compare_trajectories()'s assertions stop at the first
    violation, so without this there's no way to inspect what happens on
    steps after a failure, or to see whether a failing step is an isolated
    spike vs. part of smooth accumulation.

    Includes which single atom has the largest deviation at each step (by
    global vertex ID) and whether that step is a rebalance step, specifically
    to make it easy to check whether divergence spikes line up with
    repartition boundaries (step % REBALANCE_INTERVAL == 0).
    """
    rows = []
    for step in range(len(ref_traj)):
        ref_frame, dist_frame = ref_traj[step], dist_traj[step]
        pos_diff = dist_frame["pos"] - ref_frame["pos"]
        vel_diff = dist_frame["vel"] - ref_frame["vel"]

        pos_per_atom = np.max(np.abs(pos_diff), axis=-1)
        vel_per_atom = np.max(np.abs(vel_diff), axis=-1)
        pos_max_idx = int(np.argmax(pos_per_atom))
        vel_max_idx = int(np.argmax(vel_per_atom))

        vid_match = bool(np.array_equal(ref_frame["vid"], dist_frame["vid"]))

        rows.append({
            "step": step,
            "is_rebalance_step": step % REBALANCE_INTERVAL == 0,
            "vid_match": vid_match,
            "ref_energy": ref_frame["energy"],
            "dist_energy": dist_frame["energy"],
            "energy_diff": dist_frame["energy"] - ref_frame["energy"],
            "pos_rmsd": rmsd(dist_frame["pos"], ref_frame["pos"]),
            "vel_rmsd": rmsd(dist_frame["vel"], ref_frame["vel"]),
            "pos_max_abs": float(pos_per_atom[pos_max_idx]),
            "pos_max_atom_vid": int(ref_frame["vid"][pos_max_idx]),
            "vel_max_abs": float(vel_per_atom[vel_max_idx]),
            "vel_max_atom_vid": int(ref_frame["vid"][vel_max_idx]),
        })
    return rows


def write_comparison_report_csv(rows: list, path: str) -> None:
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote full {len(rows)}-step comparison report to {path}")

def compare_trajectories(ref_traj, dist_traj, report_path=None):
    """Raises AssertionError on the first violated check.

    If report_path is given, a full per-step diagnostic CSV (all steps,
    regardless of pass/fail) is written BEFORE any assertion runs, so a
    failure never prevents inspecting the rest of the trajectory.
    """
    if report_path is not None:
        rows = build_comparison_report(ref_traj, dist_traj)
        write_comparison_report_csv(rows, report_path)
    assert len(dist_traj) == len(ref_traj) == N_STEPS, (
        f"frame count mismatch: ref={len(ref_traj)} dist={len(dist_traj)} expected={N_STEPS}"
    )

    # Vertex ID sets must match exactly at every step, regardless of phase.
    # This is a structural check (redistribute_atoms / vertex tracking), not
    # a numerical one, so chaos tolerance doesn't apply to it.
    for step, (ref_frame, dist_frame) in enumerate(zip(ref_traj, dist_traj)):
        assert np.array_equal(ref_frame["vid"], dist_frame["vid"]), (
            f"step {step}: vertex ID sets diverged between global and distributed runs "
            "(indicates a bug in redistribute_atoms / vertex ID tracking, not numerics)"
        )

    # ---- Phase 1: tight per-atom checks, early steps only ----
    for step in range(0, EARLY_PHASE_STEPS + 1):
        ref_frame, dist_frame = ref_traj[step], dist_traj[step]

        np.testing.assert_allclose(
            dist_frame["pos"], ref_frame["pos"], atol=POS_VEL_TOL, rtol=0,
            err_msg=f"[phase 1] positions diverged beyond schedule at step {step}",
        )
        np.testing.assert_allclose(
            dist_frame["vel"], ref_frame["vel"], atol=POS_VEL_TOL, rtol=0,
            err_msg=f"[phase 1] velocities diverged beyond schedule at step {step}",
        )
        np.testing.assert_allclose(
            dist_frame["energy"], ref_frame["energy"], rtol=ENERGY_RTOL,
            err_msg=f"[phase 1] total energy diverged beyond {ENERGY_RTOL} relative at step {step}",
        )

    # ---- Phase 2a: aggregate RMSD-of-difference, later steps ----
    diffs = rmsd_between_trajectories(ref_traj, dist_traj)
    for step in range(EARLY_PHASE_STEPS + 1, N_STEPS):
        pos_tol = _pos_rmsd_tolerance(step)
        vel_tol = _vel_rmsd_tolerance(step)
        assert diffs["pos"][step] <= pos_tol, (
            f"[phase 2] position RMSD-of-difference {diffs['pos'][step]:.6f} exceeds "
            f"{pos_tol:.6f} at step {step}"
        )
        assert diffs["vel"][step] <= vel_tol, (
            f"[phase 2] velocity RMSD-of-difference {diffs['vel'][step]:.6f} exceeds "
            f"{vel_tol:.6f} at step {step}"
        )

    # ---- Phase 2b: per-run energy drift, distributed vs. reference's own noise floor ----
    ref_endpoint_drift = compute_energy_drift(ref_traj)
    dist_endpoint_drift = compute_energy_drift(dist_traj)
    print(
        f"[phase 2] energy |E(t)-E(0)| (diagnostic only): "
        f"reference max={float(np.max(ref_endpoint_drift)):.6f} eV, "
        f"distributed max={float(np.max(dist_endpoint_drift)):.6f} eV"
    )

    ref_slope = energy_drift_slope(ref_traj)
    dist_slope = energy_drift_slope(dist_traj)
    ref_std = energy_oscillation_std(ref_traj)
    dist_std = energy_oscillation_std(dist_traj)
    slope_bound = max(ENERGY_DRIFT_MARGIN * abs(ref_slope), ENERGY_SLOPE_FLOOR_EV_PER_STEP)
    std_bound = max(ENERGY_DRIFT_MARGIN * ref_std, ENERGY_STD_FLOOR_EV)
    print(
        f"[phase 2] energy drift slope: reference={ref_slope:.3e} eV/step, "
        f"distributed={dist_slope:.3e} eV/step, bound={slope_bound:.3e} eV/step"
    )
    print(
        f"[phase 2] energy oscillation std (about trend): reference={ref_std:.3e} eV, "
        f"distributed={dist_std:.3e} eV, bound={std_bound:.3e} eV"
    )
    assert abs(dist_slope) <= slope_bound, (
        f"[phase 2] distributed run's energy drift slope ({dist_slope:.3e} eV/step) exceeds "
        f"{ENERGY_DRIFT_MARGIN}x the reference run's own slope ({ref_slope:.3e} eV/step, "
        f"bound={slope_bound:.3e} eV/step) — suggests non-conservative forces specific to "
        "the distributed path (e.g. a broken ghost-gradient path), not just chaotic noise."
    )
    assert dist_std <= std_bound, (
        f"[phase 2] distributed run's energy oscillation std ({dist_std:.3e} eV) exceeds "
        f"{ENERGY_DRIFT_MARGIN}x the reference run's own std ({ref_std:.3e} eV, "
        f"bound={std_bound:.3e} eV)"
    )

    # ---- Phase 2c: RDF structural sanity check, informational-leaning ----
    r, g_ref = averaged_rdf(ref_traj, RDF_LAST_K_FRAMES, SIDE_LENGTH_ANGSTROM)
    _, g_dist = averaged_rdf(dist_traj, RDF_LAST_K_FRAMES, SIDE_LENGTH_ANGSTROM)
    rdf_diff = float(np.mean(np.abs(g_ref - g_dist)))
    print(f"[phase 2] RDF mean abs difference (last {RDF_LAST_K_FRAMES} frames): {rdf_diff:.4f}")
    assert rdf_diff <= RDF_TOL, (
        f"[phase 2] RDF mean abs difference {rdf_diff:.4f} exceeds loose sanity threshold "
        f"{RDF_TOL} — structural statistics diverged more than expected even accounting for chaos."
    )


# ---------------------------------------------------------------------------
# Calibration helper — NOT part of the pass/fail test, run manually first
# ---------------------------------------------------------------------------

def reference_vs_reference_check(structure_path: str, output_dir_a: str, output_dir_b: str):
    """Runs the world_size=1 reference twice and reports RMSD-of-difference
    per step between two runs that share zero distributed code. Use this to
    sanity-check that _pos_rmsd_tolerance() / _vel_rmsd_tolerance() above
    comfortably exceed this self-comparison at every step — if they don't,
    the Phase-2 checks will produce false failures regardless of whether the
    distributed code is correct.

    Not wired into any CLI script; call manually, e.g.:
        python -c "
            from trajectory_test_common import reference_vs_reference_check
            reference_vs_reference_check('/tmp/system.xyz', '/tmp/a', '/tmp/b')
        "
    """
    from dynamis import Simulator
    from dynamis_integrator import Verlet
    from dynamis_model.nnp import UMAModel

    ensure_test_structure(structure_path)

    def _one_run(output_dir):
        env = real_singleton_torch_dist_env()
        dist_env = env.dist_env
        model = UMAModel("uma-s-1p2", dataset="omol", device=env.device.type)
        integrator = Verlet(dt=0.5)
        config = make_config(output_dir)
        sim = Simulator(model, integrator, dist_env, config, env.device)
        sim.setup(structure_path)
        traj = run_md_loop(sim, config)
        teardown_singleton_dist_env()
        return traj

    traj_a = _one_run(output_dir_a)
    traj_b = _one_run(output_dir_b)
    diffs = rmsd_between_trajectories(traj_a, traj_b)

    print(f"{'step':>4}  {'pos_rmsd':>12}  {'pos_tol':>12}  {'vel_rmsd':>12}  {'vel_tol':>12}")
    for step in range(N_STEPS):
        if step <= EARLY_PHASE_STEPS:
            pos_tol = POS_VEL_TOL
            vel_tol = POS_VEL_TOL
        else:
            pos_tol = _pos_rmsd_tolerance(step)
            vel_tol = _vel_rmsd_tolerance(step)
        flag = "  <-- EXCEEDS TOLERANCE" if diffs["pos"][step] > pos_tol or diffs["vel"][step] > vel_tol else ""
        print(
            f"{step:>4}  {diffs['pos'][step]:>12.6f}  {pos_tol:>12.6f}  "
            f"{diffs['vel'][step]:>12.6f}  {vel_tol:>12.6f}{flag}"
        )
    return diffs
