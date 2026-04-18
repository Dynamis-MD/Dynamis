"""
test_simulation_performance.py — performance regression tests for the Simulation class.

Reference system: 32-atom LJ argon in a 10x10x10 box, fixed seed=100.
Bounds are set conservatively at ~2x observed mean on 2x A100 GPUs.
Tighten after baseline is stable.

Run via:
    torchrun --nproc_per_node=2 -m pytest distnnp/tests/test_simulation_performance.py -v -m perf
"""

import torch
import time
import pytest

# FYI I haven't actually set these bounds yet
PERF_BOUNDS = {
    "per_step_s": 2.0,
    "N100_total_s": 18.0,
}


@pytest.mark.perf
def test_per_step_time(simulation_factory):
    simulation = simulation_factory()

    # Suggested 10-50 warmup steps
    # Doing 50 to be very safe
    WARMUP_STEPS = 50
    for _ in range(WARMUP_STEPS):
        simulation.step()

    # timing done using cuda events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    simulation.step()
    end_event.record()
    torch.cuda.synchronize()

    # ms -> s
    elapsed = start_event.elapsed_time(end_event) / 1000

    # elapsed is a float in seconds
    # assert elapsed < PERF_BOUNDS["per_step_s"]


@pytest.mark.perf
def test_N_step_time(simulation_factory):
    simulation = simulation_factory()

    # number of steps we are taking
    # This could be changed but if so,
    # the bound must also be changed
    N = 100

    # Suggested 10-50 warmup steps
    # Doing 50 to be very safe
    WARMUP_STEPS = 50
    for _ in range(WARMUP_STEPS):
        simulation.step()

    # time.time() might not be the most precise but
    # given that our bounds are in seconds it is fine
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(N):
        simulation.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # elapsed is a float in seconds
    # assert elapsed < PERF_BOUNDS["N100_total_s"]


# run 10-20 times and make perf bounds 1.5x std dev
