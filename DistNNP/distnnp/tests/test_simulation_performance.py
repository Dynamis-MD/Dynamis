"""
test_simulation_performance.py — performance regression tests for the Simulation class.

Reference system: 16-atom LJ argon in an 8x8x8 box, fixed seed=100 (see conftest.py).
Bounds are mean + 1.5*std over 20 baseline runs on 2x A100 GPUs (see comments below).
Re-baseline if hardware or system size changes.

Run via:
    torchrun --nproc_per_node=2 -m pytest distnnp/tests/test_simulation_performance.py -v -m perf
"""

import torch
import torch.distributed as dist
import pytest


# Run 20 times and make perf bounds 1.5x std dev of the mean.
# The data is noisy and it shows that it can fail 
# sometimes when nothing is necessarily wrong.

# Euler    per_step:  Mean = 15.1 ms,  Std Dev = 3.06 ms,  Bound = 19.690 ms
# Euler    N100:      Mean = 1.641 s,  Std Dev = 0.066 s,  Bound = 1.740 s
# Verlet   per_step:  Mean = 33.4 ms,  Std Dev = 5.47 ms,  Bound = 41.605 ms
# Verlet   N100:      Mean = 3.288 s,  Std Dev = 0.150 s,  Bound = 3.513 s
# Langevin per_step:  Mean = 31.7 ms,  Std Dev = 7.52 ms,  Bound = 42.98 ms
# Langevin N100:      Mean = 3.307 s,  Std Dev = 0.128 s,  Bound = 3.499 s

PERF_BOUNDS = {
    "euler":    {"per_step_ms": 19.69, "N100_total_s": 1.74},
    "verlet":   {"per_step_ms": 41.605, "N100_total_s": 3.513},
    "langevin": {"per_step_ms": 42.98,  "N100_total_s": 3.499},
}


@pytest.mark.perf
def test_per_step_time(simulation_factory, request):
    integrator_name = request.node.callspec.params["simulation_factory"]
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

    # elapsed is a float in milliseconds
    elapsed = start_event.elapsed_time(end_event)
    assert elapsed < PERF_BOUNDS[integrator_name]["per_step_ms"]


@pytest.mark.perf
def test_N_step_time(simulation_factory, request):
    integrator_name = request.node.callspec.params["simulation_factory"]
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

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(N):
        simulation.step()
    end_event.record()
    torch.cuda.synchronize()

    # elapsed is a float in seconds
    elapsed = start_event.elapsed_time(end_event) / 1000  # ms -> s
    assert elapsed < PERF_BOUNDS[integrator_name]["N100_total_s"]
