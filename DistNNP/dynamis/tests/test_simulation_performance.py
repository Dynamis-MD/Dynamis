"""
test_simulation_performance.py — performance regression tests for the Simulation class.

Reference system: 16-atom LJ argon in an 8x8x8 box, fixed seed=100 (see conftest.py).
Bounds are mean + 1.5*std over 20 baseline runs on 2x A100 GPUs (see comments below).
Re-baseline if hardware or system size changes.

Run via:
    torchrun --nproc_per_node=2 -m pytest dynamis/tests/test_simulation_performance.py -v -m perf
"""

import torch
import pytest

# Run 20 times and make perf bounds 1.5x std dev of the mean.
# The data is noisy and it shows that it can fail
# sometimes when nothing is necessarily wrong.

# Euler    per_step:  Mean = 16.661 ms,  Std Dev = 4.125 ms,  Bound = 22.849 ms
# Euler    N100:      Mean = 1.972 s,  Std Dev = 0.067 s,  Bound = 2.073 s
# Verlet   per_step:  Mean = 19.791 ms,  Std Dev = 2.517 ms,  Bound = 23.567 ms
# Verlet   N100:      Mean = 1.961 s,  Std Dev = 0.103 s,  Bound = 2.115 s
# Langevin per_step:  Mean = 20.242 ms,  Std Dev = 6.058 ms,  Bound = 29.329 ms
# Langevin N100:      Mean = 2.012 s,  Std Dev = 0.113 s,  Bound = 2.181 s

PERF_BOUNDS = {
    "euler": {"per_step_ms": 22.849, "N100_total_s": 2.073},
    "verlet": {"per_step_ms": 23.567, "N100_total_s": 2.115},
    "langevin": {"per_step_ms": 29.329, "N100_total_s": 2.181},
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
