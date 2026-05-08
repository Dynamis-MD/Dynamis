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

# Run 1:
# Euler: 17 ms, 1.746s
# Verlet: 49 ms, 3.488s
# Langevin: 34 ms, 3.433s

# Run 2:
# Euler: 14 ms, 1.726s
# Verlet: 34 ms, 3.484s
# Langevin: 33 ms, 3.529s

# Run 3:
# Euler: 13 ms, 1.585s
# Verlet: 37 ms, 2.914s
# Langevin: 29 ms, 2.984s

# Run 4:
# Euler: 16 ms, 1.627s
# Verlet: 33 ms, 3.191s
# Langevin: 18 ms, 3.176s

# Run 5:
# Euler: 17 ms, 1.643s
# Verlet: 34 ms, 3.432s
# Langevin: 31 ms, 3.253s

# Run 6:
# Euler: 11 ms, 1.634s
# Verlet: 30 ms, 3.404s
# Langevin: 28 ms, 3.467s

# Run 7:
# Euler: 13 ms, 1.642s
# Verlet: 35 ms, 3.263s
# Langevin: 31 ms, 3.421s

# Run 8:
# Euler: 17 ms, 1.666s
# Verlet: 28 ms, 3.275s
# Langevin: 39 ms, 3.304s

# Run 9:
# Euler: 18 ms, 1.550s
# Verlet: 34 ms, 3.223s
# Langevin: 17 ms, 3.342s

# Run 10:
# Euler: 17 ms, 1.676s
# Verlet: 22 ms, 3.283s
# Langevin: 27 ms, 3.354s

# Run 11:
# Euler: 17 ms, 1.719s
# Verlet: 40 ms, 3.414s
# Langevin: 23 ms, 3.244s

# Run 12:
# Euler: 17 ms, 1.615s
# Verlet: 32 ms, 3.289s
# Langevin: 34 ms, 3.295s

# Run 13:
# Euler: 17 ms, 1.662s
# Verlet: 37 ms, 3.370s
# Langevin: 36 ms, 3.204s

# Run 14:
# Euler: 17 ms, 1.601s
# Verlet: 34 ms, 3.235s
# Langevin: 35 ms, 3.283s

# Run 15:
# Euler: 17 ms, 1.749s
# Verlet: 31 ms, 3.366s
# Langevin: 39 ms, 3.362s

# Run 16:
# Euler: 8 ms, 1.647s
# Verlet: 34 ms, 3.356s
# Langevin: 37 ms, 3.488s

# Run 17:
# Euler: 17 ms, 1.573s
# Verlet: 27 ms, 3.370s
# Langevin: 50 ms, 3.344s

# Run 18:
# Euler: 8 ms, 1.480s
# Verlet: 35 ms, 2.971s
# Langevin: 36 ms, 3.255s

# Run 19:
# Euler: 17 ms, 1.664s
# Verlet: 28 ms, 3.259s
# Langevin: 29 ms, 3.239s

# Run 20:
# Euler: 14 ms, 1.618s
# Verlet: 34 ms, 3.169s
# Langevin: 28 ms, 3.168s
