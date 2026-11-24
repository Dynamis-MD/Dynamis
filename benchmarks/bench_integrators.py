"""Compare integrator kernels: plain torch vs torch.compile vs Triton vs CUDA.

Times one full integrator step (Euler: step; Verlet/Langevin: halfstep1 +
halfstep2) on random tensors for a range of atom counts, single GPU.

    python benchmarks/bench_integrators.py
    python benchmarks/bench_integrators.py --atoms 1000 27000 1000000 --iters 500

All implementations are wrapped in the same dynamis_state Timer calls the
Triton/CUDA kernels make, so timing overhead is identical across them.
"""

import argparse
import math
import time

import torch
from dynamis_state import Timer, TimerClass, Where
from dynamis_integrator import _kernels

try:
    from dynamis_integrator import dynamis_integrator_impl as _cuda
except ImportError:
    _cuda = None

_timer = Timer()


# ---------------------------------------------------------------------------
# Plain torch versions of the integrator math (same equations as _kernels.py)
# ---------------------------------------------------------------------------

def euler_step(pos, vel, frc, mass, dt):
    vel.add_(frc / mass[:, None] * dt)
    pos.add_(vel * dt)


def verlet_halfstep1(pos, vel, frc, mass, dt):
    vel.add_(0.5 * frc / mass[:, None] * dt)
    pos.add_(vel * dt)


def verlet_halfstep2(vel, frc, mass, dt):
    vel.add_(0.5 * frc / mass[:, None] * dt)


def langevin_halfstep1(pos, vel, frc, mass, noise, gamma, kT, dt):
    c1 = 1.0 - 0.5 * gamma * dt
    c2 = torch.sqrt(gamma * kT * dt / mass)[:, None]
    vel.mul_(c1).add_(0.5 * frc / mass[:, None] * dt + c2 * noise)
    pos.add_(vel * dt)


def langevin_halfstep2(vel, frc, mass, noise, gamma, kT, dt):
    c3 = 1.0 / (1.0 + 0.5 * gamma * dt)
    c2 = torch.sqrt(gamma * kT * dt / mass)[:, None]
    vel.add_(0.5 * frc / mass[:, None] * dt + c2 * noise).mul_(c3)


def _timed(name, fn):
    def wrapper(*args):
        _timer.start(name, TimerClass.Compute, Where.Device)
        fn(*args)
        _timer.stop()
    return wrapper


def make_torch_classes(compile_: bool):
    """Torch integrator classes with the same interface as the Triton/CUDA ones."""
    wrap = torch.compile if compile_ else (lambda f: f)

    class EulerIntegrator:
        step = staticmethod(_timed("integrator_kernel", wrap(euler_step)))

    class VerletIntegrator:
        halfstep1 = staticmethod(_timed("integrator_halfstep1", wrap(verlet_halfstep1)))
        halfstep2 = staticmethod(_timed("integrator_halfstep2", wrap(verlet_halfstep2)))

    class LangevinIntegrator:
        halfstep1 = staticmethod(_timed("integrator_halfstep1", wrap(langevin_halfstep1)))
        halfstep2 = staticmethod(_timed("integrator_halfstep2", wrap(langevin_halfstep2)))

    return EulerIntegrator, VerletIntegrator, LangevinIntegrator


IMPLS = {
    "torch": make_torch_classes(compile_=False),
    "torch.compile": make_torch_classes(compile_=True),
    "triton": (_kernels.EulerIntegrator, _kernels.VerletIntegrator, _kernels.LangevinIntegrator),
}
if _cuda is not None:
    IMPLS["cuda"] = (_cuda.EulerIntegrator, _cuda.VerletIntegrator, _cuda.LangevinIntegrator)


# ---------------------------------------------------------------------------
# One full step per integrator
# ---------------------------------------------------------------------------

DT, GAMMA, KT = 0.5, 0.1, 0.0259


def one_step(kind, classes):
    euler_cls, verlet_cls, langevin_cls = classes
    if kind == "euler":
        integ = euler_cls()
        return lambda t: integ.step(t["pos"], t["vel"], t["frc"], t["mass"], DT)
    if kind == "verlet":
        integ = verlet_cls()

        def step(t):
            integ.halfstep1(t["pos"], t["vel"], t["frc"], t["mass"], DT)
            integ.halfstep2(t["vel"], t["frc"], t["mass"], DT)
        return step
    if kind == "langevin":
        integ = langevin_cls()

        def step(t):
            integ.halfstep1(t["pos"], t["vel"], t["frc"], t["mass"], t["noise"], GAMMA, KT, DT)
            integ.halfstep2(t["vel"], t["frc"], t["mass"], t["noise"], GAMMA, KT, DT)
        return step
    raise ValueError(kind)


def make_inputs(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    return {
        "pos": r(n, 3), "vel": r(n, 3), "frc": r(n, 3), "noise": r(n, 3),
        "mass": torch.rand(n, device="cuda", generator=g) + 0.5,
    }


def clone(t):
    return {k: v.clone() for k, v in t.items()}


def time_step(step, inputs, iters, warmup):
    """Wall-clock µs per step, including Python/launch overhead."""
    for _ in range(warmup):
        step(inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        step(inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atoms", type=int, nargs="+", default=[1_000, 4_000, 27_000, 100_000, 1_000_000])
    p.add_argument("--kinds", nargs="+", default=["euler", "verlet", "langevin"])
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}   implementations: {', '.join(IMPLS)}")
    print("Times are µs per full integrator step (lower is better).\n")

    for kind in args.kinds:
        header = f"{'atoms':>10} | " + " | ".join(f"{name:>13}" for name in IMPLS) + " | triton vs torch"
        print(f"== {kind} ==")
        print(header)
        print("-" * len(header))

        for n in args.atoms:
            inputs = make_inputs(n)

            # Correctness: every implementation must match plain torch after one step.
            ref = clone(inputs)
            one_step(kind, IMPLS["torch"])(ref)
            for name, classes in IMPLS.items():
                out = clone(inputs)
                one_step(kind, classes)(out)
                for key in ("pos", "vel"):
                    if not torch.allclose(out[key], ref[key], rtol=1e-5, atol=1e-5):
                        err = (out[key] - ref[key]).abs().max().item()
                        raise SystemExit(f"{name} {kind} n={n}: {key} differs from torch (max err {err:.2e})")

            times = {
                name: time_step(one_step(kind, classes), clone(inputs), args.iters, args.warmup)
                for name, classes in IMPLS.items()
            }
            speedup = times["torch"] / times["triton"]
            row = f"{n:>10} | " + " | ".join(f"{times[name]:>13.1f}" for name in IMPLS)
            print(f"{row} | {speedup:>14.2f}x")
        print()


if __name__ == "__main__":
    main()
