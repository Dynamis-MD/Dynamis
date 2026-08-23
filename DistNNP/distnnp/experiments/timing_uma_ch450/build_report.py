"""
Consolidate timing_rank{N}.jsonl files from the 1/2/4 GPU UMA/CH450 timing
sweep (run_timing_sweep.sh) into a single markdown scaling table.

Self-time model
----------------
Records overlap (TIMING-PLAN.md: "the raw sum of all keys is meaningless").
The instrumentation actually in place today has one concrete, known nesting,
used here instead of a generic tree solver:

    iteration (Other, host)                          <- per-step wall time
      model_forward (Compute, device)  x{1 or 2}      <- Verlet: 1 call/step after
        halo_exchange_wait (Wait, device, detail only)   the first, 2 on step 0
        halo_exchange (Comm, device)   x{L layers}
      integrator_halfstep1 (Compute, device)          <- sibling of model_forward,
      integrator_halfstep2 (Compute, device)             does not nest a model call

So, per step:
    comm_ms    = sum(halo_exchange)
    wait_ms    = sum(halo_exchange_wait)
    compute_ms = sum(model_forward) + sum(integrator_halfstep*) - comm_ms - wait_ms
    other_ms   = iteration_ms - compute_ms - comm_ms - wait_ms

other_ms captures anything not covered by the device timers above (Python/host
orchestration, and on a rebalance step, partition/build_halo_maps -- excluded
here since the experiment runs with rebalance_interval=0).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

MODEL_FORWARD_KEYS = {"model_forward"}
COMPUTE_SIBLING_KEYS = {"integrator_halfstep1", "integrator_halfstep2", "integrator_step"}
COMM_KEYS = {"halo_exchange"}
WAIT_KEYS = {"halo_exchange_wait"}


def load_rank_file(path: str):
    meta = None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["type"] == "meta":
                meta = obj
            elif obj["type"] == "rec":
                records.append(obj)
    return meta, records


def per_step_breakdown(records, warmup_steps: int):
    by_step = defaultdict(lambda: defaultdict(float))
    for r in records:
        by_step[r["step"]][r["key"]] += r["time_us"]

    steps = sorted(s for s in by_step if s >= warmup_steps)
    rows = []
    for step in steps:
        keys = by_step[step]
        iteration_us = keys.get("iteration", 0.0)
        comm_us = sum(keys.get(k, 0.0) for k in COMM_KEYS)
        wait_us = sum(keys.get(k, 0.0) for k in WAIT_KEYS)
        forward_and_siblings_us = sum(
            keys.get(k, 0.0) for k in MODEL_FORWARD_KEYS | COMPUTE_SIBLING_KEYS
        )
        compute_us = forward_and_siblings_us - comm_us - wait_us
        other_us = iteration_us - compute_us - comm_us - wait_us
        rows.append({
            "step": step,
            "iteration_us": iteration_us,
            "compute_us": compute_us,
            "comm_us": comm_us,
            "wait_us": wait_us,
            "other_us": other_us,
        })
    return rows


def summarize_gpu_count(results_dir: str, ngpu: int, warmup_steps: int):
    rank_files = sorted(glob.glob(os.path.join(results_dir, f"{ngpu}gpu", "timing_rank*.jsonl")))
    if not rank_files:
        raise FileNotFoundError(f"No timing_rank*.jsonl files found for {ngpu} GPU(s) in {results_dir}/{ngpu}gpu")

    per_rank_means = []
    for path in rank_files:
        _, records = load_rank_file(path)
        rows = per_step_breakdown(records, warmup_steps)
        if not rows:
            continue
        n = len(rows)
        per_rank_means.append({
            k: sum(r[k] for r in rows) / n
            for k in ("iteration_us", "compute_us", "comm_us", "wait_us", "other_us")
        })

    if not per_rank_means:
        raise ValueError(f"No steps past warmup ({warmup_steps}) for {ngpu} GPU(s)")

    n = len(per_rank_means)
    means = {
        k: sum(r[k] for r in per_rank_means) / n
        for k in ("iteration_us", "compute_us", "comm_us", "wait_us", "other_us")
    }
    means["n_ranks"] = n
    return means


def build_table(results_dir: str, gpu_counts, warmup_steps: int) -> str:
    summaries = {n: summarize_gpu_count(results_dir, n, warmup_steps) for n in gpu_counts}
    baseline_steps_per_s = 1_000_000.0 / summaries[gpu_counts[0]]["iteration_us"]

    header = (
        "| GPUs | ranks | steps/s | iteration (ms) | compute (ms, %) | "
        "comm (ms, %) | wait (ms, %) | other (ms, %) | speedup | efficiency |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for n in gpu_counts:
        s = summaries[n]
        iteration_ms = s["iteration_us"] / 1000.0
        steps_per_s = 1_000_000.0 / s["iteration_us"]
        speedup = steps_per_s / baseline_steps_per_s
        efficiency = speedup / n

        def cell(us):
            ms = us / 1000.0
            pct = 100.0 * us / s["iteration_us"]
            return f"{ms:.3f} ({pct:.1f}%)"

        lines.append(
            f"| {n} | {s['n_ranks']} | {steps_per_s:.2f} | {iteration_ms:.3f} | "
            f"{cell(s['compute_us'])} | {cell(s['comm_us'])} | {cell(s['wait_us'])} | "
            f"{cell(s['other_us'])} | {speedup * 100:.0f}% ({speedup:.2f}x) | {efficiency * 100:.1f}% |\n"
        )
    return "".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/timing",
                   help="Directory containing <N>gpu/timing_rank*.jsonl subdirectories.")
    p.add_argument("--gpu-counts", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--warmup-steps", type=int, default=10,
                   help="Skip this many leading steps (compile/cache warmup) before averaging.")
    p.add_argument("--out", default=None,
                   help="Write the markdown table here (default: <results-dir>/report.md).")
    return p.parse_args()


def main():
    args = parse_args()
    table = build_table(args.results_dir, args.gpu_counts, args.warmup_steps)
    out_path = args.out or os.path.join(args.results_dir, "report.md")
    with open(out_path, "w") as f:
        f.write(f"# UMA / CH450 Timing Sweep ({'/'.join(map(str, args.gpu_counts))} GPU)\n\n")
        f.write(f"Warmup steps skipped: {args.warmup_steps}\n\n")
        f.write(table)
    print(table)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
