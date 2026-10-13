#!/usr/bin/env python3
"""Read `results/tune/*.json` and print the kernel-configuration verdict.

The sweep (`scripts/tune_kernels.sbatch`) is a 2x2 factorial in `merge_mole` x
`compile` with activation checkpointing off, plus a checkpointing-on control.
This turns those five JSONs into the one thing they were run to answer: which
flag is safe, which one hangs, and what each is worth in steps/s.

    python analysis/tune_report.py --results-dir results/tune
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# tag -> (label, merge_mole, compile)
CONFIGS = [
    ("k0_tf32ac", "TF32, ackpt ON (control)", None, None),
    ("k1_noac", "TF32, ackpt OFF", False, False),
    ("k2_mole", "+ merge_mole", True, False),
    ("k3_compile", "+ compile", False, True),
    ("k4_turbo", "turbo (both)", True, True),
]

# A cell that never returned was reaped by run_cell.sbatch's `timeout` and
# recorded as status=oom with no steps_per_sec. On this sweep that is far more
# likely a deadlock than a genuine out-of-memory, so report it as HANG/OOM
# rather than asserting which.
HUNG = {"oom", "timeout"}


def load(results_dir: str) -> dict:
    rows = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            r = json.load(open(path))
        except Exception as e:  # noqa: BLE001
            print(f"skip {path}: {e!r}")
            continue
        tag = r.get("tag")
        eng = r.get("engine")
        if tag and eng:
            rows[(eng, tag)] = r
    return rows


def _cell(r: dict | None) -> str:
    if r is None:
        return "not run"
    st = r.get("status", "?")
    sps = r.get("steps_per_sec")
    if st == "ok" and sps:
        return f"{sps:.3f} st/s"
    if st in HUNG:
        return "HANG/OOM"
    return st.upper()


def _mem(r: dict | None) -> str:
    if r is None or not r.get("peak_mem_mb_max"):
        return "-"
    return f"{r['peak_mem_mb_max'] / 1024:.1f} GB"


def _backend(r: dict | None) -> str:
    s = (r or {}).get("inference_settings") or {}
    if not s:
        return "-"
    if s.get("execution_mode"):
        return s["execution_mode"]
    return "fast_gpu" if s.get("execution_mode_auto_eligible_for_fast_gpu") else "general"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/tune")
    args = ap.parse_args()

    rows = load(args.results_dir)
    if not rows:
        print(f"no result JSONs in {args.results_dir}")
        return
    engines = sorted({e for (e, _) in rows})

    for eng in engines:
        print(f"\n=== {eng} ===")
        print(f"{'config':<26} {'steps/s':>12} {'peak mem':>10} {'backend':>10}  status")
        print("-" * 74)
        base = None
        for tag, label, _mole, _cmp in CONFIGS:
            r = rows.get((eng, tag))
            speed = _cell(r)
            note = ""
            if r and r.get("status") == "ok" and r.get("steps_per_sec"):
                if tag == "k1_noac":
                    base = r["steps_per_sec"]
                elif base:
                    note = f"  ({r['steps_per_sec'] / base:.2f}x vs k1)"
            elif r and r.get("status") in HUNG:
                note = "  <-- reaped by the cell timeout"
            print(f"{label:<26} {speed:>12} {_mem(r):>10} {_backend(r):>10}  "
                  f"{(r or {}).get('status', '-')}{note}")

        # --- the verdict the sweep exists to produce -----------------------
        def ok(tag):
            r = rows.get((eng, tag))
            return bool(r and r.get("status") == "ok" and r.get("steps_per_sec"))

        have = {t: (eng, t) in rows for t, *_ in CONFIGS}
        if not (have.get("k2_mole") and have.get("k3_compile")):
            print("verdict: incomplete -- k2_mole and k3_compile are both needed.")
            continue
        m, c, both = ok("k2_mole"), ok("k3_compile"), ok("k4_turbo")
        if m and c and both:
            print("verdict: neither flag is the culprit here -> run the matrix at `turbo`.")
        elif m and c and not both:
            print("verdict: merge_mole and compile are each fine ALONE but their "
                  "COMBINATION fails -> use k2 or k3, not turbo.")
        elif m and not c:
            print("verdict: `compile` is the culprit -> "
                  "`--inference-settings turbo --compile 0` (k2 config).")
        elif c and not m:
            print("verdict: `merge_mole` is the culprit -> `--tf32 1 "
                  "--activation-checkpointing 0 --compile 1` (k3). NOTE: without "
                  "merge_mole the umas_fast_gpu backend stays disabled.")
        else:
            print("verdict: BOTH flags fail independently -> the best safe "
                  "configuration is k1 (TF32 + no activation checkpointing).")

    print("\nReminder: these cells are deliberately on DIFFERENT kernel "
          "configurations.\nDo not table them next to results/*.json, and re-run "
          "the full matrix at\nwhichever single configuration wins.")


if __name__ == "__main__":
    main()
