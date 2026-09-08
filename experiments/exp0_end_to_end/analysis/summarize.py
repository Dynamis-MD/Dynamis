#!/usr/bin/env python3
"""Merge results/*.json into a CSV + matplotlib figures for the exp0 writeup.

    python experiments/exp0_end_to_end/analysis/summarize.py \
        [--results-dir ../results] [--out-dir figures]

Produces:
  summary.csv                     one row per cell
  figures/strong_scaling.png      steps/s & ns/day vs GPU count, per engine
  figures/fill_scaling.png        steps/s + peak mem vs total atoms at ~32k/GPU
  figures/oom_ceiling.png         max atoms completed + peak mem vs atoms
  headline_table.md               matched (atoms,gpu) DistNNP vs FairChem speedup
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"


def _finish_ax(ax):
    """Shared paper-quality axis styling: no top/right spine, dashed grid
    behind the data, ticks below the default weight."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

ENGINES = ["distnnp-uma", "distnnp-uma-opt", "fairchem-uma", "lammps-mace"]
ENGINE_LABEL = {
    "distnnp-uma": "DistNNP-UMA",
    "distnnp-uma-opt": "DistNNP-UMA-opt",
    "fairchem-uma": "FairChem-UMA",
    "lammps-mace": "LAMMPS-MACE",
}
COLOR = {
    "distnnp-uma": "#1f77b4",
    "distnnp-uma-opt": "#ff7f0e",
    "fairchem-uma": "#d62728",
    "lammps-mace": "#2ca02c",
}

# distnnp-uma-opt is DELIBERATELY on a different UMA kernel path (turbo) from the
# matched distnnp-uma/fairchem-uma pair, so it is exempt from the "one kernel
# config per results dir" check in headline_table().
OPT_ENGINES = {"distnnp-uma-opt"}

FIELDS = [
    "engine", "model", "tag", "num_atoms", "box_angstrom", "world_size",
    "status", "elapsed_s", "steps_per_sec", "ns_per_day",
    "peak_mem_mb_max", "mean_temperature_k", "warmup_steps", "timed_steps",
    "inference_kernels", "git_commit", "timestamp", "input_file",
]


def _kernel_key(r: dict) -> str:
    """Compact, comparable signature of the UMA kernel configuration.

    Records written before the `inference_settings` field existed all came from
    fairchem's `default` preset -- the slowest one (tf32 off, activation
    checkpointing on, MOLE unmerged, `general` execution backend). Label them
    `legacy-default` rather than dropping them, so a results dir that mixes
    pre- and post-fix runs is visibly mixed instead of silently wrong.
    """
    s = r.get("inference_settings")
    if not s:
        return "" if r.get("engine", "").startswith("lammps") else "legacy-default"
    mode = s.get("execution_mode") or (
        "auto>fast_gpu" if s.get("execution_mode_auto_eligible_for_fast_gpu")
        else "auto>general"
    )
    return (f"tf32={int(bool(s.get('tf32')))}"
            f",ackpt={int(bool(s.get('activation_checkpointing')))}"
            f",mole={int(bool(s.get('merge_mole')))}"
            f",compile={int(bool(s.get('compile')))}"
            f",{mode}")


def load(results_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            r = json.load(open(path))
        except Exception as e:  # noqa: BLE001
            print(f"skip {path}: {e!r}")
            continue
        r["_file"] = os.path.basename(path)
        r["inference_kernels"] = _kernel_key(r)
        rows.append(r)
    return rows


def write_csv(rows: list[dict], out_path: str) -> None:
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["_file"] + FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.get("engine", ""), x.get("num_atoms", 0), x.get("world_size", 0))):
            w.writerow(r)
    print(f"wrote {out_path}  ({len(rows)} rows)")


def _ok(rows, tag=None):
    return [
        r for r in rows
        if r.get("status") == "ok"
        and (r.get("steps_per_sec") or 0) > 0
        and (tag is None or r.get("tag") == tag)
    ]


def strong_scaling(rows, out_dir):
    data = _ok(rows, tag="strong")
    if not data:
        print("strong: no data"); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # one curve per (engine, system size) -- a single tag="strong" may now cover
    # several fixed sizes (e.g. LAMMPS-MACE 8k/16k/32k), and connecting points
    # across sizes would draw nonsense vertical jumps.
    sizes = sorted({r["num_atoms"] for r in data})
    multi_size = len(sizes) > 1
    for eng in ENGINES:
        for size in sizes:
            pts = sorted((r["world_size"], r["steps_per_sec"], r["ns_per_day"])
                         for r in data
                         if r["engine"] == eng and r["num_atoms"] == size)
            if not pts:
                continue
            gpus = [p[0] for p in pts]
            lab = ENGINE_LABEL[eng] + (f" ({size // 1000}k)" if multi_size else "")
            ax1.plot(gpus, [p[1] for p in pts], "o-", color=COLOR[eng], label=lab)
            ax2.plot(gpus, [p[2] for p in pts], "o-", color=COLOR[eng], label=lab)
            # ideal-linear reference from this curve's smallest-GPU point
            base = pts[0]
            ax1.plot(gpus, [base[1] / base[0] * g for g in gpus], "--",
                     color=COLOR[eng], alpha=0.35)
    for ax, ylab in ((ax1, "steps / s"), (ax2, "ns / day")):
        ax.set_xlabel("GPUs"); ax.set_ylabel(ylab); ax.set_xticks([1, 2, 4])
        _finish_ax(ax); ax.legend(frameon=False)
    fig.suptitle("Strong Scaling — Fixed System Size")
    fig.tight_layout()
    p = os.path.join(out_dir, "strong_scaling.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def fill_scaling(rows, out_dir, atoms_per_rank=32000, tol=0.15):
    """Weak scaling at high per-GPU fill (~`atoms_per_rank` atoms/rank).

    Tag-agnostic: selects any completed cell whose atoms/world_size lands
    within `tol` of the target. This is the A100 "worst case" figure --
    isotropic dense fluid, GPUs filled near their memory ceiling, the regime
    where FairChem graph-parallel is strongest.
    """
    lo, hi = atoms_per_rank * (1 - tol), atoms_per_rank * (1 + tol)
    data = [
        r for r in rows
        if r.get("status") == "ok" and (r.get("steps_per_sec") or 0) > 0
        and r.get("world_size") and lo <= r["num_atoms"] / r["world_size"] <= hi
    ]
    if not data:
        print("fill: no data"); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for eng in ENGINES:
        pts = sorted((r["world_size"], r["num_atoms"], r["steps_per_sec"],
                      r.get("peak_mem_mb_max"))
                     for r in data if r["engine"] == eng)
        if not pts:
            continue
        ax1.plot([p[1] for p in pts], [p[2] for p in pts], "o-",
                 color=COLOR[eng], label=ENGINE_LABEL[eng])
        base = pts[0]
        ax1.plot([p[1] for p in pts],
                 [base[2] / base[0] * p[0] for p in pts], "--",
                 color=COLOR[eng], alpha=0.4)
        mem = [(p[1], p[3]) for p in pts if p[3]]
        if mem:
            ax2.plot([m[0] for m in mem], [m[1] / 1024 for m in mem], "o-",
                     color=COLOR[eng], label=ENGINE_LABEL[eng])
    ax1.set_xlabel("total atoms"); ax1.set_ylabel("steps / s")
    ax1.set_title(f"Fill Scaling — ~{atoms_per_rank // 1000}k Atoms/GPU\n"
                  "(dashed = ideal)")
    ax2.set_xlabel("total atoms"); ax2.set_ylabel("peak mem / rank (GB)")
    ax2.set_title("Peak Memory at High Fill")
    for ax in (ax1, ax2):
        _finish_ax(ax); ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "fill_scaling.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def oom_ceiling(rows, out_dir):
    data = [r for r in rows if r.get("world_size") == 4]
    if not data:
        print("oom: no 4-GPU data"); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for eng in ENGINES:
        completed = sorted(r["num_atoms"] for r in data
                           if r["engine"] == eng and r.get("status") == "ok"
                           and (r.get("steps_per_sec") or 0) > 0)
        max_atoms = max(completed) if completed else 0
        ax1.bar(ENGINE_LABEL[eng], max_atoms, color=COLOR[eng],
                edgecolor="black", linewidth=0.8)
        # peak mem: connect the cells that COMPLETED; mark OOM cells with an x
        # (their sample is the pre-kill high-water mark, i.e. the ceiling).
        ok_mem = sorted((r["num_atoms"], r["peak_mem_mb_max"])
                        for r in data if r["engine"] == eng
                        and r.get("status") == "ok" and r.get("peak_mem_mb_max"))
        oom_mem = sorted((r["num_atoms"], r["peak_mem_mb_max"])
                         for r in data if r["engine"] == eng
                         and r.get("status") == "oom" and r.get("peak_mem_mb_max"))
        if ok_mem:
            ax2.plot([m[0] for m in ok_mem], [m[1] for m in ok_mem], "o-",
                     color=COLOR[eng], label=ENGINE_LABEL[eng])
        if oom_mem:
            ax2.plot([m[0] for m in oom_mem], [m[1] for m in oom_mem], "x",
                     color=COLOR[eng], alpha=0.6,
                     label=None if ok_mem else f"{ENGINE_LABEL[eng]} (OOM)")
    ax1.set_ylabel("max atoms completed (4 GPU)")
    ax1.set_title("OOM Ceiling")
    ax1.tick_params(axis="x", rotation=20)
    ax2.set_xlabel("total atoms"); ax2.set_ylabel("peak mem / rank (MB)")
    ax2.set_title("Peak Memory vs System Size")
    _finish_ax(ax1); _finish_ax(ax2); ax2.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "oom_ceiling.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig); print(f"wrote {p}")


def _sps(r):
    """steps/s if the cell completed cleanly, else None."""
    if r and r.get("status") == "ok" and (r.get("steps_per_sec") or 0) > 0:
        return r["steps_per_sec"]
    return None


def headline_table(rows, out_path, ref_engine="distnnp-uma"):
    """One steps/s column per engine present, plus a speedup column for every
    non-reference engine (ref / other). Generalises the old 2-engine table."""
    by_cell: dict[tuple, dict] = {}
    present: set[str] = set()
    for r in rows:
        key = (r.get("num_atoms"), r.get("world_size"))
        by_cell.setdefault(key, {})[r["engine"]] = r
        present.add(r["engine"])

    engines = [e for e in ENGINES if e in present] + sorted(present - set(ENGINES))
    others = [e for e in engines if e != ref_engine]

    def cell(r):
        if r is None:
            return "-"
        if r.get("status") != "ok" or not (r.get("steps_per_sec") or 0):
            return r.get("status", "?").upper()
        return f"{r['steps_per_sec']:.3f}"

    # engine -> set of models it actually ran (usually one)
    eng_models: dict[str, set] = {}
    for r in rows:
        if r.get("model"):
            eng_models.setdefault(r["engine"], set()).add(r["model"])
    legend = "; ".join(
        f"{ENGINE_LABEL.get(e, e)} = {'/'.join(sorted(eng_models.get(e, {'?'})))}"
        for e in engines
    )

    # The MATCHED engines (everything except distnnp-uma-opt) must share one UMA
    # kernel configuration -- steps/s from `default` and `turbo` differ by the
    # kernel path, not the distribution strategy. distnnp-uma-opt is expected to
    # differ and is reported on its own line.
    kernels = sorted({r["inference_kernels"] for r in rows
                      if r.get("inference_kernels")
                      and r.get("engine") not in OPT_ENGINES})
    opt_kernels = sorted({r["inference_kernels"] for r in rows
                          if r.get("inference_kernels")
                          and r.get("engine") in OPT_ENGINES})
    kernel_note = ""
    if len(kernels) > 1:
        kernel_note = (
            "\n> **WARNING — mixed UMA kernel configurations among the matched "
            "engines; steps/s across rows are NOT comparable:**\n> "
            + "\n> ".join(f"- `{k}`" for k in kernels)
            + "\n> \n> Re-run every matched cell at one configuration before "
              "publishing this table.\n"
        )
    elif kernels:
        kernel_note = f"\n_Matched-engine UMA kernels: `{kernels[0]}`._\n"
    if opt_kernels:
        kernel_note += (f"_DistNNP-UMA-opt kernels: `{'`, `'.join(opt_kernels)}` "
                        "(separate curve, not a matched comparison)._\n")

    # For the matched engines the ratio is ref/other (the headline "DistNNP is
    # Nx FairChem"); for distnnp-uma-opt it is other/ref, so the speedup of the
    # optimised kernels over the matched DistNNP reads as a number > 1.
    def _ratio_head(e):
        if e in OPT_ENGINES:
            return f"{ENGINE_LABEL.get(e, e)}/{ENGINE_LABEL.get(ref_engine, ref_engine)}"
        return f"{ENGINE_LABEL.get(ref_engine, ref_engine)}/{ENGINE_LABEL.get(e, e)}"

    head = ["atoms", "GPUs"] + [f"{ENGINE_LABEL.get(e, e)} steps/s" for e in engines]
    head += [_ratio_head(e) for e in others]
    head += ["notes"]
    lines = [f"_Model per engine: {legend}._",
             kernel_note,
             "",
             "| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]

    for (atoms, gpus) in sorted(k for k in by_cell if k[0] is not None):
        cells = by_cell[(atoms, gpus)]
        row = [str(atoms), str(gpus)] + [cell(cells.get(e)) for e in engines]
        ref_r = cells.get(ref_engine)
        ref_s = _sps(ref_r)
        ref_model = (ref_r or {}).get("model")
        mixed_model = False
        for e in others:
            e_r = cells.get(e)
            e_s = _sps(e_r)
            # A steps/s ratio between two engines only isolates decomposition if
            # they ran the SAME model. Cross-family (UMA vs MACE) ratios are
            # model-confounded -- blank them rather than invite a false read.
            if ref_s and e_s and (e_r or {}).get("model") == ref_model:
                ratio = e_s / ref_s if e in OPT_ENGINES else ref_s / e_s
                row.append(f"{ratio:.2f}x")
            else:
                row.append("-")
                if ref_s and e_s:
                    mixed_model = True
        notes = [f"{ENGINE_LABEL.get(e, e)} OOM"
                 for e in engines if (cells.get(e) or {}).get("status") == "oom"]
        if mixed_model:
            notes.append("cross-model cells (ratio suppressed)")
        row.append(", ".join(notes))
        lines.append("| " + " | ".join(row) + " |")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(here, "..", "results"))
    ap.add_argument("--out-dir", default=os.path.join(here, "figures"))
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = load(results_dir)
    if not rows:
        print(f"no results in {results_dir}"); return
    write_csv(rows, os.path.join(here, "summary.csv"))
    strong_scaling(rows, out_dir)
    fill_scaling(rows, out_dir)
    oom_ceiling(rows, out_dir)
    headline_table(rows, os.path.join(here, "headline_table.md"))


if __name__ == "__main__":
    main()
