#!/usr/bin/env python3
"""Aggregate exp2_scaling JSON results and TimingRegistry traces into:
  - <out-dir>/summary.csv
  - <out-dir>/scaling_report.md
  - <out-dir>/scaling_table.tex
  - <out-dir>/figures/{strong_scaling,weak_scaling,memory_scaling,phase_breakdown}.{png,pdf}

Works with any model driver emitting BenchRecord JSONs (ORB, UMA, MACE, ...).

Usage:
    python experiments/exp2_scaling/analysis/summarize_scaling.py \
        --results-dir experiments/exp2_scaling/results \
        --out-dir experiments/exp2_scaling/analysis \
        --title "Dynamis Scaling"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _parse_timing_dir(timing_dir: str | None) -> dict[str, float]:
    """Return tree-exclusive class_pct (compute/comm/wait/other %) and median step ms
    from TimingRegistry JSONLs."""
    if not timing_dir or not os.path.isdir(timing_dir):
        return {}
    files = sorted(glob.glob(os.path.join(timing_dir, "timing_rank*.jsonl")))
    if not files:
        return {}
    rank_iter_ms: list[float] = []
    class_means_ns: dict[str, list[float]] = defaultdict(list)
    total_means_ns: list[float] = []
    for idx, path in enumerate(files):
        recs_by_step: dict[int, list[dict]] = defaultdict(list)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if (
                    obj.get("type") != "rec"
                    or obj.get("time_ns") is None
                    or obj.get("error") is not None
                ):
                    continue
                if (
                    idx == 0
                    and obj.get("key") == "iteration"
                    and obj.get("where") == "host"
                ):
                    rank_iter_ms.append(obj["time_ns"] / 1e6)
                if obj.get("where") == "device":
                    recs_by_step[int(obj["step"])].append(obj)
        step_tot: list[float] = []
        step_cls: dict[str, list[float]] = defaultdict(list)
        for _step, recs in recs_by_step.items():
            children_ns: dict[int, float] = defaultdict(float)
            for r in recs:
                pid = r.get("parent_id")
                if pid is not None:
                    children_ns[int(pid)] += float(r["time_ns"])
            tot = 0.0
            cls_acc: dict[str, float] = defaultdict(float)
            for r in recs:
                own = max(0.0, float(r["time_ns"]) - children_ns.get(int(r["id"]), 0.0))
                if r.get("key") == "iteration" or r.get("class") == "total":
                    tot += float(r["time_ns"])
                    cls_acc["other"] += own
                else:
                    cls_acc[r.get("class", "other")] += own
            if tot > 0:
                step_tot.append(tot)
                for c in ("compute", "comm", "wait", "other"):
                    step_cls[c].append(cls_acc.get(c, 0.0))
        if step_tot:
            total_means_ns.append(sum(step_tot) / len(step_tot))
            for c in ("compute", "comm", "wait", "other"):
                class_means_ns[c].append(sum(step_cls[c]) / len(step_cls[c]))
    out: dict[str, float] = {}
    if total_means_ns:
        denom = sum(total_means_ns) / len(total_means_ns)
        if denom > 0:
            for cls in ("compute", "comm", "wait", "other"):
                num = sum(class_means_ns[cls]) / len(class_means_ns[cls])
                out[f"pct_{cls}"] = 100.0 * num / denom
    if rank_iter_ms:
        med_ms = statistics.median(rank_iter_ms)
        out["med_step_ms"] = med_ms
        out["med_steps_per_sec"] = 1000.0 / med_ms if med_ms > 0 else float("nan")
        out["min_step_ms"] = min(rank_iter_ms)
        out["max_step_ms"] = max(rank_iter_ms)
    return out


def _mean_halo_pct(local_counts, ghost_counts) -> float | None:
    if not local_counts or not ghost_counts or len(local_counts) != len(ghost_counts):
        return None
    tot_local = sum(local_counts)
    tot_ghost = sum(ghost_counts)
    if tot_local <= 0:
        return None
    return 100.0 * tot_ghost / tot_local


def load_records(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        extra = d.get("extra") if isinstance(d.get("extra"), dict) else {}
        sps = d.get("steps_per_sec")
        natoms = d.get("num_atoms")
        w = d.get("world_size")
        us_per_atom_step = (1e6 / (sps * natoms)) if (sps and natoms) else None
        us_gpu_per_atom_step = (us_per_atom_step * w) if (us_per_atom_step and w) else None
        row = {
            "file": os.path.basename(path),
            "study": d.get("study"),
            "tag": d.get("tag"),
            "model_family": d.get("model_family"),
            "model": d.get("model"),
            "precision": d.get("precision") or d.get("orb_precision"),
            "mixed_precision": extra.get("mixed_precision", d.get("mixed_precision")),
            "edge_chunk_size": extra.get("edge_chunk_size", d.get("edge_chunk_size")),
            "fused_kernels_active": extra.get(
                "fused_kernels_active", d.get("fused_kernels_active")
            ),
            "max_neighbors": d.get("max_num_neighbors"),
            "exchanger": d.get("exchanger"),
            "use_fabric": d.get("use_fabric"),
            "num_atoms": natoms,
            "world_size": w,
            "nodes": d.get("nodes"),
            "atoms_per_gpu": d.get("atoms_per_gpu"),
            "halo_pct": _mean_halo_pct(
                d.get("local_atoms_per_rank"), d.get("ghost_atoms_per_rank")
            ),
            "status": d.get("status"),
            "steps_per_sec": sps,
            "ns_per_day": d.get("ns_per_day"),
            "us_per_atom_step": us_per_atom_step,
            "us_gpu_per_atom_step": us_gpu_per_atom_step,
            "elapsed_s": d.get("elapsed_s"),
            "peak_mem_gb": (
                d["peak_mem_mb_max"] / 1024.0
                if d.get("peak_mem_mb_max") is not None
                else None
            ),
            "mean_temp_k": d.get("mean_temperature_k"),
            "slurm_job_id": d.get("slurm_job_id"),
            "error": d.get("error"),
        }
        row.update(_parse_timing_dir(d.get("timing_dir")))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_efficiencies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speedup"] = float("nan")
    df["efficiency_pct"] = float("nan")
    df["weak_eff_norm_pct"] = float("nan")

    ok = df["status"] == "ok"
    # Strong scaling: group by (model, exchanger, tag, num_atoms)
    for _key, grp in df[ok & (df["study"] == "strong")].groupby(
        ["model", "exchanger", "tag", "num_atoms"], dropna=False
    ):
        grp_s = grp.sort_values("world_size")
        w0 = grp_s["world_size"].iloc[0]
        sps0 = grp_s["steps_per_sec"].iloc[0]
        if sps0 and sps0 > 0:
            speedup = grp_s["steps_per_sec"] / sps0
            eff = 100.0 * speedup / (grp_s["world_size"] / w0)
            df.loc[grp_s.index, "speedup"] = speedup
            df.loc[grp_s.index, "efficiency_pct"] = eff

    # Weak scaling: group by (model, exchanger, tag)
    for _key, grp in df[ok & (df["study"] == "weak")].groupby(
        ["model", "exchanger", "tag"], dropna=False
    ):
        grp_s = grp.sort_values("world_size")
        sps0 = grp_s["steps_per_sec"].iloc[0]
        u0 = grp_s["us_gpu_per_atom_step"].iloc[0]
        if sps0 and sps0 > 0:
            df.loc[grp_s.index, "efficiency_pct"] = 100.0 * grp_s["steps_per_sec"] / sps0
        if u0 and u0 > 0:
            df.loc[grp_s.index, "weak_eff_norm_pct"] = (
                100.0 * u0 / grp_s["us_gpu_per_atom_step"]
            )

    return df


def plot_strong(df: pd.DataFrame, out_dir: str, title: str) -> None:
    sub = df[(df["study"] == "strong") & (df["status"] == "ok")]
    if sub.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for (model, exch, tag, natoms), grp in sub.groupby(
        ["model", "exchanger", "tag", "num_atoms"], dropna=False
    ):
        g = grp.sort_values("world_size")
        size_lbl = f"{natoms / 1e6:.2f}M" if natoms >= 1_000_000 else f"{natoms / 1e3:.0f}K"
        lbl = f"{model} {size_lbl} ({tag or exch})"
        line = ax1.plot(g["world_size"], g["steps_per_sec"], marker="o", lw=2, label=lbl)[0]
        w0 = g["world_size"].iloc[0]
        s0 = g["steps_per_sec"].iloc[0]
        ax1.plot(
            g["world_size"],
            s0 * (g["world_size"] / w0),
            ls="--",
            lw=1,
            alpha=0.5,
            color=line.get_color(),
        )
        ax2.plot(g["world_size"], g["efficiency_pct"], marker="o", lw=2, label=lbl)

    xticks = sorted({int(w) for w in sub["world_size"].dropna()})
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("GPUs")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax1.set_yscale("log")
    ax1.set_ylabel("Throughput (steps/s)")
    ax1.set_title(f"{title} — Strong Throughput")

    ax2.axhline(100.0, ls="--", color="gray", lw=1)
    ax2.set_ylabel("Parallel Efficiency (%)")
    ax2.set_title(f"{title} — Strong Efficiency")
    ax2.set_ylim(0, 115)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"strong_scaling.{ext}"), dpi=200)
    plt.close(fig)


def plot_weak(df: pd.DataFrame, out_dir: str, title: str) -> None:
    sub = df[(df["study"] == "weak") & (df["status"] == "ok")]
    if sub.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for (model, exch, tag), grp in sub.groupby(
        ["model", "exchanger", "tag"], dropna=False
    ):
        g = grp.sort_values("world_size")
        lbl = f"{model} {tag or 'weak'} ({exch})"
        ax1.plot(g["world_size"], g["steps_per_sec"], marker="s", lw=2, label=lbl)
        ax2.plot(
            g["world_size"],
            g["weak_eff_norm_pct"],
            marker="s",
            lw=2,
            label=f"{lbl} (atom-norm)",
        )
        ax2.plot(
            g["world_size"],
            g["efficiency_pct"],
            marker="^",
            ls="--",
            lw=1.5,
            alpha=0.75,
            label=f"{lbl} (raw T₁/Tₙ)",
        )

    xticks = sorted({int(w) for w in sub["world_size"].dropna()})
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("GPUs")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax1.set_ylabel("Throughput (steps/s)")
    ax1.set_title(f"{title} — Weak Throughput")

    ax2.axhline(100.0, ls="--", color="gray", lw=1)
    ax2.set_ylabel("Weak Scaling Efficiency (%)")
    ax2.set_title(f"{title} — Weak Efficiency")
    ax2.set_ylim(0, 115)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"weak_scaling.{ext}"), dpi=200)
    plt.close(fig)


def plot_memory(df: pd.DataFrame, out_dir: str, title: str) -> None:
    sub = df[(df["status"] == "ok") & df["peak_mem_gb"].notna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (model, study, tag), grp in sub.groupby(["model", "study", "tag"], dropna=False):
        g = grp.sort_values("atoms_per_gpu")
        ax.plot(
            g["atoms_per_gpu"] / 1000.0,
            g["peak_mem_gb"],
            marker="o",
            lw=1.8,
            label=f"{model} ({study}:{tag})",
        )
    ax.set_xlabel("Local Atoms per GPU (thousands)")
    ax.set_ylabel("Peak GPU HBM (GiB)")
    ax.set_title(f"{title} — Per-GPU Peak HBM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"memory_scaling.{ext}"), dpi=200)
    plt.close(fig)


def plot_breakdown(df: pd.DataFrame, out_dir: str, title: str) -> None:
    sub = df[(df["status"] == "ok") & df.get("pct_compute", pd.Series(dtype=float)).notna()]
    if sub.empty:
        return
    groups = list(sub.groupby(["model", "study", "tag"], dropna=False))
    fig, axes = plt.subplots(1, len(groups), figsize=(5.0 * len(groups), 4.5), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, ((model, study, tag), grp) in zip(axes, groups):
        g = grp.sort_values("world_size")
        xs = [str(int(w)) for w in g["world_size"]]
        comp = g["pct_compute"].fillna(0).values
        comm = g["pct_comm"].fillna(0).values
        wait = g["pct_wait"].fillna(0).values
        other = g["pct_other"].fillna(0).values
        ax.bar(xs, comp, label="Compute")
        ax.bar(xs, comm, bottom=comp, label="Comm")
        ax.bar(xs, wait, bottom=comp + comm, label="Wait")
        ax.bar(xs, other, bottom=comp + comm + wait, label="Other")
        ax.set_title(f"{model}\n{study}: {tag}")
        ax.set_xlabel("GPUs")
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Step Time Share (%)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{title} — Per-Step Timing Breakdown")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"phase_breakdown.{ext}"), dpi=200)
    plt.close(fig)


def _df_to_markdown(df: pd.DataFrame, cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [header, sep]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r.get(c)
            if pd.isna(v):
                vals.append("--")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}" if "us_" in c else f"{v:.2f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_markdown_report(df: pd.DataFrame, path: str, title: str) -> None:
    lines = [f"# {title}\n"]
    cols = [
        "study",
        "model",
        "tag",
        "num_atoms",
        "world_size",
        "nodes",
        "atoms_per_gpu",
        "halo_pct",
        "use_fabric",
        "status",
        "steps_per_sec",
        "med_steps_per_sec",
        "ns_per_day",
        "us_per_atom_step",
        "us_gpu_per_atom_step",
        "speedup",
        "efficiency_pct",
        "weak_eff_norm_pct",
        "peak_mem_gb",
        "pct_compute",
        "pct_comm",
        "mean_temp_k",
    ]
    present = [c for c in cols if c in df.columns]
    for study in ("strong", "weak", "hero", "probe"):
        sub = df[df["study"] == study].sort_values(
            ["model", "exchanger", "tag", "num_atoms", "world_size"]
        )
        if sub.empty:
            continue
        lines.append(f"\n## {study.capitalize()}\n")
        lines.append(_df_to_markdown(sub, present))
        lines.append("\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_latex_table(df: pd.DataFrame, path: str, title: str) -> None:
    ok = df[df["status"] == "ok"].sort_values(["model", "study", "tag", "world_size"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{title} ($\Delta t = 0.5\,\text{{fs}}$).}}",
        r"\label{tab:exp2_scaling}",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Study & Config & Atoms & GPUs & Steps/s & ns/day & $\mu\text{s}/(\text{atom}\cdot\text{step})$ & $\mu\text{s}\cdot\text{GPU}/(\text{atom}\cdot\text{step})$ & Eff. (\%) & Halo (\%) & Peak HBM (GiB) \\",
        r"\midrule",
    ]
    prev_grp = None
    for _, r in ok.iterrows():
        grp = (r["model"], r["study"], r["tag"])
        if prev_grp is not None and grp != prev_grp:
            lines.append(r"\midrule")
        prev_grp = grp
        eff = (
            r["weak_eff_norm_pct"]
            if r["study"] == "weak" and pd.notna(r["weak_eff_norm_pct"])
            else r["efficiency_pct"]
        )
        eff_s = f"{eff:.1f}" if pd.notna(eff) else "--"
        halo_s = f"{r['halo_pct']:.1f}" if pd.notna(r["halo_pct"]) else "--"
        mem_s = f"{r['peak_mem_gb']:.2f}" if pd.notna(r["peak_mem_gb"]) else "--"
        lines.append(
            f"{str(r['study']).capitalize()} & {r['tag'] or r['model']} & {int(r['num_atoms']):,} & "
            f"{int(r['world_size'])} & {r['steps_per_sec']:.2f} & {r['ns_per_day']:.2f} & "
            f"{r['us_per_atom_step']:.3f} & {r['us_gpu_per_atom_step']:.2f} & "
            f"{eff_s} & {halo_s} & {mem_s} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--results-dir", default=os.path.join(exp_dir, "results"))
    p.add_argument("--out-dir", default=os.path.join(exp_dir, "analysis"))
    p.add_argument("--title", default="exp2_scaling — Dynamis Scaling")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = load_records(args.results_dir)
    if df.empty:
        print(f"No JSON result files found in {args.results_dir}")
        return

    df = compute_efficiencies(df)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} records)")

    md_path = os.path.join(args.out_dir, "scaling_report.md")
    write_markdown_report(df, md_path, args.title)
    print(f"Wrote {md_path}")

    tex_path = os.path.join(args.out_dir, "scaling_table.tex")
    write_latex_table(df, tex_path, args.title)
    print(f"Wrote {tex_path}")

    plot_strong(df, fig_dir, args.title)
    plot_weak(df, fig_dir, args.title)
    plot_memory(df, fig_dir, args.title)
    plot_breakdown(df, fig_dir, args.title)
    print(f"Plots written to {fig_dir}")


if __name__ == "__main__":
    main()
