"""Summarize Dynamis timing_rank*.jsonl files.

Reads the per-rank JSONL files written by dynamis_state's TimingRegistry
(see dynamis_state/include/dynamis_state/timing.hpp) and prints:

  - a Compute/Comm/Wait/Other breakdown as a fraction of the `Total`
    ("iteration") self time, aggregated over all ranks
  - per-key call counts and self/inclusive time, in first-seen order
  - a per-rank imbalance view (max/mean ratio) for keys present on
    more than one rank -- useful for spotting a straggler rank

"Self time" is a record's own time_ns minus the time_ns of its children
(records whose parent_id points at it, matched on the same host/device
clock the child was measured on) -- see timing.hpp's discussion of why
summing time_ns directly double-counts nested regions.

USAGE:
    python3 timing_summary.py <dir_with_timing_rank*.jsonl | single_file.jsonl>
    python3 timing_summary.py <dir> --json summary.json
    python3 timing_summary.py <dir> --by-rank
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

CLASS_ORDER = ["compute", "comm", "wait", "other"]


def load_rank_file(path):
    meta = None
    records = []
    call_order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind == "meta":
                meta = obj
            elif kind == "rec":
                records.append(obj)
            elif kind == "call_order":
                call_order = obj["order"]  # last one in the file wins
    return meta, records, call_order


def find_input_files(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "timing_rank*.jsonl")))
        if not files:
            raise SystemExit(f"no timing_rank*.jsonl files found in {path}")
        return files
    return [path]


def self_times_ns(records):
    """Maps (id, where) -> self time_ns for every completed record."""
    children_ns = defaultdict(int)
    for r in records:
        if r.get("time_ns") is None or r.get("parent_id") is None:
            continue
        children_ns[(r["parent_id"], r["where"])] += r["time_ns"]

    out = {}
    for r in records:
        if r.get("time_ns") is None:
            continue
        key = (r["id"], r["where"])
        out[key] = r["time_ns"] - children_ns.get(key, 0)
    return out


def summarize_rank(records):
    """Per-key stats plus the rank's total `Total`-class self time."""
    self_ns = self_times_ns(records)
    stats = defaultdict(
        lambda: {
            "class": None,
            "count": 0,
            "errors": 0,
            "incl_ns": 0,
            "self_ns": 0,
        }
    )
    total_self_ns = 0

    for r in records:
        s = stats[r["key"]]
        s["class"] = r["class"]
        if r.get("error") is not None:
            s["errors"] += 1
            continue
        if r.get("time_ns") is None:
            continue
        s["count"] += 1
        s["incl_ns"] += r["time_ns"]
        own_self = self_ns.get((r["id"], r["where"]), r["time_ns"])
        s["self_ns"] += own_self
        if r["class"] == "total":
            total_self_ns += own_self

    return stats, total_self_ns


def ns_to_ms(ns):
    return ns / 1e6


def print_class_breakdown(class_self_ns, total_self_ns):
    print("\n== Compute / Comm / Wait / Other (fraction of iteration self time) ==")
    if total_self_ns <= 0:
        print(
            "  (no `Total`-class records found; run with the `iteration` timer active)"
        )
        return
    for cls in CLASS_ORDER:
        ns = class_self_ns.get(cls, 0)
        pct = 100.0 * ns / total_self_ns
        print(f"  {cls:<8} {ns_to_ms(ns):>12.2f} ms  {pct:6.2f}%")


def print_key_table(order, combined, total_self_ns):
    print("\n== Per-key totals across all ranks (call_order) ==")
    header = f"  {'key':<24} {'class':<8} {'calls':>8} {'self ms':>12} {'self/call us':>14} {'incl ms':>12} {'% iter':>7} {'errs':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in order:
        if key not in combined:
            continue
        s = combined[key]
        pct = (
            (100.0 * s["self_ns"] / total_self_ns)
            if total_self_ns > 0
            else float("nan")
        )
        per_call_us = (s["self_ns"] / s["count"] / 1e3) if s["count"] else float("nan")
        print(
            f"  {key:<24} {s['class'] or '?':<8} {s['count']:>8} "
            f"{ns_to_ms(s['self_ns']):>12.2f} {per_call_us:>14.2f} "
            f"{ns_to_ms(s['incl_ns']):>12.2f} {pct:>6.2f}% {s['errors']:>6}"
        )


def print_imbalance(order, per_rank_stats):
    ranks = sorted(per_rank_stats)
    if len(ranks) < 2:
        return
    print("\n== Rank imbalance (self time, max/mean across ranks) ==")
    header = f"  {'key':<24} {'min ms':>10} {'mean ms':>10} {'max ms':>10} {'max/mean':>9} {'max rank':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in order:
        values = []
        for rank in ranks:
            s = per_rank_stats[rank].get(key)
            values.append(ns_to_ms(s["self_ns"]) if s else 0.0)
        if not any(values):
            continue
        mean = sum(values) / len(values)
        max_val = max(values)
        max_rank = ranks[values.index(max_val)]
        ratio = (max_val / mean) if mean > 0 else float("inf")
        print(
            f"  {key:<24} {min(values):>10.2f} {mean:>10.2f} {max_val:>10.2f} "
            f"{ratio:>9.2f} {max_rank:>9}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", help="directory of timing_rank*.jsonl files, or a single file"
    )
    parser.add_argument("--json", metavar="PATH", help="also write the summary as JSON")
    parser.add_argument(
        "--by-rank",
        action="store_true",
        help="print each rank's per-key table individually",
    )
    args = parser.parse_args()

    files = find_input_files(args.input)

    call_order = []
    seen_keys = set()
    per_rank_stats = {}
    per_rank_meta = {}
    combined = defaultdict(
        lambda: {"class": None, "count": 0, "errors": 0, "incl_ns": 0, "self_ns": 0}
    )
    class_self_ns = defaultdict(int)
    total_self_ns_all = 0

    for path in files:
        meta, records, order = load_rank_file(path)
        if meta is None:
            print(f"warning: {path} has no meta header, skipping", file=sys.stderr)
            continue
        rank = meta["rank"]
        per_rank_meta[rank] = meta
        for key in order:
            if key not in seen_keys:
                seen_keys.add(key)
                call_order.append(key)

        stats, total_self_ns = summarize_rank(records)
        per_rank_stats[rank] = stats
        total_self_ns_all += total_self_ns

        for key, s in stats.items():
            c = combined[key]
            c["class"] = s["class"]
            c["count"] += s["count"]
            c["errors"] += s["errors"]
            c["incl_ns"] += s["incl_ns"]
            c["self_ns"] += s["self_ns"]
            if s["class"] != "total":
                class_self_ns[s["class"]] += s["self_ns"]

    if not per_rank_meta:
        raise SystemExit("no valid timing files found")

    any_meta = next(iter(per_rank_meta.values()))
    print(
        f"run_id={any_meta.get('run_id')} world_size={any_meta.get('world_size')} "
        f"ranks_loaded={sorted(per_rank_meta)}"
    )

    print_class_breakdown(class_self_ns, total_self_ns_all)
    print_key_table(call_order, combined, total_self_ns_all)
    print_imbalance(call_order, per_rank_stats)

    if args.by_rank:
        for rank in sorted(per_rank_stats):
            print(f"\n== Rank {rank} ==")
            rank_total_self_ns = sum(
                s["self_ns"]
                for s in per_rank_stats[rank].values()
                if s["class"] == "total"
            )
            print_key_table(call_order, per_rank_stats[rank], rank_total_self_ns)

    if args.json:
        out = {
            "run_id": any_meta.get("run_id"),
            "world_size": any_meta.get("world_size"),
            "ranks": sorted(per_rank_meta),
            "class_self_ns": dict(class_self_ns),
            "total_self_ns": total_self_ns_all,
            "keys": {key: combined[key] for key in call_order if key in combined},
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
