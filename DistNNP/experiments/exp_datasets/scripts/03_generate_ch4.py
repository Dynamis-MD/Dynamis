#!/usr/bin/env python3
"""03_generate_ch4.py — Version-independent CH4/O2 baseline generator.

Generates the CH4:O2 = 1:2 mixture systems used as scaling baselines, at every
size in the reference table. This replaces the old RDKit-based generator and is
**self-contained** (no rdkit, no dependency on ../../../baseline/generate_xyz.py).

------------------------------------------------------------------------------
Why this rewrite exists — reproducibility across machines / library versions
------------------------------------------------------------------------------
The previous generator built each molecule with RDKit's ETKDG distance-geometry
embedding + UFF force-field optimization, then rotated it with libm sin/cos.
None of that is reproducible: ETKDG/UFF change between RDKit releases, and libm
transcendental functions differ in the last bit between platforms. So "same
seed" did NOT give the same data on a different machine or a different RDKit.

For CH4 (a rigid tetrahedron) and O2 (a rigid dimer) the conformer search was
pure overkill. This version hardcodes the geometry and draws only the things
that are provably stable, giving a clean two-tier guarantee:

  POSITIONS  — bit-identical on **any platform and any NumPy version**.
               They use only:
                 * IEEE-754 correctly-rounded ops (+ - * / sqrt) for geometry,
                 * integer-only RNG (RandomState.permutation / randint) for the
                   grid shuffle and orientation choice,
                 * exact signed-permutation rotation matrices (entries -1/0/1),
                   i.e. NO sin/cos/exp/log anywhere on the position path.

  MOMENTA    — identical across **NumPy versions** (legacy RandomState stream is
               frozen by NEP-19) and across platforms up to the last ULP (the
               Gaussian polar method calls libm `log`, which is not correctly
               rounded). This is physically irrelevant: any MD engine re-draws
               or thermalizes initial velocities. Positions are drawn from a
               SEPARATE RNG stream, so this last-bit momenta variance can never
               perturb the (bit-exact) positions.

Net effect: you can delete the committed .xyz files and regenerate byte-for-byte
identical positions anywhere. See scripts/README.md for the full policy.

------------------------------------------------------------------------------
Variants
------------------------------------------------------------------------------
  splited  (default)  — normal-density baseline: CH4 at x>0, O2 at x<0, both at
                        physical spacing (min interatomic distance ~1.09 Å).
                        This is the ONLY variant generated / checksummed by
                        default and the one the datasets ship.
  cluster  (opt-in)   — an *intentional extreme* for partition-imbalance stress:
                        CH4 is concentrated into a small corner, which packs
                        atoms far past contact (they OVERLAP by design). It is
                        NOT a normal-density system, is not part of the pinned
                        dataset, and must not be fed to an energy model. Enable
                        explicitly with `--variants cluster`.
  mixed    (opt-in)   — both species uniform across the box, interleaved.

Usage
------------------------------------------------------------------------------
    # regenerate the normal-density baselines (splited) into ../baselines/
    python 03_generate_ch4.py

    # a subset of sizes
    python 03_generate_ch4.py --sizes 1 2 4 8

    # opt in to the extreme imbalance-stress variant (see "Variants" above)
    python 03_generate_ch4.py --variants cluster

    # a single custom system
    python 03_generate_ch4.py --single --num-c 111 --box 30 --variant splited \
        --out /tmp/ch4_1k.xyz
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── output schema (unchanged from the original, drop-in compatible) ──────────
TEMPERATURE = 298.0  # written to header as T_K
MOMENTA_STD = 0.5  # std of Gaussian momenta components
PBC = "T T T"
ATOM_FMT = "%-2s%14.8f%14.8f%14.8f%14.8f%14.8f%14.8f"
MOMENTA_STREAM_OFFSET = 1_000_003  # momenta RNG = RandomState(seed + this)

# ── reference size table (identical to the original 03 wrapper) ──────────────
# 1 CH4 = 5 atoms (C + 4H); per CH4 there are 2 O2 = 4 atoms -> 9 atoms / unit.
#   num_ch4  = round(k * 1000 / 9)
#   box_side = round(60 * (k/8)^(1/3))   [Å, constant density scaling]
SIZES_K = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# Default deliverable: normal-density "splited" only.
DEFAULT_VARIANTS = ["splited"]
# `cluster` is opt-in (see the "Variants" note in the module docstring): CH4 is
# packed into CLUSTER_FRACTION of the box, which overlaps atoms by design — an
# extreme partition-imbalance artifact, not a normal-density system.
CLUSTER_FRACTION = 0.3  # CH4 concentrated in 30% of box linear extent
SEED_SPLITED = 42
SEED_CLUSTER = 43


def _size_params(target_k: int) -> Tuple[int, int, int]:
    """(num_ch4, box_side_A, total_atoms) for a target size in thousands."""
    num_ch4 = round(target_k * 1000 / 9)
    box_side = round(60.0 * (target_k / 8) ** (1 / 3))
    total_atoms = num_ch4 * 5 + num_ch4 * 2 * 2
    return num_ch4, box_side, total_atoms


# ── hardcoded rigid geometry (centred at the molecular centroid) ─────────────
def ch4_geometry() -> Tuple[List[str], np.ndarray]:
    """CH4: C at origin, 4 H on tetrahedron vertices, C-H = 1.09 Å."""
    bond = 1.09
    s = bond / math.sqrt(3.0)  # correctly-rounded sqrt -> exact
    h = (
        np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)
        * s
    )
    coords = np.vstack([np.zeros((1, 3)), h])  # centroid already at origin
    return ["C", "H", "H", "H", "H"], coords


def o2_geometry() -> Tuple[List[str], np.ndarray]:
    """O2: two O along x, bond = 1.21 Å, centred."""
    half = 1.21 / 2.0
    coords = np.array([[half, 0.0, 0.0], [-half, 0.0, 0.0]], dtype=np.float64)
    return ["O", "O"], coords


def max_radius(coords: np.ndarray) -> float:
    return float(np.sqrt((coords**2).sum(axis=1)).max())


# ── the 24 proper rotations of the cube (exact integer matrices) ─────────────
# Every entry is in {-1, 0, 1}; applying one to coordinates is a pure
# axis-permutation-with-signs, so it introduces NO floating-point error and NO
# transcendental calls -> positions stay bit-identical across platforms.
def _cube_rotations() -> np.ndarray:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=np.float64)
            for row, (col, sgn) in enumerate(zip(perm, signs)):
                m[row, col] = sgn
            if round(np.linalg.det(m)) == 1:  # proper rotations only
                mats.append(m)
    return np.array(mats)  # (24, 3, 3)


ROTATIONS_24 = _cube_rotations()


def _place_molecules(
    rotations: np.ndarray, local: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Rotate `local` (A,3) by per-molecule `rotations` (M,3,3) and translate to
    `centers` (M,3). Returns (M*A, 3) in molecule-major order."""
    rotated = np.einsum("mij,aj->mai", rotations, local)  # (M, A, 3)
    atoms = rotated + centers[:, None, :]  # (M, A, 3)
    return atoms.reshape(-1, 3)


# ── deterministic grid center generators (ported from the reference) ─────────
def _grid_counts(
    lengths: Tuple[float, float, float], n: int, largest_spacing: bool
) -> Tuple[int, int, int]:
    """Integer grid counts (nx,ny,nz) with capacity >= n. Pure integer/float
    comparisons -> platform independent."""
    lx, ly, lz = lengths
    total = lx + ly + lz
    k = math.ceil(n ** (1.0 / 3.0))
    nx = max(1, math.ceil(k * (lx / total) * 3))
    ny = max(1, math.ceil(k * (ly / total) * 3))
    nz = max(1, math.ceil(k * (lz / total) * 3))
    while nx * ny * nz < n:
        if largest_spacing:
            sx, sy, sz = lx / nx, ly / ny, lz / nz
            if sx >= sy and sx >= sz:
                nx += 1
            elif sy >= sx and sy >= sz:
                ny += 1
            else:
                nz += 1
        else:
            if lx / nx <= ly / ny and lx / nx <= lz / nz:
                nx += 1
            elif ly / ny <= lx / nx and ly / ny <= lz / nz:
                ny += 1
            else:
                nz += 1
    return nx, ny, nz


def _enumerate_centers(
    bounds: Tuple[float, float, float, float, float, float],
    counts: Tuple[int, int, int],
) -> np.ndarray:
    """All cell centres inside the given [x0,x1]x[y0,y1]x[z0,z1] box, x-major."""
    x0, x1, y0, y1, z0, z1 = bounds
    nx, ny, nz = counts
    lx, ly, lz = max(x1 - x0, 0.0), max(y1 - y0, 0.0), max(z1 - z0, 0.0)
    xs = x0 + (np.arange(nx) + 0.5) * (lx / nx)
    ys = y0 + (np.arange(ny) + 0.5) * (ly / ny)
    zs = z0 + (np.arange(nz) + 0.5) * (lz / nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def grid_full_box(
    n: int, a: float, b: float, c: float, margin: float, rng: np.random.RandomState
) -> np.ndarray:
    """n shuffled cell centres in [0,a]x[0,b]x[0,c] (margin from the walls)."""
    counts = _grid_counts((a, b, c), n, largest_spacing=False)
    centers = _enumerate_centers(
        (margin, a - margin, margin, b - margin, margin, c - margin), counts
    )
    perm = rng.permutation(len(centers))
    return centers[perm][:n]


def grid_in_range(
    n: int,
    bounds: Tuple[float, float, float, float, float, float],
    rng: np.random.RandomState,
) -> np.ndarray:
    """n shuffled cell centres inside an explicit bounding box."""
    x0, x1, y0, y1, z0, z1 = bounds
    lengths = (max(x1 - x0, 1e-9), max(y1 - y0, 1e-9), max(z1 - z0, 1e-9))
    counts = _grid_counts(lengths, n, largest_spacing=True)
    centers = _enumerate_centers(bounds, counts)
    perm = rng.permutation(len(centers))
    return centers[perm][:n]


# ── system assembly ──────────────────────────────────────────────────────────
def build_system(
    num_ch4: int, a: float, b: float, c: float, variant: str, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (species (N,), positions (N,3)) for one CH4/O2 system.

    variant: 'splited' (CH4 x>0, O2 x<0), 'cluster' (CH4 in a corner,
             O2 uniform) or 'mixed' (both uniform, interleaved).
    Uses one RNG stream `seed` for all geometry decisions.
    """
    geom = np.random.RandomState(seed)
    ch4_sym, ch4_local = ch4_geometry()
    o2_sym, o2_local = o2_geometry()
    max_r = max(max_radius(ch4_local), max_radius(o2_local)) + 0.5

    num_o2 = 2 * num_ch4

    def rot(m: int) -> np.ndarray:
        return ROTATIONS_24[geom.randint(0, len(ROTATIONS_24), size=m)]

    if variant == "cluster":
        # CH4 concentrated in a [0, f*side] corner; O2 uniform in the full box.
        f = CLUSTER_FRACTION
        ch4_centers = grid_full_box(num_ch4, a * f, b * f, c * f, max_r, geom)
        o2_centers = grid_full_box(num_o2, a, b, c, max_r, geom)
        ch4_atoms = _place_molecules(rot(num_ch4), ch4_local, ch4_centers)
        o2_atoms = _place_molecules(rot(num_o2), o2_local, o2_centers)
        species = np.array(ch4_sym * num_ch4 + o2_sym * num_o2)
        positions = np.vstack([ch4_atoms, o2_atoms])

    elif variant == "splited":
        # CH4 at x>0, O2 at x<0; coordinates centred at the origin.
        ha, hb, hc = a / 2.0, b / 2.0, c / 2.0
        if ha - 2.0 * max_r <= 0:
            raise SystemExit("Box too small in x for split placement.")
        o2_centers = grid_in_range(
            num_o2,
            (-ha + max_r, -max_r, -hb + max_r, hb - max_r, -hc + max_r, hc - max_r),
            geom,
        )
        ch4_centers = grid_in_range(
            num_ch4,
            (max_r, ha - max_r, -hb + max_r, hb - max_r, -hc + max_r, hc - max_r),
            geom,
        )
        ch4_atoms = _place_molecules(rot(num_ch4), ch4_local, ch4_centers)
        o2_atoms = _place_molecules(rot(num_o2), o2_local, o2_centers)
        species = np.array(ch4_sym * num_ch4 + o2_sym * num_o2)
        positions = np.vstack([ch4_atoms, o2_atoms])

    elif variant == "mixed":
        # Both species uniform across the whole box, interleaved by a shuffle.
        total = num_ch4 + num_o2
        centers = grid_full_box(total, a, b, c, max_r, geom)
        is_ch4 = np.zeros(total, dtype=bool)
        is_ch4[:num_ch4] = True
        is_ch4 = is_ch4[geom.permutation(total)]
        ch4_centers = centers[is_ch4]
        o2_centers = centers[~is_ch4]
        ch4_atoms = _place_molecules(rot(len(ch4_centers)), ch4_local, ch4_centers)
        o2_atoms = _place_molecules(rot(len(o2_centers)), o2_local, o2_centers)
        # keep species aligned with the interleaved center order
        species_blocks: List[str] = []
        positions_blocks: List[np.ndarray] = []
        ci = oi = 0
        for flag in is_ch4:
            if flag:
                species_blocks.extend(ch4_sym)
                positions_blocks.append(ch4_atoms[ci * 5 : (ci + 1) * 5])
                ci += 1
            else:
                species_blocks.extend(o2_sym)
                positions_blocks.append(o2_atoms[oi * 2 : (oi + 1) * 2])
                oi += 1
        species = np.array(species_blocks)
        positions = np.vstack(positions_blocks)

    else:
        raise SystemExit(f"Unknown variant: {variant}")

    return species, positions


def build_momenta(num_atoms: int, seed: int, momenta_std: float) -> np.ndarray:
    """Gaussian momenta from a SEPARATE frozen RNG stream (see module docstring).
    Independent of the geometry stream, so positions stay bit-exact."""
    mom = np.random.RandomState(seed + MOMENTA_STREAM_OFFSET)
    return mom.standard_normal((num_atoms, 3)) * momenta_std


def write_xyz(
    out_path: Path,
    species: np.ndarray,
    positions: np.ndarray,
    momenta: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> None:
    n = len(species)
    header = (
        f'Lattice="{a:.1f} 0.0 0.0 0.0 {b:.1f} 0.0 0.0 0.0 {c:.1f}" '
        f"Properties=species:S:1:pos:R:3:momenta:R:3 "
        f'T_K={TEMPERATURE:g} pbc="{PBC}"'
    )
    rec = np.empty(
        n,
        dtype=[
            ("s", "U2"),
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("px", "f8"),
            ("py", "f8"),
            ("pz", "f8"),
        ],
    )
    rec["s"] = species
    rec["x"], rec["y"], rec["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    rec["px"], rec["py"], rec["pz"] = momenta[:, 0], momenta[:, 1], momenta[:, 2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        f.write(header + "\n")
        np.savetxt(f, rec, fmt=ATOM_FMT)


def generate_one(
    num_ch4: int,
    box: float,
    out_path: Path,
    variant: str,
    seed: int,
    momenta_std: float,
) -> int:
    species, positions = build_system(num_ch4, box, box, box, variant, seed)
    momenta = build_momenta(len(species), seed, momenta_std)
    write_xyz(out_path, species, positions, momenta, box, box, box)
    return len(species)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Version-independent CH4/O2 generator")
    p.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=None,
        help="Target sizes in thousands (default: all in the table)",
    )
    p.add_argument(
        "--variants",
        nargs="*",
        default=DEFAULT_VARIANTS,
        choices=["splited", "cluster", "mixed"],
        help="Default: splited (normal density). 'cluster' is an "
        "opt-in extreme imbalance artifact (overlapping).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".." / "baselines",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Regenerate files that already exist"
    )
    p.add_argument("--momenta-std", type=float, default=MOMENTA_STD)
    # single-system mode
    p.add_argument(
        "--single",
        action="store_true",
        help="Generate one custom system (needs --num-c/--box/--out)",
    )
    p.add_argument("--num-c", type=int, help="[single] number of CH4 molecules")
    p.add_argument("--box", type=float, help="[single] cubic box side (Å)")
    p.add_argument(
        "--variant",
        default="splited",
        choices=["splited", "cluster", "mixed"],
        help="[single]",
    )
    p.add_argument("--out", type=Path, help="[single] output path")
    p.add_argument("--seed", type=int, help="[single] RNG seed override")
    args = p.parse_args()

    if args.single:
        if args.num_c is None or args.box is None or args.out is None:
            p.error("--single requires --num-c, --box and --out")
        seed = (
            args.seed
            if args.seed is not None
            else (SEED_SPLITED if args.variant == "splited" else SEED_CLUSTER)
        )
        n = generate_one(
            args.num_c, args.box, args.out, args.variant, seed, args.momenta_std
        )
        print(f"Wrote {args.out}  ({n} atoms)")
        return

    sizes = args.sizes if args.sizes is not None else SIZES_K
    out_dir: Path = args.out_dir.resolve()
    print(f"Output dir: {out_dir}")
    for k in sizes:
        num_ch4, box, expected = _size_params(k)
        for variant in args.variants:
            out_path = out_dir / f"ch4_{k}k_{variant}.xyz"
            if out_path.exists() and not args.overwrite:
                print(f"  SKIP (exists): {out_path.name}")
                continue
            seed = SEED_SPLITED if variant == "splited" else SEED_CLUSTER
            n = generate_one(num_ch4, box, out_path, variant, seed, args.momenta_std)
            status = "OK" if n == expected else f"WARN expected {expected}"
            print(
                f"  {out_path.name:28s} {num_ch4:>7d} CH4  box {box:>3g}Å  "
                f"{n:>9d} atoms  [{status}]"
            )


if __name__ == "__main__":
    main()
