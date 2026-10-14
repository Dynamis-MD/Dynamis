#!/usr/bin/env python3
"""06_generate_electrochemical.py — Pt(111) slab + explicit water layer.

Builds a Pt(111) slab using ASE and fills the space above with TIP3P water
placed on a grid at liquid density. The result demonstrates 1D layered
density non-uniformity (Pt dense near z=0, water above).

Reproduces:
    electrochemical/pt_water_12k_density.xyz   (Pt 20×20×4 + 40 Å water, ~14.2k atoms)
    electrochemical/pt_water_17k_density.xyz   (Pt 24×24×4 + 40 Å water, ~21.2k atoms)

Usage:
    python scripts/06_generate_electrochemical.py [--systems 12k 17k]
    python scripts/06_generate_electrochemical.py --dry-run

Requires:
    pip install ase scipy numpy
"""

import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / ".." / "electrochemical"

SYSTEMS = {
    "12k": dict(
        nx=20,
        ny=20,
        nz=4,
        water_height=40.0,
        out="pt_water_12k_density.xyz",
        desc="Pt 20×20×4 + 40 Å water (~14k atoms)",
    ),
    "17k": dict(
        nx=24,
        ny=24,
        nz=4,
        water_height=40.0,
        out="pt_water_17k_density.xyz",
        desc="Pt 24×24×4 + 40 Å water (~21k atoms)",
    ),
}

LATTICE_A = 3.92  # Pt FCC lattice constant (Å)
BUFFER_Z = 2.0  # Gap above/below slab before water starts (Å)
WATER_SPACING = 3.103  # Grid spacing → ~0.0334 mol/Å³ liquid density
PT_O_MIN_DIST = 2.8  # Minimum Pt–O distance (Å)
SEED = 42


def generate_pt_water(nx, ny, nz, water_height, out_path, dry_run=False):
    from ase.build import fcc111
    from ase import Atoms
    from ase.io import write
    from scipy.spatial import cKDTree

    print(f"  Building Pt({nx}×{ny}×{nz}) slab ...")
    slab = fcc111("Pt", size=(nx, ny, nz), a=LATTICE_A, vacuum=0.0, periodic=True)
    pt_pos = slab.get_positions()
    pt_sym = np.array(slab.get_chemical_symbols())
    cell = np.array(slab.get_cell())
    n_pt = len(slab)

    z_min_slab = pt_pos[:, 2].min()
    z_max_slab = pt_pos[:, 2].max()
    slab_thick = z_max_slab - z_min_slab

    total_z = slab_thick + 2 * BUFFER_Z + water_height
    cell[2, 2] = total_z

    pt_pos[:, 2] -= z_min_slab
    pt_pos[:, 2] += BUFFER_Z

    water_z_start = pt_pos[:, 2].max() + BUFFER_Z
    water_z_end = total_z - BUFFER_Z

    cell_2d = cell[:2]  # 2×3
    lx = np.linalg.norm(cell[0])
    ly = np.linalg.norm(cell[1])
    surface_area = np.linalg.norm(np.cross(cell[0], cell[1]))

    print(
        f"  Slab: {n_pt} atoms, thickness {slab_thick:.1f} Å, "
        f"surface {lx:.1f}×{ly:.1f} Å²={surface_area:.0f} Å²"
    )
    print(f"  Water region: z = {water_z_start:.1f} – {water_z_end:.1f} Å")

    if dry_run:
        print("  [DRY-RUN] skipping water placement and file write.")
        return

    rng = np.random.default_rng(SEED)
    nx_w = int(np.ceil(lx / WATER_SPACING))
    ny_w = int(np.ceil(ly / WATER_SPACING))
    nz_w = int(np.ceil((water_z_end - water_z_start) / WATER_SPACING))

    o_positions = []
    for ix in range(nx_w):
        for iy in range(ny_w):
            for iz in range(nz_w):
                fx = (ix + 0.5) / nx_w
                fy = (iy + 0.5) / ny_w
                fz = water_z_start + (iz + 0.5) * WATER_SPACING
                pos = fx * cell[0] + fy * cell[1]
                pos[2] = fz
                pos += rng.uniform(-0.3, 0.3, 3)
                pos[2] = np.clip(pos[2], water_z_start + 0.5, water_z_end - 0.5)
                o_positions.append(pos)

    o_positions = np.array(o_positions)
    tree = cKDTree(pt_pos)
    dists, _ = tree.query(o_positions)
    o_positions = o_positions[dists > PT_O_MIN_DIST]
    n_mol = len(o_positions)

    # H positions with random orientations
    bond_len = 0.9572
    half_angle = 104.52 / 2 * np.pi / 180
    h1_local = np.array(
        [bond_len * np.sin(half_angle), bond_len * np.cos(half_angle), 0.0]
    )
    h2_local = np.array(
        [-bond_len * np.sin(half_angle), bond_len * np.cos(half_angle), 0.0]
    )

    phi = rng.uniform(0, 2 * np.pi, n_mol)
    cos_theta = rng.uniform(-1, 1, n_mol)
    sin_theta = np.sqrt(1 - cos_theta**2)
    psi = rng.uniform(0, 2 * np.pi, n_mol)

    h1_all = np.zeros_like(o_positions)
    h2_all = np.zeros_like(o_positions)
    for i in range(n_mol):
        cp, sp = np.cos(phi[i]), np.sin(phi[i])
        ct, st = cos_theta[i], sin_theta[i]
        cps, sps = np.cos(psi[i]), np.sin(psi[i])
        R = np.array(
            [
                [cp * ct * cps - sp * sps, -cp * ct * sps - sp * cps, cp * st],
                [sp * ct * cps + cp * sps, -sp * ct * sps + cp * cps, sp * st],
                [-st * cps, st * sps, ct],
            ]
        )
        h1_all[i] = o_positions[i] + R @ h1_local
        h2_all[i] = o_positions[i] + R @ h2_local

    water_pos = np.vstack([o_positions, h1_all, h2_all])
    water_sym = ["O"] * n_mol + ["H"] * (2 * n_mol)

    all_sym = list(pt_sym) + water_sym
    all_pos = np.vstack([pt_pos, water_pos])
    n_total = len(all_sym)

    pt_vol = surface_area * (slab_thick + 2 * BUFFER_Z)
    water_vol = surface_area * (water_z_end - water_z_start)
    print(f"  Pt local density:    {n_pt / pt_vol:.4f} atoms/Å³")
    print(f"  Water local density: {len(water_sym) / water_vol:.4f} atoms/Å³")
    print(f"  Total: {n_total} atoms  (Pt {n_pt}, water {len(water_sym)})")

    result = Atoms(symbols=all_sym, positions=all_pos, cell=cell, pbc=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), result, format="extxyz")
    print(f"  Written: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pt(111) slab + water electrochemical systems"
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        choices=list(SYSTEMS.keys()),
        help="Systems to generate (default: all)",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available systems:")
        for k, cfg in SYSTEMS.items():
            print(f"  {k:<6} {cfg['out']:<40} ({cfg['desc']})")
        return

    keys = args.systems or list(SYSTEMS.keys())
    for key in keys:
        cfg = SYSTEMS[key]
        out_path = args.out_dir / cfg["out"]
        print(f"\n=== {key}: {cfg['desc']} ===")

        if out_path.exists() and not args.overwrite and not args.dry_run:
            from ase.io import read

            n = len(read(str(out_path)))
            print(f"  SKIP (exists): {out_path.name} ({n} atoms)")
            continue

        generate_pt_water(
            nx=cfg["nx"],
            ny=cfg["ny"],
            nz=cfg["nz"],
            water_height=cfg["water_height"],
            out_path=out_path,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
