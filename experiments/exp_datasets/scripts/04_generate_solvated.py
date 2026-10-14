#!/usr/bin/env python3
"""04_generate_solvated.py — Protein + explicit water at liquid density.

Reads vacuum protein XYZ files (from 02_convert_vacuum.py), extracts protein
atoms via covalent-bond BFS, fills the remaining box volume with TIP3P water
placed on a grid at ~liquid water density (0.0334 molecules/Å³).

The resulting systems are NOT MD-equilibrated (grid water placement). Use the
OpenMM-equilibrated DHFR/ApoA1 systems (05_generate_openmm.py) for equilibrated runs.

Reproduces:
    protein_solvated/1aki_density.xyz   (~25.8k atoms)
    protein_solvated/2rik_density.xyz   (~112.7k atoms)
    protein_solvated/9gmo_density.xyz   (~1.54M atoms)  ← slow, ~10 min

Usage:
    python scripts/04_generate_solvated.py [--systems 1aki 2rik]
    python scripts/04_generate_solvated.py --list

Requires:
    pip install ase scipy numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VAC_DIR = SCRIPT_DIR / ".." / "protein_vacuum"
OUT_DIR = SCRIPT_DIR / ".." / "protein_solvated"

# Buffer (Å) added on each side of protein bounding box to set box size
BUFFER = 12.0

SYSTEMS = {
    "1aki": ("1aki_vacuum_openmm.xyz", "1aki_density.xyz", "Lysozyme"),
    "2rik": ("2rik_vacuum_openmm.xyz", "2rik_density.xyz", "Titin I-band"),
    "9gmo": (
        "9gmo_vacuum.xyz",
        "9gmo_density.xyz",
        "Ribosome 9GMO (~1.5M atoms, slow)",
    ),
}

PROTEIN_ELEMENTS = {"C", "N", "S", "P", "Fe", "Zn", "Ca", "Mg", "Na", "Cl", "K"}


def extract_protein(atoms):
    """Extract protein heavy atoms + bonded H/O via covalent-bond BFS."""
    from scipy.spatial import cKDTree

    species = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()

    heavy_mask = np.isin(species, list(PROTEIN_ELEMENTS))
    if heavy_mask.sum() == 0:
        return species, positions

    is_protein = heavy_mask.copy()
    ho_mask = np.isin(species, ["H", "O"])
    ho_idx = np.where(ho_mask)[0]
    ho_pos = positions[ho_idx]

    for _ in range(3):
        prot_pos = positions[is_protein]
        tree = cKDTree(prot_pos)
        dists, _ = tree.query(ho_pos)
        old_count = is_protein.sum()
        is_protein[ho_idx[dists < 1.5]] = True
        if is_protein.sum() == old_count:
            break

    return species[is_protein], positions[is_protein]


def fill_water(
    box_size: np.ndarray, protein_pos: np.ndarray, min_dist: float = 2.4, seed: int = 42
):
    """Place water molecules on grid at liquid density, excluding protein region."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    spacing = 3.103  # Å, gives ~0.0334 molecules/Å³

    nx = int(np.ceil(box_size[0] / spacing))
    ny = int(np.ceil(box_size[1] / spacing))
    nz = int(np.ceil(box_size[2] / spacing))

    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    o_pos = np.stack(
        [
            (ix.ravel() + 0.5) * spacing,
            (iy.ravel() + 0.5) * spacing,
            (iz.ravel() + 0.5) * spacing,
        ],
        axis=1,
    )
    o_pos += rng.uniform(-0.3, 0.3, size=o_pos.shape)

    if len(protein_pos) > 0:
        tree = cKDTree(protein_pos)
        dists, _ = tree.query(o_pos)
        o_pos = o_pos[dists > min_dist]

    n_mol = len(o_pos)
    bond_len = 0.9572
    half_angle = 104.52 / 2 * np.pi / 180

    phi = rng.uniform(0, 2 * np.pi, n_mol)
    cos_theta = rng.uniform(-1, 1, n_mol)
    sin_theta = np.sqrt(1 - cos_theta**2)
    psi = rng.uniform(0, 2 * np.pi, n_mol)

    h1_local = np.array(
        [bond_len * np.sin(half_angle), bond_len * np.cos(half_angle), 0.0]
    )
    h2_local = np.array(
        [-bond_len * np.sin(half_angle), bond_len * np.cos(half_angle), 0.0]
    )

    h1_all = np.zeros_like(o_pos)
    h2_all = np.zeros_like(o_pos)
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
        h1_all[i] = o_pos[i] + R @ h1_local
        h2_all[i] = o_pos[i] + R @ h2_local

    species = ["O"] * n_mol + ["H"] * n_mol + ["H"] * n_mol
    positions = np.vstack([o_pos, h1_all, h2_all])
    return np.array(species), positions, n_mol


def generate_solvated(vac_xyz: Path, out_xyz: Path, buffer: float = BUFFER):
    from ase import Atoms
    from ase.io import read, write

    print(f"  Reading {vac_xyz.name} ...")
    atoms = read(str(vac_xyz))
    prot_species, prot_pos = extract_protein(atoms)
    n_prot = len(prot_species)

    pmin, pmax = prot_pos.min(0), prot_pos.max(0)
    extent = pmax - pmin
    box_size = extent + 2 * buffer
    prot_shifted = prot_pos - pmin + buffer

    print(
        f"  Protein: {n_prot} atoms, "
        f"extent {extent[0]:.1f}×{extent[1]:.1f}×{extent[2]:.1f} Å"
    )
    print(f"  Box: {box_size[0]:.1f}×{box_size[1]:.1f}×{box_size[2]:.1f} Å")
    print("  Filling with water ...")

    water_species, water_pos, n_mol = fill_water(box_size, prot_shifted)
    all_species = list(prot_species) + list(water_species)
    all_pos = np.vstack([prot_shifted, water_pos])
    n_total = len(all_species)

    prot_frac = 100 * n_prot / n_total
    density = n_total / np.prod(box_size)
    print(f"  Water: {n_mol} molecules ({n_mol * 3} atoms)")
    print(
        f"  Total: {n_total} atoms, protein {prot_frac:.1f}%, "
        f"density {density:.4f} atoms/Å³"
    )

    cell = np.diag(box_size)
    result = Atoms(symbols=all_species, positions=all_pos, cell=cell, pbc=True)
    out_xyz.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_xyz), result, format="extxyz")
    print(f"  Written: {out_xyz.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate solvated protein systems (protein + grid water)"
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="System keys (default: all). See --list.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--vac-dir", type=Path, default=VAC_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--buffer",
        type=float,
        default=BUFFER,
        help="Box buffer in Å around protein bounding box (default: 12)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available systems:")
        for k, (src, out, desc) in SYSTEMS.items():
            print(f"  {k:<8} {src:<30} → {out:<30} ({desc})")
        return

    try:
        import scipy  # noqa: F401
    except ImportError:
        sys.exit("Missing: pip install scipy")

    keys = args.systems or list(SYSTEMS.keys())
    for key in keys:
        if key not in SYSTEMS:
            print(f"Unknown system: {key}")
            continue
        src_name, out_name, desc = SYSTEMS[key]
        vac_path = args.vac_dir / src_name
        out_path = args.out_dir / out_name

        print(f"\n=== {key}: {desc} ===")

        if not vac_path.exists():
            print(f"  SKIP: vacuum XYZ not found: {vac_path}")
            print("  Run 02_convert_vacuum.py first.")
            continue

        if out_path.exists() and not args.overwrite:
            from ase.io import read

            n = len(read(str(out_path)))
            print(f"  SKIP (exists): {out_name} ({n} atoms)")
            continue

        generate_solvated(vac_path, out_path, buffer=args.buffer)


if __name__ == "__main__":
    main()
