#!/usr/bin/env python3
"""07_retrieve_hiv.py — Fetch the HIV-1 capsid (PDB 3J3Q) and cut shell subsets.

The full capsid is a *deterministic* parse of the RCSB mmCIF (coordinates copied
verbatim, no pdbfixer/OpenMM), so it is byte-reproducible as long as RCSB does
not re-version the entry. Hydrogens are NOT added — the 3J3Q cryo-EM structure
is heavy-atom only, matching the production loader (setup_system.load_capsid).

Because the download is ~40 MB and the parse of ~2.4 M atoms takes ~1 min, the
outputs are also shipped in the hosted archive (download_datasets.sh) and cached
in the Delta shared directory — this script is the provenance / regeneration
path, not the default way to obtain the data.

Usage:
    python scripts/07_retrieve_hiv.py                 # download + write full capsid
    python scripts/07_retrieve_hiv.py --subsets       # also re-cut sub150 / sub350
    python scripts/07_retrieve_hiv.py --subsets-only   # cut subsets from an existing full xyz
    python scripts/07_retrieve_hiv.py --overwrite

Requires only the Python standard library (urllib, gzip).
"""

import argparse
import gzip
import urllib.request
from pathlib import Path
import numpy as np
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / ".." / "capsid"

PDB_ID = "3J3Q"
FULL_NAME = "3j3q_capsid.xyz"

# Nested shell-cube cut-outs sharing one lower corner. The corner sits on a
# DENSE part of the capsid shell (the capsid interior is hollow, so a cube at the
# geometric centre would be nearly empty); the two sizes are nested
# (sub150 ⊂ sub350). This exact corner + sizes reproduce the pinned files in
# scripts/checksums.txt (byte-for-byte, given a fixed full capsid).
SHELL_CORNER = (297.0, 216.0, 184.0)
SUBSETS = {
    "sub150": ("3j3q_capsid_sub150.xyz", 150.0),  # ~33.2 k atoms
    "sub350": ("3j3q_capsid_sub350.xyz", 350.0),  # ~175 k atoms
}


def download_and_convert(out_path: Path):
    """Download 3J3Q.cif.gz from RCSB and write heavy-atom XYZ.

    Coordinates are copied verbatim from the mmCIF (the raw column strings), so
    re-running yields byte-identical output for a given RCSB revision.
    """
    url = f"https://files.rcsb.org/download/{PDB_ID}.cif.gz"
    cif_path = out_path.parent / f"{PDB_ID}.cif.gz"

    print(f"  Downloading {PDB_ID} from {url} (~40 MB compressed) ...")
    urllib.request.urlretrieve(url, cif_path)
    print("  Parsing mmCIF (this may take a minute for ~2.4 M atoms) ...")

    atom_lines = []
    with gzip.open(cif_path, "rt") as f:
        in_atom_site = False
        columns = []
        sym_idx = x_idx = y_idx = z_idx = -1
        for line in f:
            line = line.strip()
            if line.startswith("loop_"):
                columns = []
                in_atom_site = False
            elif line.startswith("_atom_site."):
                in_atom_site = True
                columns.append(line)
            elif in_atom_site and (
                line.startswith("ATOM") or line.startswith("HETATM")
            ):
                if sym_idx == -1:
                    sym_idx = columns.index("_atom_site.type_symbol")
                    x_idx = columns.index("_atom_site.Cartn_x")
                    y_idx = columns.index("_atom_site.Cartn_y")
                    z_idx = columns.index("_atom_site.Cartn_z")
                parts = line.split()
                atom_lines.append(
                    f"{parts[sym_idx]} {parts[x_idx]} {parts[y_idx]} {parts[z_idx]}\n"
                )

    print(f"  Extracted {len(atom_lines)} atoms; writing {out_path.name} ...")
    with open(out_path, "w") as out_f:
        out_f.write(f"{len(atom_lines)}\n")
        out_f.write(f"HIV-1 capsid {PDB_ID}\n")
        out_f.writelines(atom_lines)
    print(f"  CIF downloaded to: {cif_path}")
    print(f"  Written: {out_path}")

    return cif_path


def protonate_and_solvate(cif_path: Path):
    import openmm
    from openmm import app, unit

    """Protonate and solvate the HIV-1 capsid subset."""
    ff = app.ForceField("charmm36.xml", "charmm36/water.xml")

    with gzip.open(cif_path, "rt") as f:
        cif = app.PDBxFile(f)

    # 3. Initialize the Modeller
    # The topology and positions abstract away the file format seamlessly
    modeller = app.Modeller(cif.topology, cif.positions)

    # 4. Add Hydrogens (pH 7.4)
    print("Adding hydrogens...")
    modeller.addHydrogens(ff, pH=7.4)

    # 5. Add Solvent (Water + Ions)
    print("Solvating system...")
    modeller.addSolvent(
        ff,
        model="tip3p",
        padding=1.5 * unit.nanometers,
        ionicStrength=0.15 * unit.molar,
    )

    # -------------------------------------------------------------------------
    # ENERGY MINIMIZATION (CLASSICAL PRE-RELAXATION)
    # -------------------------------------------------------------------------
    print("Building classical system for minimization...")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometers,
        constraints=app.HBonds,
    )

    print("Applying position restraints to protein heavy atoms...")
    k = 1000.0 * unit.kilojoules_per_mole / unit.nanometers**2
    restraint = openmm.CustomExternalForce("(k/2) * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter("k", k)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    solvent_resnames = ["HOH", "NA", "CL"]
    for atom in modeller.topology.atoms():
        if atom.residue.name not in solvent_resnames and atom.element.symbol != "H":
            idx = atom.index
            pos = modeller.positions[idx]
            restraint.addParticle(idx, pos)

    system.addForce(restraint)

    print("Setting up simulation context on CUDA...")
    platform = openmm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0"}
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)

    context = openmm.Context(system, integrator, platform, properties)
    context.setPositions(modeller.positions)

    print("Running energy minimization to resolve steric clashes...")
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=10000)

    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    atomic_numbers = np.array(
        [atom.element.atomic_number for atom in modeller.topology.atoms()],
        dtype=np.int32,
    )

    box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)

    print(f"Final minimized system size: {len(atomic_numbers):,} atoms")
    print(f"Box dimensions (Angstroms):\n{box_vectors}")

    return {
        "positions": positions,
        "atom_types": atomic_numbers,
        "box_vectors": box_vectors,
    }


def cut_subset(full_path: Path, out_path: Path, corner, side: float):
    """Write the atoms of the full capsid inside the axis-aligned cube
    [corner, corner + side) on every axis. Deterministic (a pure coordinate
    filter over the file order), so byte-reproducible from a fixed full xyz.
    """
    cx, cy, cz = corner
    kept = []
    with open(full_path) as f:
        f.readline()  # atom count
        f.readline()  # comment
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            x, y, z = float(p[1]), float(p[2]), float(p[3])
            if (
                (cx <= x < cx + side)
                and (cy <= y < cy + side)
                and (cz <= z < cz + side)
            ):
                kept.append(line if line.endswith("\n") else line + "\n")
    with open(out_path, "w") as out_f:
        out_f.write(f"{len(kept)}\n")
        out_f.write(f"{PDB_ID} subset cube L={side:.1f}\n")
        out_f.writelines(kept)
    print(f"  {out_path.name}: {len(kept)} atoms (cube L={side:.0f} at {corner})")


def main():
    ap = argparse.ArgumentParser(description="Retrieve HIV-1 capsid (3J3Q) + subsets")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--subsets",
        action="store_true",
        help="Also cut the sub150 / sub350 shell cubes",
    )
    ap.add_argument(
        "--subsets-only",
        action="store_true",
        help="Only cut subsets from an existing full xyz (no download)",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--solvate", action="store_true", help="Protonate and solvate the capsid subset"
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / FULL_NAME

    cif_path = None
    if not args.subsets_only:
        if full_path.exists() and not args.overwrite:
            print(f"SKIP (exists): {full_path.name} — use --overwrite to refetch")
            cif_path = out_dir / f"{PDB_ID}.cif.gz"
            if not cif_path.exists():
                cif_path = None
        else:
            cif_path = download_and_convert(full_path)

    if args.subsets or args.subsets_only:
        if not full_path.exists():
            sys.exit(f"ERROR: {full_path} missing — run without --subsets-only first.")
        for _, (name, side) in SUBSETS.items():
            out_path = out_dir / name
            if out_path.exists() and not args.overwrite:
                print(f"SKIP (exists): {name} — use --overwrite to recut")
                continue
            cut_subset(full_path, out_path, SHELL_CORNER, side)

    if args.solvate and cif_path is not None:
        print("Protonating and solvating the HIV-1 capsid subset...")
        result = protonate_and_solvate(cif_path)
        print("Protonation and solvation complete. Extracted data:")
        np.savez(
            out_dir / "solvated_capsid.npz",
            positions=result["positions"],
            atom_types=result["atom_types"].astype(np.uint8),
            box_vectors=result["box_vectors"],
        )
        print(f"Saved solvated capsid data to {out_dir / 'solvated_capsid.npz'}")


if __name__ == "__main__":
    main()
