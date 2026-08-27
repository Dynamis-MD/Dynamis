#!/usr/bin/env python3
"""02_convert_vacuum.py — Convert PDB structures to vacuum XYZ.

Uses pdbfixer to repair missing residues/atoms and add hydrogens at pH 7.0,
then writes extended XYZ with a tight box (3 Å buffer on each side).

Reproduces the protein_vacuum/*.xyz datasets committed in this repository.

Usage:
    python scripts/02_convert_vacuum.py [--systems 1aki 2rik ...]
    python scripts/02_convert_vacuum.py --list

Requires:
    pip install pdbfixer openmm ase
    # or: conda install -c conda-forge pdbfixer openmm
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PDB_DIR = SCRIPT_DIR / ".." / "pdb"
OUT_DIR = SCRIPT_DIR / ".." / "protein_vacuum"


# ── system definitions ────────────────────────────────────────────────────
# Each entry: (pdb_source, output_name, buffer_angstrom, description)
# buffer: padding added to protein bounding box on each side
SYSTEMS = {
    # Each entry: (pdb_source, output_name, buffer_angstrom, description)
    # Use *_fixed.pdb when available (pre-repaired, reproduces committed datasets exactly).
    # Fall back to raw PDB if fixed version is absent — pdbfixer will repair on the fly.
    "1aki": ("1AKI_fixed.pdb|1AKI.pdb",  "1aki_vacuum_openmm.xyz",  3.0, "Lysozyme (PDB 1AKI)"),
    "2rik": ("2RIK_fixed.pdb|2RIK.pdb",  "2rik_vacuum_openmm.xyz",  3.0, "Titin I-band (PDB 2RIK)"),
    "1grl": ("1GRL_fixed.pdb|1GRL.pdb",  "groel_vacuum_openmm.xyz", 3.0, "GroEL chaperonin (PDB 1GRL)"),
    "2w49": ("2W49_fixed.pdb|2W49.pdb",  "2w49_vacuum_openmm.xyz",  3.0, "Filament (PDB 2W49, DistMLIP)"),
    "6p9x": ("6P9X_fixed.pdb|6P9X.pdb",  "6p9x_vacuum_openmm.xyz",  3.0, "GPCR (PDB 6P9X, DistMLIP)"),
    # 9GMO is available as mmCIF only; pdbfixer reads .cif directly
    "9gmo": ("9GMO_fixed.pdb|9GMO.cif",  "9gmo_vacuum.xyz",          3.0, "Ribosome complex (PDB 9GMO)"),
}


def check_deps():
    missing = []
    try:
        import pdbfixer  # noqa: F401
    except ImportError:
        missing.append("pdbfixer")
    try:
        from openmm import unit  # noqa: F401
    except ImportError:
        missing.append("openmm")
    try:
        import ase  # noqa: F401
    except ImportError:
        missing.append("ase")
    if missing:
        sys.exit(f"Missing dependencies: {', '.join(missing)}\n"
                 "Install with: conda install -c conda-forge pdbfixer openmm\n"
                 "             pip install ase")


def convert_pdb_to_vacuum_xyz(pdb_path: Path, out_path: Path, buffer: float = 3.0):
    """Fix PDB, add hydrogens, write vacuum XYZ with tight box.

    Steps:
      1. pdbfixer: find & add missing residues + atoms
      2. pdbfixer: addMissingHydrogens at pH 7.0
      3. Build box: protein_extent + 2 * buffer (Å) on each axis
      4. Write extended XYZ via ASE
    """
    from pdbfixer import PDBFixer
    from openmm.unit import angstrom
    from ase import Atoms
    from ase.io import write

    suffix = pdb_path.suffix.lower()
    print(f"  Loading {pdb_path.name} ...")

    if suffix == ".cif":
        # mmCIF / PDBx must be parsed via pdbxfile=; filename= is PDB-format only.
        with open(pdb_path) as cif_fh:
            fixer = PDBFixer(pdbxfile=cif_fh)
    else:
        fixer = PDBFixer(filename=str(pdb_path))

    print("  Finding missing residues and atoms ...")
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Remove crystal waters and ligands (HETATM) — vacuum protein only
    fixer.removeHeterogens(keepWater=False)

    print("  Adding hydrogens at pH 7.0 ...")
    fixer.addMissingHydrogens(7.0)

    pos = np.array([
        [p.value_in_unit(angstrom) for p in v]
        for v in fixer.positions
    ])
    elements = [a.element.symbol for a in fixer.topology.atoms()]
    n_atoms = len(elements)

    pmin, pmax = pos.min(0), pos.max(0)
    extent = pmax - pmin
    box = extent + 2 * buffer
    shifted = pos - pmin + buffer

    cell = np.diag(box)
    atoms = Atoms(symbols=elements, positions=shifted, cell=cell, pbc=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), atoms, format="extxyz")

    density = n_atoms / np.prod(box)
    print(f"  {n_atoms} atoms, box {box[0]:.1f}×{box[1]:.1f}×{box[2]:.1f} Å, "
          f"density {density:.5f} atoms/Å³")
    print(f"  Written: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Convert PDB → vacuum XYZ")
    parser.add_argument("--systems", nargs="*", default=None,
                        help="System keys to process (default: all). See --list.")
    parser.add_argument("--list", action="store_true",
                        help="Print available systems and exit")
    parser.add_argument("--pdb-dir", type=Path, default=PDB_DIR,
                        help="Directory containing PDB/CIF source files")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Output directory for vacuum XYZ files")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    if args.list:
        print("Available systems:")
        for key, (src_spec, out, buf, desc) in SYSTEMS.items():
            src_display = src_spec.split("|")[0]  # show preferred source
            print(f"  {key:<8} {src_display:<20} → {out:<35} ({desc})")
        return

    check_deps()

    keys = args.systems or list(SYSTEMS.keys())
    for key in keys:
        if key not in SYSTEMS:
            print(f"Unknown system: {key}. Use --list to see options.")
            continue

        src_spec, out_name, buffer, desc = SYSTEMS[key]
        out_path = args.out_dir / out_name

        # Resolve source: "preferred|fallback" — use first that exists
        pdb_path = None
        for candidate in src_spec.split("|"):
            p = args.pdb_dir / candidate.strip()
            if p.exists():
                pdb_path = p
                break

        print(f"\n=== {key}: {desc} ===")

        if pdb_path is None:
            print(f"  SKIP: no source found for: {src_spec}")
            print(f"  Run 01_download_pdbs.sh first.")
            continue
        print(f"  Source: {pdb_path.name}")

        if out_path.exists() and not args.overwrite:
            import ase.io
            existing = ase.io.read(str(out_path))
            print(f"  SKIP (exists): {out_name} ({len(existing)} atoms)")
            continue

        try:
            convert_pdb_to_vacuum_xyz(pdb_path, out_path, buffer=buffer)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
