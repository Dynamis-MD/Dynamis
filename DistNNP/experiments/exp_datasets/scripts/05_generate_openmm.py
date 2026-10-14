#!/usr/bin/env python3
"""05_generate_openmm.py — Generate pre-equilibrated systems via OpenMM.

Produces three datasets:
  1. DHFR + explicit water (~23.6k atoms) — from openmmtools testsystems
  2. ApoA1 + POPC membrane + water (~92.2k atoms) — from OpenMM examples PDB
  3. DPPC bilayer + water (~28.2k atoms) — from OpenMM built-in data

All systems are pre-equilibrated by the OpenMM/openmmtools developers.

Reproduces:
    protein_solvated/dhfr_23k_openmm.xyz
    membrane/apoa1_92k_openmm.xyz
    membrane/dppc_membrane_28k.xyz

Usage:
    python scripts/05_generate_openmm.py [--systems dhfr apoa1 dppc]
    python scripts/05_generate_openmm.py --list

Requires:
    conda install -c conda-forge openmm openmmtools
    pip install ase
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOL_DIR = SCRIPT_DIR / ".." / "protein_solvated"
MEM_DIR = SCRIPT_DIR / ".." / "membrane"
PDB_DIR = SCRIPT_DIR / ".." / "pdb"


def check_openmm():
    try:
        import openmm  # noqa: F401
        from openmm import unit  # noqa: F401
        from openmm.app import PDBFile  # noqa: F401
    except ImportError:
        sys.exit(
            "Missing: conda install -c conda-forge openmm\n" "         pip install ase"
        )


def check_openmmtools():
    try:
        import openmmtools  # noqa: F401
    except ImportError:
        sys.exit("Missing openmmtools: conda install -c conda-forge openmmtools")


# ── shared helper ─────────────────────────────────────────────────────────


def topology_to_ase(topology, positions_nm, cell_vectors_nm=None):
    """Convert OpenMM Topology + positions (nm) to ASE Atoms."""
    import numpy as np
    from ase import Atoms

    nm_to_ang = 10.0
    pos = np.array([[v for v in p] for p in positions_nm]) * nm_to_ang
    elements = [a.element.symbol for a in topology.atoms()]

    if cell_vectors_nm is not None:
        cell = np.array([[v for v in row] for row in cell_vectors_nm]) * nm_to_ang
        atoms = Atoms(symbols=elements, positions=pos, cell=cell, pbc=True)
        atoms.wrap()
    else:
        atoms = Atoms(symbols=elements, positions=pos)

    return atoms


# ── DHFR ─────────────────────────────────────────────────────────────────


def generate_dhfr(out_path: Path):
    """DHFR + TIP3P water from openmmtools testsystems."""
    check_openmmtools()
    from openmmtools.testsystems import DHFRExplicit
    from openmm.unit import angstrom as ang_unit
    import numpy as np
    from ase import Atoms
    from ase.io import write

    print("  Loading openmmtools DHFRExplicit() ...")
    test = DHFRExplicit()

    # test.positions is an OpenMM Quantity[nm]; convert to Å numpy array
    pos = np.array([[v.value_in_unit(ang_unit) for v in p] for p in test.positions])
    elements = [a.element.symbol for a in test.topology.atoms()]

    box_vecs = test.topology.getPeriodicBoxVectors()
    cell = np.array([[v.value_in_unit(ang_unit) for v in row] for row in box_vecs])

    atoms = Atoms(symbols=elements, positions=pos, cell=cell, pbc=True)
    atoms.wrap()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), atoms, format="extxyz")
    print(f"  {len(atoms)} atoms → {out_path.name}")


# ── ApoA1 ────────────────────────────────────────────────────────────────


def generate_apoa1(pdb_path: Path, out_path: Path):
    """ApoA1 + POPC membrane + water from OpenMM example PDB."""
    from openmm.app import PDBFile
    from openmm.unit import angstrom
    import numpy as np
    from ase import Atoms
    from ase.io import write

    print(f"  Reading {pdb_path.name} ...")
    pdb = PDBFile(str(pdb_path))

    pos = np.array([[p.value_in_unit(angstrom) for p in v] for v in pdb.getPositions()])
    elements = [a.element.symbol for a in pdb.topology.atoms()]

    box_vectors = pdb.topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        cell = np.array(
            [[v.value_in_unit(angstrom) for v in row] for row in box_vectors]
        )
        atoms = Atoms(symbols=elements, positions=pos, cell=cell, pbc=True)
        atoms.wrap()
    else:
        atoms = Atoms(symbols=elements, positions=pos)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), atoms, format="extxyz")
    print(f"  {len(atoms)} atoms → {out_path.name}")


# ── DPPC membrane ─────────────────────────────────────────────────────────


def generate_dppc(out_path: Path):
    """DPPC bilayer + water from OpenMM built-in data files."""
    import openmm
    from openmm.app import PDBFile
    from openmm.unit import angstrom
    import numpy as np
    from ase import Atoms
    from ase.io import write

    # Find OpenMM data directory
    openmm_data = Path(openmm.__file__).parent / "app" / "data"
    dppc_pdb = openmm_data / "DPPC.pdb"

    if not dppc_pdb.exists():
        # Try alternate name
        candidates = list(openmm_data.glob("dppc*.pdb")) + list(
            openmm_data.glob("DPPC*.pdb")
        )
        if not candidates:
            sys.exit(
                f"DPPC.pdb not found in {openmm_data}. "
                "List available membranes: ls " + str(openmm_data) + "/*.pdb"
            )
        dppc_pdb = candidates[0]

    print(f"  Reading {dppc_pdb.name} from OpenMM data ...")
    pdb = PDBFile(str(dppc_pdb))

    pos = np.array([[p.value_in_unit(angstrom) for p in v] for v in pdb.getPositions()])
    elements = [a.element.symbol for a in pdb.topology.atoms()]

    box_vectors = pdb.topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        cell = np.array(
            [[v.value_in_unit(angstrom) for v in row] for row in box_vectors]
        )
        atoms = Atoms(symbols=elements, positions=pos, cell=cell, pbc=True)
        atoms.wrap()
    else:
        atoms = Atoms(symbols=elements, positions=pos)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), atoms, format="extxyz")
    print(f"  {len(atoms)} atoms → {out_path.name}")


# ── main ─────────────────────────────────────────────────────────────────

SYSTEMS = {
    "dhfr": ("DHFR + TIP3P water (openmmtools)", SOL_DIR / "dhfr_23k_openmm.xyz", None),
    "apoa1": (
        "ApoA1 + POPC + water (OpenMM example)",
        MEM_DIR / "apoa1_92k_openmm.xyz",
        PDB_DIR / "apoa1.pdb",
    ),
    "dppc": (
        "DPPC bilayer + water (OpenMM built-in)",
        MEM_DIR / "dppc_membrane_28k.xyz",
        None,
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-equilibrated systems via OpenMM"
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        choices=list(SYSTEMS.keys()),
        help="Systems to generate (default: all)",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sol-dir",
        type=Path,
        default=None,
        help="Override output dir for solvated systems (default: protein_solvated/)",
    )
    parser.add_argument(
        "--mem-dir",
        type=Path,
        default=None,
        help="Override output dir for membrane systems (default: membrane/)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Override directory for source PDB files (default: pdb/)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available systems:")
        for k, (desc, out, src) in SYSTEMS.items():
            src_str = src.name if src else "built-in"
            print(f"  {k:<8} {src_str:<20} → {out.name:<35} ({desc})")
        return

    check_openmm()

    keys = args.systems or list(SYSTEMS.keys())
    for key in keys:
        desc, default_out, default_src = SYSTEMS[key]

        # Apply directory overrides
        if key == "dhfr":
            out_path = (args.sol_dir or SOL_DIR) / default_out.name
            src_path = None
        elif key == "apoa1":
            out_path = (args.mem_dir or MEM_DIR) / default_out.name
            src_path = (args.pdb_dir or PDB_DIR) / default_src.name
        elif key == "dppc":
            out_path = (args.mem_dir or MEM_DIR) / default_out.name
            src_path = None

        print(f"\n=== {key}: {desc} ===")

        if out_path.exists() and not args.overwrite:
            from ase.io import read

            n = len(read(str(out_path)))
            print(f"  SKIP (exists): {out_path.name} ({n} atoms)")
            continue

        try:
            if key == "dhfr":
                check_openmmtools()
                generate_dhfr(out_path)
            elif key == "apoa1":
                if not src_path.exists():
                    print(
                        f"  SKIP: {src_path.name} not found. Run 01_download_pdbs.sh first."
                    )
                    continue
                generate_apoa1(src_path, out_path)
            elif key == "dppc":
                generate_dppc(out_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
