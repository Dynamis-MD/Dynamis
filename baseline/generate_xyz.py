#!/usr/bin/env python3
"""
Generate an .xyz file following the CH4.xyz-like format.

This version builds molecules using RDKit conformers and places them in the box.

Header:
  1) <num_atoms>
  2) Lattice and metadata, e.g.
     Lattice="15.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0" Properties=species:S:1:pos:R:3:momenta:R:3 T_K=298 pbc="T T T"

Lines 3+: atoms as
  species  pos_x  pos_y  pos_z  momenta_x  momenta_y  momenta_z

Key behavior:
  - Use --num-c to specify the number of CH4 molecules (i.e., number of C atoms)
  - O2 molecules are automatically created with ratio CH4:O2 = 1:2
  - RDKit generates one 3D conformer for CH4 and one for O2, which are then copied,
    randomly rotated and translated onto a grid inside the simulation box
  - Momenta are sampled from a zero-mean Gaussian distribution

Usage examples:
  python generate_xyz.py --num-c 10 --out sample.xyz --box 15 30 30 --temperature 298 --seed 1
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple

import math

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception as rdkit_exc:  # pragma: no cover - provide friendly error
    Chem = None
    AllChem = None
    _rdkit_import_error = rdkit_exc
else:
    _rdkit_import_error = None


def parse_box(values: List[str]) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("--box requires exactly 3 numbers: a b c")
    try:
        a, b, c = (float(values[0]), float(values[1]), float(values[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--box values must be floats") from exc
    if a <= 0 or b <= 0 or c <= 0:
        raise argparse.ArgumentTypeError("--box side lengths must be positive")
    return a, b, c


def parse_species_list(spec: str) -> List[str]:
    items = [s.strip() for s in spec.split(',') if s.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--species-list must contain at least one symbol")
    # Basic sanity check: symbols are alphanumeric with optional case
    for s in items:
        if not s.isalpha():
            raise argparse.ArgumentTypeError(f"Invalid species symbol: {s}")
    return items


def parse_weights(spec: str, length: int) -> List[float]:
    raw = [w.strip() for w in spec.split(',') if w.strip()]
    if len(raw) != length:
        raise argparse.ArgumentTypeError(
            f"--species-weights must provide exactly {length} weights"
        )
    try:
        weights = [float(w) for w in raw]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--species-weights must be numeric") from exc
    if any(w < 0 for w in weights):
        raise argparse.ArgumentTypeError("--species-weights must be non-negative")
    if all(w == 0 for w in weights):
        raise argparse.ArgumentTypeError("At least one species weight must be > 0")
    return weights


def format_header(a: float, b: float, c: float, temperature: float, pbc: str) -> str:
    lattice = f"Lattice=\"{a:.1f} 0.0 0.0 0.0 {b:.1f} 0.0 0.0 0.0 {c:.1f}\""
    props = "Properties=species:S:1:pos:R:3:momenta:R:3"
    tval = f"T_K={temperature:g}"
    pbc_str = f"pbc=\"{pbc}\""
    return f"{lattice} {props} {tval} {pbc_str}"


def format_atom_line(symbol: str, x: float, y: float, z: float, px: float, py: float, pz: float) -> str:
    # Match typical fixed-width-ish formatting with 8 decimals
    return (
        f"{symbol:<2s}"
        f"{x:14.8f}{y:14.8f}{z:14.8f}"
        f"{px:14.8f}{py:14.8f}{pz:14.8f}"
    )


def random_rotation_matrix() -> List[List[float]]:
    """Generate a random 3D rotation matrix using a random unit quaternion.

    Returns a 3x3 matrix as list of rows.
    """
    u1 = random.random()
    u2 = random.random()
    u3 = random.random()
    q1 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    q2 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)

    x = q1
    y = q2
    z = q3
    w = q4

    # Rotation matrix from unit quaternion
    r11 = 1 - 2 * (y * y + z * z)
    r12 = 2 * (x * y - z * w)
    r13 = 2 * (x * z + y * w)
    r21 = 2 * (x * y + z * w)
    r22 = 1 - 2 * (x * x + z * z)
    r23 = 2 * (y * z - x * w)
    r31 = 2 * (x * z - y * w)
    r32 = 2 * (y * z + x * w)
    r33 = 1 - 2 * (x * x + y * y)
    return [
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33],
    ]


def matvec(R: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


def compute_centroid(coords: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    sx = sum(p[0] for p in coords)
    sy = sum(p[1] for p in coords)
    sz = sum(p[2] for p in coords)
    n = float(len(coords))
    return (sx / n, sy / n, sz / n)


def recentre(coords: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    cx, cy, cz = compute_centroid(coords)
    return [(x - cx, y - cy, z - cz) for (x, y, z) in coords]


def max_radius(coords: List[Tuple[float, float, float]]) -> float:
    return max(math.sqrt(x * x + y * y + z * z) for (x, y, z) in coords)


def build_conformer_ch4() -> Tuple[List[str], List[Tuple[float, float, float]]]:
    if Chem is None or AllChem is None:
        raise RuntimeError(
            f"RDKit is required but not available: {_rdkit_import_error}"
        )
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random.randint(1, 2**31 - 1)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("Failed to embed CH4 molecule with RDKit")
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z) for i in range(mol.GetNumAtoms())]
    coords = recentre(coords)
    return symbols, coords


def build_conformer_o2() -> Tuple[List[str], List[Tuple[float, float, float]]]:
    if Chem is None or AllChem is None:
        raise RuntimeError(
            f"RDKit is required but not available: {_rdkit_import_error}"
        )
    mol = Chem.MolFromSmiles("O=O")
    params = AllChem.ETKDGv3()
    params.randomSeed = random.randint(1, 2**31 - 1)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("Failed to embed O2 molecule with RDKit")
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z) for i in range(mol.GetNumAtoms())]
    coords = recentre(coords)
    return symbols, coords


def place_molecules_on_grid(num_slots: int, a: float, b: float, c: float, margin: float) -> List[Tuple[float, float, float]]:
    """Generate grid centers inside the box with at least a small margin to walls.

    Returns a list of translation vectors (x,y,z) of length num_slots.
    """
    # Choose grid counts roughly proportional to box dimensions
    k = math.ceil(num_slots ** (1.0 / 3.0))
    # Proportional counts by side length ratios
    ratios = [a, b, c]
    total = sum(ratios)
    nx = max(1, math.ceil(k * (a / total) * 3))
    ny = max(1, math.ceil(k * (b / total) * 3))
    nz = max(1, math.ceil(k * (c / total) * 3))
    # Ensure capacity
    while nx * ny * nz < num_slots:
        # Increment the dimension that increases capacity with least relative spacing degradation
        if a / nx <= b / ny and a / nx <= c / nz:
            nx += 1
        elif b / ny <= a / nx and b / ny <= c / nz:
            ny += 1
        else:
            nz += 1

    xs = [margin + (i + 0.5) * (max(a - 2 * margin, 0.0) / nx) for i in range(nx)]
    ys = [margin + (j + 0.5) * (max(b - 2 * margin, 0.0) / ny) for j in range(ny)]
    zs = [margin + (k_ + 0.5) * (max(c - 2 * margin, 0.0) / nz) for k_ in range(nz)]

    centers: List[Tuple[float, float, float]] = []
    for xi in xs:
        for yj in ys:
            for zk in zs:
                centers.append((xi, yj, zk))
    # Shuffle and take the first num_slots
    random.shuffle(centers)
    return centers[:num_slots]


def generate_mixture_ch4_o2(
    num_ch4: int,
    a: float,
    b: float,
    c: float,
    momenta_std: float,
    split: bool = False,
    cluster_ch4: float = 0.0,
) -> List[str]:
    """Generate atom lines for a CH4:O2 = 1:2 mixture given number of CH4 molecules.

    Args:
        cluster_ch4: If > 0, concentrate CH4 in a corner region of this fraction (0-1).
                     E.g., 0.2 means CH4 in 20% of box volume, O2 uniform in whole box.

    Returns formatted lines matching the output schema.
    """
    # Build base conformers
    ch4_symbols, ch4_coords = build_conformer_ch4()
    o2_symbols, o2_coords = build_conformer_o2()

    # Radii for packing (use max distance from centroid and add tiny gap)
    ch4_r = max_radius(ch4_coords) + 0.5
    o2_r = max_radius(o2_coords) + 0.5
    max_r = max(ch4_r, o2_r)

    num_o2 = 2 * num_ch4
    total_molecules = num_ch4 + num_o2

    lines: List[str] = []

    if not split:
        # Mixed random placement across the whole box
        centers = place_molecules_on_grid(total_molecules, a, b, c, margin=max_r)

        mol_kinds: List[str] = ["CH4"] * num_ch4 + ["O2"] * num_o2
        random.shuffle(mol_kinds)

        center_idx = 0
        for kind in mol_kinds:
            R = random_rotation_matrix()
            tx, ty, tz = centers[center_idx]
            center_idx += 1
            if kind == "CH4":
                for sym, base in zip(ch4_symbols, ch4_coords):
                    rx, ry, rz = matvec(R, base)
                    x = rx + tx
                    y = ry + ty
                    z = rz + tz
                    px = random.gauss(0.0, momenta_std)
                    py = random.gauss(0.0, momenta_std)
                    pz = random.gauss(0.0, momenta_std)
                    lines.append(format_atom_line(sym, x, y, z, px, py, pz))
            else:
                for sym, base in zip(o2_symbols, o2_coords):
                    rx, ry, rz = matvec(R, base)
                    x = rx + tx
                    y = ry + ty
                    z = rz + tz
                    px = random.gauss(0.0, momenta_std)
                    py = random.gauss(0.0, momenta_std)
                    pz = random.gauss(0.0, momenta_std)
                    lines.append(format_atom_line(sym, x, y, z, px, py, pz))
        return lines

    # Cluster mode: CH4 concentrated in small region, O2 uniform across full box
    if cluster_ch4 > 0:
        # CH4 goes in a corner region of size (cluster_frac * side) for each dimension
        cluster_frac = cluster_ch4
        ch4_a = a * cluster_frac
        ch4_b = b * cluster_frac  
        ch4_c = c * cluster_frac
        
        # Place CH4 in corner region [0, ch4_a] x [0, ch4_b] x [0, ch4_c]
        ch4_centers = place_molecules_on_grid(num_ch4, ch4_a, ch4_b, ch4_c, margin=max_r)
        # O2 uniform across full box
        o2_centers = place_molecules_on_grid(num_o2, a, b, c, margin=max_r)
        
        # Emit CH4 atoms
        for tx, ty, tz in ch4_centers:
            R = random_rotation_matrix()
            for sym, base in zip(ch4_symbols, ch4_coords):
                rx, ry, rz = matvec(R, base)
                x = rx + tx
                y = ry + ty
                z = rz + tz
                px = random.gauss(0.0, momenta_std)
                py = random.gauss(0.0, momenta_std)
                pz = random.gauss(0.0, momenta_std)
                lines.append(format_atom_line(sym, x, y, z, px, py, pz))
        
        # Emit O2 atoms
        for tx, ty, tz in o2_centers:
            R = random_rotation_matrix()
            for sym, base in zip(o2_symbols, o2_coords):
                rx, ry, rz = matvec(R, base)
                x = rx + tx
                y = ry + ty
                z = rz + tz
                px = random.gauss(0.0, momenta_std)
                py = random.gauss(0.0, momenta_std)
                pz = random.gauss(0.0, momenta_std)
                lines.append(format_atom_line(sym, x, y, z, px, py, pz))
        
        return lines

    # Split placement: CH4 and O2 in separated x regions.
    # Use coordinates centered at 0 so that O2 have x<0 and CH4 x>0.
    half_a = a / 2.0
    half_b = b / 2.0
    half_c = c / 2.0

    # Validate available widths on each half
    usable_half_x = half_a - 2.0 * max_r
    if usable_half_x <= 0:
        raise SystemExit(
            "Box too small in x for split placement. Increase --box A ... or reduce molecule count."
        )

    # Helper to create evenly spaced centers in a range [lo, hi]
    def grid_in_range(n: int, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> List[Tuple[float, float, float]]:
        lx = max(x1 - x0, 1e-9)
        ly = max(y1 - y0, 1e-9)
        lz = max(z1 - z0, 1e-9)
        k = math.ceil(n ** (1.0 / 3.0))
        # Allocate grid counts proportional to lengths
        total = lx + ly + lz
        nx = max(1, math.ceil(k * (lx / total) * 3))
        ny = max(1, math.ceil(k * (ly / total) * 3))
        nz = max(1, math.ceil(k * (lz / total) * 3))
        while nx * ny * nz < n:
            # Increase the dimension with largest available spacing
            sx = lx / nx
            sy = ly / ny
            sz = lz / nz
            if sx >= sy and sx >= sz:
                nx += 1
            elif sy >= sx and sy >= sz:
                ny += 1
            else:
                nz += 1
        xs = [x0 + (i + 0.5) * (lx / nx) for i in range(nx)]
        ys = [y0 + (j + 0.5) * (ly / ny) for j in range(ny)]
        zs = [z0 + (k_ + 0.5) * (lz / nz) for k_ in range(nz)]
        centers_: List[Tuple[float, float, float]] = []
        for xi in xs:
            for yj in ys:
                for zk in zs:
                    centers_.append((xi, yj, zk))
        random.shuffle(centers_)
        return centers_[:n]

    # Left (O2): x in [-half_a + max_r, -max_r]
    o2_centers = grid_in_range(
        num_o2,
        -half_a + max_r,
        -max_r,
        -half_b + max_r,
        half_b - max_r,
        -half_c + max_r,
        half_c - max_r,
    )
    # Right (CH4): x in [max_r, half_a - max_r]
    ch4_centers = grid_in_range(
        num_ch4,
        max_r,
        half_a - max_r,
        -half_b + max_r,
        half_b - max_r,
        -half_c + max_r,
        half_c - max_r,
    )

    # Emit lines: CH4 with x>0, O2 with x<0
    for tx, ty, tz in ch4_centers:
        R = random_rotation_matrix()
        for sym, base in zip(ch4_symbols, ch4_coords):
            rx, ry, rz = matvec(R, base)
            x = rx + tx
            y = ry + ty
            z = rz + tz
            px = random.gauss(0.0, momenta_std)
            py = random.gauss(0.0, momenta_std)
            pz = random.gauss(0.0, momenta_std)
            lines.append(format_atom_line(sym, x, y, z, px, py, pz))

    for tx, ty, tz in o2_centers:
        R = random_rotation_matrix()
        for sym, base in zip(o2_symbols, o2_coords):
            rx, ry, rz = matvec(R, base)
            x = rx + tx
            y = ry + ty
            z = rz + tz
            px = random.gauss(0.0, momenta_std)
            py = random.gauss(0.0, momenta_std)
            pz = random.gauss(0.0, momenta_std)
            lines.append(format_atom_line(sym, x, y, z, px, py, pz))

    return lines


def generate_atoms(
    num_atoms: int,
    a: float,
    b: float,
    c: float,
    momenta_std: float,
    species_mode: str,
    species_list: List[str],
    species_weights: List[float] | None,
) -> List[str]:
    if species_mode == "fixed":
        chosen_species = [species_list[0]] * num_atoms
    else:
        # Random choice according to weights (equal if not provided)
        weights = species_weights if species_weights is not None else [1.0] * len(species_list)
        chosen_species = random.choices(species_list, weights=weights, k=num_atoms)

    lines: List[str] = []
    for sym in chosen_species:
        x = random.uniform(0.0, a)
        y = random.uniform(0.0, b)
        z = random.uniform(0.0, c)
        px = random.gauss(0.0, momenta_std)
        py = random.gauss(0.0, momenta_std)
        pz = random.gauss(0.0, momenta_std)
        lines.append(format_atom_line(sym, x, y, z, px, py, pz))
    return lines


def write_xyz(
    output_path: str,
    num_atoms: int,
    a: float,
    b: float,
    c: float,
    temperature: float,
    pbc: str,
    atom_lines: List[str],
) -> None:
    header = format_header(a, b, c, temperature, pbc)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{num_atoms}\n")
        f.write(header + "\n")
        for line in atom_lines:
            f.write(line + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an .xyz file following the CH4.xyz-like format."
    )
    # New primary control: number of CH4 molecules (equals number of C atoms)
    parser.add_argument("--num-c", type=int, required=True, help="Number of CH4 molecules (i.e., number of C atoms)")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .xyz file path",
    )
    parser.add_argument(
        "--box",
        nargs=3,
        metavar=("A", "B", "C"),
        default=["15", "30", "30"],
        help="Cell side lengths (Lattice uses A 0 0  0 B 0  0 0 C)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="Temperature value to write in header (T_K)",
    )
    parser.add_argument(
        "--pbc",
        type=str,
        default="T T T",
        help="Periodic boundary conditions string (e.g., 'T T T' or 'F F F')",
    )
    parser.add_argument(
        "--momenta-std",
        type=float,
        default=0.5,
        help="Standard deviation for Gaussian-sampled momenta components",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Place CH4 and O2 on separated x halves: CH4 at x>0, O2 at x<0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--cluster-ch4",
        type=float,
        default=0.0,
        help="Concentrate CH4 in corner region of this fraction (0-1). E.g., 0.3 = 30%% of box. O2 remains uniform.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    a, b, c = parse_box(args.box)

    # Mixture assembly (CH4:O2 = 1:2)
    num_ch4 = int(args.num_c)
    if num_ch4 <= 0:
        raise SystemExit("--num-c must be a positive integer")
    num_o2 = 2 * num_ch4

    # Build atom lines via RDKit conformers and packing
    atom_lines = generate_mixture_ch4_o2(
        num_ch4=num_ch4,
        a=a,
        b=b,
        c=c,
        momenta_std=args.momenta_std,
        split=bool(args.split),
        cluster_ch4=args.cluster_ch4,
    )

    total_atoms = num_ch4 * 5 + num_o2 * 2
    if total_atoms != len(atom_lines):
        raise RuntimeError(
            f"Internal error: expected {total_atoms} atoms but generated {len(atom_lines)} lines"
        )

    write_xyz(
        output_path=args.out,
        num_atoms=total_atoms,
        a=a,
        b=b,
        c=c,
        temperature=args.temperature,
        pbc=args.pbc,
        atom_lines=atom_lines,
    )


if __name__ == "__main__":
    main()

