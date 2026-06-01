import torch
from distnnp_state import MolecularInfo

ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'I': 53,
}

# Buffer (Å) added on each side of the bounding box to set the simulation cell.
BOX_BUFFER_ANGSTROM = 20.0


def load_capsid_from_xyz(xyz_path: str, device: torch.device) -> tuple:
    """
    Load a PDB-derived XYZ file (Element X Y Z per line, no velocities) produced
    by retrieve_data.py and return a MolecularInfo ready for DistNNP.

    Positions are shifted so the structure's minimum corner sits at
    BOX_BUFFER_ANGSTROM on each axis, placing all atoms inside a cubic box of
    side length (max_extent + 2*BOX_BUFFER_ANGSTROM). Velocities are zeroed.

    Args:
        xyz_path: Path to .xyz file produced by retrieve_data.py
        device:   Torch device to place all tensors on

    Returns:
        (MolecularInfo, box_side_length_angstrom)
    """
    print(f"Loading structure from {xyz_path} ...")

    with open(xyz_path, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    # lines[1] is the comment line; atom data starts at lines[2]

    elements = []
    raw_positions = []
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        elements.append(parts[0])
        raw_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    pos = torch.tensor(raw_positions, dtype=torch.float32)

    # Shift positions into [BOX_BUFFER, extent + BOX_BUFFER] on each axis and
    # compute the side length of the enclosing cubic cell.
    pos_min = pos.min(dim=0).values
    extents = pos.max(dim=0).values - pos_min
    box_size = float(extents.max()) + 2.0 * BOX_BUFFER_ANGSTROM
    pos = pos - pos_min + BOX_BUFFER_ANGSTROM

    atom_types = torch.tensor(
        [ELEMENT_TO_Z.get(el, 6) for el in elements], dtype=torch.long
    )

    mol_info = MolecularInfo()
    mol_info.num_atoms = num_atoms
    mol_info.atom_vertex_numbers = torch.arange(num_atoms, dtype=torch.long, device=device)
    mol_info.atom_positions = pos.to(device)
    mol_info.atom_velocities = torch.zeros(num_atoms, 3, dtype=torch.float32, device=device)
    mol_info.atom_types = atom_types.to(device)
    mol_info.atom_forces = torch.zeros(num_atoms, 3, dtype=torch.float32, device=device)

    print(f"Loaded {num_atoms:,} atoms | bounding box: {box_size:.1f} Å")
    return mol_info, box_size
