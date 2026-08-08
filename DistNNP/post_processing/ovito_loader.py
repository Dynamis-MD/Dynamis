import os
import numpy as np
from ovito.data import DataCollection, ParticleType
from ovito.io import FileReaderInterface, import_file
from typing import Callable, Any

ELEMENT_SYMBOLS = {
    1: 'H',   2: 'He',  3: 'Li',  4: 'Be',  5: 'B',   6: 'C',   7: 'N',   8: 'O',
    9: 'F',  10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',  16: 'S',
    17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V',  24: 'Cr',
    25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge',
    33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',  40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
    49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs', 56: 'Ba',
    57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
    65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf',
    73: 'Ta', 74: 'W',  75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb',
}

# Byte layout of the .distnnp front matter. Everything up to field_count is
# fixed-size regardless of which fields a writer selected; field_count and
# field_descriptors then determine where front_matter_size (and everything
# after it) actually lives, so those offsets must be computed, not assumed.
MAGIC = b"DISTNNP\x00"
OFFSET_NUM_ATOMS = 92
OFFSET_BYTES_PER_ATOM = 108
OFFSET_FIELD_COUNT = 112
OFFSET_FIELD_DESCRIPTORS = 116
FIELD_DESCRIPTOR_SIZE = 32

# Per-atom field layout, keyed by the name written into field_descriptors.
# (dtype, number of components, bytes per atom) -- mirrors kFieldOrder in
# trajectory_writer.hpp.
FIELD_LAYOUT = {
    "id": (np.int64, 1, 8),
    "position": (np.float32, 3, 12),
    "velocity": (np.float32, 3, 12),
}


def _read_header(f):
    """Reads and validates the front matter shared by scan()/parse().

    Returns (num_atoms, bytes_per_atom, front_matter_size, field_names).
    """
    f.seek(0)
    if f.read(8) != MAGIC:
        raise ValueError("not a valid distnnp file (bad magic)")

    f.seek(OFFSET_NUM_ATOMS)
    num_atoms = int(np.fromfile(f, dtype=np.uint64, count=1)[0])

    f.seek(OFFSET_BYTES_PER_ATOM)
    bytes_per_atom = int(np.fromfile(f, dtype=np.uint32, count=1)[0])

    f.seek(OFFSET_FIELD_COUNT)
    field_count = int(np.fromfile(f, dtype=np.uint32, count=1)[0])

    field_names = []
    for i in range(field_count):
        f.seek(OFFSET_FIELD_DESCRIPTORS + i * FIELD_DESCRIPTOR_SIZE)
        raw = f.read(FIELD_DESCRIPTOR_SIZE)
        name = raw.split(b"\x00", 1)[0].decode("ascii")
        if name not in FIELD_LAYOUT:
            raise ValueError(f"unknown field descriptor '{name}' in distnnp file")
        field_names.append(name)

    if "position" not in field_names:
        raise ValueError("'position' field missing from distnnp file, cannot load")

    # front_matter_size immediately follows the field_descriptors block, whose
    # size depends on field_count, so this offset moves with the field set.
    f.seek(OFFSET_FIELD_DESCRIPTORS + field_count * FIELD_DESCRIPTOR_SIZE)
    front_matter_size = int(np.fromfile(f, dtype=np.uint64, count=1)[0])

    return num_atoms, bytes_per_atom, front_matter_size, field_names


class BinaryReader(FileReaderInterface):

    @staticmethod
    def detect(filename: str):
        try:
            is_correct_ext = os.path.splitext(filename)[1].lower() == ".distnnp"
            if not is_correct_ext or os.path.getsize(filename) < OFFSET_FIELD_DESCRIPTORS:
                return False
            with open(filename, "rb") as f:
                return f.read(8) == MAGIC
        except OSError:
            return False

    def scan(self, filename: str, register_frame: Callable[..., None]):
        with open(filename, "rb") as f:
            num_atoms, bytes_per_atom, front_matter_size, _ = _read_header(f)
            bytes_per_frame = num_atoms * bytes_per_atom

            # Loop through the file to register frames
            file_size = os.path.getsize(filename)
            current_offset = front_matter_size
            frame_index = 0

            while current_offset + bytes_per_frame <= file_size:
                label = f"Frame {frame_index}"
                # Pass the exact byte offset and atom count to the parse method
                register_frame(frame_info=(current_offset, num_atoms), label=label)
                current_offset += bytes_per_frame
                frame_index += 1

    def parse(self, data: DataCollection, filename: str, frame_info: tuple[int, int], **kwargs: Any):
        frame_offset, num_atoms = frame_info
        num_atoms = int(num_atoms)

        # Create OVITO standard particle properties
        particles = data.create_particles(count=num_atoms)
        positions_prop = particles.create_property("Position")
        velocities_prop = particles.create_property("Velocity")
        indices_prop = particles.create_property("Particle Identifier")
        types_prop = particles.create_property("Particle Type")

        with open(filename, "rb") as f:
            _, _, front_matter_size, field_names = _read_header(f)

            # 1. Read Atom Types (Global list, located right before front_matter_size
            # boundary, padded to an 8-byte multiple). front_matter_size varies with
            # how many fields the writer selected, so this offset is computed rather
            # than assumed to be fixed.
            atom_type_padding = 8 - (num_atoms % 8)
            atom_types_offset = front_matter_size - num_atoms - atom_type_padding
            f.seek(atom_types_offset)
            atom_types = np.fromfile(f, dtype=np.uint8, count=num_atoms)
            for atomic_num in np.unique(atom_types):
                name = ELEMENT_SYMBOLS.get(int(atomic_num), str(atomic_num))
                types_prop.types.append(ParticleType(id=int(atomic_num), name=name))
            types_prop[...] = atom_types

            # 2. Walk the field descriptors (in file order) to find each active
            # field's byte offset within this frame; fields absent from
            # field_names simply aren't in the map. Mirrors the cursor logic in
            # TrajectoryWriter::load_distnnp_file.
            field_offsets = {}
            cursor = frame_offset
            for name in field_names:
                _, _, size = FIELD_LAYOUT[name]
                field_offsets[name] = cursor
                cursor += num_atoms * size

            # 3. Indices default to sequential order if the field is absent.
            if "id" in field_offsets:
                f.seek(field_offsets["id"])
                frame_indices = np.fromfile(f, dtype=np.int64, count=num_atoms)
            else:
                frame_indices = np.arange(num_atoms, dtype=np.int64)

            # 4. Positions are mandatory (already validated by _read_header).
            f.seek(field_offsets["position"])
            frame_positions = np.fromfile(f, dtype=np.float32, count=num_atoms * 3).reshape(num_atoms, 3)

            # 5. Velocities default to zero if the field is absent.
            if "velocity" in field_offsets:
                f.seek(field_offsets["velocity"])
                frame_velocities = np.fromfile(f, dtype=np.float32, count=num_atoms * 3).reshape(num_atoms, 3)
            else:
                frame_velocities = np.zeros((num_atoms, 3), dtype=np.float32)

            # 6. Create a sorting map based on the particle indices
            # This finds the permutation needed to sort frame_indices into [0, 1, 2, ... num_atoms-1]
            sort_map = np.argsort(frame_indices)

            # 7. Assign sorted data to OVITO so everything aligns with atom_types
            indices_prop[...] = frame_indices[sort_map]
            positions_prop[...] = frame_positions[sort_map]
            velocities_prop[...] = frame_velocities[sort_map]
