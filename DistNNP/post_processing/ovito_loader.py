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

class BinaryReader(FileReaderInterface):

    @staticmethod
    def detect(filename: str):
        try:
            # Check file extension to ensure it doesn't grab random files
            is_correct_ext = os.path.splitext(filename)[1].lower() == ".distnnp"
            # 264 bytes minimum front matter (id/position/velocity/pbc_cell
            # field descriptors) + a 40 byte pbc_cell block for one frame
            return is_correct_ext and os.path.getsize(filename) >= 304
        except OSError:
            return False

    def scan(self, filename: str, register_frame: Callable[..., None]):
        with open(filename, "rb") as f:
            # read in front matter info
            num_atoms = np.fromfile(f, dtype=np.uint64, count=1, offset=92)[0]
            # file position is now at 100
            bytes_per_atom = np.fromfile(f, dtype=np.uint32, count=1, offset=8)[0]
            # file position is now at 112
            # front_matter_size is at absolute offset 244 now that the field
            # descriptor list has 4 entries (pbc_cell, id, position, velocity)
            front_matter_size = np.fromfile(f, dtype=np.uint64, count=1, offset=132)[0]
            # file position is now at 252

            # every frame starts with a fixed 40 byte pbc_cell block (9
            # float32 cell + 3 uint8 pbc + 1 padding byte) before the
            # id/position/velocity per-atom data
            pbc_cell_block_size = 40
            bytes_per_frame = pbc_cell_block_size + (num_atoms * bytes_per_atom)

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

        # Create OVITO standard particle properties
        particles = data.create_particles(count=num_atoms)
        positions_prop = particles.create_property("Position")
        velocities_prop = particles.create_property("Velocity")
        indices_prop = particles.create_property("Particle Identifier")
        types_prop = particles.create_property("Particle Type")

        with open(filename, "rb") as f:
            # 1. Read Atom Types (Global list, located right after front matter)
            f.seek(256) # atom types always start at 256
            atom_types = np.fromfile(f, dtype=np.uint8, count=num_atoms)
            for atomic_num in np.unique(atom_types):
                name = ELEMENT_SYMBOLS.get(int(atomic_num), str(atomic_num))
                types_prop.types.append(ParticleType(id=int(atomic_num), name=name))
            types_prop[...] = atom_types

            # 2. Jump straight to this specific frame's chunk
            f.seek(frame_offset)

            # 3. Read the pbc_cell block (9 float32 cell + 3 uint8 pbc + 1
            # padding byte = 40 bytes), then the rest of the frame data
            frame_cell = np.fromfile(f, dtype=np.float32, count=9).reshape(3, 3)
            frame_pbc = np.fromfile(f, dtype=np.uint8, count=3)
            f.seek(frame_offset + 40)

            frame_indices = np.fromfile(f, dtype=np.uint64, count=num_atoms)
            frame_positions = np.fromfile(f, dtype=np.float32, count=num_atoms * 3).reshape(num_atoms, 3)
            frame_velocities = np.fromfile(f, dtype=np.float32, count=num_atoms * 3).reshape(num_atoms, 3)

            # 4. Create a sorting map based on the particle indices
            # This finds the permutation needed to sort frame_indices into [0, 1, 2, ... num_atoms-1]
            sort_map = np.argsort(frame_indices)

            # 5. Assign sorted data to OVITO so everything aligns with atom_types
            indices_prop[...] = frame_indices[sort_map]
            positions_prop[...] = frame_positions[sort_map]
            velocities_prop[...] = frame_velocities[sort_map]

            # 6. Surface the simulation cell / periodicity for this frame.
            # frame_cell is row-major (each row is a lattice vector, the
            # ASE/extxyz convention); OVITO's cell matrix stores lattice
            # vectors as columns, hence the transpose.
            cell_matrix = np.zeros((3, 4), dtype=np.float32)
            cell_matrix[:, :3] = frame_cell.T
            data.create_cell(cell_matrix, tuple(bool(x) for x in frame_pbc))
            
