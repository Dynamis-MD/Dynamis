import os
import numpy as np
from ovito.data import DataCollection
from ovito.io import FileReaderInterface, import_file
from typing import Callable, Any

class BinaryReader(FileReaderInterface):

    @staticmethod
    def detect(filename: str):
        try:
            # Check file extension to ensure it doesn't grab random files
            is_correct_ext = os.path.splitext(filename)[1].lower() == ".distnnp"
            return is_correct_ext and os.path.getsize(filename) >= 244
        except OSError:
            return False

    def scan(self, filename: str, register_frame: Callable[..., None]):
        with open(filename, "rb") as f:
            # read in front matter info
            num_atoms = np.fromfile(f, dtype=np.uint64, count=1, offset=92)[0]
            # file position is now at 100
            bytes_per_atom = np.fromfile(f, dtype=np.uint32, count=1, offset=8)[0]
            # file position is now at 112
            front_matter_size = np.fromfile(f, dtype=np.uint64, count=1, offset=100)[0]
            # file position is now at 220
            
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

        # Create OVITO standard particle properties
        particles = data.create_particles(count=num_atoms)
        positions_prop = particles.create_property("Position")
        velocities_prop = particles.create_property("Velocity")
        indices_prop = particles.create_property("Particle Identifier")
        types_prop = particles.create_property("Particle Type")

        with open(filename, "rb") as f:
            # 1. Read Atom Types (Global list, located right after front matter)
            f.seek(224) # atom types always start at 224
            atom_types = np.fromfile(f, dtype=np.uint8, count=num_atoms)
            types_prop[...] = atom_types

            # 2. Jump straight to this specific frame's chunk
            f.seek(frame_offset)
            
            # 3. Read the frame data sequentially
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
            
            # Since the frame data is now sorted sequentially (0 to N-1), 
            # it perfectly matches the global type array.
            types_prop[...] = atom_types
