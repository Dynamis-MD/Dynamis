from distnnp_state import MolecularInfo
import os.path as osp
import torch


class XYZParser:
    @staticmethod
    def parse(file_path: str) -> MolecularInfo:
        """
        Parses an XYZ file and returns a MolecularInfo object containing the atom indices, positions, velocities, and edge counts.

        Args:
            file_path (str): Path to the XYZ file.
        Returns:
            MolecularInfo: An object containing the parsed molecular information.
        """

        # TODO: THis is a placeholder and will be replaced with more performance-optimized C++ code that directly fills the MolecularInfo struct.

        assert osp.isfile(file_path), f"File not found: {file_path}"
        assert file_path.endswith(".xyz"), f"Expected an XYZ file, got: {file_path}"

        with open(file_path, "r") as f:
            lines = f.readlines()

        num_atoms = int(lines[0].strip())
        assert (
            len(lines) >= num_atoms + 2
        ), f"File {file_path} does not contain enough lines for {num_atoms} atoms."

        atom_indices = torch.arange(num_atoms, dtype=torch.long)
        atom_positions = torch.zeros((num_atoms, 3), dtype=torch.float)
        atom_velocities = torch.zeros((num_atoms, 3), dtype=torch.float)
