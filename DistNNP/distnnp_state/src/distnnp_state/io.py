import os
import tempfile

import torch

from distnnp_state import MolecularInfo, TrajectoryWriter


def is_pathlike(file_path) -> bool:
    return isinstance(file_path, (str, os.PathLike))


class MolecularInfoWriter:
    @staticmethod
    def write_struct(info: MolecularInfo, file_path: str, opts: list[str] | None = None):
        """Writes a MolecularInfo struct to a .distnnp file in a single call.

        opts: which fields to write, from {"indices", "positions", "velocities"}.
        None or an empty list writes everything; otherwise only the listed fields
        are written.
        """
        directory, filename = os.path.split(file_path)
        with TrajectoryWriter(0, bytes(16), "MolInfoWriter", directory or ".", filename) as writer:
            writer.initialize(info, 1.0, 1, opts or [])
            # Single standalone writer, so all atoms belong to rank 0.
            writer.update_atom_counts(torch.tensor([info.num_atoms], dtype=torch.uint32))
            writer.submit(0, info)
            writer.flush()


class MolecularInfoReader:
    @staticmethod
    def read_struct(file_path: str) -> MolecularInfo:
        """Reads a MolecularInfo struct from a .xyz or .distnnp file in a single call.

        Missing indices are assumed sequential and missing velocities default to
        zero; see TrajectoryWriter for the full self-describing read behavior.
        """
        assert is_pathlike(file_path) and os.path.exists(file_path), f"{file_path} could not be found"
        with tempfile.TemporaryDirectory() as scratch_dir:
            # TrajectoryWriter always opens/truncates an output file on
            # construction, so reading goes through a throwaway scratch file
            # rather than the path being read.
            with TrajectoryWriter(0, bytes(16), "MolInfoReader", scratch_dir, "_scratch.distnnp") as writer:
                # make paths a string
                return writer.initialize(os.fspath(file_path), 1.0, 1)
