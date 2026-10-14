import os
from typing import Optional

import torch

import dynamis_state.dynamis_state_impl as dynamis_state_impl
from dynamis_state.dynamis_state_impl import MolecularInfo


def is_pathlike(file_path) -> bool:
    return isinstance(file_path, (str, os.PathLike))


class TrajectoryReader:
    """Stateless reader for .xyz and .dynamis trajectory files."""

    @staticmethod
    def read(path, frame_index: int = -1):
        """Reads one frame from a .xyz or .dynamis file.

        Dispatches on extension. frame_index defaults to -1 (the last frame)
        and indexes real frames uniformly for both formats; an out-of-range
        index raises RuntimeError. Missing indices are assumed sequential and
        missing velocities default to zero.

        Returns (MolecularInfo, cell, pbc).
        """
        return dynamis_state_impl.read_molecular_info(os.fspath(path), frame_index)


class TrajectoryWriter:
    """Thin wrapper around TrajectoryWriterImpl: opens/truncates the output
    file and writes front matter on construction, then accepts submit() calls."""

    def __init__(
        self,
        global_rank: int,
        run_id: bytes,
        author: str,
        output_dir,
        output_file,
        molecular_info: MolecularInfo,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        time_per_step: float,
        steps_per_write: int,
        opts: Optional[list] = None,
    ):
        self._impl = dynamis_state_impl.TrajectoryWriterImpl(
            global_rank,
            run_id,
            author,
            os.fspath(output_dir),
            os.fspath(output_file),
            molecular_info,
            cell,
            pbc,
            time_per_step,
            steps_per_write,
            opts or [],
        )

    def submit(self, step, state, cell=None, pbc=None):
        self._impl.submit(step, state, cell, pbc)

    def flush(self):
        self._impl.flush()

    def update_atom_counts(self, counts):
        self._impl.update_atom_counts(counts)

    def get_cell(self):
        return self._impl.get_cell()

    def get_pbc(self):
        return self._impl.get_pbc()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()


def translate_dynamis_to_xyz(
    dynamis_path, xyz_path, opts: Optional[list] = None
) -> None:
    """Translates every frame of a .dynamis file into a multi-frame extended-xyz file."""
    dynamis_state_impl.translate_dynamis_to_xyz(
        os.fspath(dynamis_path), os.fspath(xyz_path), opts or []
    )
