"""Thin Python wrapper around the CUDA radius_graph_pbc C++ extension.

Drop-in replacement for baseline's radius_graph_pbc_v2. Extracts pos, cell,
pbc, natoms from the data object and calls the C++ function.
"""

from __future__ import annotations

import torch
import dynamis_state.dynamis_state_impl as _impl


def radius_graph_pbc_cuda(
    data,
    radius: float,
    max_num_neighbors_threshold: int,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
):
    """Build a radius graph with periodic boundary conditions using CUDA cell-list.

    Args:
        data: Object with attributes pos [N, 3], cell [batch, 3, 3],
              natoms [batch], pbc [batch, 3].
        radius: Cutoff distance for neighbor search.
        max_num_neighbors_threshold: Maximum neighbors per atom (0 = no limit).
        enforce_max_neighbors_strictly: If True, hard cap; if False, allow
            degeneracy tolerance.
        pbc: Optional override for PBC flags [3] bool tensor.

    Returns:
        edge_index: [2, E] int64 tensor (source_idx, target_idx)
        cell_offsets: [E, 3] float32 tensor (unit cell offsets per edge)
        num_neighbors_image: [batch_size] int64 tensor (edges per image)
    """
    # Resolve PBC flags
    if pbc is None:
        data_pbc = torch.atleast_2d(data.pbc)
        all_true = torch.all(data_pbc, dim=0)
        all_false = torch.all(~data_pbc, dim=0)
        mixed = ~(all_true | all_false)
        if mixed.any():
            raise RuntimeError(
                "Different structures in the batch have different PBC "
                "configurations. This is not currently supported."
            )
        pbc_flags = all_true
    else:
        pbc_flags = pbc.to(torch.bool)

    # Currently supports single-image batches (batch_size=1)
    # For multi-image, call per-image and concatenate
    batch_size = len(data.natoms)

    if batch_size == 1:
        cell_matrix = data.cell[0]  # [3, 3]
        edge_index, cell_offsets, num_neighbors = _impl.radius_graph_pbc(
            data.pos,
            cell_matrix,
            pbc_flags,
            float(radius),
            int(max_num_neighbors_threshold),
            enforce_max_neighbors_strictly,
        )
        num_neighbors_image = num_neighbors  # [1]
        return edge_index, cell_offsets, num_neighbors_image
    else:
        # Multi-image: process each image separately and concatenate
        all_edge_index = []
        all_cell_offsets = []
        all_num_neighbors = []

        atom_offset = 0
        for i in range(batch_size):
            n = data.natoms[i].item()
            pos_i = data.pos[atom_offset : atom_offset + n]
            cell_i = data.cell[i]  # [3, 3]

            ei, co, nn = _impl.radius_graph_pbc(
                pos_i,
                cell_i,
                pbc_flags,
                float(radius),
                int(max_num_neighbors_threshold),
                enforce_max_neighbors_strictly,
            )

            # Offset indices by atom_offset
            if ei.size(1) > 0:
                ei = ei + atom_offset

            all_edge_index.append(ei)
            all_cell_offsets.append(co)
            all_num_neighbors.append(nn)

            atom_offset += n

        if all_edge_index:
            edge_index = torch.cat(all_edge_index, dim=1)
            cell_offsets = torch.cat(all_cell_offsets, dim=0)
            num_neighbors_image = torch.cat(all_num_neighbors, dim=0)
        else:
            device = data.pos.device
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            cell_offsets = torch.empty(0, 3, dtype=torch.float, device=device)
            num_neighbors_image = torch.zeros(batch_size, dtype=torch.long, device=device)

        return edge_index, cell_offsets, num_neighbors_image
