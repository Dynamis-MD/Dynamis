"""Phase 5 v5: Distributed build_graph with graph-aware halo.

Aligns send/recv/interior semantics with `main` branch's
`build_halo_maps.cpp` while doing the work per-rank (each rank only
builds edges where `dst ∈ my_outer`). The send plan is derived locally
from the same edges via geometric symmetry — no NCCL exchange of plans.

Mode A (refresh, 1/K steps):
  1. PBC wrap.
  2. Cell-block partition: n_cells/W cells per rank (deterministic, no NCCL).
  3. count-only @ r on atoms whose cell ∈ my_cells (target=these atoms),
     source=full pos (no halo gather; pos is globally broadcast).
  4. all_reduce(degrees) → full_degrees[N].
  5. SFC weighted partition → pid_outer, my_outer_atoms.
  6. radius_graph_pbc_ex(pos, r+skin, target=my_outer) → edges_at_r_skin.
  7. Locally extract recv/send/interior from edges via geometric symmetry
     (Fix 1: sort by global atom ID; no NCCL exchange).
  8. Output edges_at_r (filter r+skin → r).

Mode B (steady, K-1/K steps):
  radius_graph_pbc_ex(pos[my_outer ∪ halo], r, target=my_outer).
  No NCCL.

See `docs/v4_scaling_results.md` §10 for the design discussion.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.distributed as dist

import distnnp_state.distnnp_state_impl as _impl


class BuildMode(Enum):
    REFRESH = "A"
    STEADY = "B"


@dataclass
class DistBuildGraphResultV5:
    """Per-step output. `halo_info` matches `main::build_halo_maps` semantics."""
    edges_outer: torch.Tensor             # [2, E_local] global ids, dst ∈ my_outer
    cell_shifts_outer: torch.Tensor       # [E_local, 3] float
    pid_outer: torch.Tensor               # [N] int32
    my_outer_atoms: torch.Tensor          # [num_local] int64 global ids
    halo_info: _impl.PartitionMessagePassingInfo
    interior_mask: torch.Tensor           # [num_local] bool
    full_degrees: torch.Tensor            # [N] int32 — r-degrees from last refresh
    halo_recv_global_ids: torch.Tensor    # [total_recv] int64, ordered by peer then atom-id
    mode: BuildMode


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _cell_block_partition(
    pos_w: torch.Tensor,
    L_vec: torch.Tensor,
    cell_size: float,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Step 2: assign cells to ranks block-wise; return my target atom indices.

    Each rank owns a contiguous block of cells: cell_id ∈ [r * n/W, (r+1) * n/W).
    Pure local computation, deterministic across ranks.

    Args:
      pos_w     — [N, 3] PBC-wrapped positions in [0, L)
      L_vec     — [3] box edge lengths
      cell_size — Å, e.g. r + skin
      rank      — my rank id
      world_size — total ranks

    Returns:
      my_target_atoms[num_in_my_cells] int64 global indices.
    """
    device = pos_w.device

    nx = max(1, int(torch.floor(L_vec[0] / cell_size).item()))
    ny = max(1, int(torch.floor(L_vec[1] / cell_size).item()))
    nz = max(1, int(torch.floor(L_vec[2] / cell_size).item()))
    n_cells = nx * ny * nz

    # Actual cell sizes so cell grid exactly covers the box
    cx_size = L_vec[0] / nx
    cy_size = L_vec[1] / ny
    cz_size = L_vec[2] / nz

    cx = torch.clamp((pos_w[:, 0] / cx_size).floor().to(torch.int64), 0, nx - 1)
    cy = torch.clamp((pos_w[:, 1] / cy_size).floor().to(torch.int64), 0, ny - 1)
    cz = torch.clamp((pos_w[:, 2] / cz_size).floor().to(torch.int64), 0, nz - 1)
    cell_id = cx + cy * nx + cz * (nx * ny)            # [N] int64

    # Block-assign cells to ranks. ceil-div so last rank may get fewer cells.
    my_cells_lo = (rank * n_cells) // world_size
    my_cells_hi = ((rank + 1) * n_cells) // world_size

    in_my_block = (cell_id >= my_cells_lo) & (cell_id < my_cells_hi)
    my_target_atoms = torch.nonzero(in_my_block, as_tuple=False).squeeze(-1)
    return my_target_atoms


def _all_reduce_degrees(
    degrees_local: torch.Tensor,
    my_target_atoms: torch.Tensor,
    N: int,
    group=None,
) -> torch.Tensor:
    """Scatter local r-degrees into [N] tensor, then NCCL all_reduce-sum."""
    device = degrees_local.device
    full = torch.zeros(N, dtype=torch.int32, device=device)
    if degrees_local.numel() > 0:
        full.scatter_(0, my_target_atoms.to(torch.int64), degrees_local)
    if dist.is_initialized():
        dist.all_reduce(full, op=dist.ReduceOp.SUM, group=group)
    return full


def _compute_sfc_weighted_partition(
    pos: torch.Tensor,
    full_degrees: torch.Tensor,
    env: _impl.DistEnvironment,
    curve_type: _impl.SFCCurveType = _impl.SFCCurveType.Morton,
) -> torch.Tensor:
    """SFC weighted partition. Deterministic given inputs; no NCCL."""
    N = pos.shape[0]
    device = pos.device
    mol = _impl.MolecularInfo()
    mol.num_atoms = N
    mol.atom_positions = pos.contiguous()
    mol.atom_vertex_numbers = torch.arange(N, dtype=torch.int64, device=device)
    mol.edge_counts = full_degrees.to(torch.int64).contiguous()
    partitioner = _impl.SFCPartitioner(curve_type)
    info = partitioner.partition(mol, env)
    return info.partition_id


def _extract_halo_info_local(
    edges: torch.Tensor,
    pid_outer: torch.Tensor,
    my_outer_atoms: torch.Tensor,
    my_rank: int,
    world_size: int,
    N: int,
    device,
) -> Tuple[_impl.PartitionMessagePassingInfo, torch.Tensor, torch.Tensor]:
    """Step 7: from edges (dst ∈ my_outer guaranteed), derive recv/send/interior
    purely locally using geometric symmetry — see docs §10.3.

    Returns (halo_info, interior_mask, halo_recv_global_ids).

    Conventions matching `main::build_halo_maps`:
      - send_local_indices : positions 0..num_local-1 into my_outer_atoms
      - recv_ghost_indices : arange(num_local, num_local + total_recv)
      - both CSRs ordered by peer rank; within each peer segment, atoms are
        sorted by global atom ID (Fix 1 — ensures sender/receiver alignment
        without NCCL exchange of plans).
    """
    # Heavy lifting (boundary detection, recv/send CSR via sort+unique, interior
    # mask) is done by the fused CUDA op `_impl.extract_halo_info`; Python only
    # assembles the PartitionMessagePassingInfo struct around the returned
    # tensors. Bit-exact with the previous ATen implementation.
    (send_offsets, send_local_indices, recv_offsets, recv_ghost_indices,
     interior_mask, recv_atom) = _impl.extract_halo_info(
        edges, pid_outer, my_outer_atoms, my_rank, world_size, N,
    )

    info = _impl.PartitionMessagePassingInfo()
    info.partition_id = pid_outer
    info.partition_indices = my_outer_atoms
    info.global_send_recv_matrix = torch.zeros(
        world_size, world_size, dtype=torch.int64, device=device,
    )
    info.send_offsets = send_offsets
    info.send_local_indices = send_local_indices
    info.recv_offsets = recv_offsets
    info.recv_ghost_indices = recv_ghost_indices
    info.interior_mask = interior_mask

    return info, interior_mask, recv_atom


def _filter_edges_to_radius(
    edges: torch.Tensor,
    cell_shifts: torch.Tensor,
    pos: torch.Tensor,
    cell: torch.Tensor,
    r: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Step 8: from edges at r+skin, keep only those with true distance < r.

    pos is PBC-wrapped; cell_shifts[i] is the integer shift applied to src to
    produce the PBC image used for that edge. Distance = ||(pos[src] + shift @ cell) - pos[dst]||.

    Delegates to the fused CUDA op `_impl.filter_edges_to_radius` (one distance
    kernel + compaction), replacing the ATen recompute that dominated step 8 at
    low world size.
    """
    if edges.numel() == 0:
        return edges, cell_shifts
    return _impl.filter_edges_to_radius(edges, cell_shifts, pos, cell, float(r))


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────


class DistGraphBuilderV5:
    """Distributed graph builder with graph-aware halo (v5).

    Output `halo_info` matches `main::build_halo_maps`'s
    `PartitionMessagePassingInfo` semantics — see docs §10 + §10.8.

    Args:
      cell, pbc            — simulation cell, PBC flags.
      cutoff, ghost_buffer — graph radius (r) and Verlet skin.
      env                  — DistEnvironment.
      refresh_interval     — K. Mode A every K calls (default 100).
      max_num_neighbors    — passed to radius_graph_pbc_ex (0 = no truncation).
      enforce_max_strictly — passed to radius_graph_pbc_ex.
      drift_threshold      — if set, force early Mode A when max PBC-min-image
                             drift exceeds this value (Å). Default skin/2.
      process_group        — NCCL group (default WORLD).
      sfc_curve_type       — Morton (default) or Hilbert.
      partition_cell_size  — cell size for the step-2 block partition.
                             Must be ≥ r+skin so cells map cleanly to halo
                             neighborhoods (default = r + skin).
    """

    def __init__(
        self,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float,
        ghost_buffer: float,
        env: _impl.DistEnvironment,
        refresh_interval: int = 100,
        max_num_neighbors: int = 0,
        enforce_max_strictly: bool = False,
        drift_threshold: Optional[float] = None,
        process_group=None,
        sfc_curve_type: _impl.SFCCurveType = _impl.SFCCurveType.Morton,
        partition_cell_size: Optional[float] = None,
    ):
        self.cell = cell
        self.pbc = pbc
        self.cutoff = float(cutoff)
        self.ghost_buffer = float(ghost_buffer)
        self.r_plus_skin = self.cutoff + self.ghost_buffer
        self.env = env
        self.refresh_interval = int(refresh_interval)
        self.max_num_neighbors = int(max_num_neighbors)
        self.enforce_max_strictly = bool(enforce_max_strictly)
        self.drift_threshold = (
            float(drift_threshold) if drift_threshold is not None
            else self.ghost_buffer / 2.0
        )
        self.process_group = process_group
        self.sfc_curve_type = sfc_curve_type
        self.partition_cell_size = (
            float(partition_cell_size) if partition_cell_size is not None
            else self.r_plus_skin
        )

        # cached state across K-1 Mode B steps
        self._pid_outer: Optional[torch.Tensor] = None
        self._my_outer_atoms: Optional[torch.Tensor] = None
        self._halo_recv_global_ids: Optional[torch.Tensor] = None
        self._halo_info: Optional[_impl.PartitionMessagePassingInfo] = None
        self._interior_mask: Optional[torch.Tensor] = None
        self._full_degrees: Optional[torch.Tensor] = None
        self._pos_w_at_refresh: Optional[torch.Tensor] = None
        self._steps_since_refresh = self.refresh_interval     # force first call → Mode A

        self._L_vec = cell.diag().contiguous()

    @property
    def next_is_refresh(self) -> bool:
        return self._steps_since_refresh >= self.refresh_interval

    def reset(self):
        self._pid_outer = None
        self._my_outer_atoms = None
        self._halo_recv_global_ids = None
        self._halo_info = None
        self._interior_mask = None
        self._full_degrees = None
        self._pos_w_at_refresh = None
        self._steps_since_refresh = self.refresh_interval

    def _pbc_wrap(self, pos: torch.Tensor) -> torch.Tensor:
        L = self._L_vec.to(pos.device)
        return pos - (pos / L.unsqueeze(0)).floor() * L.unsqueeze(0)

    def _live_drift_exceeded(self, pos_w: torch.Tensor) -> bool:
        if self._pos_w_at_refresh is None:
            return False
        L = self._L_vec.to(pos_w.device)
        delta = (pos_w - self._pos_w_at_refresh).abs()
        delta = torch.minimum(delta, L.unsqueeze(0) - delta)
        return float(delta.max().item()) > self.drift_threshold

    # ── Mode A ──────────────────────────────────────────────────────────────

    def _mode_a(self, pos: torch.Tensor) -> DistBuildGraphResultV5:
        env = self.env
        R = int(env.global_rank)
        W = int(env.world_size)
        device = pos.device
        N = pos.shape[0]
        pbc = self.pbc.to(device)
        cell = self.cell.to(device).to(pos.dtype)
        L_vec = self._L_vec.to(device)

        # 1. PBC wrap
        pos_w = self._pbc_wrap(pos)

        # 2. Cell-block partition
        my_target_atoms = _cell_block_partition(
            pos_w, L_vec, self.partition_cell_size, R, W,
        )

        # 3. Local r-degrees (count-only kernel; full pos as source)
        if my_target_atoms.numel() > 0:
            degrees_local = _impl.radius_graph_pbc_counts_ex(
                pos_w, cell, pbc, self.cutoff, my_target_atoms,
            )
        else:
            degrees_local = torch.empty(0, dtype=torch.int32, device=device)

        # 4. all_reduce(degrees)
        full_degrees = _all_reduce_degrees(
            degrees_local, my_target_atoms, N, group=self.process_group,
        )

        # 5. SFC weighted partition
        pid_outer = _compute_sfc_weighted_partition(
            pos_w, full_degrees, env, curve_type=self.sfc_curve_type,
        )
        my_outer_atoms = torch.nonzero(pid_outer.eq(R), as_tuple=False).squeeze(-1)

        # 6. Edge build at r+skin with target=my_outer (full pos as source)
        edges_at_r_skin, cell_shifts_skin, _, _ = _impl.radius_graph_pbc_ex(
            pos_w, cell, pbc, self.r_plus_skin,
            self.max_num_neighbors, self.enforce_max_strictly,
            my_outer_atoms,
        )

        # 7. Extract halo / send / recv / interior locally (no NCCL)
        halo_info, interior_mask, halo_recv_global_ids = _extract_halo_info_local(
            edges_at_r_skin, pid_outer, my_outer_atoms,
            R, W, N, device,
        )

        # 8. Filter edges r+skin → r as Mode A's output
        edges_outer, cell_shifts_outer = _filter_edges_to_radius(
            edges_at_r_skin, cell_shifts_skin, pos_w, cell, self.cutoff,
        )

        # Cache for K-1 Mode B steps
        self._pid_outer = pid_outer
        self._my_outer_atoms = my_outer_atoms
        self._halo_recv_global_ids = halo_recv_global_ids
        self._halo_info = halo_info
        self._interior_mask = interior_mask
        self._full_degrees = full_degrees
        self._pos_w_at_refresh = pos_w.detach().clone()
        self._steps_since_refresh = 0

        return DistBuildGraphResultV5(
            edges_outer=edges_outer,
            cell_shifts_outer=cell_shifts_outer,
            pid_outer=pid_outer,
            my_outer_atoms=my_outer_atoms,
            halo_info=halo_info,
            interior_mask=interior_mask,
            full_degrees=full_degrees,
            halo_recv_global_ids=halo_recv_global_ids,
            mode=BuildMode.REFRESH,
        )

    # ── Mode B ──────────────────────────────────────────────────────────────

    def _mode_b(self, pos: torch.Tensor) -> DistBuildGraphResultV5:
        env = self.env
        device = pos.device
        pbc = self.pbc.to(device)
        cell = self.cell.to(device).to(pos.dtype)

        pos_w = self._pbc_wrap(pos)

        # drift watchdog
        if self._live_drift_exceeded(pos_w):
            self._steps_since_refresh = self.refresh_interval
            return self._mode_a(pos)

        # Build edges at r on (my_outer ∪ halo_recv) with target=my_outer
        my_full_set = torch.cat([self._my_outer_atoms, self._halo_recv_global_ids])
        local_pos = pos_w.index_select(0, my_full_set).contiguous()
        target_local = torch.arange(
            self._my_outer_atoms.numel(),
            dtype=torch.int64, device=device,
        )
        edges_local, cell_shifts, _, _ = _impl.radius_graph_pbc_ex(
            local_pos, cell, pbc, self.cutoff,
            self.max_num_neighbors, self.enforce_max_strictly,
            target_local,
        )
        if edges_local.numel() == 0:
            edges_global = torch.empty(2, 0, dtype=torch.int64, device=device)
        else:
            src_global = my_full_set.index_select(0, edges_local[0])
            tgt_global = my_full_set.index_select(0, edges_local[1])
            edges_global = torch.stack([src_global, tgt_global], dim=0)

        self._steps_since_refresh += 1

        return DistBuildGraphResultV5(
            edges_outer=edges_global,
            cell_shifts_outer=cell_shifts,
            pid_outer=self._pid_outer,
            my_outer_atoms=self._my_outer_atoms,
            halo_info=self._halo_info,
            interior_mask=self._interior_mask,
            full_degrees=self._full_degrees,
            halo_recv_global_ids=self._halo_recv_global_ids,
            mode=BuildMode.STEADY,
        )

    # ── Public entry ────────────────────────────────────────────────────────

    def build(self, pos: torch.Tensor) -> DistBuildGraphResultV5:
        if self.next_is_refresh or self._halo_info is None:
            return self._mode_a(pos)
        return self._mode_b(pos)
