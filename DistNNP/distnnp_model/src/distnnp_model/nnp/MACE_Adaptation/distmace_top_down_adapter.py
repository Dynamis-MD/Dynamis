"""Distributed MACE adapter with DistNNP-owned graph construction.

DistNNP supplies ownership and halo membership. ``radius_graph_pbc_cuda`` then
builds one graph over the exchanged ``[local | halo]`` coordinates, and only
edges whose destination is locally owned enter MACE. The distributed path does
not use ``AtomicData.from_config`` for neighbor generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from distnnp_model.comm.exchanger import GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_model.potential import DistNNPModel
from distnnp_state import DistNNPState, PartitionMessagePassingInfo, radius_graph_pbc_cuda

# Keep model, structural exchange, graph geometry, and recovered forces aligned
# with DistNNP's standard communication dtype.
COMPUTE_DTYPE = torch.float32

try:
    from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar
    from mace.modules import ScaleShiftMACE
    from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
    from mace.tools.scatter import scatter_sum

    _MACE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional cluster dependency
    mace_mp = mace_off = mace_omol = mace_polar = None
    ScaleShiftMACE = None  # type: ignore[assignment]
    get_atomic_virials_stresses = get_outputs = prepare_graph = scatter_sum = None
    _MACE_IMPORT_ERROR = exc


def _require_mace() -> None:
    if _MACE_IMPORT_ERROR is not None:
        raise ImportError("The DistMACE adapter requires the MACE package.") from _MACE_IMPORT_ERROR


def load_raw_mace_model(
    model_family: str = "mp",
    model: str = "medium-mpa-0",
    device: str = "cuda",
):
    """Load a public MACE foundation model without its ASE calculator wrapper."""
    _require_mace()
    loaders = {"mp": mace_mp, "off": mace_off, "omol": mace_omol, "polar": mace_polar}
    if model_family not in loaders:
        raise ValueError(f"Unsupported MACE model family: {model_family}")
    loaded_model = loaders[model_family](
        model=model,
        device=device,
        default_dtype="float32",
        return_raw_model=True,
    )
    return loaded_model.to(dtype=COMPUTE_DTYPE)


@dataclass
class StructuralHaloView:
    local_positions: torch.Tensor
    exchanged_positions: torch.Tensor
    exchanged_atomic_numbers: torch.Tensor
    local_global_indices: torch.Tensor
    halo_global_indices: torch.Tensor
    expanded_global_indices: torch.Tensor
    num_local: int
    num_total: int

    @property
    def num_halo(self) -> int:
        return self.num_total - self.num_local


@dataclass
class ExplicitOwnedGraph:
    edge_index: torch.Tensor
    unit_shifts: torch.Tensor
    shifts: torch.Tensor
    batch_dict: Dict[str, torch.Tensor]


def canonicalize_nonperiodic_geometry(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    padding: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place nonperiodic coordinates inside an orthorhombic padded cell.

    DistNNP's current CUDA radius graph wraps coordinates on every axis. A
    rigid translation into the positive cell prevents that implementation
    detail from changing nonperiodic distances before halo construction.
    """
    positions = positions.to(dtype=COMPUTE_DTYPE)
    cell = cell.reshape(3, 3).to(device=positions.device, dtype=COMPUTE_DTYPE)
    pbc = pbc.reshape(3).to(device=positions.device, dtype=torch.bool)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [num_atoms, 3].")
    if padding <= 0:
        raise ValueError("nonperiodic padding must be positive.")
    if bool(torch.all(pbc)):
        return positions, cell, torch.zeros(3, dtype=COMPUTE_DTYPE, device=positions.device)

    off_diagonal = cell - torch.diag(torch.diagonal(cell))
    if not bool(torch.allclose(off_diagonal, torch.zeros_like(off_diagonal))):
        raise ValueError(
            "Nonperiodic coordinate canonicalization currently requires an "
            "orthorhombic cell."
        )

    adjusted_positions = positions.clone()
    adjusted_cell = cell.clone()
    translation = torch.zeros(3, dtype=COMPUTE_DTYPE, device=positions.device)
    for axis in range(3):
        if bool(pbc[axis]):
            continue
        axis_min = positions[:, axis].min()
        axis_max = positions[:, axis].max()
        translation[axis] = float(padding) - axis_min
        adjusted_positions[:, axis] += translation[axis]
        required_length = (axis_max - axis_min) + 2.0 * float(padding)
        adjusted_cell[axis, axis] = torch.maximum(
            adjusted_cell[axis, axis], required_length
        )
    return adjusted_positions, adjusted_cell, translation


# Does it get used once when we first partition? Why does it refer to multiple exchanges?
def validate_partition_info(
    partition_info: PartitionMessagePassingInfo,
    world_size: int,
) -> Dict[str, Any]:
    """Validate the CSR metadata consumed by every DistNNP exchange."""
    send_offsets = partition_info.send_offsets.to(torch.long)
    recv_offsets = partition_info.recv_offsets.to(torch.long)
    if send_offsets.numel() != world_size + 1 or recv_offsets.numel() != world_size + 1:
        raise ValueError("Halo offsets must contain world_size + 1 entries.")
    if send_offsets[0].item() != 0 or recv_offsets[0].item() != 0:
        raise ValueError("Halo offsets must start at zero.")
    if not bool(torch.all(send_offsets[1:] >= send_offsets[:-1])):
        raise ValueError("send_offsets must be monotonic.")
    if not bool(torch.all(recv_offsets[1:] >= recv_offsets[:-1])):
        raise ValueError("recv_offsets must be monotonic.")
    total_send = int(send_offsets[-1].item())
    total_recv = int(recv_offsets[-1].item())
    if partition_info.send_local_indices.numel() != total_send:
        raise ValueError("send_local_indices does not match send_offsets.")
    if partition_info.recv_ghost_indices.numel() != total_recv:
        raise ValueError("recv_ghost_indices does not match recv_offsets.")
    return {
        "total_send": total_send,
        "total_recv": total_recv,
        "send_counts_by_peer": (send_offsets[1:] - send_offsets[:-1]).cpu().tolist(),
        "recv_counts_by_peer": (recv_offsets[1:] - recv_offsets[:-1]).cpu().tolist(),
    }


def _edge_records(
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
    global_indices: torch.Tensor,
) -> set[Tuple[int, int, int, int, int]]:
    if edge_index.numel() == 0:
        return set()
    src = global_indices.index_select(0, edge_index[0].long()).cpu().tolist()
    dst = global_indices.index_select(0, edge_index[1].long()).cpu().tolist()
    offsets = unit_shifts.round().long().cpu().tolist()
    return {
        (int(i), int(j), int(offset[0]), int(offset[1]), int(offset[2]))
        for i, j, offset in zip(src, dst, offsets)
    }


class _RadiusGraphInput:
    def __init__(self, positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor) -> None:
        self.pos = positions
        self.cell = cell.reshape(1, 3, 3)
        self.pbc = pbc.reshape(1, 3)
        self.natoms = torch.tensor([positions.shape[0]], dtype=torch.long, device=positions.device)


# This is where the featureexchanger is inserted into MACE forward
# How does it subclass though? did that logic get removed??
class DistScaleShiftMACERuntimeMixin:
    """Small forward override that inserts one feature exchange into MACE."""

    __distmace_explicit_graph__ = True

    def _configure_dist_runtime(self) -> None:
        self._distmace_context: Dict[str, Any] = {
            "state": None,
            "exchange_enabled": False,
            "observation": None,
        }
        self._distmace_feature_exchanger = GPUDirectExchanger()

    def _begin_dist_forward(
        self, state: DistNNPState, exchange_enabled: bool, observation: Dict[str, Any]
    ) -> None:
        self._distmace_context.update(
            state=state, exchange_enabled=exchange_enabled, observation=observation
        )

    def _end_dist_forward(self) -> None:
        self._distmace_context.update(state=None, exchange_enabled=False, observation=None)

    def _exchange_after_first_layer(
        self, node_feats: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        observation = self._distmace_context["observation"]
        if layer_index != 0 or not self._distmace_context["exchange_enabled"]:
            if observation is not None:
                observation.setdefault("layer_records", []).append(
                    {
                        "layer_index": layer_index,
                        "executed": False,
                        "reason": "only_layer0_is_exchanged" if layer_index else "no_structural_halo",
                    }
                )
            return node_feats
        state = self._distmace_context["state"]
        if state is None:
            raise RuntimeError("Layer-0 exchange reached without an active DistNNPState.")
        num_local = int(state.local_molecular_info.num_atoms)
        exchanged = FeaturePushToExchange.apply(
            # Only the features corresponding the local rank and not the ghost?
            node_feats[:num_local].contiguous(), state, self._distmace_feature_exchanger
        ).to(dtype=node_feats.dtype)
        if exchanged.shape[0] != node_feats.shape[0]:
            raise ValueError("Layer-0 feature exchange changed the structural row count.")
        if observation is not None:
            observation["feature_exchange_used_ipc"] = True
            observation.setdefault("layer_records", []).append(
                {
                    "layer_index": 0,
                    "executed": True,
                    "owned_shape": list(node_feats[:num_local].shape),
                    "exchanged_shape": list(exchanged.shape),
                }
            )
        return exchanged

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        _require_mace()
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        node_heads = ctx.node_heads.to(torch.int64)
        atom_range = ctx.num_atoms_arange.to(torch.int64)
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[atom_range, node_heads]
        e0 = scatter_sum(node_e0, data["batch"], dim=0, dim_size=ctx.num_graphs).to(ctx.vectors.dtype)
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = self.radial_embedding(
            ctx.lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        pair_node_energy = (
            self.pair_repulsion_fn(
                ctx.lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if hasattr(self, "pair_repulsion")
            else torch.zeros_like(node_e0)
        )
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []
        for layer_index, (interaction, product) in enumerate(zip(self.interactions, self.products)):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(layer_index == 0),
                lammps_class=ctx.interaction_kwargs.lammps_class,
                lammps_natoms=ctx.interaction_kwargs.lammps_natoms,
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"])
            #Exact point of exchange
            #Can modify it for multiple layers without any issue
            node_feats = self._exchange_after_first_layer(node_feats, layer_index)
            node_feats_list.append(node_feats)


        for readout_index, readout in enumerate(self.readouts):
            feature_index = -1 if len(self.readouts) == 1 else readout_index
            node_es_list.append(
                readout(node_feats_list[feature_index], node_heads)[atom_range, node_heads]
            )
        node_inter_es = self.scale_shift(torch.sum(torch.stack(node_es_list), dim=0), node_heads)
        interaction_energy = scatter_sum(
            node_inter_es, data["batch"], dim=-1, dim_size=ctx.num_graphs
        )
        total_energy = e0 + interaction_energy
        node_energy = node_e0.clone() + node_inter_es.clone()
        state = self._distmace_context["state"]
        owned_node_energy = owned_local_energy = None
        if state is not None:
            num_local = int(state.local_molecular_info.num_atoms)
            owned_node_energy = node_energy.reshape(-1)[:num_local]
            owned_local_energy = owned_node_energy.sum()

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=interaction_energy,
            positions=ctx.positions,
            displacement=ctx.displacement,
            vectors=ctx.vectors,
            cell=ctx.cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )
        atomic_virials = atomic_stresses = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=ctx.vectors,
                num_atoms=ctx.positions.shape[0],
                batch=data["batch"],
                cell=ctx.cell,
            )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": interaction_energy,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": ctx.displacement,
            "node_feats": torch.cat(node_feats_list, dim=-1),
            "owned_node_energy": owned_node_energy,
            "owned_local_energy": owned_local_energy,
        }


# The DistNNP model contract facing adapter class
class DistMACETopDownAdapter(DistNNPModel):
    """MACE model adapter using DistNNP partitions, halos, graphs, and exchange."""

    def __init__(
        self,
        model,
        atomic_number_map: Optional[Mapping[int, int]] = None,
        diagnostics_enabled: bool = True,
    ) -> None:
        super().__init__()
        _require_mace()
        if ScaleShiftMACE is None or not isinstance(model, ScaleShiftMACE):
            raise TypeError("DistMACETopDownAdapter currently supports ScaleShiftMACE.")
        self.model = model.to(dtype=COMPUTE_DTYPE).eval()
        self.atomic_number_map = dict(atomic_number_map or {})
        self.cutoff = float(model.r_max.detach().cpu().item())
        self.position_exchanger = GPUDirectExchanger()
        self.atom_type_exchanger = GPUDirectExchanger()
        self.exchange_enabled = False
        self.diagnostics_enabled = diagnostics_enabled
        self.partition_validation: Dict[str, Any] = {}
        self.last_observation: Dict[str, Any] = {}
        self._install_runtime_subclass()

    def set_diagnostics(self, enabled: bool) -> None:
        """Enable correctness records or the synchronization-free timed path."""
        self.diagnostics_enabled = bool(enabled)

    def _install_runtime_subclass(self) -> None:
        if not getattr(self.model.__class__, "__distmace_explicit_graph__", False):
            runtime_class = type(
                f"DistMACE{self.model.__class__.__name__}",
                (DistScaleShiftMACERuntimeMixin, self.model.__class__),
                {"__distmace_explicit_graph__": True},
            )
            self.model.__class__ = runtime_class
        self.model._configure_dist_runtime()

    # List all the buffers that we're using for this adaptation
    def initialize_buffers(
        self,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
        world_size: Optional[int] = None,
    ) -> None:
        if world_size is None:
            world_size = int(partition_info.global_send_recv_matrix.shape[0])
        self.partition_validation = validate_partition_info(partition_info, world_size)
        total_send = self.partition_validation["total_send"]
        total_recv = self.partition_validation["total_recv"]
        if total_send == 0 and total_recv == 0:
            self.exchange_enabled = False
            return
        if total_send == 0 or total_recv == 0:
            raise RuntimeError("Asymmetric halo traffic cannot form a local DistNNP view.")
        self.position_exchanger.setup_exchanger(3, partition_info, device)
        self.atom_type_exchanger.setup_exchanger(1, partition_info, device)
        self.exchange_enabled = True

    @staticmethod
    def _halo_global_indices(state: DistNNPState) -> torch.Tensor:
        info = state.local_partition_info
        edge_index = info.edge_index_halo
        device = state.local_molecular_info.atom_positions.device
        if edge_index is None or edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        src, dst = edge_index.to(torch.long)
        owner = info.partition_id.to(torch.long)
        rank = int(state.dist_env.global_rank)
        remote = src[owner.index_select(0, dst).eq(rank) & owner.index_select(0, src).ne(rank)]
        groups = []
        for peer in range(int(state.dist_env.world_size)):
            if peer == rank:
                continue
            peer_atoms = remote[owner.index_select(0, remote).eq(peer)]
            if peer_atoms.numel() > 0:
                groups.append(torch.unique(peer_atoms.sort()[0]))
        return torch.cat(groups) if groups else torch.empty(0, dtype=torch.long, device=device)

    def _map_atomic_numbers(self, atom_types: torch.Tensor) -> torch.Tensor:
        atom_types = atom_types.to(torch.long)
        if not self.atomic_number_map:
            return atom_types
        return torch.tensor(
            [self.atomic_number_map[int(value)] for value in atom_types.cpu()],
            dtype=torch.long,
            device=atom_types.device,
        )

    #key logic right here
    def _build_structural_view(self, state: DistNNPState) -> StructuralHaloView:
        info = state.local_partition_info
        if not self.partition_validation:
            self.partition_validation = validate_partition_info(
                info, int(state.dist_env.world_size)
            )
        local_positions = (
            state.local_molecular_info.atom_positions.detach()
            .to(dtype=COMPUTE_DTYPE)
            .clone()
            .requires_grad_(True)
        )
        local_types = state.local_molecular_info.atom_types.to(torch.long)
        if self.exchange_enabled:
            exchanged_positions = FeaturePushToExchange.apply(
                local_positions, state, self.position_exchanger
            )
            with torch.no_grad():
                exchanged_types = FeaturePushToExchange.apply(
                    local_types.float().unsqueeze(-1), state, self.atom_type_exchanger
                ).squeeze(-1).long()
        else:
            exchanged_positions, exchanged_types = local_positions, local_types
        exchanged_positions.retain_grad()
        local_global = info.partition_indices.long()
        expected_total = int(local_global.numel() + self.partition_validation["total_recv"])
        if exchanged_positions.shape[0] != expected_total:
            raise ValueError("Structural rows disagree with DistNNP recv metadata.")
        if self.diagnostics_enabled:
            halo_global = self._halo_global_indices(state)
            expanded_global = torch.cat((local_global, halo_global))
            if expanded_global.shape[0] != expected_total:
                raise ValueError("Halo indices disagree with DistNNP recv metadata.")
        else:
            halo_global = torch.empty(0, dtype=torch.long, device=local_global.device)
            expanded_global = torch.empty(0, dtype=torch.long, device=local_global.device)
        return StructuralHaloView(
            local_positions,
            exchanged_positions,
            self._map_atomic_numbers(exchanged_types),
            local_global,
            halo_global,
            expanded_global,
            int(local_global.numel()),
            expected_total,
        )

    # radius_graph_pbc_cuda wrapper
    def _radius_graph(
        self, positions: torch.Tensor, state: DistNNPState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = state.sim_config
        with torch.no_grad():
            edge_index, cell_offsets, _ = radius_graph_pbc_cuda(
                _RadiusGraphInput(positions, sim.cell, sim.pbc),
                radius=self.cutoff,
                max_num_neighbors_threshold=0,
                enforce_max_neighbors_strictly=False,
            )
        # DistNNP shifts the source image; MACE shifts source-to-destination vectors.
        return edge_index.long(), -cell_offsets

    def _build_owned_graph(
        self, view: StructuralHaloView, state: DistNNPState
    ) -> ExplicitOwnedGraph:
        edge_index, unit_shifts = self._radius_graph(view.exchanged_positions, state)
        owned_destination = edge_index[1] < view.num_local
        edge_index = edge_index[:, owned_destination].contiguous()
        unit_shifts = unit_shifts[owned_destination].to(dtype=COMPUTE_DTYPE)
        cell = state.sim_config.cell.reshape(3, 3).to(dtype=COMPUTE_DTYPE)
        shifts = torch.matmul(unit_shifts, cell)
        model_atomic_numbers = self.model.atomic_numbers.to(
            device=view.exchanged_atomic_numbers.device, dtype=torch.long
        )
        node_attrs = view.exchanged_atomic_numbers[:, None].eq(
            model_atomic_numbers[None, :]
        ).to(dtype=COMPUTE_DTYPE)
        if self.diagnostics_enabled and not bool(torch.all(node_attrs.sum(dim=1) == 1)):
            missing = view.exchanged_atomic_numbers[node_attrs.sum(dim=1) == 0].unique()
            raise ValueError(f"MACE model does not support atomic numbers {missing.tolist()}.")
        device = view.exchanged_positions.device
        heads = list(getattr(self.model, "heads", ["Default"]))
        head_index = heads.index("Default") if "Default" in heads else len(heads) - 1
        batch_dict = {
            "positions": view.exchanged_positions,
            "edge_index": edge_index,
            "unit_shifts": unit_shifts,
            "shifts": shifts,
            "cell": cell,
            "node_attrs": node_attrs,
            "batch": torch.zeros(view.num_total, dtype=torch.long, device=device),
            "ptr": torch.tensor([0, view.num_total], dtype=torch.long, device=device),
            "head": torch.tensor([head_index], dtype=torch.long, device=device),
        }
        batch_dict = {
            key: value.to(COMPUTE_DTYPE) if torch.is_floating_point(value) else value
            for key, value in batch_dict.items()
        }
        return ExplicitOwnedGraph(edge_index, unit_shifts, shifts, batch_dict)

    def _verify_graph_authority(
        self,
        view: StructuralHaloView,
        local_graph: ExplicitOwnedGraph,
        state: DistNNPState,
    ) -> Dict[str, Any]:
        global_positions = state.global_molecular_info.atom_positions
        global_edge_index, global_unit_shifts = self._radius_graph(global_positions, state)
        owner = state.local_partition_info.partition_id.long()
        rank = int(state.dist_env.global_rank)
        mask = owner.index_select(0, global_edge_index[1]).eq(rank)
        global_ids = torch.arange(global_positions.shape[0], device=global_positions.device)
        expected = _edge_records(global_edge_index[:, mask], global_unit_shifts[mask], global_ids)
        actual = _edge_records(
            local_graph.edge_index, local_graph.unit_shifts, view.expanded_global_indices
        )
        owned = set(view.local_global_indices.cpu().tolist())
        required_remote = sorted(
            {src for src, dst, *_ in expected if dst in owned and src not in owned}
        )
        actual_halo = set(view.halo_global_indices.cpu().tolist())
        missing_halo = sorted(set(required_remote) - actual_halo)
        missing_edges = sorted(expected - actual)
        extra_edges = sorted(actual - expected)
        halo_issue = bool(missing_halo)
        reconstruction_issue = not halo_issue and bool(missing_edges or extra_edges)
        return {
            "stage": "pre_forward_graph_authority",
            "radius_graph_authority": "distnnp_state.radius_graph_pbc_cuda",
            "required_remote_global_indices": required_remote,
            "actual_halo_global_indices": sorted(actual_halo),
            "missing_required_halo_global_indices": missing_halo,
            "halo_builder_issue": halo_issue,
            "local_graph_matches_global_owned_graph": not (missing_edges or extra_edges),
            "local_graph_reconstruction_issue": reconstruction_issue,
            "expected_owned_edge_count": len(expected),
            "actual_owned_edge_count": len(actual),
            "missing_owned_edge_preview": [list(item) for item in missing_edges[:12]],
            "extra_owned_edge_preview": [list(item) for item in extra_edges[:12]],
            "structural_classification": (
                "halo_builder_issue"
                if halo_issue
                else "local_graph_reconstruction_issue"
                if reconstruction_issue
                else "structural_graph_ok"
            ),
        }

    @staticmethod
    def _recover_forces(
        owned_energy: torch.Tensor,
        view: StructuralHaloView,
        diagnostics_enabled: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        local_grad, exchanged_grad = torch.autograd.grad(
            owned_energy, [view.local_positions, view.exchanged_positions], allow_unused=False
        )
        forces = -local_grad
        checks = {}
        if diagnostics_enabled:
            checks = {
                "local_grad_is_finite": bool(torch.isfinite(local_grad).all()),
                "exchanged_grad_is_finite": bool(torch.isfinite(exchanged_grad).all()),
                "recovered_force_shape": list(forces.shape),
            }
        return forces, checks

    def __call__(self, state: DistNNPState, *args, **kwargs):
        view = self._build_structural_view(state)
        graph = self._build_owned_graph(view, state)
        observation = None
        if self.diagnostics_enabled:
            verification = self._verify_graph_authority(view, graph, state)
            observation = {
                "rank": int(state.dist_env.global_rank),
                "feature_exchange_used_ipc": False,
                "layer_records": [],
                "graph_verification": verification,
            }
        self.last_observation = observation or {}
        self.model._begin_dist_forward(state, self.exchange_enabled, observation)
        try:
            outputs = self.model(
                graph.batch_dict,
                training=False,
                compute_force=False,
                compute_stress=False,
                compute_hessian=False,
            )
            owned_energy = outputs.get("owned_local_energy")
            if owned_energy is None:
                raise KeyError("Distributed MACE forward did not return owned_local_energy.")
            owned_forces, force_checks = self._recover_forces(
                owned_energy, view, self.diagnostics_enabled
            )
        finally:
            self.model._end_dist_forward()
        if self.diagnostics_enabled:
            global_edges = _edge_records(
                graph.edge_index, graph.unit_shifts, view.expanded_global_indices
            )
            self.last_observation.update({
                "n_local": view.num_local,
                "n_halo": view.num_halo,
                "n_total_after_exchange": view.num_total,
                "local_global_indices": view.local_global_indices.cpu().tolist(),
                "halo_global_indices": view.halo_global_indices.cpu().tolist(),
                "send_counts_by_peer": self.partition_validation["send_counts_by_peer"],
                "recv_counts_by_peer": self.partition_validation["recv_counts_by_peer"],
                "owned_destination_edge_count": int(graph.edge_index.shape[1]),
                "compute_dtype": str(COMPUTE_DTYPE),
                "model_parameter_dtype": str(next(self.model.parameters()).dtype),
                "position_dtype": str(view.exchanged_positions.dtype),
                "force_dtype": str(owned_forces.dtype),
                "owned_destination_edge_records": [list(item) for item in sorted(global_edges)],
                "owned_node_energy_values": outputs["owned_node_energy"].detach().cpu().tolist(),
                "owned_local_energy": float(owned_energy.detach().cpu()),
                "owned_local_force_values": owned_forces.detach().cpu().tolist(),
                **force_checks,
            })
        return owned_energy.detach(), owned_forces.detach()


__all__ = [
    "COMPUTE_DTYPE",
    "DistMACETopDownAdapter",
    "canonicalize_nonperiodic_geometry",
    "load_raw_mace_model",
    "validate_partition_info",
]
