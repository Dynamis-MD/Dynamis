import contextlib
import os
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from dynamis_model.potential import DynamisModel

try:
    from orb_models.forcefield import pretrained
    from orb_models.common.atoms.batch.graph_batch import AtomGraphs
    from orb_models.common.models.nn_util import get_cutoff
    from orb_models.forcefield.models.direct_regressor import (
        DirectForcefieldRegressor,
    )
except ImportError as e:
    raise ImportError(
        "orb-models must be installed to use ORBModel: "
        "pip install -e ./dynamis_model[orb]"
    ) from e
from dynamis_state import DynamisState, radius_graph_pbc_cuda
from dynamis_model.comm.exchanger import Exchanger, GPUDirectExchanger
import dynamis_state.dynamis_state_impl as _impl


from dynamis_model.units import EV_PER_A_TO_AMU_A_PER_FS2

# Re-exported under the historical private name so existing
# `from dynamis_model.nnp.orb import _EV_PER_A_TO_AMU_A_PER_FS2` keeps resolving
# to the one definition in dynamis_model.units.
_EV_PER_A_TO_AMU_A_PER_FS2 = EV_PER_A_TO_AMU_A_PER_FS2

# Direct and conservative checkpoints share an identical MoleculeGNS backbone and
# EnergyHead; they differ only in how forces are recovered. Direct predicts them
# with a ForceHead (plus analytic ZBL forces) instead of -dE/dx, so the whole
# call runs under torch.no_grad() with no backward halo exchanges.


@dataclass
class _Backbone:
    nodes: torch.Tensor  # [n_local + n_ghost, hidden_dim] after the last MP layer
    local_positions: torch.Tensor  # autograd leaf for conservative forces
    local_atom_types: torch.Tensor
    n_local: int
    mp_graph: AtomGraphs | None  # sender OR receiver local (None on chunked direct path)
    zbl_graph: AtomGraphs | None  # receiver local (None on chunked direct path)
    world_size: int
    zbl_energy: torch.Tensor | None = None  # precomputed scalar eV (chunked direct path)
    zbl_forces: torch.Tensor | None = None  # precomputed [n_local, 3] eV/A (chunked direct path)


# Default edge chunk size for direct-family inference (torch.no_grad()).
# 512k edges keeps transient edge-MLP scratch to ~6.5 GiB in FP32 (~3.3 GiB in BF16)
# while fully saturating Tensor Cores, cutting peak HBM ~5-10x on large systems.
DEFAULT_EDGE_CHUNK_SIZE = 524_288


class ORBModel(DynamisModel, nn.Module):
    def __init__(
        self,
        model_name: Literal[
            "orb_v3_conservative_inf_omat",
            "orb_v3_conservative_inf_mpa",
            "orb_v3_conservative_20_omat",
            "orb_v3_conservative_20_mpa",
            "orb_v3_direct_inf_omat",
            "orb_v3_direct_inf_mpa",
            "orb_v3_direct_20_omat",
            "orb_v3_direct_20_mpa",
        ],
        # todo: v3 omol
        precision: Literal["float32-high", "float32-highest", "float64"],
        device: Literal["cuda", "cpu"] | None = None,
        edge_chunk_size: int | None = DEFAULT_EDGE_CHUNK_SIZE,
        mixed_precision: Literal["bf16"] | None = None,
        use_fused_kernels: bool | Literal["auto"] = "auto",
        **kwargs,
    ):
        nn.Module.__init__(self)
        DynamisModel.__init__(self)

        if mixed_precision not in (None, "bf16"):
            raise ValueError(
                f"Unsupported mixed_precision={mixed_precision!r}; expected None or 'bf16'."
            )
        if edge_chunk_size is not None and edge_chunk_size < 0:
            raise ValueError(
                f"edge_chunk_size must be >= 0 or None, got {edge_chunk_size}."
            )
        if use_fused_kernels not in (True, False, "auto"):
            raise ValueError(
                f"Unsupported use_fused_kernels={use_fused_kernels!r}; expected True, False, or 'auto'."
            )

        self.device = device
        self.edge_chunk_size = (
            int(edge_chunk_size) if edge_chunk_size and edge_chunk_size > 0 else None
        )
        self.mixed_precision = mixed_precision
        self.use_fused_kernels = use_fused_kernels
        self._fused_state = None

        # Let the ORB factory build the model, heads, and load weights
        factory_function = getattr(pretrained, model_name)
        self.orbff, adapter = (
            factory_function(  # atoms_adapter is only used for attributes
                compile=False, device=device, precision=precision
            )
        )
        self.orbff.eval()  # Freeze weights for inference

        self.is_direct = isinstance(self.orbff, DirectForcefieldRegressor)
        if self.mixed_precision is not None and not self.is_direct:
            raise ValueError(
                "mixed_precision='bf16' is only supported for direct-family ORB models."
            )

        self.num_mp_steps = self.orbff.model.num_message_passing_steps
        self.hidden_dim = self.orbff.model.gnn_stacks[0].latent_dim

        self.model_radius = adapter.radius  # default 6.0
        self.model_max_num_neighbors = (
            adapter.max_num_neighbors
        )  # 120 for the inf family, 20 otherwise
        # Orb keeps, per centre atom, its k nearest neighbours (hard cap).
        self.max_num_neighbors = int(self.model_max_num_neighbors)
        self.enforce_max_strictly = True

        exchanger = kwargs.get("exchanger", None)
        max_dim = max(3, int(self.hidden_dim))
        if exchanger is None:
            self.exchanger: Exchanger = GPUDirectExchanger(max_feature_dim=max_dim)
        elif isinstance(exchanger, type):
            self.exchanger = exchanger(max_feature_dim=max_dim)
        else:
            self.exchanger = exchanger

    def initialize_buffers(
        self,
        partition_info,
        device: Literal["cuda", "cpu"] | None = None,
    ) -> None:
        device = device or self.device
        self.exchanger.setup_exchanger(
            feature_dim=max(3, int(self.hidden_dim)),
            partition_info=partition_info,
            device=device,
        )

    def _make_atomgraph(
        self,
        atom_types,
        n_atoms,
        senders,
        receivers,
        displacement_vectors,
        cell,
        cutoff,
        device,
    ):
        return AtomGraphs(
            senders=senders,
            receivers=receivers,
            n_node=torch.tensor([n_atoms], device=device),
            n_edge=torch.tensor([senders.shape[0]], device=device),
            node_features={
                # Integer element ids. Consumed by the GNN atom embedding
                "atomic_numbers": atom_types,
                # One-hot of the element ids. Consumed only by ZBL in native orb impl. Non-differentiable
                "atomic_numbers_embedding": torch.nn.functional.one_hot(
                    atom_types.long(), num_classes=118
                ).to(displacement_vectors.dtype),
            },
            edge_features={"vectors": displacement_vectors},
            system_features={"cell": cell.unsqueeze(0).to(displacement_vectors.dtype)},
            node_targets={},
            edge_targets={},
            system_targets={},
            system_id=None,
            fix_atoms=None,
            tags=None,
            radius=cutoff,
            max_num_neighbors=torch.tensor(
                [self.model_max_num_neighbors], device=device
            ),
            half_supercell=False,
        )

    def __call__(self, distributed_state: DynamisState, *args, **kwargs):
        # no_grad rather than inference_mode: the integrator keeps these forces as
        # prev_forces, and inference tensors reject later in-place use.
        grad_mode = torch.no_grad() if self.is_direct else contextlib.nullcontext()
        with grad_mode:
            backbone = self._backbone(distributed_state)
            local_energy = self._local_energy(backbone)
            if self.is_direct:
                local_forces = self._direct_forces(backbone)
            else:
                local_forces = self._conservative_forces(local_energy, backbone)
        return local_energy.detach(), local_forces.detach()

    def _can_use_chunked_direct(self) -> bool:
        """True when the edge-chunked direct inference path can run identically."""
        if not self.is_direct:
            return False
        if self.edge_chunk_size is None:
            return False
        if os.environ.get("ORB_DEBUG") == "1":
            return False
        gns = self.orbff.model
        if gns.conditioner is not None:
            return False
        if not gns.use_embedding or gns.expects_atom_type_embedding:
            return False
        if any(k != "feat" for k in gns.node_feature_names):
            return False
        if any(k != "feat" for k in gns.edge_feature_names):
            return False
        for gnn in gns.gnn_stacks:
            if (
                gnn._attention_gate != "sigmoid"
                or gnn._node_cond != "none"
                or gnn._edge_cond != "none"
            ):
                return False
        if getattr(self.orbff, "pair_repulsion", False):
            zbl = self.orbff.pair_repulsion_fn
            if getattr(zbl, "node_aggregation", None) != "sum":
                return False
        return True

    def _accumulate_zbl_chunk(
        self,
        zbl,
        atom_types_long: torch.Tensor,
        snd_c: torch.Tensor,
        rcv_c: torch.Tensor,
        vec_c: torch.Tensor,
        lengths_c: torch.Tensor,
        n_local_atoms: int,
        zbl_energy_local: torch.Tensor,
        zbl_forces_local: torch.Tensor,
    ) -> None:
        """Accumulate exact ZBL energy (receiver-local) and forces (union -> local) for one edge chunk.

        Mirrors orb_models.forcefield.models.pair_repulsion.ZBLBasis.forward without
        materializing [N, 118] one-hot node tensors or [E, 3, 3] unused virial stress.
        """
        # ZBLBasis does argmax(one_hot(atom_types)) + 1 == atom_types + 1.
        z_u = atom_types_long[snd_c] + 1
        z_v = atom_types_long[rcv_c] + 1
        a = (
            zbl.a_prefactor
            * 0.529
            / (torch.pow(z_u, zbl.a_exp) + torch.pow(z_v, zbl.a_exp))
        )
        r_hat = vec_c / lengths_c.unsqueeze(1)
        r_over_a = lengths_c / a
        exp_term = torch.exp(-zbl.d * r_over_a.unsqueeze(0))
        phi = torch.sum(zbl.c * exp_term, dim=0)

        coulomb_term = 14.3996 * z_u * z_v / lengths_c
        v_edges_raw = coulomb_term * phi
        r_max = zbl.covalent_radii[z_u] + zbl.covalent_radii[z_v]
        envelope, denvelope_dr = zbl._polynomial_cutoff_with_derivative(
            lengths_c, r_max, zbl.p
        )
        v_edges = 0.5 * v_edges_raw * envelope

        # Energy: receiver-local edges only (one copy of every pair job-wide).
        rcv_local = rcv_c < n_local_atoms
        if rcv_local.any():
            zbl_energy_local.add_(v_edges[rcv_local].sum(dtype=torch.float64))

        # Forces: union graph -> accumulate onto local rows only.
        dphi_dr_over_a = torch.sum(-zbl.d * zbl.c * exp_term, dim=0)
        dphi_dr = dphi_dr_over_a / a
        dv_dr_raw = -coulomb_term / lengths_c * phi + coulomb_term * dphi_dr
        dv_dr = 0.5 * (dv_dr_raw * envelope + v_edges_raw * denvelope_dr)
        force_vectors = (-dv_dr).unsqueeze(1) * r_hat

        snd_local = snd_c < n_local_atoms
        if snd_local.any():
            zbl_forces_local.index_add_(
                0, snd_c[snd_local], -force_vectors[snd_local]
            )
        if rcv_local.any():
            zbl_forces_local.index_add_(
                0, rcv_c[rcv_local], force_vectors[rcv_local]
            )

    def _backbone_direct_chunked(
        self,
        distributed_state: DynamisState,
        local_positions: torch.Tensor,
        local_atom_types: torch.Tensor,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        mp_offsets: torch.Tensor,
        cell_2d: torch.Tensor,
        n_local_atoms: int,
        world_size: int,
    ) -> _Backbone:
        """Memory-optimized direct-family backbone using edge-chunked encoder, ZBL, and GNN layers.

        Avoids materializing full [E, 768] / [E, 1024] transient edge-MLP activations,
        full [E, 3] displacement vectors, duplicate [N, 118] one-hot tables, and
        extra [E, 256] edge residual copies.
        """
        gns = self.orbff.model
        chunk_size = self.edge_chunk_size
        assert chunk_size is not None
        n_edges = senders.shape[0]
        pos_dtype = positions.dtype
        use_bf16 = (
            self.mixed_precision == "bf16" and positions.device.type == "cuda"
        )
        edge_dtype = torch.bfloat16 if use_bf16 else pos_dtype
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else contextlib.nullcontext()
        )

        # 1. Chunked node embedding + encoder ([N, latent_dim], kept in FP32)
        n_atoms_total = atom_types.shape[0]
        nodes = torch.empty(
            (n_atoms_total, self.hidden_dim),
            device=positions.device,
            dtype=pos_dtype,
        )
        for s in range(0, n_atoms_total, chunk_size):
            e = min(s + chunk_size, n_atoms_total)
            with amp_ctx:
                nodes[s:e] = gns._encoder._node_fn(
                    gns.atom_emb.embeddings(atom_types[s:e])
                ).to(pos_dtype)

        # 2. Chunked edge featurization + edge encoder + exact analytic ZBL
        edges = torch.empty(
            (n_edges, self.hidden_dim), device=positions.device, dtype=edge_dtype
        )
        edge_cutoff = torch.empty(
            (n_edges, 1), device=positions.device, dtype=pos_dtype
        )
        has_zbl = bool(getattr(self.orbff, "pair_repulsion", False))
        zbl = self.orbff.pair_repulsion_fn if has_zbl else None
        zbl_energy_local = torch.zeros(
            (), device=positions.device, dtype=torch.float64
        )
        zbl_forces_local = (
            torch.zeros(
                (n_local_atoms, 3), device=positions.device, dtype=pos_dtype
            )
            if has_zbl
            else None
        )

        for s in range(0, n_edges, chunk_size):
            e = min(s + chunk_size, n_edges)
            snd_c = senders[s:e]
            rcv_c = receivers[s:e]
            shifts_c = torch.mm(mp_offsets[s:e].to(pos_dtype), cell_2d)
            vec_c = positions[rcv_c] - positions[snd_c] - shifts_c
            lengths_c = vec_c.norm(dim=1)
            edge_cutoff[s:e] = get_cutoff(lengths_c, r_max=self.model_radius)

            if has_zbl:
                self._accumulate_zbl_chunk(
                    zbl,
                    atom_types,
                    snd_c,
                    rcv_c,
                    vec_c,
                    lengths_c,
                    n_local_atoms,
                    zbl_energy_local,
                    zbl_forces_local,
                )

            ang_c = gns.angular_transform(vec_c)
            rbf_c = gns.rbf_transform(lengths_c)
            if gns.outer_product_with_cutoff:
                cut_enc = get_cutoff(lengths_c)
                outer = rbf_c[:, :, None] * ang_c[:, None, :]
                ef_c = cut_enc * outer.view(e - s, gns.edge_embed_size)
            else:
                ef_c = torch.cat([rbf_c, ang_c], dim=1)

            with amp_ctx:
                edges[s:e] = gns._encoder._edge_fn(ef_c).to(edge_dtype)

        # Free geometry buffers no longer needed during message passing.
        del mp_offsets, positions, atom_types

        # 3. Chunked message-passing stack (in-place residual update on `edges`)
        for i, gnn in enumerate(gns.gnn_stacks):
            if i > 0:
                nodes = self.exchanger.push(
                    nodes[:n_local_atoms], distributed_state
                )

            sent_agg = torch.zeros_like(nodes)
            recv_agg = torch.zeros_like(nodes)
            nodes_in = nodes.to(edge_dtype) if edge_dtype != nodes.dtype else nodes

            for s in range(0, n_edges, chunk_size):
                e = min(s + chunk_size, n_edges)
                e_c = edges[s:e]
                snd_c = senders[s:e]
                rcv_c = receivers[s:e]
                cut_c = edge_cutoff[s:e]

                with amp_ctx:
                    recv_attn = F.sigmoid(gnn._receive_attn(e_c))
                    send_attn = F.sigmoid(gnn._send_attn(e_c))
                    if gnn._distance_cutoff:
                        recv_attn = recv_attn * cut_c
                        send_attn = send_attn * cut_c
                    ef_c = torch.cat(
                        [e_c, nodes_in[snd_c], nodes_in[rcv_c]], dim=1
                    )
                    upd_e = gnn._edge_mlp(ef_c)

                sent_agg.index_add_(
                    0, snd_c, (upd_e * send_attn).to(sent_agg.dtype)
                )
                recv_agg.index_add_(
                    0, rcv_c, (upd_e * recv_attn).to(recv_agg.dtype)
                )
                edges[s:e].add_(upd_e.to(edge_dtype))

            del nodes_in
            n_nodes = nodes.shape[0]
            for s in range(0, n_nodes, chunk_size):
                e = min(s + chunk_size, n_nodes)
                nf_c = torch.cat(
                    [nodes[s:e], recv_agg[s:e], sent_agg[s:e]], dim=1
                )
                with amp_ctx:
                    nodes[s:e].add_(gnn._node_mlp(nf_c).to(nodes.dtype))
            del sent_agg, recv_agg

        del edges, senders, receivers, edge_cutoff

        return _Backbone(
            nodes=nodes,
            local_positions=local_positions,
            local_atom_types=local_atom_types,
            n_local=n_local_atoms,
            mp_graph=None,
            zbl_graph=None,
            world_size=world_size,
            zbl_energy=zbl_energy_local,
            zbl_forces=zbl_forces_local,
        )

    def _backbone(self, distributed_state: DynamisState) -> _Backbone:
        prev_local_positions = distributed_state.local_molecular_info.atom_positions
        # requires_grad is inert under the direct path's no_grad, so the backbone
        # needs no knowledge of which family it is running.
        local_positions = prev_local_positions.clone().detach().requires_grad_(True)
        local_atom_types = distributed_state.local_molecular_info.atom_types
        n_local_atoms = local_positions.shape[0]

        positions = self.exchanger.push(local_positions, distributed_state)

        n_atoms = positions.shape[0]
        sim_config = distributed_state.sim_config
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        graph_args = (
            positions.detach(),
            sim_config.cell,
            sim_config.pbc.to(torch.bool),
            float(sim_config.cutoff),
        )

        # Orb's graph: centre c → neighbour n iff n is among c's k nearest. An edge whose
        # centre is a ghost but whose neighbour is local still feeds the local atom, and
        # deciding it needs the ghost's full neighbourhood, which only its owner has. So:
        #   pass 1 — tau (k-th-neighbour threshold on d^2) for local centres;
        #   push tau with the atom types (the push this model already does every step);
        #   pass 2 — build with every centre's tau, own or received.
        # Detached from autograd, graph topology is not differentiable.
        with torch.no_grad():
            tau_local, _ = _impl.radius_graph_pbc_select(
                *graph_args,
                self.max_num_neighbors,
                self.enforce_max_strictly,
                torch.arange(n_local_atoms, device=positions.device),
            )
        types_tau = self.exchanger.push(
            torch.stack([local_atom_types.float(), tau_local], dim=1),
            distributed_state,
        ).detach()
        atom_types = types_tau[:, 0].long()
        with torch.no_grad():
            knn_edge_index, knn_cell_offsets, _, _ = _impl.radius_graph_pbc_ex(
                *graph_args, 0, False, None, types_tau[:, 1].contiguous()
            )
        del tau_local, types_tau

        if self._can_use_chunked_direct():
            # Kernel caps per target (knn_edge_index[1]); Orb caps per sender (centre),
            # so sender = knn_edge_index[1], receiver = knn_edge_index[0], offset = -knn_cell_offsets.
            mp_mask = (knn_edge_index[1] < n_local_atoms) | (
                knn_edge_index[0] < n_local_atoms
            )
            senders = knn_edge_index[1, mp_mask].contiguous()
            receivers = knn_edge_index[0, mp_mask].contiguous()
            mp_offsets = (-knn_cell_offsets[mp_mask]).contiguous()
            del knn_edge_index, knn_cell_offsets, mp_mask
            cell_2d = sim_config.cell.squeeze(0).to(positions.dtype)
            if self.use_fused_kernels is True or (
                self.use_fused_kernels == "auto" and self.mixed_precision == "bf16"
            ):
                from dynamis_model.nnp import orb_fused_kernels as _fused_orb

                if _fused_orb.can_use_fused_orb_direct(
                    self.orbff,
                    positions,
                    mixed_precision=self.mixed_precision,
                    use_fused_kernels=self.use_fused_kernels,
                ):
                    if (
                        self._fused_state is None
                        or self._fused_state.bessel_weights.device != positions.device
                    ):
                        self._fused_state = _fused_orb.prepare_orb_fused_state(
                            self.orbff, positions.device
                        )
                    nodes, zbl_energy, zbl_forces = _fused_orb.run_fused_orb_backbone(
                        orbff=self.orbff,
                        fused=self._fused_state,
                        exchanger=self.exchanger,
                        distributed_state=distributed_state,
                        positions=positions,
                        atom_types=atom_types,
                        senders=senders,
                        receivers=receivers,
                        mp_offsets=mp_offsets,
                        cell_2d=cell_2d,
                        n_local_atoms=n_local_atoms,
                        model_radius=float(self.model_radius),
                        chunk_size=self.edge_chunk_size or DEFAULT_EDGE_CHUNK_SIZE,
                    )
                    return _Backbone(
                        nodes=nodes,
                        local_positions=local_positions,
                        local_atom_types=local_atom_types,
                        n_local=n_local_atoms,
                        mp_graph=None,
                        zbl_graph=None,
                        world_size=world_size,
                        zbl_energy=zbl_energy,
                        zbl_forces=zbl_forces,
                    )
            return self._backbone_direct_chunked(
                distributed_state=distributed_state,
                local_positions=local_positions,
                local_atom_types=local_atom_types,
                positions=positions,
                atom_types=atom_types,
                senders=senders,
                receivers=receivers,
                mp_offsets=mp_offsets,
                cell_2d=cell_2d,
                n_local_atoms=n_local_atoms,
                world_size=world_size,
            )

        # The kernel caps per target (edge_index[1]); Orb caps per sender (the centre), so
        # flip: sender = centre, receiver = neighbour, shift re-attached to the receiver.
        edge_index = knn_edge_index.flip(0)
        cell_offsets = -knn_cell_offsets

        def _displacement(ei, co):  # lattice-vector → Cartesian conversion
            shifts = torch.mm(
                co.to(positions.dtype), sim_config.cell.squeeze(0).to(positions.dtype)
            )
            return positions[ei[1]] - positions[ei[0]] - shifts

        # Orb's AttentionInteractionNetwork does segment_sum over senders and receivers,
        # so message passing is effectively undirected.
        mp_mask = (edge_index[0] < n_local_atoms) | (
            edge_index[1] < n_local_atoms
        )  # is this edge's sender local or edge's receiver local
        mp_edge_index = edge_index[
            :, mp_mask
        ]  # drops any edge connecting atoms in my halo but not local
        senders = mp_edge_index[0]
        receivers = mp_edge_index[1]
        displacement_vectors = _displacement(mp_edge_index, cell_offsets[mp_mask])

        # ZBL energy edges are receiver-local only: one copy of every pair job-wide.
        recv_local_mask = edge_index[1] < n_local_atoms
        zbl_edge_index = edge_index[:, recv_local_mask]
        zbl_displacement = _displacement(zbl_edge_index, cell_offsets[recv_local_mask])

        atomgraph = self._make_atomgraph(
            atom_types,
            n_atoms,
            senders,
            receivers,
            displacement_vectors,
            sim_config.cell,
            sim_config.cutoff,
            self.device,
        )
        zbl_graph = self._make_atomgraph(
            atom_types,
            n_atoms,
            zbl_edge_index[0],
            zbl_edge_index[1],
            zbl_displacement,
            sim_config.cell,
            sim_config.cutoff,
            self.device,
        )

        gns = self.orbff.model

        edge_features = gns.featurize_edges(atomgraph)  # [n_edges, edge_feat_dim]
        node_features = gns.featurize_nodes(atomgraph)  # [n_atoms, node_feat_dim]

        if gns.conditioner is not None:
            cond_nodes, cond_edges = gns.conditioner(atomgraph)
        else:
            cond_nodes, cond_edges = None, None

        # Initial Encoder Pass
        nodes, edges = gns._encoder(node_features, edge_features)

        # Calculate cutoffs for attention masking, with default to 6A
        edge_cutoff = get_cutoff(
            displacement_vectors.norm(dim=-1), r_max=self.model_radius
        )

        for i, gnn in enumerate(gns.gnn_stacks):
            if i > 0:
                local_nodes_to_push = nodes[:n_local_atoms]  # [n_local, hidden_dim]

                nodes = self.exchanger.push(
                    local_nodes_to_push, distributed_state
                )  # [n_local + n_ghost, hidden_dim]

            if os.environ.get("ORB_DEBUG") == "1" and n_atoms > n_local_atoms:
                _gh = nodes[n_local_atoms:]
                _lo = nodes[:n_local_atoms]
                print(
                    f"[orb.py dbg] layer {i} post-exchange: "
                    f"local_norm={_lo.detach().norm().item():.4f} "
                    f"ghost_norm={_gh.detach().norm().item():.4f} "
                    f"ghost_mean_abs={_gh.detach().abs().mean().item():.5f}",
                    flush=True,
                )

            # Run the interaction layer
            nodes, edges = gnn(
                nodes,
                edges,
                senders,
                receivers,
                edge_cutoff,
                cond_nodes=cond_nodes,
                cond_edges=cond_edges,
            )

        if os.environ.get("ORB_DEBUG") == "1" and world_size > 1:
            with torch.no_grad():
                _fs = edge_index[0]
                _fr = edge_index[1]
                _fshift = torch.mm(
                    cell_offsets.to(positions.dtype),
                    sim_config.cell.squeeze(0).to(positions.dtype),
                )
                _fdisp = positions[_fr] - positions[_fs] - _fshift
                _fgraph = self._make_atomgraph(
                    atom_types,
                    n_atoms,
                    _fs,
                    _fr,
                    _fdisp,
                    sim_config.cell,
                    sim_config.cutoff,
                    self.device,
                )
                _fout = gns(_fgraph)
                _fint = self.orbff.heads["energy"](_fout["node_features"], _fgraph)
                print(
                    f"[orb.py dbg] FULL-forward over {n_atoms} atoms "
                    f"({_fs.shape[0]} edges): interaction={_fint.item():.5f}",
                    flush=True,
                )

        if os.environ.get("ORB_DEBUG") == "1":
            _elen = displacement_vectors.norm(dim=-1)
            print(
                f"[orb.py dbg] n_local={n_local_atoms} n_atoms(+ghost)={n_atoms} "
                f"n_edges(mp)={senders.shape[0]} "
                f"edge_len_sum={_elen.sum().item():.4f} "
                f"edge_len_min={_elen.min().item():.4f} edge_len_max={_elen.max().item():.4f}",
                flush=True,
            )

        return _Backbone(
            nodes=nodes,
            local_positions=local_positions,
            local_atom_types=local_atom_types,
            n_local=n_local_atoms,
            mp_graph=atomgraph,
            zbl_graph=zbl_graph,
            world_size=world_size,
        )

    def _local_energy(self, backbone: _Backbone) -> torch.Tensor:
        # 1. EnergyHead: heads["energy"] works for both conservative and direct
        # regressors (DirectForcefieldRegressor has no energy_name attribute).
        energy_head = self.orbff.heads["energy"]
        world_size = backbone.world_size

        local_emb_sum = backbone.nodes[: backbone.n_local].sum(
            dim=0, keepdim=True
        )  # [1, latent_dim]
        local_count = torch.tensor(
            [float(backbone.n_local)],
            device=local_emb_sum.device,
            dtype=local_emb_sum.dtype,
        )
        # dist_nn.all_reduce is an autograd function; under the direct path's
        # no_grad its forward is a plain all_reduce.
        if world_size > 1:
            global_emb_sum = dist_nn.all_reduce(local_emb_sum, op=dist.ReduceOp.SUM)
            global_count = dist_nn.all_reduce(local_count, op=dist.ReduceOp.SUM)
        else:
            global_emb_sum, global_count = local_emb_sum, local_count

        global_mean = global_emb_sum / global_count  # [1, latent_dim]
        mlp_out = energy_head.mlp(global_mean).squeeze(-1)

        interaction_global = energy_head.normalizer.inverse(mlp_out)
        if getattr(energy_head, "atom_avg", False):
            interaction_global = interaction_global * global_count
        interaction_global = interaction_global.squeeze()
        # sum over ranks == interaction_global.
        interaction_contribution = interaction_global / world_size

        # 2. Pair repulsion (receiver-local edge set: one rank per pair)
        zbl_local = backbone.nodes.new_zeros(())
        if getattr(self.orbff, "pair_repulsion", False):
            if backbone.zbl_energy is not None:
                zbl_local = backbone.zbl_energy
            else:
                zbl_local = self.orbff.pair_repulsion_fn(backbone.zbl_graph)[
                    "energy"
                ].squeeze()

        # 3. Atom reference energy: lookup table over this rank's atoms only
        reference_local = energy_head.reference(
            backbone.local_atom_types,
            torch.tensor([backbone.n_local], device=self.device),
        ).squeeze()

        # Sum in fp64, mirroring ORB's EnergyHead.absolute_energy.
        local_energy = (
            interaction_contribution.double()
            + zbl_local.double()
            + reference_local.double()
        ) * _EV_PER_A_TO_AMU_A_PER_FS2  # eV -> amu * A^2 / fs^2

        if os.environ.get("ORB_DEBUG") == "1":
            print(
                f"[orb.py dbg] direct={self.is_direct} "
                f"interaction_global={interaction_global.item():.5f} "
                f"interaction_share={interaction_contribution.item():.5f} "
                f"zbl={zbl_local.item():.5f} "
                f"reference={reference_local.item():.5f} "
                f"total={local_energy.item():.5f}",
                flush=True,
            )

        return local_energy

    def _conservative_forces(
        self, local_energy: torch.Tensor, backbone: _Backbone
    ) -> torch.Tensor:
        (grad_local_positions,) = torch.autograd.grad(
            local_energy,
            backbone.local_positions,
            create_graph=False,
            retain_graph=False,
        )
        return -grad_local_positions

    def _direct_forces(self, backbone: _Backbone) -> torch.Tensor:
        force_head = self.orbff.heads["forces"]
        forces = force_head.normalizer.inverse(
            force_head.mlp(backbone.nodes[: backbone.n_local])
        )

        # ForceHead.remove_mean subtracts the mean over the WHOLE system.
        # Torque removal only fires for cell == 0, so it never applies under PBC.
        if force_head.remove_mean:
            force_stats = torch.cat(
                [forces.sum(dim=0), forces.new_tensor([float(backbone.n_local)])]
            )
            if backbone.world_size > 1:
                dist.all_reduce(force_stats, op=dist.ReduceOp.SUM)
            forces = forces - force_stats[:3] / force_stats[3]

        # ZBL forces need the UNION graph: a local atom gets half its pair force
        # as sender of i->j and half as receiver of j->i (pair_repulsion.py:149-151).
        # Ghost rows hold partial sums and are dropped. Added after mean removal,
        # matching DirectForcefieldRegressor.predict.
        if getattr(self.orbff, "pair_repulsion", False):
            if backbone.zbl_forces is not None:
                forces = forces + backbone.zbl_forces
            else:
                zbl_forces = self.orbff.pair_repulsion_fn(backbone.mp_graph)["forces"]
                forces = forces + zbl_forces[: backbone.n_local]

        return forces * _EV_PER_A_TO_AMU_A_PER_FS2
