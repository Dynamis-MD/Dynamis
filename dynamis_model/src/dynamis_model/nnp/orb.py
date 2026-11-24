import contextlib
import os
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
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
    mp_graph: AtomGraphs  # sender OR receiver local
    zbl_graph: AtomGraphs  # receiver local: one copy of every pair job-wide
    world_size: int


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
        **kwargs,
    ):
        nn.Module.__init__(self)
        DynamisModel.__init__(self)

        self.device = device

        # Let the ORB factory build the model, heads, and load weights
        factory_function = getattr(pretrained, model_name)
        self.orbff, adapter = (
            factory_function(  # atoms_adapter is only used for attributes
                compile=False, device=device, precision=precision
            )
        )
        self.orbff.eval()  # Freeze weights for inference

        self.is_direct = isinstance(self.orbff, DirectForcefieldRegressor)

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
        # Position exchange: dim=3
        self._feature_exchanger.setup_exchanger(
            feature_dim=3, partition_info=partition_info, device=device
        )
        # Atom-type + kNN-threshold exchange: dim=2 ([type, tau], see _backbone)
        self._atom_type_exchanger.setup_exchanger(
            feature_dim=2, partition_info=partition_info, device=device
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

    def _backbone(self, distributed_state: DynamisState) -> _Backbone:
        prev_local_positions = distributed_state.local_molecular_info.atom_positions
        # requires_grad is inert under the direct path's no_grad, so the backbone
        # needs no knowledge of which family it is running.
        local_positions = prev_local_positions.clone().detach().requires_grad_(True)
        local_atom_types = distributed_state.local_molecular_info.atom_types
        n_local_atoms = local_positions.shape[0]

        positions = self.exchanger.push(local_positions, distributed_state)
        atom_types = self.exchanger.push(local_atom_types, distributed_state).long()

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
        types_tau = FeaturePushToExchange.apply(
            torch.stack([local_atom_types.float(), tau_local], dim=1),
            distributed_state,
            self._atom_type_exchanger,
        ).detach()
        atom_types = types_tau[:, 0].long()
        with torch.no_grad():
            knn_edge_index, knn_cell_offsets, _, _ = _impl.radius_graph_pbc_ex(
                *graph_args, 0, False, None, types_tau[:, 1].contiguous()
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
            zbl_forces = self.orbff.pair_repulsion_fn(backbone.mp_graph)["forces"]
            forces = forces + zbl_forces[: backbone.n_local]

        return forces * _EV_PER_A_TO_AMU_A_PER_FS2
