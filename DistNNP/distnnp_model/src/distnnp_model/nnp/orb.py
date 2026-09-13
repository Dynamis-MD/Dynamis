import os
from typing import Literal
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from distnnp_model.potential import DistNNPModel
try:
    from orb_models.forcefield import pretrained
    from orb_models.common.atoms.batch.graph_batch import AtomGraphs
    from orb_models.common.models.nn_util import get_cutoff
except ImportError as e:
    raise ImportError(
        "orb-models must be installed to use ORBModel: "
        "pip install -e ./distnnp_model[orb]"
    ) from e
from distnnp_state import DistNNPState, radius_graph_pbc_cuda
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_model.comm.exchanger import GPUDirectExchanger

from distnnp_model.units import EV_PER_A_TO_AMU_A_PER_FS2

# Re-exported under the historical private name so existing
# `from distnnp_model.nnp.orb import _EV_PER_A_TO_AMU_A_PER_FS2` keeps resolving
# to the one definition in distnnp_model.units.
_EV_PER_A_TO_AMU_A_PER_FS2 = EV_PER_A_TO_AMU_A_PER_FS2
# missing conservative architechure params

# when has_electrostatic is true the energy head is replace with a ChargeConditionedEnergyHead + LatentChargeHEad + CoulombModule(conservative regressor) (long range electrostatics)
# use_per_atom_spins adds a per-atom spin prediction through LatentSpinHead.
# has_charge_spin_cond or has_electrostatics adds ChargeSpinConditioner
# has_stress (not sure how this is calculated in conservative regressor)

# orb conservative_inf_omat/mpa only set has_stress = true.

class ORBModel(DistNNPModel, nn.Module):
    def __init__(
        self,
        model_name: Literal["orb_v3_conservative_inf_omat", "orb_v3_conservative_inf_map", "orb_v3_conservative_20_omat", "orb_v3_conservative_20_mpa"],
        # todo: v3 omol and direct
        precision: Literal["float32-high","float32-highest", "float64"],
        device: Literal["cuda", "cpu"] | None = None,
        **kwargs
    ):
        nn.Module.__init__(self)
        DistNNPModel.__init__(self)
        
        self.device = device
        
        # Let the ORB factory build the model, heads, and load weights
        factory_function = getattr(pretrained, model_name)
        self.orbff, adapter = factory_function( # atoms_adapter is only used for attributes
            compile=False, 
            device=device,
            precision=precision
        )
        self.orbff.eval() # Freeze weights for inference
        
        self.num_mp_steps = self.orbff.model.num_message_passing_steps
        self.hidden_dim = self.orbff.model.gnn_stacks[0].latent_dim

        self.model_radius = adapter.radius                        # default 6.0
        self.model_max_num_neighbors = adapter.max_num_neighbors  # 120 for the inf family, 20 otherwise

        self._feature_exchanger = GPUDirectExchanger()
        self._atom_type_exchanger = GPUDirectExchanger()
        # One latent exchanger PER message-passing layer (dim=hidden_dim)
        # gnn_stacks are 5 distinct layers
        self._latent_exchangers = [
            GPUDirectExchanger() for _ in range(self.num_mp_steps)
        ]

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
        # Atom-type exchange: dim=1
        self._atom_type_exchanger.setup_exchanger(
            feature_dim=1, partition_info=partition_info, device=device
        )
        # Per-layer latent exchange: dim=hidden_dim (256, uniform across layers)
        for latent_exchanger in self._latent_exchangers:
            latent_exchanger.setup_exchanger(
                feature_dim=self.hidden_dim, partition_info=partition_info, device=device
            )
        
    def _make_atomgraph(self, atom_types, n_atoms, senders, receivers,
                               displacement_vectors, cell, cutoff, device):
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
            max_num_neighbors=torch.tensor([self.model_max_num_neighbors], device=device),
            half_supercell=False,
        )
        
    def __call__(self, distributed_state: DistNNPState, *args, **kwargs):
        
        prev_local_positions = distributed_state.local_molecular_info.atom_positions
        local_positions = prev_local_positions.clone().detach().requires_grad_(True)
        local_atom_types = distributed_state.local_molecular_info.atom_types
        n_local_atoms = local_positions.shape[0]
        

        positions = FeaturePushToExchange.apply(
            local_positions, distributed_state, self._feature_exchanger
        )
        atom_types = FeaturePushToExchange.apply(
            local_atom_types.float().unsqueeze(-1),
            distributed_state,
            self._atom_type_exchanger,
        ).squeeze(-1).long()

        n_atoms = positions.shape[0]
        sim_config = distributed_state.sim_config
        
        # Detached from autograd, graph topology is not differentiable
        with torch.no_grad():
            edge_index, cell_offsets, _ = radius_graph_pbc_cuda(
                data=type("Data", (object,),
                    {
                        "pos": positions.detach(),
                        "cell": sim_config.cell.unsqueeze(0),  # Shape [1, 3, 3]
                        "natoms": [n_atoms],
                        "pbc": sim_config.pbc.unsqueeze(0),  # Shape [1, 3]
                    },
                )(),
                radius=sim_config.cutoff,
                max_num_neighbors_threshold=self.model_max_num_neighbors,
                enforce_max_neighbors_strictly=True,
            )

        if os.environ.get("ORB_DEBUG") == "1":
            _dbg_edge_index_full = edge_index
            _dbg_cell_offsets_full = cell_offsets


        def _displacement(ei, co): # lattice-vector → Cartesian conversion
            shifts = torch.mm(
                co.to(positions.dtype), sim_config.cell.squeeze(0).to(positions.dtype)
            )
            return positions[ei[1]] - positions[ei[0]] - shifts
        
        #  Orb's AttentionInteractionNetwork does segment_sum over senders and receivers, so message passing is effectively undirected
        mp_mask = (edge_index[0] < n_local_atoms) | (edge_index[1] < n_local_atoms) # is this edge's sender local or edge's receiver local
        mp_edge_index = edge_index[:, mp_mask] # drops any edge connecting atoms in my halo but not local
        senders = mp_edge_index[0]
        receivers = mp_edge_index[1]
        displacement_vectors = _displacement(mp_edge_index, cell_offsets[mp_mask])

        # ZBL edges are receiver local only.
        recv_local_mask = edge_index[1] < n_local_atoms
        zbl_edge_index = edge_index[:, recv_local_mask]
        zbl_displacement = _displacement(zbl_edge_index, cell_offsets[recv_local_mask])
        
        # Build AtomGraphs.
        # featurize_nodes for orb_v3 (use_embedding=True) calls atom_emb(batch) which looks up "atomic_numbers" from node_features. 
        # featurize_edges reads "vectors" from edge_features. 

        atomgraph = self._make_atomgraph(
            atom_types, n_atoms, senders, receivers,
            displacement_vectors, sim_config.cell, sim_config.cutoff, self.device,
        )
        
        
        gns = self.orbff.model
        
        edge_features = gns.featurize_edges(atomgraph) # [n_edges, edge_feat_dim]

        node_features = gns.featurize_nodes(atomgraph) # [n_atoms, node_feat_dim]
        
        if gns.conditioner is not None:
            cond_nodes, cond_edges = gns.conditioner(atomgraph)
        else:
            cond_nodes, cond_edges = None, None
        # A Conditioner is a neural network component used to inject global global state properties—such as ambient temperature, macroscopic pressure, total charge, or external magnetic/electric fields—into local node and edge states.

        # initial Encoder Pass
        nodes, edges = gns._encoder(node_features, edge_features)
        
        # calculate cutoffs for attention masking, with deafult to 6A
        edge_cutoff = get_cutoff(displacement_vectors.norm(dim=-1), r_max=self.model_radius)

        
        for i, gnn in enumerate(gns.gnn_stacks):
            if i > 0:
                local_nodes_to_push = nodes[:n_local_atoms] # [n_local, hidden_dim]

                nodes = FeaturePushToExchange.apply(
                    local_nodes_to_push, distributed_state, self._latent_exchangers[i]
                ) # [n_local + n_ghost, hidden_dim]

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
                nodes, edges, senders, receivers, edge_cutoff,
                cond_nodes=cond_nodes, cond_edges=cond_edges,
            )

        # Build a local-only graph for the energy head so n_node, atomic_numbers,
        # and aggregation all see exactly this rank's atoms.
        # EnergyHead aggregates over batch.n_node and adds reference(atomic_numbers).
        local_node_embeddings = nodes[:n_local_atoms]
        local_graph = AtomGraphs(
            senders=senders[:0],  # empty edge list (head ignores edges); [:0] keeps int64 dtype
            receivers=receivers[:0],
            n_node=torch.tensor([n_local_atoms], device=self.device),
            n_edge=torch.tensor([0], device=self.device),
            node_features={"atomic_numbers": local_atom_types},
            edge_features={"vectors": displacement_vectors.new_zeros((0, 3))},
            system_features={"cell": sim_config.cell.unsqueeze(0).to(positions.dtype)},
            node_targets={}, edge_targets={}, system_targets={},
            system_id=None, fix_atoms=None, tags=None,
            radius=sim_config.cutoff,
            max_num_neighbors=torch.tensor([self.model_max_num_neighbors], device=self.device),
            half_supercell=False,
        )
        # 1. EnergyHead: below replicates the energyhead.forward() call
        energy_head = self.orbff.heads[self.orbff.energy_name]

        world_size = dist.get_world_size() if dist.is_initialized() else 1

        local_emb_sum = local_node_embeddings.sum(dim=0, keepdim=True)  # [1, latent_dim]
        local_count = torch.tensor(
            [float(n_local_atoms)], device=local_emb_sum.device, dtype=local_emb_sum.dtype
        )
        # dist_nn.all_reduce is an autograd function 
        if world_size > 1: 
            global_emb_sum = dist_nn.all_reduce(local_emb_sum, op=dist.ReduceOp.SUM)
            global_count = dist_nn.all_reduce(local_count, op=dist.ReduceOp.SUM)
        else:
            global_emb_sum, global_count = local_emb_sum, local_count

        global_mean = global_emb_sum / global_count  # [1, latent_dim]

        mlp_out = energy_head.mlp(global_mean).squeeze(-1)   
        
        # look at _denormalize in forcefield_heads.py
        interaction_global = energy_head.normalizer.inverse(mlp_out)
        if getattr(energy_head, "atom_avg", False):
            interaction_global = interaction_global * global_count
        interaction_global = interaction_global.squeeze()
        # sum over ranks == interaction_global.
        interaction_contribution = interaction_global / world_size

        # 2. Pair repulsion
        zbl_local = displacement_vectors.new_zeros(())
        if getattr(self.orbff, "pair_repulsion", False):
            # RECEIVER-local edge set, not the union MP graph: one rank per pair.
            zbl_graph = self._make_atomgraph(
                atom_types, n_atoms, zbl_edge_index[0], zbl_edge_index[1],
                zbl_displacement, sim_config.cell, sim_config.cutoff, self.device,
            )
            zbl_local = self.orbff.pair_repulsion_fn(zbl_graph)["energy"].squeeze()

        # 3. Atom reference energy: lookup table
        reference_local = energy_head.reference(
            local_graph.atomic_numbers, local_graph.n_node
        ).squeeze()

        # Sum in fp64, mirroring ORB's EnergyHead.absolute_energy. The element reference is about 1e4 to 1e5 eV against a ~10 eV interaction
        # Free here — all matmuls are done — and autograd casts back at the fp32 leaf.
        local_energy = (
            interaction_contribution.double() + zbl_local.double() + reference_local.double()
        ) * _EV_PER_A_TO_AMU_A_PER_FS2 # ev -> amu * A^2 / fs^2

        if os.environ.get("ORB_DEBUG") == "1" and world_size > 1:
            # Unfiltered gns.forward over all n_atoms this rank sees. Valid only
            # while every rank sees every atom (small systems), where it reproduces
            # the W=1 computation with no filtering and no exchange. If it matches
            # W=1 but the distributed value does not, the bug is in the
            # filter/exchange path, not the model or the graph.
            with torch.no_grad():
                _fs = _dbg_edge_index_full[0]
                _fr = _dbg_edge_index_full[1]
                _fshift = torch.mm(
                    _dbg_cell_offsets_full.to(positions.dtype),
                    sim_config.cell.squeeze(0).to(positions.dtype),
                )
                _fdisp = positions[_fr] - positions[_fs] - _fshift
                _fgraph = self._make_atomgraph(
                    atom_types, n_atoms, _fs, _fr, _fdisp,
                    sim_config.cell, sim_config.cutoff, self.device,
                )
                _fout = gns(_fgraph)
                _fint = energy_head(_fout["node_features"], _fgraph)
                print(
                    f"[orb.py dbg] FULL-forward over {n_atoms} atoms "
                    f"({_fs.shape[0]} edges): interaction={_fint.item():.5f}",
                    flush=True,
                )

        if os.environ.get("ORB_DEBUG") == "1":
            _elen = displacement_vectors.norm(dim=-1)
            print(
                f"[orb.py dbg] n_local={n_local_atoms} n_atoms(+ghost)={n_atoms} "
                f"n_edges(recv-local)={senders.shape[0]} "
                f"edge_len_sum={_elen.sum().item():.4f} "
                f"edge_len_min={_elen.min().item():.4f} edge_len_max={_elen.max().item():.4f} "
                f"interaction_global={interaction_global.item():.5f} "
                f"interaction_share={interaction_contribution.item():.5f} "
                f"zbl={zbl_local.item():.5f} "
                f"reference={reference_local.item():.5f} "
                f"total={local_energy.item():.5f}",
                flush=True,
            )

        (grad_local_positions,) = torch.autograd.grad(
            local_energy, local_positions,
            create_graph=False, retain_graph=False,
        )
        local_forces = -grad_local_positions

        return local_energy.detach(), local_forces.detach()
