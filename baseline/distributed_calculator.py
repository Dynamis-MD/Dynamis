"""
Distributed Calculator for multi-GPU molecular dynamics.

Wraps an inner calculator (FAIRChem, MACE, etc.) and handles
ghost atom exchange and force reduction across MPI ranks.
"""

import numpy as np
import torch
import torch.distributed as dist
import ase
from typing import Dict, Tuple, Optional
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError


from baseline.profiler import CUDAEventProfiler, StepProfiler
from baseline.halo_exchange import HaloExchanger


class DistributedCalculator(Calculator):
    """ASE Calculator wrapper for distributed MD with ghost atom handling."""
    
    implemented_properties = ["energy", "forces"]
    
    def __init__(self, calculator, cutoff: float, comm_map: Dict[int, torch.Tensor] = None, 
                 interior_mask: torch.Tensor = None, enable_overlap: bool = False,
                 recv_ghost_indices: Dict[int, torch.Tensor] = None,
                 hierarchical_params: dict = None,
                 per_node_params: dict = None):
        super().__init__()
        self.calculator = calculator
        self.cutoff = cutoff
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exchanger = HaloExchanger(
            self.rank, 
            self.world_size, 
            comm_map if comm_map else {}, 
            device,
            recv_ghost_indices=recv_ghost_indices
        )
        self.num_local = 0
        
        # Store interior mask and overlap flag for optimization
        self.interior_mask = interior_mask.to(device) if interior_mask is not None else None
        self.enable_overlap = enable_overlap
        
        # Hierarchical aggregation params (Area 2 / PXN)
        self.hierarchical_params = hierarchical_params
        
        # Per-node partitioning params
        self.per_node_params = per_node_params
        self._intra_group = per_node_params['intra_group'] if per_node_params else None
        
        # OPTIMIZATION: Cache comm maps to avoid rebuilding every step
        self._cached_recv_ghost_indices = None
        self._cached_device_send_map = None
        self._cached_comm_map_version = -1  # Version counter to invalidate cache on update
        self._comm_map_version = 0
        
        # OPTIMIZATION: Cache atomic numbers (they don't change during MD)
        self._cached_local_Z_tensor = None
        
        # Profiler (optional)
        self.profiler = None
        self.measure_bw = False
        self.differentiable_exchange = False
        
    def set_profiler(self, profiler: StepProfiler):
        """Set profiler for detailed timing analysis."""
        self.profiler = profiler
        # Also pass profiler to inner FAIRChemCalculator
        self.calculator._profiler = profiler
        
    def update_comm_map(self, comm_map: Dict[int, torch.Tensor], interior_mask: torch.Tensor = None):
        self.exchanger = HaloExchanger(self.rank, self.world_size, comm_map, self.exchanger.device)
        # Update interior mask
        if interior_mask is not None:
            self.interior_mask = interior_mask.to(self.exchanger.device)
        # OPTIMIZATION: Invalidate cached comm maps
        self._comm_map_version += 1

    def _prepare_combined_atoms(self, atoms: ase.Atoms) -> Tuple[ase.Atoms, Dict[str, torch.Tensor]]:
        if self.per_node_params is not None:
            return self._prepare_combined_atoms_per_node(atoms)
        return self._prepare_combined_atoms_default(atoms)
    
    def _prepare_combined_atoms_per_node(self, atoms: ase.Atoms) -> Tuple[ase.Atoms, Dict[str, torch.Tensor]]:
        """
        Per-node mode: build combined_atoms = [all_node_atoms | inter_node_ghosts].

        1. Intra-node all_gather to get all node atoms (pos, Z)
        2. Inter-node ghost exchange (only cross-node ghosts)
        3. Build combined_atoms for graph/edge computation
        """

        pnp = self.per_node_params
        intra_group = pnp['intra_group']
        gpus_per_node = pnp['gpus_per_node']
        local_gpu_id = pnp['local_gpu_id']

        # 1. All_gather positions and atomic numbers from all GPUs in node
        local_pos = torch.from_numpy(atoms.get_positions()).float().to(self.exchanger.device)
        local_Z = torch.from_numpy(atoms.get_atomic_numbers()).long().to(self.exchanger.device)
        N_local = local_pos.size(0)

        # Step 1a: Gather actual atom counts from all GPUs in node (needed for variable-size all_gather)
        local_count = torch.tensor([N_local], dtype=torch.long, device=self.exchanger.device)
        count_recv = [torch.zeros(1, dtype=torch.long, device=self.exchanger.device)
                      for _ in range(gpus_per_node)]
        dist.all_gather(count_recv, local_count, group=intra_group)
        node_atom_counts = [c.item() for c in count_recv]

        # Update per_node_params with current counts (may change after rebalance)
        pnp['node_atom_counts'] = node_atom_counts

        # Step 1b: All_gather positions using padding (all_gather requires same-size tensors)
        max_atoms = max(node_atom_counts)

        # Pad local_pos to max_atoms
        if N_local < max_atoms:
            pad_pos = torch.zeros(max_atoms - N_local, 3, device=self.exchanger.device)
            local_pos_padded = torch.cat([local_pos, pad_pos], dim=0)
        else:
            local_pos_padded = local_pos

        pos_recv = [torch.zeros(max_atoms, 3, device=self.exchanger.device)
                    for _ in range(gpus_per_node)]
        dist.all_gather(pos_recv, local_pos_padded.contiguous(), group=intra_group)

        # Unpad and concatenate
        node_pos_list = [pos_recv[i][:node_atom_counts[i]] for i in range(gpus_per_node)]
        node_pos = torch.cat(node_pos_list, dim=0)  # [N_node_total, 3]

        # Step 1c: All_gather atomic numbers (same padding approach)
        if N_local < max_atoms:
            pad_Z = torch.zeros(max_atoms - N_local, device=self.exchanger.device, dtype=torch.long)
            local_Z_padded = torch.cat([local_Z, pad_Z], dim=0)
        else:
            local_Z_padded = local_Z

        Z_recv = [torch.zeros(max_atoms, device=self.exchanger.device, dtype=torch.long)
                  for _ in range(gpus_per_node)]
        dist.all_gather(Z_recv, local_Z_padded.contiguous(), group=intra_group)

        # Unpad and concatenate
        node_Z_list = [Z_recv[i][:node_atom_counts[i]] for i in range(gpus_per_node)]
        node_Z = torch.cat(node_Z_list, dim=0)  # [N_node_total]

        # 2. Inter-node ghost exchange (skip in single-node case)
        num_nodes = pnp.get('num_nodes', 1)
        if num_nodes > 1:
            ghosts = self.exchanger.exchange(atoms)
        else:
            # Single node: no inter-node ghosts, all atoms are within the node
            ghosts = {}
        
        # 3. Build combined_atoms: [all_node_atoms | inter_node_ghosts]
        num_node_total = node_pos.size(0)
        node_pos_np = node_pos.cpu().numpy()
        node_Z_np = node_Z.cpu().numpy()

        self.num_local = len(atoms)  # per-GPU local count
        self.num_node_atoms = num_node_total

        if ghosts:
            ghost_pos_np = ghosts['pos'].cpu().numpy()
            ghost_Z_np = ghosts['Z'].cpu().numpy()
            combined_pos = np.concatenate([node_pos_np, ghost_pos_np])
            combined_Z = np.concatenate([node_Z_np, ghost_Z_np])
        else:
            combined_pos = node_pos_np
            combined_Z = node_Z_np
        
        combined_atoms = ase.Atoms(
            numbers=combined_Z,
            positions=combined_pos,
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc()
        )
        
        # Update model comm maps
        model = self.calculator.predictor.model
        if hasattr(model, 'module'):
            model = model.module
        
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'set_comm_maps'):
            if self._cached_comm_map_version != self._comm_map_version or self._cached_recv_ghost_indices is None:
                recv_ghost_indices = {}
                # In per-node mode, ghost slots start after num_node_total (not num_local)
                current_idx = num_node_total
                for r in range(self.world_size):
                    count = self.exchanger.recv_counts.get(r, 0)
                    if count > 0:
                        indices = torch.arange(current_idx, current_idx + count, device=self.exchanger.device)
                        recv_ghost_indices[r] = indices
                        current_idx += count
                
                device_send_map = {r: idx.to(self.exchanger.device) for r, idx in self.exchanger.comm_map.items()}
                
                self._cached_recv_ghost_indices = recv_ghost_indices
                self._cached_device_send_map = device_send_map
                self._cached_comm_map_version = self._comm_map_version
            
            model.backbone.set_comm_maps(
                self._cached_device_send_map, self._cached_recv_ghost_indices,
                self.rank, self.world_size, self.interior_mask, self.enable_overlap,
                per_node_params=self.per_node_params
            )
            # num_local for the model = num_node_total (the model treats all node atoms as "local")
            model.backbone.num_local = num_node_total
            model.backbone.differentiable_exchange = self.differentiable_exchange
            if hasattr(model.backbone, 'set_profiler'):
                model.backbone.set_profiler(self.profiler)
            if self.measure_bw and hasattr(model.backbone, 'exchanger'):
                model.backbone.exchanger._measure_bw = True
        
        # Wire profiler to model heads
        if self.profiler and hasattr(model, 'output_heads'):
            for head in model.output_heads.values():
                if hasattr(head, 'profiler'):
                    head.profiler = self.profiler
        
        return combined_atoms, ghosts
    
    def _prepare_combined_atoms_default(self, atoms: ase.Atoms) -> Tuple[ase.Atoms, Dict[str, torch.Tensor]]:
        # 1. Exchange ghosts
        ghosts = self.exchanger.exchange(atoms)
        
        self.num_local = len(atoms)
        num_local = self.num_local
        
        # Update model comm maps (only rebuild if version changed)
        model = self.calculator.predictor.model
        if hasattr(model, "module"):
            model = model.module
        
        if hasattr(model, "backbone") and hasattr(model.backbone, "set_comm_maps"):
            if self._cached_comm_map_version != self._comm_map_version or self._cached_recv_ghost_indices is None:
                recv_ghost_indices = {}
                current_idx = num_local
                
                for r in range(self.world_size):
                    count = self.exchanger.recv_counts.get(r, 0)
                    if count > 0:
                        indices = torch.arange(current_idx, current_idx + count, device=self.exchanger.device)
                        recv_ghost_indices[r] = indices
                        current_idx += count
                
                device_send_map = {r: idx.to(self.exchanger.device) for r, idx in self.exchanger.comm_map.items()}
                
                self._cached_recv_ghost_indices = recv_ghost_indices
                self._cached_device_send_map = device_send_map
                self._cached_comm_map_version = self._comm_map_version
            
            model.backbone.set_comm_maps(self._cached_device_send_map, self._cached_recv_ghost_indices, self.rank, self.world_size, self.interior_mask, self.enable_overlap, self.hierarchical_params)
            model.backbone.num_local = num_local
            model.backbone.differentiable_exchange = self.differentiable_exchange
            # Wire profiler to model backbone for detailed timing
            if hasattr(model.backbone, 'set_profiler'):
                model.backbone.set_profiler(self.profiler)
            if self.measure_bw and hasattr(model.backbone, 'exchanger'):
                model.backbone.exchanger._measure_bw = True
        
        # Wire profiler to model heads for backward pass timing
        if self.profiler and hasattr(model, 'output_heads'):
            for head in model.output_heads.values():
                if hasattr(head, 'profiler'):
                    head.profiler = self.profiler
        
        # 2. Build combined atoms
        # OPTIMIZATION: Cache combined_atoms to avoid expensive object creation
        if not hasattr(self, "_cached_combined_atoms"):
             self._cached_combined_atoms = None
             self._cached_num_ghosts = -1
             self._cached_num_local = -1

        num_ghosts = 0
        if ghosts:
            ghost_pos = ghosts["pos"].cpu().numpy()
            ghost_Z = ghosts["Z"].cpu().numpy()
            num_ghosts = len(ghost_pos)

        # Check if cache is valid
        # We also need to check if atoms length changed (rebalance or just different atoms object)
        cache_valid = False
        if (self._cached_combined_atoms is not None and 
            self._cached_num_local == len(atoms) and 
            self._cached_num_ghosts == num_ghosts):
            cache_valid = True
        
        if cache_valid:
            combined_atoms = self._cached_combined_atoms
            # Update local part (in-place)
            combined_atoms.positions[:len(atoms)] = atoms.get_positions()
            # CRITICAL: Also update local atomic numbers (they can change after repartition!)
            combined_atoms.numbers[:len(atoms)] = atoms.get_atomic_numbers()
            # Cell/PBC usually constant in NVE/NVT but might change in NPT.
            # Safe to update them (cheap).
            combined_atoms.cell[:] = atoms.get_cell()
            
            # Update ghost part
            if num_ghosts > 0:
                combined_atoms.positions[len(atoms):] = ghost_pos
                # Z updates if needed (usually static, but safe to update)
                combined_atoms.numbers[len(atoms):] = ghost_Z
                
        else:
            # Rebuild cache
            if ghosts:
                combined_atoms = atoms.copy()
                combined_atoms.extend(ase.Atoms(numbers=ghost_Z, positions=ghost_pos))
                combined_atoms.set_cell(atoms.get_cell())
                combined_atoms.set_pbc(atoms.get_pbc())
            else:
                combined_atoms = atoms.copy()

            self._cached_combined_atoms = combined_atoms
            self._cached_num_local = len(atoms)
            self._cached_num_ghosts = num_ghosts
        
        return combined_atoms, ghosts
        

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        if atoms is None:
            atoms = self.atoms
        
        # Profile: Halo exchange (includes ghost data preparation)
        if self.profiler:
            self.profiler.start("halo_exchange")
        combined_atoms, ghosts = self._prepare_combined_atoms(atoms)
        if self.profiler:
            self.profiler.stop("halo_exchange")
        
        # Run calculation on combined atoms
        combined_atoms.calc = self.calculator
        # In per_node mode, edge indices are in node-wide coordinates [0, num_node_atoms).
        # The model filters edges by target < num_local, so we must pass the node total,
        # not the per-GPU count. Otherwise GPUs with my_start > 0 lose ALL their edges.
        if self.per_node_params is not None:
            self.calculator.num_local = self.num_node_atoms
        else:
            self.calculator.num_local = self.num_local
        
        # Note: Detailed profiling (atom2graph, predict, result_transfer) happens
        # inside FAIRChemCalculator.calculate() using self.calculator._profiler
        
        # Calculate forces
        all_forces_tensor = None
        if hasattr(self.calculator, "results_raw") and "forces" in self.calculator.results_raw:
            if self.calculator.calculation_required(combined_atoms, ["forces"]):
                self.calculator.calculate(combined_atoms, ["forces"], [])
            all_forces_tensor = self.calculator.results_raw.get("forces")

        if all_forces_tensor is None:
            all_forces = self.calculator.get_forces(combined_atoms)
            if isinstance(all_forces, np.ndarray):
                all_forces_tensor = torch.from_numpy(all_forces).to(self.exchanger.device)
            else:
                all_forces_tensor = all_forces
            
        # In per-node mode, all_forces_tensor has layout [all_node_atoms | inter_node_ghosts]
        # We need to extract forces for all node atoms first, then slice to get our GPU's portion
        if self.per_node_params is not None:
            num_forces_local = self.num_node_atoms  # All node atoms
        else:
            num_forces_local = self.num_local  # Per-GPU atoms
        local_forces = all_forces_tensor[:num_forces_local]

        # Per-node: each GPU computes partial forces (only from edges targeting
        # its atoms). Sum them across GPUs in the node for the node atoms portion.
        # IMPORTANT: Only all_reduce the node atoms portion, NOT the ghost forces.
        # Each GPU has a different number of inter-node ghosts, so all_forces_tensor
        # sizes differ per GPU. NCCL all_reduce requires identical sizes.
        # Ghost forces remain partial per-GPU; the receiver accumulates contributions
        # from all GPUs via index_add_ in reduce_forces_wait.
        if self.per_node_params is not None and self._intra_group is not None:
            dist.all_reduce(all_forces_tensor[:self.num_node_atoms], op=dist.ReduceOp.SUM, group=self._intra_group)

        # === Area 4: Overlap force_reduction with energy_allreduce ===
        # Both are independent collective operations. Start force reduction async,
        # compute energy while it runs, then wait for both.
        force_handle = None
        if self.world_size > 1:
            if self.profiler:
                self.profiler.start("force_reduction")
            ghost_forces = all_forces_tensor[num_forces_local:] if ghosts else None
            force_handle = self.exchanger.reduce_forces_start(local_forces, ghost_forces)

        # Calculate energy (overlaps with force_reduction all_to_all)
        # Try per-atom energies first; fall back to total energy from results_raw/results
        energies = None
        try:
            energies = self.calculator.get_potential_energies(combined_atoms)
        except (PropertyNotImplementedError, AttributeError):
            pass

        # Optimization #2+#3: Skip energy all-reduce on non-logging steps.
        # MD integration only needs forces; energy is for monitoring/logging.
        # Skipping saves ~7ms/step by removing an NCCL all_reduce from the
        # critical path, which also unblocks force_reduction on the NCCL stream.
        self._calc_step = getattr(self, '_calc_step', 0) + 1
        energy_interval = getattr(self, '_energy_allreduce_interval', 1)
        need_energy_allreduce = (energy_interval <= 1) or (self._calc_step % energy_interval == 0) or (self._calc_step == 1)

        if need_energy_allreduce and self.profiler:
            self.profiler.start("energy_allreduce")

        if energies is not None:
            # Per-atom energies available — slice to avoid double-counting
            if self.per_node_params is not None:
                pnp = self.per_node_params
                my_start = sum(pnp['node_atom_counts'][:pnp['local_gpu_id']])
                my_end = my_start + pnp['node_atom_counts'][pnp['local_gpu_id']]
                energy_slice = slice(my_start, my_end)
            else:
                energy_slice = slice(0, self.num_local)

            if isinstance(energies, torch.Tensor):
                local_energy_tensor = energies[energy_slice].sum().to(dtype=torch.float64)
                self.results['local_energy'] = local_energy_tensor.item()
                if need_energy_allreduce:
                    global_energy_tensor = local_energy_tensor.clone()
                    if dist.is_initialized():
                        dist.all_reduce(global_energy_tensor, op=dist.ReduceOp.SUM)
                    self.results['energy'] = global_energy_tensor.item()
                else:
                    # Store local energy as placeholder (not globally reduced)
                    self.results['energy'] = local_energy_tensor.item()
            else:
                local_energy_sum = np.sum(energies[energy_slice])
                self.results['local_energy'] = local_energy_sum
                if need_energy_allreduce:
                    if dist.is_initialized():
                        energy_tensor = torch.tensor([local_energy_sum], dtype=torch.float64, device=self.exchanger.device)
                        dist.all_reduce(energy_tensor, op=dist.ReduceOp.SUM)
                        self.results['energy'] = energy_tensor.item()
                    else:
                        self.results['energy'] = local_energy_sum
                else:
                    self.results['energy'] = local_energy_sum
        else:
            # Fallback: use total energy from inner calculator (already summed over all atoms)
            total_e = None
            if hasattr(self.calculator, 'results_raw') and 'energy' in self.calculator.results_raw:
                total_e = self.calculator.results_raw['energy']
            elif hasattr(self.calculator, 'results') and 'energy' in self.calculator.results:
                total_e = self.calculator.results['energy']

            if total_e is not None:
                if isinstance(total_e, torch.Tensor):
                    local_energy_tensor = total_e.to(dtype=torch.float64).view(-1)[0]
                else:
                    local_energy_tensor = torch.tensor([float(total_e)], dtype=torch.float64,
                                                        device=self.exchanger.device)
                if self.per_node_params is not None:
                    if self.per_node_params['local_gpu_id'] != 0:
                        local_energy_tensor = local_energy_tensor * 0

                self.results['local_energy'] = local_energy_tensor.item()
                if need_energy_allreduce:
                    global_energy_tensor = local_energy_tensor.clone()
                    if dist.is_initialized():
                        dist.all_reduce(global_energy_tensor, op=dist.ReduceOp.SUM)
                    self.results['energy'] = global_energy_tensor.item()
                else:
                    self.results['energy'] = local_energy_tensor.item()
            else:
                self.results['energy'] = 0.0
                self.results['local_energy'] = 0.0

        if need_energy_allreduce and self.profiler:
            self.profiler.stop("energy_allreduce")

        # Wait for force reduction to complete (if started)
        if force_handle is not None:
            local_forces = self.exchanger.reduce_forces_wait(force_handle)
            if self.profiler:
                self.profiler.stop("force_reduction")
        
        # In per-node mode, extract only this GPU's portion of the node forces
        if self.per_node_params is not None and self._intra_group is not None:
            pnp = self.per_node_params
            my_start = sum(pnp['node_atom_counts'][:pnp['local_gpu_id']])
            my_end = my_start + pnp['node_atom_counts'][pnp['local_gpu_id']]
            local_forces = local_forces[my_start:my_end]
            
        self.results["forces"] = local_forces.cpu().numpy()

    def get_potential_energies(self, atoms=None):
        # Delegate to the inner calculator
        # We assume atoms is the same as self.atoms (which is local_atoms)
        # The inner calculator has results for combined_atoms (local + ghosts)
        # We should return energies for all atoms in the inner calculator's results
        # The caller (verify_forces.py) will handle slicing/gathering.
        
        if self.calculation_required(atoms, ["energy"]):
             # We can't easily trigger a full distributed calc from here without atoms being passed correctly
             # But usually calculate() has already been called.
             pass
             
        # Check if inner calculator has energies
        if hasattr(self.calculator, "results") and "energies" in self.calculator.results:
            return self.calculator.results["energies"]
        else:
            raise PropertyNotImplementedError("energies property not implemented by the inner calculator")
