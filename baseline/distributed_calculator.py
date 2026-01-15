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
                 recv_ghost_indices: Dict[int, torch.Tensor] = None):
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
        
        # OPTIMIZATION: Cache comm maps to avoid rebuilding every step
        self._cached_recv_ghost_indices = None
        self._cached_device_send_map = None
        self._cached_comm_map_version = -1  # Version counter to invalidate cache on update
        self._comm_map_version = 0
        
        # OPTIMIZATION: Cache atomic numbers (they don't change during MD)
        self._cached_local_Z_tensor = None
        
        # Profiler (optional)
        self.profiler = None
        
    def set_profiler(self, profiler: StepProfiler):
        """Set profiler for detailed timing analysis."""
        self.profiler = profiler
        # Also pass profiler to inner FAIRChemCalculator
        if hasattr(self.calculator, '_profiler'):
            self.calculator._profiler = profiler
        else:
            self.calculator._profiler = profiler
        
    def update_comm_map(self, comm_map: Dict[int, torch.Tensor], interior_mask: torch.Tensor = None):
        self.exchanger = HaloExchanger(self.rank, self.world_size, comm_map, self.exchanger.device)
        # Update interior mask
        if interior_mask is not None:
            self.interior_mask = interior_mask.to(self.exchanger.device)
        # OPTIMIZATION: Invalidate cached comm maps
        self._comm_map_version += 1

    def _prepare_combined_atoms(self, atoms: ase.Atoms) -> Tuple[ase.Atoms, Dict[str, torch.Tensor]]:
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
            
            model.backbone.set_comm_maps(self._cached_device_send_map, self._cached_recv_ghost_indices, self.rank, self.world_size, self.interior_mask, self.enable_overlap)
            model.backbone.num_local = num_local
            # Wire profiler to model backbone for detailed timing
            if hasattr(model.backbone, 'set_profiler'):
                model.backbone.set_profiler(self.profiler)
        
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
            
        local_forces = all_forces_tensor[:self.num_local]
        
        # Profile: Force reduction
        if ghosts:
            if self.profiler:
                self.profiler.start("force_reduction")
            ghost_forces = all_forces_tensor[self.num_local:]
            local_forces = self.exchanger.reduce_forces(local_forces, ghost_forces)
            if self.profiler:
                self.profiler.stop("force_reduction")
            
        self.results["forces"] = local_forces.cpu().numpy()
        
        # Calculate energy
        energies = self.calculator.get_potential_energies(combined_atoms)
        
        # Profile: Energy all-reduce
        if self.profiler:
            self.profiler.start("energy_allreduce")
        
        if isinstance(energies, torch.Tensor):
            local_energy_tensor = energies[:self.num_local].sum().to(dtype=torch.float64)
            global_energy_tensor = local_energy_tensor.clone()
            if dist.is_initialized():
                dist.all_reduce(global_energy_tensor, op=dist.ReduceOp.SUM)
            
            self.results['energy'] = global_energy_tensor.item()
            self.results['local_energy'] = local_energy_tensor.item()
        else:
            local_energy_sum = np.sum(energies[:self.num_local])
            self.results['local_energy'] = local_energy_sum
            if dist.is_initialized():
                energy_tensor = torch.tensor([local_energy_sum], dtype=torch.float64, device=self.exchanger.device)
                dist.all_reduce(energy_tensor, op=dist.ReduceOp.SUM)
                self.results['energy'] = energy_tensor.item()
            else:
                self.results['energy'] = local_energy_sum
        
        if self.profiler:
            self.profiler.stop("energy_allreduce")

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
