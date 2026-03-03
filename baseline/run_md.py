#!/usr/bin/env python3
"""
Serial (single-GPU) MD simulation script.
Counterpart to dist_md.py for comparison and debugging.
"""

import os
import time
import numpy as np
import torch
import ase
import ase.io
import ase.md
import ase.md.langevin
import ase.md.velocitydistribution
from ase import units
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.calculate.pretrained_mlip import get_predict_unit
from fairchem.core.units.mlip_unit.api.inference import inference_settings_turbo

# Import profiler from same directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from profiler import CUDAEventProfiler


# Configure torch inductor
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
# Force Inductor to use g++ instead of Cray Wrapper (CC) which causes linker errors
torch._inductor.config.cpp.cxx = ["g++"]


@dataclass
class MDConfig:
    cell_size: float = 30.0
    temperature: float = 850.0
    timestep: float = 0.1
    friction: float = 0.01
    total_steps: int = 1000
    steps_per_episode: int = 1
    traj_interval: int = 100
    log_interval: int = 100


@dataclass
class UMAConfig:
    model: str = "uma-s-1p1"
    activation_checkpointing: bool = True
    compile: bool = False
    wigner_cuda: bool = True
    merge_mole: bool = True
    cache_path: Optional[str] = None
    dynamo_cache_size: int = 64
    workers: int = 0 # number of workers for dataloader， 0 means use all available gpus


@dataclass
class ProfilerConfig:
    enabled: bool = False


@dataclass
class AppConfig:
    input_file: str = ""
    output_dir: str = "./results"
    prefix: str = "serial_md"
    md: MDConfig = field(default_factory=MDConfig)
    uma: UMAConfig = field(default_factory=UMAConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=AppConfig)


def run_serial_md(calc, cfg: AppConfig, profiler: CUDAEventProfiler = None) -> None:
    """
    Run serial (single-GPU) MD simulation.
    Saves trajectory data (positions, energies, forces) to NPZ files each step.
    Prints total energy each step.
    """
    # Setup output directory
    output_dir = os.path.join(cfg.output_dir, cfg.prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_path = os.path.join(output_dir, "trajectory.xyz")
    md_log_path = os.path.join(output_dir, "md.log")
    
    # Load atoms from input file
    atoms = ase.io.read(cfg.input_file)
    
    # Set cell if needed
    current_cell_lengths = atoms.get_cell().lengths()
    if not (current_cell_lengths > 0).all() and cfg.md.cell_size > 0:
        atoms.set_cell([cfg.md.cell_size] * 3)
    atoms.set_pbc([True] * 3)
    
    # Initialize velocities
    ase.md.velocitydistribution.MaxwellBoltzmannDistribution(
        atoms, temperature_K=cfg.md.temperature
    )
    
    # Set OMOL task info
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    
    # Attach calculator
    atoms.calc = calc
    
    # Simulation parameters
    timestep = cfg.md.timestep * units.fs
    friction = cfg.md.friction / units.fs
    
    print(f"Starting serial MD simulation with {len(atoms)} atoms")
    print(f"Temperature: {cfg.md.temperature} K, Timestep: {cfg.md.timestep} fs")
    print(f"Total steps: {cfg.md.total_steps}")
    print(f"Output directory: {output_dir}")
    
    # Save initial forces (before any MD steps) for force comparison
    init_energy = atoms.get_potential_energy()
    init_forces = atoms.get_forces()
    init_positions = atoms.get_positions()
    n_atoms = len(atoms)
    init_path = os.path.join(output_dir, "traj_step_init.npz")
    np.savez(init_path,
             step=-1,
             positions=init_positions,
             energy=init_energy,
             forces=init_forces,
             atomic_numbers=atoms.get_atomic_numbers(),
             global_id=np.arange(n_atoms, dtype=np.int64))
    print(f"Step 0 (initial): Total Energy = {init_energy:.4f} eV")

    # Run MD
    total_done = 0
    total_started = time.time()

    step = 0
    while step < cfg.md.total_steps:
        # Create dynamics integrator
        dyn = ase.md.langevin.Langevin(
            atoms, timestep, 
            temperature_K=cfg.md.temperature, 
            friction=friction
        )
        
        def output_cb():
            atoms.info["Step"] = total_done
            atoms.info["Time"] = total_done * (timestep / units.fs)
            atoms.info["T_K"] = atoms.get_temperature()
            atoms.write(output_file_path, append=True)
            
        dyn.attach(output_cb, interval=cfg.md.traj_interval)
        dyn.attach(ase.md.MDLogger(dyn, atoms, md_log_path), interval=cfg.md.log_interval)
        
        start_time = time.time()
        
        # Start profiling for this episode
        if profiler:
            profiler.start_episode()
        
        # Run step-by-step for profiling
        if profiler and profiler.enabled:
            for step_i in range(cfg.md.steps_per_episode):
                dyn.run(steps=1)
                profiler.step_done()
        else:
            dyn.run(steps=cfg.md.steps_per_episode)
        
        run_time = time.time() - start_time
        
        # Get and save trajectory data
        try:
            step_energy = atoms.get_potential_energy()
            step_forces = atoms.get_forces()
            step_positions = atoms.get_positions()
            
            # Save trajectory data to NPZ file
            traj_data_path = os.path.join(output_dir, f"traj_step_{total_done}.npz")
            n_atoms = len(atoms)
            np.savez(traj_data_path,
                     step=total_done,
                     positions=step_positions,
                     energy=step_energy,
                     forces=step_forces,
                     atomic_numbers=atoms.get_atomic_numbers(),
                     global_id=np.arange(n_atoms, dtype=np.int64))
            
            print(f"Step {total_done}: Total Energy = {step_energy:.4f} eV")
        except Exception as e:
            print(f"Step {total_done}: Failed to get energy/forces: {e}")
        
        step += cfg.md.steps_per_episode
        total_done += cfg.md.steps_per_episode
        print(f"Ran {cfg.md.steps_per_episode} steps in {run_time:.2f}s. (perf: {cfg.md.steps_per_episode/run_time:.2f} steps/s, {total_done}/{cfg.md.total_steps})")
        
        # Print profiling report for this episode
        if profiler:
            profiler.report()
    
    total_time = time.time() - total_started
    print(f"\nSerial MD completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average performance: {cfg.md.total_steps / total_time:.2f} steps/s")


@hydra.main(version_base=None, config_name="config")
def main(cfg: AppConfig) -> None:
    if not cfg.input_file:
        raise ValueError("input_file must be provided")
    if not os.path.exists(cfg.input_file):
        raise ValueError(f"input_file {cfg.input_file} does not exist")
    
    if cfg.uma.workers == 0:
        cfg.uma.workers = torch.cuda.device_count()
    print(f"Using {cfg.uma.workers} workers")
    
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Build calculator using pretrained_mlip API
    settings = inference_settings_turbo()
    settings.activation_checkpointing = bool(cfg.uma.activation_checkpointing)
    settings.merge_mole = bool(cfg.uma.merge_mole)
    settings.wigner_cuda = cfg.uma.wigner_cuda
    settings.compile = bool(cfg.uma.compile)
    
    print(f"Merge Mole: {settings.merge_mole}")
    print(f"Wigner CUDA: {settings.wigner_cuda}")
    print(f"Torch Compile: {settings.compile}")
    
    if settings.compile:
        torch._dynamo.config.cache_size_limit = cfg.uma.dynamo_cache_size
    
    print(f"[Setting up calculator ({cfg.uma.model})]")
    start_time = time.time()
    predictor = get_predict_unit(cfg.uma.model, device=device, inference_settings=settings, workers=cfg.uma.workers)
    calc = FAIRChemCalculator(predictor, task_name="omol")
    print(f"[Setting up calculator ({cfg.uma.model})] took {time.time() - start_time:.4f}s")
    
    # Setup profiler
    profiler = CUDAEventProfiler(enabled=cfg.profiler.enabled, rank=0)
    
    # Set profiler on predictor if it supports it (ParallelMLIPPredictUnit)
    if hasattr(predictor, 'set_profiler'):
        predictor.set_profiler(profiler)
    
    # Run MD
    run_serial_md(calc, cfg, profiler=profiler)


if __name__ == "__main__":
    main()
