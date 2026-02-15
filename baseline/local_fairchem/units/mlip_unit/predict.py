"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import TYPE_CHECKING, Protocol, Sequence

import hydra
import numpy as np
import torch
import torch.distributed as dist
from monty.dev import requires
from torch.distributed.elastic.utils.distributed import get_free_port
from torchtnt.framework import PredictUnit, State

from baseline.local_fairchem.common import gp_utils
from baseline.local_fairchem.common.distutils import (
    CURRENT_DEVICE_TYPE_STR,
    assign_device_for_local_rank,
    get_device_for_local_rank,
    setup_env_local_multi_gpu,
    broadcast,
)
from baseline.local_fairchem.datasets.atomic_data import AtomicData
from baseline.local_fairchem.units.mlip_unit import InferenceSettings
from baseline.local_fairchem.units.mlip_unit.utils import (
    load_inference_model,
    tf32_context_manager,
)

if TYPE_CHECKING:
    from baseline.local_fairchem.units.mlip_unit.mlip_unit import Task

try:
    import ray
    from ray import remote
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray_installed = True
except ImportError:
    ray = None

    def remote(cls):
        # dummy
        return cls

    ray_installed = False


def collate_predictions(predict_fn):
    @wraps(predict_fn)
    def collated_predict(
        predict_unit, data: AtomicData, undo_element_references: bool = True
    ):
        # Get the full prediction dictionary from the original predict method
        preds = predict_fn(predict_unit, data, undo_element_references)
        collated_preds = defaultdict(list)
        for i, dataset in enumerate(data.dataset):
            for task in predict_unit.dataset_to_tasks[dataset]:
                if task.level == "system":
                    collated_preds[task.property].append(
                        preds[task.name][i].unsqueeze(0)
                    )
                elif task.level == "atom":
                    collated_preds[task.property].append(
                        preds[task.name][data.batch == i]
                    )
                else:
                    raise RuntimeError(
                        f"Unrecognized task level={task.level} found in data batch at position {i}"
                    )
            
            # Manually collate atomic_energies if present
            if "atomic_energies" in preds:
                collated_preds["atomic_energies"].append(
                    preds["atomic_energies"][data.batch == i]
                )

        return {prop: torch.cat(val) for prop, val in collated_preds.items()}

    return collated_predict


class MLIPPredictUnitProtocol(Protocol):
    def predict(self, data: AtomicData, undo_element_references: bool) -> dict: ...

    @property
    def dataset_to_tasks(self) -> dict[str, list]: ...


def merge_uma_model(model, data):
    # merge the backbone
    model.backbone = model.backbone.merge_MOLE_model(data)

    # merge any heads
    new_output_heads = torch.nn.ModuleDict()
    for head_name, head in model.output_heads.items():
        if hasattr(head, "merge_MOLE_model"):
            new_output_heads[head_name] = head.merge_MOLE_model(data)
        else:
            new_output_heads[head_name] = head
    model.output_heads = new_output_heads


class MLIPPredictUnit(PredictUnit[AtomicData], MLIPPredictUnitProtocol):
    def __init__(
        self,
        inference_model_path: str,
        device: str = "cpu",
        overrides: dict | None = None,
        inference_settings: InferenceSettings | None = None,
        seed: int = 41,
        atom_refs: dict | None = None,
        assert_on_nans: bool = False,
    ):
        import os
        _rank = os.environ.get("SLURM_PROCID", "?")
        # print(f"[DEBUG] Rank {_rank}: MLIPPredictUnit.__init__ START, file={__file__}", flush=True)
        
        super().__init__()
        device = str(device)
        os.environ[CURRENT_DEVICE_TYPE_STR] = device
        # print(f"[DEBUG] Rank {_rank}: device={device}, inference_model_path={inference_model_path}", flush=True)

        # print(f"[DEBUG] Rank {_rank}: set_seed({seed})...", flush=True)
        self.set_seed(seed)
        # note these are different from the element references used for model training
        self.atom_refs = (
            {task.replace("_elem_refs", ""): refs for task, refs in atom_refs.items()}
            if atom_refs is not None
            else {}
        )

        if inference_settings is None:
            inference_settings = InferenceSettings()
        if inference_settings.torch_num_threads is not None:
            torch.set_num_threads(inference_settings.torch_num_threads)
            torch.set_num_interop_threads(inference_settings.torch_num_threads)

        if overrides is None:
            overrides = {}
        if "backbone" not in overrides:
            overrides["backbone"] = {}
        # always disable always_use_pbc for inference
        overrides["backbone"]["always_use_pbc"] = False
        if inference_settings.activation_checkpointing is not None:
            overrides["backbone"]["activation_checkpointing"] = (
                inference_settings.activation_checkpointing
            )
        if inference_settings.edge_chunk_size is not None:
            overrides["backbone"]["edge_chunk_size"] = (
                inference_settings.edge_chunk_size
            )
        if inference_settings.external_graph_gen is not None:
            overrides["backbone"][
                "otf_graph"
            ] = not inference_settings.external_graph_gen

        if inference_settings.internal_graph_gen_version is not None:
            overrides["backbone"]["radius_pbc_version"] = (
                inference_settings.internal_graph_gen_version
            )

        if inference_settings.wigner_cuda:
            logging.warning(
                "The wigner_cuda flag is deprecated and will be removed in future versions."
            )

        # print(f"[DEBUG] Rank {_rank}: Calling load_inference_model...", flush=True)
        self.model, checkpoint = load_inference_model(
            inference_model_path, use_ema=True, overrides=overrides
        )
        # print(f"[DEBUG] Rank {_rank}: load_inference_model completed", flush=True)
        
        # print(f"[DEBUG] Rank {_rank}: Instantiating tasks from checkpoint...", flush=True)
        tasks = [
            hydra.utils.instantiate(task_config)
            for task_config in checkpoint.tasks_config
        ]
        self.tasks = {t.name: t for t in tasks}
        # print(f"[DEBUG] Rank {_rank}: Tasks instantiated: {list(self.tasks.keys())}", flush=True)

        self._dataset_to_tasks = get_dataset_to_tasks_map(self.tasks.values())
        assert set(self._dataset_to_tasks.keys()).issubset(
            set(self.model.module.backbone.dataset_list)
        ), "Datasets in tasks is not a strict subset of datasets in backbone."
        
        is_cuda = device.startswith("cuda")
        assert device == "cpu" or is_cuda, "device must be 'cpu' or 'cuda'"

        if is_cuda:
             if ":" in device:
                 self.device = device
             else:
                 self.device = get_device_for_local_rank()
        else:
             self.device = "cpu"
        # print(f"[DEBUG] Rank {_rank}: Final device={self.device}", flush=True)

        # print(f"[DEBUG] Rank {_rank}: Calling model.eval()...", flush=True)
        self.model.eval()
        # print(f"[DEBUG] Rank {_rank}: model.eval() completed", flush=True)

        self.lazy_model_intialized = False
        self.inference_settings = inference_settings

        # store composition embedding of system the model was merged on
        self.merged_on = None

        self.assert_on_nans = assert_on_nans

        if self.direct_forces:
            logging.warning(
                "This is a direct-force model. Direct force predictions may lead to discontinuities in the potential "
                "energy surface and energy conservation errors."
            )
        
        self._dataset_cache = {}
        # print(f"[DEBUG] Rank {_rank}: MLIPPredictUnit.__init__ END", flush=True)

    @property
    def direct_forces(self) -> bool:
        return self.model.module.backbone.direct_forces

    @property
    def dataset_to_tasks(self) -> dict[str, list]:
        return self._dataset_to_tasks

    def set_seed(self, seed: int):
        logging.debug(f"Setting random seed to {seed}")
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def move_to_device(self):
        self.model.to(self.device)
        for task in self.tasks.values():
            task.normalizer.to(self.device)
            if task.element_references is not None:
                task.element_references.to(self.device)
        
        # Populate Cache
        self._dataset_cache = {}
        for task_name, task in self.tasks.items():
            if task.element_references is not None:
                for dset in task.datasets:
                    if dset not in self._dataset_cache:
                        cache_entry = {"refs": task.element_references.element_references.to(self.device)}
                        
                        # Cache Normalizer stats
                        if task.normalizer is not None:
                            if hasattr(task.normalizer, "rmsd"):
                                val = task.normalizer.rmsd
                                if isinstance(val, torch.Tensor): val = val.to(self.device)
                                cache_entry["rmsd"] = val
                            elif hasattr(task.normalizer, "std"):
                                val = task.normalizer.std
                                if isinstance(val, torch.Tensor): val = val.to(self.device)
                                cache_entry["rmsd"] = val # Use rmsd key for consistent access
                            
                            if hasattr(task.normalizer, "mean"):
                                val = task.normalizer.mean
                                if isinstance(val, torch.Tensor): val = val.to(self.device)
                                cache_entry["mean"] = val
                        
                        self._dataset_cache[dset] = cache_entry

    def predict_step(self, state: State, data: AtomicData) -> dict[str, torch.tensor]:
        return self.predict(data)

    def get_composition_charge_spin_dataset(self, data):
        composition_sum = data.atomic_numbers.new_zeros(
            self.model.module.backbone.max_num_elements,
            dtype=torch.int,
        ).index_add(
            0,
            data.atomic_numbers.to(torch.int),
            data.atomic_numbers.new_ones(data.atomic_numbers.shape[0], dtype=torch.int),
        )
        comp_charge_spin = (
            composition_sum,
            getattr(data, "charge", None),
            getattr(data, "spin", None),
        )
        return comp_charge_spin, getattr(data, "dataset", [None])

    @collate_predictions
    def predict(
        self, data: AtomicData, undo_element_references: bool = True
    ) -> dict[str, torch.tensor]:
        if not self.lazy_model_intialized:
            # merge everything on CPU
            if self.inference_settings.merge_mole:
                # replace backbone with non MOE version
                assert (
                    data.natoms.numel() == 1
                ), f"Cannot merge model with multiple systems in batch. Must be exactly 1 system, found {data.natoms.numel()}"

                merge_uma_model(self.model.module, data.clone())
            
            self.move_to_device()
            if hasattr(self.inference_settings, 'compile') and self.inference_settings.compile:
                logging.warning(
                    "Model is being compiled this might take a while for the first time"
                )
                
                # Distributed compile cache sharing using save_cache_artifacts/load_cache_artifacts
                # API: save_cache_artifacts() -> tuple[bytes, CacheInfo], use [0] for bytes
                #      load_cache_artifacts(bytes) -> CacheInfo
                cache_path = getattr(self.inference_settings, 'compile_cache_path', None)
                rank = dist.get_rank() if dist.is_initialized() else 0
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                
                if cache_path and world_size > 1:
                    import os
                    cache_dir = os.path.dirname(cache_path)
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                    
                    # Check if PyTorch version supports save/load cache artifacts (>= 2.6)
                    has_cache_api = hasattr(torch.compiler, 'save_cache_artifacts') and hasattr(torch.compiler, 'load_cache_artifacts')
                    
                    if has_cache_api:
                        # Check if cache already exists (from previous run)
                        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                            logging.info(f"Rank {rank}: Found existing compile cache, loading from {cache_path}")
                            try:
                                with open(cache_path, 'rb') as f:
                                    cache_bytes = f.read()
                                torch.compiler.load_cache_artifacts(cache_bytes)
                                logging.info(f"Rank {rank}: Compile cache loaded successfully")
                            except Exception as e:
                                logging.warning(f"Rank {rank}: Failed to load existing cache: {e}")
                            
                            # All ranks compile with loaded cache (should be fast)
                            self.model = torch.compile(self.model, dynamic=True)
                            self._cache_saved = True  # Skip save since we loaded existing cache
                        elif rank == 0:
                            # Rank 0 compiles first (no existing cache)
                            logging.info(f"Rank 0: Compiling model (no existing cache)...")
                            self.model = torch.compile(self.model, dynamic=True)
                            
                            # Store cache path for saving after first forward pass
                            self._compile_cache_path = cache_path
                            self._cache_saved = False
                        else:
                            # Other ranks wait for cache to be available
                            # They will compile after first barrier (triggered by rank 0's first inference)
                            self._waiting_for_compile_cache = True
                            self._compile_cache_path = cache_path
                            self._cache_saved = False  # Ensure they enter cache handling block
                    else:
                        # Fallback for older PyTorch: use TORCHINDUCTOR_CACHE_DIR env var only
                        logging.info(f"Rank {rank}: Using TORCHINDUCTOR_CACHE_DIR for cache sharing (older PyTorch)")
                        self.model = torch.compile(self.model, dynamic=True)
                else:
                    # Single rank or no cache path specified
                    self.model = torch.compile(self.model, dynamic=True)
            self.lazy_model_intialized = True

        # OPTIMIZATION: Skip .clone() when merge_mole=True (MOLE experts are merged, no in-place modification)
        # The clone is only needed for MOLE routing which modifies data in-place
        if hasattr(self.inference_settings, 'merge_mole') and self.inference_settings.merge_mole:
            data_device = data.to(self.device)  # No clone needed
        else:
            # Original: needs clone to avoid issues with graph parallel modifying this data with MOLE
            data_device = data.to(self.device).clone()
             
        # Force move all properties to device (workaround for AtomicData.to issue)
        # Force move all properties to device (workaround for AtomicData.to issue)
        # Try a list of potential keys
        potential_keys = ['pos', 'cell', 'atomic_numbers', 'batch', 'natoms', 'charge', 'spin', 'dataset', 'edge_index', 'edge_weight']
        for key in potential_keys:
            # Attribute access
            if hasattr(data_device, key):
                val = getattr(data_device, key)
                if isinstance(val, torch.Tensor):
                     setattr(data_device, key, val.to(self.device))
            # Item access (for PyG/Dict compatibility)
            try:
                if isinstance(data_device, dict) or hasattr(data_device, '__getitem__'):
                     if key in data_device:
                         val = data_device[key]
                         if isinstance(val, torch.Tensor):
                             data_device[key] = val.to(self.device)
            except Exception:
                pass
                
        # Optimization: Removed generic dir() iteration which is very slow
        # The explicit list above should cover all standard FairChem properties.


        if self.inference_settings.merge_mole:

            if self.merged_on is None:
                # only get embeddings after moved to final device to get right types
                self.merged_on = self.get_composition_charge_spin_dataset(data_device)
            else:
                this_sys = self.get_composition_charge_spin_dataset(data_device)
                assert (
                    data_device.natoms.numel() == 1
                ), f"Cannot run merged model on batch with multiple systems. Must be exactly 1 system, found {data_device.natoms.numel()}"

                # Normalize compositions by total number of atoms to allow same reduced composition
                merged_comp = self.merged_on[0][0].float()
                this_comp = this_sys[0][0].float()
                merged_comp_norm = merged_comp / merged_comp.sum()
                this_comp_norm = this_comp / this_comp.sum()

                # NOTE: In distributed inference with repartition, local+ghosts composition
                # may differ from global composition. This is expected and safe because
                # MOLE weights are merged based on global composition.
                # Skip strict check; only warn if significantly different.
                if False and not merged_comp_norm.isclose(this_comp_norm, rtol=0.1).all():
                    logging.warning(
                        f"Composition differs from merged model: merged={merged_comp_norm.nonzero()}, "
                        f"current={this_comp_norm.nonzero()}. This is expected in distributed inference."
                    )
                assert (
                    self.merged_on[0][1] == this_sys[0][1]
                ), f"Cannot run on merged model on system. Charge is different {self.merged_on[0][1]} vs {this_sys[0][1]}"
                assert (
                    self.merged_on[0][2] == this_sys[0][2]
                ), f"Cannot run on merged model on system. Spin is different {self.merged_on[0][2]} vs {this_sys[0][2]}"
                assert (
                    self.merged_on[1] == this_sys[1]
                ), f"Cannot run on merged model on system. Dataset is different {self.merged_on[1]} vs {this_sys[1]}"

        inference_context = torch.no_grad() if self.direct_forces else nullcontext()
        tf32_context = (
            tf32_context_manager() if self.inference_settings.tf32 else nullcontext()
        )

        pred_output = {}
        with inference_context, tf32_context:
            output = self.model(data_device)
            for task_name, task in self.tasks.items():
                pred_output[task_name] = task.normalizer.denorm(
                    output[task_name][task.property]
                )
                if self.assert_on_nans:
                    assert torch.isfinite(
                        pred_output[task_name]
                    ).all(), f"NaNs/Infs found in prediction for task {task_name}.{task.property}"
                if undo_element_references and task.element_references is not None:
                    pred_output[task_name] = task.element_references.undo_refs(
                        data_device, pred_output[task_name]
                    )
                
                # Propagate atomic energies if available
                if "atomic_energies" in output[task_name]:
                    pred_output["atomic_energies"] = output[task_name]["atomic_energies"]
                
        # Scan for orphaned atomic_energies keys (from DatasetSpecificSingleHeadWrapper)
        # Priority: omol > oc20 > others. Skip NaN datasets.
        if "atomic_energies" not in pred_output:
            # Look for omol first
            preferred_order = ["omol", "oc20", "omat", "odac", "omc"]
            for dataset_name in preferred_order:
                key = f"{dataset_name}_atomic_energies"
                if key in output and "atomic_energies" in output[key]:
                    atomic_energies = output[key]["atomic_energies"]
                    
                    # Skip if contains NaN
                    if torch.isnan(atomic_energies).any():
                        continue
                    
                    # Ensure atomic_energies is 1D for addition
                    if atomic_energies.dim() > 1:
                        atomic_energies = atomic_energies.squeeze()
                    
                    if dataset_name in self._dataset_cache:
                        cache = self._dataset_cache[dataset_name]
                        try:
                            # Apply Norm (Scale)
                            if "rmsd" in cache:
                                atomic_energies = atomic_energies * cache["rmsd"]
                            
                            # Apply Mean
                            if "mean" in cache:
                                mean_val = cache["mean"]
                                if isinstance(mean_val, float) and mean_val != 0.0:
                                     atomic_energies = atomic_energies + (mean_val / data.natoms.sum())
                                elif isinstance(mean_val, torch.Tensor) and mean_val.abs().sum() > 1e-6:
                                     atomic_energies = atomic_energies + (mean_val / data.natoms.sum())

                            # add refs
                            refs = cache["refs"][data.atomic_numbers.long()]
                            atomic_energies = atomic_energies + refs
                        except Exception:
                            pass
                    
                    pred_output["atomic_energies"] = atomic_energies
                    break  # Use first valid dataset

        # Recompute total energy if num_local is present (Distributed Mode)
        if hasattr(data, "num_local") and "atomic_energies" in pred_output:
             num_local = data.num_local
             try:
                 if isinstance(num_local, torch.Tensor):
                     num_local = int(num_local.item())
                 elif isinstance(num_local, (list, tuple)):
                     num_local = int(num_local[0])
                 elif hasattr(num_local, "item"):
                     num_local = int(num_local.item())
                 else:
                     num_local = int(num_local)
             except Exception:
                 num_local = None
             
             if num_local is not None:
                 energies = pred_output["atomic_energies"]
                 if energies.dim() > 1: energies = energies.squeeze()
                 
                 # Sum only first num_local
                 local_energy = energies[:num_local].sum()

                 # Overwrite task output if task property is energy
                 for task_name, task in self.tasks.items():
                     if task.property == "energy":
                          pred_output[task_name] = local_energy.view(1)

        # Save torch.compile cache after first inference (rank 0 only)
        # Then allow other ranks to load cache and compile
        if hasattr(self, '_compile_cache_path') and not getattr(self, '_cache_saved', True):
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                try:
                    logging.info(f"Rank 0: Saving compile cache to {self._compile_cache_path}")
                    # save_cache_artifacts() returns tuple[bytes, CacheInfo], use [0] for bytes
                    artifacts = torch.compiler.save_cache_artifacts()
                    if artifacts is not None:
                        cache_bytes = artifacts[0]
                        with open(self._compile_cache_path, 'wb') as f:
                            f.write(cache_bytes)
                        logging.info(f"Rank 0: Compile cache saved successfully ({len(cache_bytes)} bytes)")
                    else:
                        logging.warning("Rank 0: save_cache_artifacts() returned None")
                except Exception as e:
                    logging.warning(f"Rank 0: Failed to save compile cache: {e}")
            self._cache_saved = True
            
            # Barrier to ensure cache is saved before other ranks proceed
            if dist.is_initialized():
                dist.barrier()
            
            # Now other ranks can load cache and compile
            if rank != 0 and getattr(self, '_waiting_for_compile_cache', False):
                import os
                if os.path.exists(self._compile_cache_path) and os.path.getsize(self._compile_cache_path) > 0:
                    logging.info(f"Rank {rank}: Loading compile cache from {self._compile_cache_path}")
                    try:
                        with open(self._compile_cache_path, 'rb') as f:
                            cache_bytes = f.read()
                        torch.compiler.load_cache_artifacts(cache_bytes)
                        logging.info(f"Rank {rank}: Compile cache loaded successfully")
                    except Exception as e:
                        logging.warning(f"Rank {rank}: Failed to load cache: {e}")
                else:
                    logging.warning(f"Rank {rank}: Cache file not found or empty, compiling without cache")
                
                logging.info(f"Rank {rank}: Compiling model...")
                self.model = torch.compile(self.model, dynamic=True)
                self._waiting_for_compile_cache = False
            
            # Final barrier to ensure all ranks are done compiling before proceeding
            if dist.is_initialized():
                dist.barrier()

        return pred_output


def get_dataset_to_tasks_map(tasks: Sequence[Task]) -> dict[str, list[Task]]:
    """Create a mapping from dataset names to their associated tasks.

    Args:
        tasks: A sequence of Task objects to be organized by dataset

    Returns:
        A dictionary mapping dataset names (str) to lists of Task objects
        that are associated with that dataset
    """
    dset_to_tasks_map = defaultdict(list)
    for task in tasks:
        for dataset_name in task.datasets:
            dset_to_tasks_map[dataset_name].append(task)
    return dict(dset_to_tasks_map)


def move_tensors_to_cpu(data):
    """
    Recursively move all PyTorch tensors in a nested data structure to CPU.

    Args:
        data: Input data structure (dict, list, tuple, tensor, or other)

    Returns:
        Data structure with all tensors moved to CPU
    """
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: move_tensors_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_cpu(item) for item in data)
    else:
        # Return as-is for non-tensor types (str, int, float, etc.)
        return data


class MLIPWorkerLocal:
    def __init__(
        self,
        worker_id: int,
        world_size: int,
        predictor_config: dict,
        master_port: int | None = None,
        master_address: str | None = None,
    ):
        if ray_installed is False:
            raise RuntimeError("Requires `ray` to be installed")

        self.worker_id = worker_id
        self.world_size = world_size
        self.predictor_config = predictor_config
        self.master_address = (
            ray.util.get_node_ip_address() if master_address is None else master_address
        )
        self.master_port = get_free_port() if master_port is None else master_port
        self.is_setup = False
        self.last_received_atomic_data = None

    def get_master_address_and_port(self):
        return (self.master_address, self.master_port)

    def get_device_for_local_rank(self):
        return get_device_for_local_rank()

    def _distributed_setup(
        self,
    ):
        # initialize distributed environment
        # TODO, this wont work for multi-node, need to fix master addr
        logging.info(f"Initializing worker {self.worker_id}...")
        setup_env_local_multi_gpu(self.worker_id, self.master_port, self.master_address)

        device = self.predictor_config.get("device", "cpu")
        assign_device_for_local_rank(device == "cpu", 0)
        backend = "gloo" if device == "cpu" else "nccl"
        dist.init_process_group(
            backend=backend,
            rank=self.worker_id,
            world_size=self.world_size,
        )
        gp_utils.setup_graph_parallel_groups(self.world_size, backend)
        self.predict_unit = hydra.utils.instantiate(self.predictor_config)
        self.device = get_device_for_local_rank()
        logging.info(
            f"Worker {self.worker_id}, gpu_id: {ray.get_gpu_ids()}, loaded predict unit: {self.predict_unit}, "
            f"on port {self.master_port}, with device: {self.device}, config: {self.predictor_config}"
        )
        self.is_setup = True

    def predict(
        self, data: AtomicData, use_nccl: bool = False
    ) -> dict[str, torch.tensor] | None:
        if not self.is_setup:
            self._distributed_setup()

        out = self.predict_unit.predict(data)
        if self.worker_id == 0:
            return move_tensors_to_cpu(out)

        if self.worker_id != 0 and use_nccl:
            self.last_received_atomic_data = data.to(self.device)
            while True:
                broadcast(self.last_received_atomic_data.pos, src=0)
                self.predict_unit.predict(self.last_received_atomic_data)

        return None


@remote
class MLIPWorker(MLIPWorkerLocal):
    pass


@requires(ray_installed, message="Requires `ray` to be installed")
class ParallelMLIPPredictUnit(MLIPPredictUnitProtocol):
    def __init__(
        self,
        inference_model_path: str,
        device: str = "cpu",
        overrides: dict | None = None,
        inference_settings: InferenceSettings | None = None,
        seed: int = 41,
        atom_refs: dict | None = None,
        assert_on_nans: bool = False,
        num_workers: int = 1,
        num_workers_per_node: int = 8,
        log_level: int = logging.INFO,
    ):
        super().__init__()
        _mlip_pred_unit = MLIPPredictUnit(
            inference_model_path=inference_model_path,
            device="cpu",
            overrides=overrides,
            inference_settings=inference_settings,
            seed=seed,
            atom_refs=atom_refs,
        )
        self.inference_settings = inference_settings
        self._dataset_to_tasks = copy.deepcopy(_mlip_pred_unit.dataset_to_tasks)

        predict_unit_config = {
            "_target_": "baseline.local_fairchem.units.mlip_unit.predict.MLIPPredictUnit",
            "inference_model_path": inference_model_path,
            "device": device,
            "overrides": overrides,
            "inference_settings": inference_settings,
            "seed": seed,
            "atom_refs": atom_refs,
            "assert_on_nans": assert_on_nans,
        }

        logging.basicConfig(
            level=log_level,
            force=True,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
        )
        # Optional: keep Ray/uvicorn chatty logs in check
        logging.getLogger("ray").setLevel(log_level)
        logging.getLogger("uvicorn").setLevel(log_level)
        if not ray.is_initialized():
            # in CI envrionment, we want to control the number of CPUs allocated to limit the pool of IDLE ray workers
            if os.environ.get("CI"):
                logging.info(
                    f"CI environment detected, initializing ray with limited CPUs: {num_workers_per_node}"
                )
                ray.init(
                    logging_level=log_level,
                    num_cpus=num_workers_per_node,
                    runtime_env={
                        "env_vars": {"RAY_DEBUG": "1"},
                    },
                )
            else:
                ray.init(logging_level=log_level)

        self.atomic_data_on_device = None

        num_nodes = math.ceil(num_workers / num_workers_per_node)
        num_workers_on_node_array = [num_workers_per_node] * num_nodes
        if num_workers % num_workers_per_node > 0:
            num_workers_on_node_array[-1] = num_workers % num_workers_per_node
        logging.info(
            f"Creating placement groups with {num_workers_on_node_array} workers on {device}"
        )

        # first create one placement group for each node
        num_gpu_per_worker = 1 if device == "cuda" else 0
        placement_groups = []
        for workers in num_workers_on_node_array:
            bundle = {"CPU": workers}
            if device == "cuda":
                bundle["GPU"] = workers
            pg = ray.util.placement_group([bundle], strategy="STRICT_PACK")
            placement_groups.append(pg)
        ray.get(pg.ready())  # Wait for each placement group to be scheduled

        # Need to still place worker to occupy space, otherwise ray double books this GPU
        rank0_worker = MLIPWorker.options(
            num_gpus=num_gpu_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_groups[0],
                placement_group_bundle_index=0,  # Use the first (and only) bundle in the PG
                placement_group_capture_child_tasks=True,  # Ensure child tasks also run in this PG
            ),
        ).remote(0, num_workers, predict_unit_config)

        local_gpu_or_cpu = ray.get(rank0_worker.get_device_for_local_rank.remote())
        os.environ[CURRENT_DEVICE_TYPE_STR] = local_gpu_or_cpu

        self.workers = []
        self.local_rank0 = MLIPWorkerLocal(
            worker_id=0,
            world_size=num_workers,
            predictor_config=predict_unit_config,
        )
        master_addr, master_port = self.local_rank0.get_master_address_and_port()
        logging.info(f"Started rank0 on {master_addr}:{master_port}")

        # next place all ranks in order and pack them on placement groups
        # ie: rank0-7 -> placement group 0, 8->15 -> placement group 1 etc.
        worker_id = 0
        for pg_idx, pg in enumerate(placement_groups):
            workers = num_workers_on_node_array[pg_idx]
            logging.info(
                f"Launching workers for placement group {pg_idx} (Node {pg_idx}), workers={workers}"
            )

            for i in range(workers):
                # skip the first one because it's already been initialized above
                if pg_idx == 0 and i == 0:
                    worker_id += 1
                    continue
                # Each actor requests 1 worker worth of resources and uses the specific placement group
                actor = MLIPWorker.options(
                    num_gpus=num_gpu_per_worker,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=0,  # Use the first (and only) bundle in the PG
                        placement_group_capture_child_tasks=True,  # Ensure child tasks also run in this PG
                    ),
                ).remote(
                    worker_id,
                    num_workers,
                    predict_unit_config,
                    master_port,
                    master_addr,
                )
                self.workers.append(actor)
                worker_id += 1

    def predict(self, data: AtomicData) -> dict[str, torch.tensor]:
        # put the reference in the object store only once
        if not self.inference_settings.merge_mole or self.atomic_data_on_device is None:
            data_ref = ray.put(data)
            # this will put the ray works into an infinite loop listening for broadcasts
            _futures = [
                w.predict.remote(data_ref, use_nccl=self.inference_settings.merge_mole)
                for w in self.workers
            ]
            self.atomic_data_on_device = data.clone()
        else:
            self.atomic_data_on_device.pos = data.pos.to(self.local_rank0.device)
            broadcast(self.atomic_data_on_device.pos, src=0)

        return self.local_rank0.predict(self.atomic_data_on_device)

    @property
    def dataset_to_tasks(self) -> dict[str, list]:
        return self._dataset_to_tasks
