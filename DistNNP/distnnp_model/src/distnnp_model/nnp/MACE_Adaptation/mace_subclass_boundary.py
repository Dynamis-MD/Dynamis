"""Explicit MACE subclass boundary for distributed DistMACE adaptation.

Step 2 establishes a dedicated internal home for distributed-aware MACE
behavior so the adapter can stay focused on DistNNP orchestration while the
chosen interaction boundary owns the model-internal exchange seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from distnnp_model.comm.exchanger import GPUDirectExchanger
from distnnp_model.comm.feature_push import FeaturePushToExchange
from distnnp_state import DistNNPState


@dataclass
class DistMACEExecutionContext:
    """Runtime state shared between the adapter and the model subclass seam."""

    distributed_state: Optional[DistNNPState] = None
    exchange_mode: str = "uninitialized"
    learned_feature_exchanger: Optional[GPUDirectExchanger] = None
    observation: Optional[Dict[str, Any]] = None


class DistMACEModelSubclassBoundary:
    """Install a dedicated subclass seam on the selected MACE interaction block."""

    model_contract = "subclass_first_mace_internal_adaptation"
    model_status = "installed"
    ownership_contract = "adapter_owned_energy_head"

    def __init__(self, feature_exchange_policy) -> None:
        self.feature_exchange_policy = feature_exchange_policy
        self.execution_context = DistMACEExecutionContext()
        self._installed_model_class_name: Optional[str] = None
        self._installed_interaction_class_name: Optional[str] = None
        self._installed_interaction_index: Optional[int] = None

    @staticmethod
    def _ensure_model_subclass(model) -> str:
        model_cls = model.__class__
        if getattr(model_cls, "__distmace_subclass__", False):
            return model_cls.__name__

        subclass_name = f"DistMACE{model_cls.__name__}"
        subclass_cls = type(
            subclass_name,
            (model_cls,),
            {
                "__distmace_subclass__": True,
                "__distmace_subclass_role__": "model_root",
            },
        )
        model.__class__ = subclass_cls
        return subclass_cls.__name__

    def install(self, model) -> None:
        interactions = getattr(model, "interactions", None)
        if interactions is None:
            raise AttributeError(
                "Loaded MACE model does not expose an 'interactions' stack; "
                "cannot install the Step 2 subclass boundary."
            )

        layer_index = int(self.feature_exchange_policy.interaction_layer)
        if layer_index < 0 or layer_index >= len(interactions):
            raise ValueError(
                f"feature_exchange_layer={layer_index} is out of range for "
                f"{len(interactions)} interaction blocks."
            )

        target_block = interactions[layer_index]
        base_cls = target_block.__class__
        if getattr(base_cls, "__distmace_subclass__", False):
            self._installed_model_class_name = model.__class__.__name__
            self._installed_interaction_class_name = base_cls.__name__
            self._installed_interaction_index = layer_index
            return

        boundary = self.feature_exchange_policy.boundary
        execution_context = self.execution_context
        original_handle_lammps = base_cls.handle_lammps

        def handle_lammps(
            block_self,
            node_feats: torch.Tensor,
            lammps_class,
            lammps_natoms,
            first_layer: bool,
        ) -> torch.Tensor:
            if lammps_class is not None:
                return original_handle_lammps(
                    block_self,
                    node_feats,
                    lammps_class=lammps_class,
                    lammps_natoms=lammps_natoms,
                    first_layer=first_layer,
                )
            if first_layer or execution_context.exchange_mode != "ipc":
                return node_feats
            if execution_context.distributed_state is None:
                raise RuntimeError(
                    "DistMACE subclass boundary was invoked without an active "
                    "DistNNPState execution context."
                )
            if execution_context.learned_feature_exchanger is None:
                raise RuntimeError(
                    "DistMACE subclass boundary was invoked without a learned "
                    "feature exchanger."
                )

            distributed_state = execution_context.distributed_state
            num_local = int(
                distributed_state.local_partition_info.partition_indices.shape[0]
            )
            learned_local_node_feats = node_feats[:num_local].contiguous()
            input_dtype = node_feats.dtype
            exchanged_node_feats = FeaturePushToExchange.apply(
                learned_local_node_feats,
                distributed_state,
                execution_context.learned_feature_exchanger,
            )
            if exchanged_node_feats.dtype != input_dtype:
                exchanged_node_feats = exchanged_node_feats.to(dtype=input_dtype)

            if execution_context.observation is not None:
                execution_context.observation.update(
                    {
                        "feature_exchange_layer": layer_index,
                        "feature_exchange_boundary": boundary,
                        "input_feature_shape": list(node_feats.shape),
                        "owned_local_feature_shape": list(
                            learned_local_node_feats.shape
                        ),
                        "exchanged_feature_shape": list(exchanged_node_feats.shape),
                        "input_feature_dtype": str(input_dtype),
                        "exchanged_feature_dtype": str(exchanged_node_feats.dtype),
                        "feature_exchange_used_ipc": True,
                        "subclass_boundary_executed": True,
                    }
                )
            return exchanged_node_feats

        subclass_name = f"DistMACE{base_cls.__name__}"
        subclass_cls = type(
            subclass_name,
            (base_cls,),
            {
                "__distmace_subclass__": True,
                "__distmace_subclass_role__": "interaction_boundary",
                "__distmace_boundary_kind__": "feature_exchange",
                "handle_lammps": handle_lammps,
            },
        )
        target_block.__class__ = subclass_cls

        self._installed_model_class_name = self._ensure_model_subclass(model)
        self._installed_interaction_class_name = subclass_cls.__name__
        self._installed_interaction_index = layer_index

    def begin_forward(
        self,
        distributed_state: DistNNPState,
        exchange_mode: str,
        learned_feature_exchanger: GPUDirectExchanger,
        observation: Dict[str, Any],
    ) -> None:
        self.execution_context.distributed_state = distributed_state
        self.execution_context.exchange_mode = exchange_mode
        self.execution_context.learned_feature_exchanger = learned_feature_exchanger
        self.execution_context.observation = observation

    def end_forward(self) -> None:
        self.execution_context.distributed_state = None
        self.execution_context.exchange_mode = "uninitialized"
        self.execution_context.learned_feature_exchanger = None
        self.execution_context.observation = None

    def describe(self) -> Dict[str, Any]:
        return {
            "model_subclass_contract": self.model_contract,
            "model_subclass_status": self.model_status,
            "ownership_contract": self.ownership_contract,
            "installed_model_class_name": self._installed_model_class_name,
            "installed_interaction_class_name": self._installed_interaction_class_name,
            "installed_interaction_index": self._installed_interaction_index,
            "subclass_boundary_installed": self._installed_interaction_class_name
            is not None,
        }
