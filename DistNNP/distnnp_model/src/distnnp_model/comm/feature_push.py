import torch

# Uncomment when the communication engine is implemented
# from distnnp_model.distnnp_model_impl import feature_push

from dynamis_state import DistNNPState, PartitionMessagePassingInfo, Timer, TimerClass, Where
from torch.autograd import Function
from distnnp_model.comm.exchanger import Exchanger
from typing import Tuple


class FeaturePushToExchange(Function):
    @staticmethod
    def forward(
        ctx,
        features_to_push: torch.Tensor,
        state: DistNNPState,
        exchanger: Exchanger,
    ) -> torch.Tensor:
        """Implements the forward pass for the feature push operation.

        Args:
            features_to_push (torch.Tensor): The features that need to be pushed to neighboring partitions.
            state (DistNNPState): The current state of the distributed system, containing necessary information for communication.
        Returns:
            torch.Tensor: The features received from neighboring partitions after the push to exchange operation.
        """
        # Uncomment when the communication engine is implemented
        # return feature_push(state, config, partition_info)
        ctx._state = state
        ctx._features_to_push_shape = features_to_push.shape
        ctx._exchanger = exchanger

        partition_info: PartitionMessagePassingInfo = state.local_partition_info

        # Callers must push a 2-D [n_local, feature_dim] tensor, one row per local
        # atom. feature_push_forward reads feature_dim off the last axis to size
        # the IPC buffers, so any other rank silently mis-sizes them instead of
        # failing here.
        assert features_to_push.dim() == 2, (
            "features_to_push must be 2-D [n_local, feature_dim], got shape "
            f"{tuple(features_to_push.shape)}"
        )

        # Captured now, before this region opens, so backward() — which runs on a
        # different thread with no region open — can still nest under model_forward.
        ctx._timing_parent_span = Timer.current_span_id()

        t = Timer()
        t.start("halo_exchange", TimerClass.Comm, Where.Device)
        received_features = exchanger.feature_push_forward(
            features_to_push, partition_info
        )
        t.stop()

        return received_features

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, None, None]:
        """Implements the backward pass for the feature push operation.

        Args:
            grad_output (torch.Tensor): The gradients received from the neighboring partitions for the features that were
                pushed in the forward pass.
        Returns:
            Tuple[torch.Tensor, None]: The gradients for the input features that were pushed in the forward pass,
                and None for the state (as it is not learnable).
        """
        # Implement the backward pass for the feature push operation

        state = ctx._state
        input_feature_shape = ctx._features_to_push_shape
        exchanger = ctx._exchanger

        # Runs on the autograd worker thread, which has no open region of its own —
        # parent_id (captured in forward()) supplies the link to model_forward.
        t = Timer()
        t.start("halo_exchange", TimerClass.Comm, Where.Device,
                parent_id=getattr(ctx, "_timing_parent_span", 0))
        recevied_features_grad = exchanger.feature_push_backward(
            grad_output, input_feature_shape, state.local_partition_info
        )
        t.stop()

        return recevied_features_grad, None, None  # No gradients for state or exchanger
