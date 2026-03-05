import torch

# Uncomment when the communication engine is implemented
# from distnnp_model.distnnp_model_impl import feature_push

from distnnp_state import DistNNPState, SimulationConfig, PartitionMessagePassingInfo
from torch.autograd import Function


class FeaturePushToExchange(Function):
    @staticmethod
    def forward(
        ctx,
        features_to_push: torch.Tensor,
        state: DistNNPState,
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
        num_sends = state.local_partition_info.send_local_indices.shape[0]
        feature_dim = features_to_push.shape[1]

        received_features = torch.empty(
            num_sends,
            feature_dim,
            device=features_to_push.device,
            dtype=features_to_push.dtype,
        )

        # TODO: Perform the push to exchange operation
        # Uncomment when the communication engine is implemented

        # send_buffer = torch.gather(features_to_push, 0, state.local_partition_info.send_local_indices)
        # distnnp_model_impl.feature_push(send_buffer, received_features, state)

        return received_features

    @staticmethod
    def backward(ctx, grad_output):
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

        recevied_features_grad = torch.zeros(
            *input_feature_shape, device=grad_output.device, dtype=grad_output.dtype
        )

        # TODO: Compute the gradients for the input features based on the gradients received from neighboring partitions
        # This will involve a push operation of the local gradients on all ranks to a temporary buffer.
        # Then the gradients in the temporary buffer will be accumulated locally.
        # A single vertex may be sent to multiple ranks, so once recevied the gradients must be
        # accumulated across the rank dimension as well.
        #

        return recevied_features_grad, None  # No gradients for state
