import torch
from distnnp_state import PartitionMessagePassingInfo, DistEnvironment
from abc import ABC, abstractmethod


class Exchanger(ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def setup_exchanger(self,
        distributed_env: DistEnvironment,
        partition_info: PartitionMessagePassingInfo,
        max_feature_dim: int) -> None:
        """Helper function to setup the feature exchanger for testing.

        This function performs the necessary steps to initialize the feature exchanger,
        including allocating IPC buffers, exchanging handles, and setting up events.

        Args:
            distributed_env: The current distributed world description
            partition_info: The partition information containing send/recv details.
            max_feature_dim: Maximum feature size of any row sent
        """
        ...

    @abstractmethod
    def reset_buffers(self,
        feature_dim: int,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
    ) -> None:
        """Reset the internal buffers of the exchanger to prepare for the next communication round."""
        ...

    @abstractmethod
    def feature_push_forward(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Perform the forward feature push operation.

        Args:
            features_to_push (torch.Tensor): The features that need to be pushed to neighboring partitions.
            partition_info: The partition information containing send/recv details.

        Returns:
            torch.Tensor: The features received from neighboring partitions after the push to exchange operation concatenated 
                          with the local features_to_push. 
        """
        ...


    @abstractmethod
    def feature_push_backward(
        self,
        grad_output: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        """Perform the backward pass for the feature push operation.

        Args:
            grad_output (torch.Tensor): The gradients calculated locally for the local vertices as well as the vertices 
                                        received from the neighboring partitions for the features that were
                                        pushed in the forward pass.
            partition_info: The partition information containing send/recv details.

        Returns:
            torch.Tensor: The gradients for the input features that were pushed in the forward pass. 
                          Gradients received from are added into the local gradients.
        """
