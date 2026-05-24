import torch
from distnnp_state import PartitionMessagePassingInfo
from abc import ABC, abstractmethod


class Exchanger(ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def setup_exchanger(
        self,
        feature_dim: int,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
    ) -> None:
        """Helper function to setup the feature exchanger for testing.

        This function performs the necessary steps to initialize the feature exchanger,
        including allocating IPC buffers, exchanging handles, and setting up events.

        Args:
            exchanger (FeatureExchanger): The feature exchanger instance to setup.
            local_features (torch.Tensor): The local features tensor for the current rank.
            partition_info: The partition information containing send/recv details.
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
            torch.Tensor: The features received from neighboring partitions after the push to exchange operation.
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
            grad_output (torch.Tensor): The gradients received from the neighboring partitions for the features that were
                pushed in the forward pass.
            partition_info: The partition information containing send/recv details.

        Returns:
            torch.Tensor: The gradients for the input features that were pushed in the forward pass.
        """
