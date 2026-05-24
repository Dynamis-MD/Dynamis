import torch
from distnnp_model.comm.exchanger import Exchanger
from distnnp_state import PartitionMessagePassingInfo


class SingleRankExchanger(Exchanger):
    def __init__(self) -> None:
        super().__init__()

    def setup_exchanger(self, feature_dim: int, partition_info: PartitionMessagePassingInfo, device: torch.device):
        pass

    def reset_buffers(self, feature_dim: int, partition_info: PartitionMessagePassingInfo, device: torch.device) -> None:
        pass 

    def feature_push_forward(self, features_to_push: torch.Tensor, partition_info: PartitionMessagePassingInfo) -> torch.Tensor:
        return features_to_push
    
    def feature_push_backward(self, grad_output: torch.Tensor, partition_info: PartitionMessagePassingInfo) -> torch.Tensor:
        return grad_output