import torch
from distnnp_model.comm.exchanger import Exchanger
from distnnp_state import PartitionMessagePassingInfo, DistEnvironment
import torch.distributed as dist


class TorchNCCLExchanger(Exchanger):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self._send_buffer = torch.empty(0)
        self._recv_buffer = torch.empty(0)
        self._device = device
        self._world_size = 0
        self._send_split_sizes: list = []  # atom counts per rank (forward send)
        self._recv_split_sizes: list = []  # atom counts per rank (forward recv)

    def setup_exchanger(
        self,
        distributed_env: DistEnvironment,
        partition_info: PartitionMessagePassingInfo,
        max_feature_dim: int,
    ) -> None:
        assert dist.is_initialized(), "Distributed process group is not initialized"

        self._world_size = dist.get_world_size()

        send_offsets = partition_info.send_offsets.tolist()
        recv_offsets = partition_info.recv_offsets.tolist()

        self._send_split_sizes = [
            send_offsets[r + 1] - send_offsets[r] for r in range(self._world_size)
        ]
        self._recv_split_sizes = [
            recv_offsets[r + 1] - recv_offsets[r] for r in range(self._world_size)
        ]

        # Flat 1-D buffers: a 1-D slice is always contiguous, which is required by
        # all_to_all_single. The actual feature_dim used each call may be <= max_feature_dim.
        self._send_buffer = torch.zeros(
            partition_info.num_out_halo_atoms * max_feature_dim, device=self._device
        )
        self._recv_buffer = torch.zeros(
            partition_info.num_in_halo_atoms * max_feature_dim, device=self._device
        )

    def reset_buffers(
        self,
        max_feature_dim: int,
        partition_info: PartitionMessagePassingInfo,
        device: torch.device,
    ) -> None:
        self._send_buffer.zero_()
        self._recv_buffer.zero_()

    def feature_push_forward(
        self,
        features_to_push: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        feature_dim = features_to_push.shape[1]
        n_out = partition_info.num_out_halo_atoms
        n_in = partition_info.num_in_halo_atoms

        # Gather atoms and write into the flat send buffer as a contiguous slice.
        send_flat = self._send_buffer[: n_out * feature_dim]
        send_flat.copy_(features_to_push[partition_info.send_local_indices].view(-1))

        recv_flat = self._recv_buffer[: n_in * feature_dim]

        dist.all_to_all_single(
            recv_flat,
            send_flat,
            output_split_sizes=[s * feature_dim for s in self._recv_split_sizes],
            input_split_sizes=[s * feature_dim for s in self._send_split_sizes],
        )

        halo = recv_flat.view(n_in, feature_dim)
        return torch.cat([features_to_push, halo], dim=0)

    def feature_push_backward(
        self,
        grad_output: torch.Tensor,
        partition_info: PartitionMessagePassingInfo,
    ) -> torch.Tensor:
        n_local = partition_info.num_local_atoms
        n_in = partition_info.num_in_halo_atoms
        n_out = partition_info.num_out_halo_atoms
        feature_dim = grad_output.shape[1]

        grad_local = grad_output[:n_local].clone()
        grad_halo = grad_output[n_local:]  # [n_in, feature_dim]

        # Pack halo grads into recv_buffer (symmetric: this is the input side now).
        recv_flat = self._recv_buffer[: n_in * feature_dim]
        recv_flat.copy_(grad_halo.view(-1))

        send_flat = self._send_buffer[: n_out * feature_dim]

        # Swapped split sizes: send recv_count[r] grads to rank r, get send_count[r] back.
        dist.all_to_all_single(
            send_flat,
            recv_flat,
            output_split_sizes=[s * feature_dim for s in self._send_split_sizes],
            input_split_sizes=[s * feature_dim for s in self._recv_split_sizes],
        )

        # Accumulate returned grads; scatter_add_ handles atoms sent to multiple ranks.
        grad_local.scatter_add_(
            0,
            partition_info.send_local_indices.unsqueeze(1).expand(-1, feature_dim),
            send_flat.view(n_out, feature_dim),
        )

        return grad_local
