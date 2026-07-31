from distnnp_state import (
    DistEnvironment,
    SFCPartitioner,
    SimpleSpatialPartitioner,
    KDTreePartitioner,
)
import torch
import torch.distributed as dist
from typing import Union
import os

ABSTRACT_PARTITIONERS = Union[
    SFCPartitioner, SimpleSpatialPartitioner, KDTreePartitioner
]


def _infer_is_distributed() -> bool:
    if dist.is_initialized():
        return True
    launcher_markers = (
        "WORLD_SIZE", "RANK", "LOCAL_RANK",
        "SLURM_PROCID", "SLURM_NTASKS", "MASTER_ADDR",
    )
    return any(m in os.environ for m in launcher_markers)


class TorchDistEnv:
    """Helper class to initialize distributed environment and set device.
    This is used by SimState but can also be used by users directly if they want to create their own DistNNPState outside of SimState.
    """

    def __init__(self, is_distributed: bool | None = None, backend="cuda:nccl,cpu:gloo", **kwargs) -> None:
        if is_distributed is None:
            is_distributed = _infer_is_distributed()
            print(f"is_distributed inferred as {is_distributed} from env")

        if is_distributed:
            if not dist.is_initialized():
                print("Initializing distributed environment...")
                dist.init_process_group(backend=backend, **kwargs)
            global_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            global_rank = 0
            world_size = 1

        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", -1)))
        num_nodes = int(os.environ.get("SLURM_NNODES", -1))

        if local_rank == -1 or num_nodes == -1:
            num_gpus = torch.cuda.device_count()
            gpus_per_node = num_gpus if num_gpus > 0 else 1
            if local_rank == -1:
                local_rank = global_rank % gpus_per_node
            if num_nodes == -1:
                num_nodes = max(1, (world_size + gpus_per_node - 1) // gpus_per_node)
        else:
            gpus_per_node = max(1, world_size // num_nodes)

        if num_gpus > 0:
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}" if num_gpus > 0 else "cpu")

        self.device = device

        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.is_distributed = is_distributed

    @property
    def dist_env(self) -> DistEnvironment:
        obj = DistEnvironment()
        obj.global_rank = self.global_rank
        obj.world_size = self.world_size
        obj.gpus_per_node = self.gpus_per_node
        obj.num_nodes = self.num_nodes
        obj.local_rank = self.local_rank
        return obj

    def barrier(self):
        if self.is_distributed:
            dist.barrier()
        else:
            pass
