"""
Minimal cross-node transport check for PythonNCCLExchanger.

Launched under torchrun (multi-node). Builds a tiny 4-rank ring halo
topology - rank r sends 2 atoms to rank (r+1)%W and receives 2 from
rank (r-1)%W - so at W=4 on 2 nodes at least one send/recv pair crosses
the node boundary. Runs feature_push_forward + feature_push_backward and
checks the received ghost values and scattered gradients on every rank.

Exit 0 + "TRANSPORT OK" on success; exit 1 otherwise.
"""
import os
import sys

import torch
import torch.distributed as dist

from distnnp_state import PartitionMessagePassingInfo
from distnnp_model.comm.exchanger import PythonNCCLExchanger

N_LOCAL = 10
F = 8
K = 2  # atoms exchanged per neighbour


def build_info(rank: int, world_size: int, device: torch.device) -> PartitionMessagePassingInfo:
    dst = (rank + 1) % world_size
    src = (rank - 1) % world_size

    send_offsets = [0] * (world_size + 1)
    for r in range(world_size):
        send_offsets[r + 1] = send_offsets[r] + (K if r == dst else 0)

    recv_offsets = [0] * (world_size + 1)
    for r in range(world_size):
        recv_offsets[r + 1] = recv_offsets[r] + (K if r == src else 0)

    info = PartitionMessagePassingInfo()
    info.partition_indices = torch.arange(N_LOCAL, dtype=torch.int64, device=device)
    info.send_offsets = torch.tensor(send_offsets, dtype=torch.int64, device=device)
    info.send_local_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
    info.recv_offsets = torch.tensor(recv_offsets, dtype=torch.int64, device=device)
    return info


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    src = (rank - 1) % world_size

    info = build_info(rank, world_size, device)

    feats = torch.zeros(N_LOCAL, F, dtype=torch.float32, device=device)
    for i in range(N_LOCAL):
        feats[i, :] = rank * 100.0 + i

    ex = PythonNCCLExchanger()

    out = ex.feature_push_forward(feats, info)
    torch.cuda.synchronize()

    ok = out.shape == torch.Size((N_LOCAL + K, F))
    ok &= torch.allclose(out[:N_LOCAL], feats)
    ghost_expected = torch.zeros(K, F, dtype=torch.float32, device=device)
    for j in range(K):
        ghost_expected[j, :] = src * 100.0 + j
    ok &= torch.allclose(out[N_LOCAL:], ghost_expected)

    grad_out = torch.ones(N_LOCAL + K, F, dtype=torch.float32, device=device)
    grad_in = ex.feature_push_backward(grad_out, torch.Size([N_LOCAL, F]), info)
    torch.cuda.synchronize()

    grad_expected = torch.ones(N_LOCAL, F, dtype=torch.float32, device=device)
    grad_expected[0, :] = 2.0
    grad_expected[1, :] = 2.0
    ok &= grad_in.shape == torch.Size((N_LOCAL, F))
    ok &= torch.allclose(grad_in, grad_expected)

    hostname = os.uname().nodename
    print(f"[rank {rank} @ {hostname}] recv from {src}: {'ok' if ok else 'MISMATCH'}", flush=True)

    verdict = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        if verdict.item() == 1:
            print("TRANSPORT OK", flush=True)
            sys.exit(0)
        print("TRANSPORT FAILED", flush=True)
        sys.exit(1)
    sys.exit(0 if verdict.item() == 1 else 1)


if __name__ == "__main__":
    main()
