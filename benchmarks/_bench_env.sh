#!/bin/bash
# Common env setup for distributed benchmark sbatch scripts on Delta gpuA100x4.
# Sourced by run_v5_scaling_w{2,4,8,16}.sh and other launchers.

set -e
mkdir -p benchmarks/logs

# ── Delta NCCL/CXI module stack ─────────────────────────────────────────────
ml cudatoolkit
ml libfabric
ml aws-ofi-nccl

# CXI/Slingshot fabric for inter-node NCCL
export FI_PROVIDER=cxi
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

# NCCL settings — multi-node + GDR off (GPU-NIC distance 8-9 on Delta would
# otherwise segfault if GDRDMA is attempted)
export NCCL_CROSS_NIC=1
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0

# Python / PyTorch
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TORCH_CUDA_ARCH_LIST="8.0"

# Rendezvous master: first node in the allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

cd /projects/bcpt/hqin1/git/DistNNP

echo "============================================================"
echo "  DistNNP distributed benchmark — W=$BENCH_W"
echo "  Job:    $SLURM_JOB_ID"
echo "  Nodes:  $SLURM_JOB_NODELIST"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  Time:   $(date)"
echo "============================================================"
