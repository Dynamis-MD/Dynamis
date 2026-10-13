#include "dynamis_state/partitioner.hpp"
#include "dynamis_state/timing.hpp"

namespace dynamis_state
{

PartitionMessagePassingInfo SimpleSpatialPartitioner::partition(
    const MolecularInfo &state, const DistEnvironment &env)
{
    Timer t;
    t.start("partition", TimerClass::Compute,
            state.atom_positions.is_cuda() ? Where::Device : Where::Host);

    PartitionMessagePassingInfo info;
    auto device = state.atom_positions.device();
    auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(device);

    long N = static_cast<long>(state.num_atoms);
    long W = static_cast<long>(env.world_size);
    long R = static_cast<long>(env.global_rank);

    // Build partition_id: atom i → rank, contiguous split.
    // Rank r gets atoms [r * atoms_per_rank, (r+1) * atoms_per_rank),
    // with the last rank absorbing any remainder.
    long atoms_per_rank = N / W;
    // clamp(arange // atoms_per_rank, max=W-1) handles remainder on last rank
    info.partition_id = torch::clamp(
        torch::div(torch::arange(N, long_opts), atoms_per_rank, "trunc"),
        /*min=*/0, /*max=*/W - 1);

    // Contiguous split: rank R gets [start, end)
    long start = R * atoms_per_rank;
    long end = (R == W - 1) ? N : start + atoms_per_rank;

    info.partition_indices     = state.atom_vertex_numbers.slice(0, start, end).contiguous();

    // Send/recv matrix: diagonal only (each rank keeps its own atoms, no exchange)
    info.global_send_recv_matrix = torch::zeros({W, W}, long_opts);
    long num_local = end - start;
    info.global_send_recv_matrix.index_put_({R, R}, num_local);

    info.atom_counts = torch::bincount(info.partition_id, /*weights=*/{}, /*minlength=*/env.world_size);

    t.stop();

    return info;
}

} // namespace dynamis_state
