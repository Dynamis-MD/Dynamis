#include "dynamis_state/dist_graph_ops.hpp"
#include "dynamis_state/dynamis_state.hpp"
#include "dynamis_state/partitioner.hpp"
#include "dynamis_state/radius_graph.hpp"
#include "dynamis_state/timing.hpp"
#include "dynamis_state/trajectory_reader.hpp"
#include "dynamis_state/trajectory_writer.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <string_view>
#include <torch/extension.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(dynamis_state_impl, m)
{
    py::class_<dynamis_state::MolecularInfo>(m, "MolecularInfo")
            .def(py::init<>())
            .def_readwrite("num_atoms", &dynamis_state::MolecularInfo::num_atoms)
            .def_readwrite("atom_vertex_numbers",
                           &dynamis_state::MolecularInfo::atom_vertex_numbers)
            .def_readwrite("atom_types", &dynamis_state::MolecularInfo::atom_types)
            .def_readwrite("atom_positions", &dynamis_state::MolecularInfo::atom_positions)
            .def_readwrite("atom_velocities", &dynamis_state::MolecularInfo::atom_velocities)
            .def_readwrite("edge_counts", &dynamis_state::MolecularInfo::edge_counts);

    py::class_<dynamis_state::DistEnvironment>(m, "DistEnvironment")
            .def(py::init<>())
            .def_readwrite("global_rank", &dynamis_state::DistEnvironment::global_rank)
            .def_readwrite("world_size", &dynamis_state::DistEnvironment::world_size)
            .def_readwrite("gpus_per_node", &dynamis_state::DistEnvironment::gpus_per_node)
            .def_readwrite("num_nodes", &dynamis_state::DistEnvironment::num_nodes)
            .def_readwrite("local_rank", &dynamis_state::DistEnvironment::local_rank);

    py::class_<dynamis_state::SimulationConfig>(m, "SimulationConfig")
            .def(py::init<>())
            .def_readwrite("cutoff", &dynamis_state::SimulationConfig::cutoff)
            .def_readwrite("ghost_cutoff_buffer",
                           &dynamis_state::SimulationConfig::ghost_cutoff_buffer)
            .def_readwrite("cell", &dynamis_state::SimulationConfig::cell)
            .def_readwrite("pbc", &dynamis_state::SimulationConfig::pbc);

    py::class_<dynamis_state::PartitionMessagePassingInfo>(m, "PartitionMessagePassingInfo")
            .def(py::init<>())
            .def_readwrite("partition_id",
                           &dynamis_state::PartitionMessagePassingInfo::partition_id)
            .def_readwrite("partition_indices",
                           &dynamis_state::PartitionMessagePassingInfo::partition_indices)
            .def_readwrite("global_send_recv_matrix",
                           &dynamis_state::PartitionMessagePassingInfo::global_send_recv_matrix)
            .def_readwrite("send_offsets",
                           &dynamis_state::PartitionMessagePassingInfo::send_offsets)
            .def_readwrite("send_local_indices",
                           &dynamis_state::PartitionMessagePassingInfo::send_local_indices)
            .def_readwrite("recv_offsets",
                           &dynamis_state::PartitionMessagePassingInfo::recv_offsets)
            .def_readwrite("recv_ghost_indices",
                           &dynamis_state::PartitionMessagePassingInfo::recv_ghost_indices)
            .def_readwrite("interior_mask",
                           &dynamis_state::PartitionMessagePassingInfo::interior_mask)
            .def_readwrite("edge_index", &dynamis_state::PartitionMessagePassingInfo::edge_index)
            .def_readwrite("edge_index_halo",
                           &dynamis_state::PartitionMessagePassingInfo::edge_index_halo)
            .def_readwrite("atom_counts", &dynamis_state::PartitionMessagePassingInfo::atom_counts);

    py::class_<dynamis_state::DynamisState>(m, "DynamisState")
            .def(py::init<>())
            .def_readwrite("dist_env", &dynamis_state::DynamisState::dist_env)
            .def_readwrite("sim_config", &dynamis_state::DynamisState::sim_config)
            .def_readwrite("global_molecular_info",
                           &dynamis_state::DynamisState::global_molecular_info)
            .def_readwrite("local_molecular_info",
                           &dynamis_state::DynamisState::local_molecular_info)
            .def_readwrite("local_partition_info",
                           &dynamis_state::DynamisState::local_partition_info);

    py::class_<dynamis_state::KDTreePartitioner>(m, "KDTreePartitioner")
            .def(py::init<>())
            .def("partition",
                 py::overload_cast<const dynamis_state::MolecularInfo &,
                                   const dynamis_state::DistEnvironment &>(
                         &dynamis_state::KDTreePartitioner::partition),
                 "Partition: unweighted (default) or user-weighted (if edge_counts "
                 "set)")
            .def("partition",
                 py::overload_cast<const dynamis_state::MolecularInfo &,
                                   const dynamis_state::DistEnvironment &,
                                   const dynamis_state::SimulationConfig &>(
                         &dynamis_state::KDTreePartitioner::partition),
                 "Partition: auto-compute edge weights via radius_graph_pbc (CUDA)")
            .def("build_halo_maps", &dynamis_state::KDTreePartitioner::build_halo_maps);

    py::class_<dynamis_state::SimpleSpatialPartitioner>(m, "SimpleSpatialPartitioner")
            .def(py::init<>())
            .def("partition", &dynamis_state::SimpleSpatialPartitioner::partition)
            .def("build_halo_maps", &dynamis_state::SimpleSpatialPartitioner::build_halo_maps);

    py::enum_<dynamis_state::SFCCurveType>(m, "SFCCurveType")
            .value("Morton", dynamis_state::SFCCurveType::Morton)
            .value("Hilbert", dynamis_state::SFCCurveType::Hilbert);

    py::class_<dynamis_state::SFCPartitioner>(m, "SFCPartitioner")
            .def(py::init<dynamis_state::SFCCurveType>(),
                 py::arg("curve_type") = dynamis_state::SFCCurveType::Morton)
            .def("partition",
                 py::overload_cast<const dynamis_state::MolecularInfo &,
                                   const dynamis_state::DistEnvironment &>(
                         &dynamis_state::SFCPartitioner::partition),
                 "Partition: unweighted (default) or user-weighted (if edge_counts "
                 "set)")
            .def("partition",
                 py::overload_cast<const dynamis_state::MolecularInfo &,
                                   const dynamis_state::DistEnvironment &,
                                   const dynamis_state::SimulationConfig &>(
                         &dynamis_state::SFCPartitioner::partition),
                 "Partition: auto-compute edge weights via radius_graph_pbc (CUDA)")
            .def("build_halo_maps", &dynamis_state::SFCPartitioner::build_halo_maps);

    py::class_<dynamis_state::TrajectoryWriterImpl>(m, "TrajectoryWriterImpl")
            .def(py::init<uint32_t, py::bytes, const std::string &, const std::string &,
                          const std::string &, const dynamis_state::MolecularInfo &, torch::Tensor,
                          torch::Tensor, const float, const uint32_t, std::vector<std::string>>(),
                 "Opens/truncates the output file and writes front matter, in one call",
                 py::arg("global_rank"), py::arg("run_id"), py::arg("author"),
                 py::arg("output_dir"), py::arg("output_file"), py::arg("molecular_info"),
                 py::arg("cell"), py::arg("pbc"), py::arg("time_per_step"),
                 py::arg("steps_per_write"), py::arg("opts") = std::vector<std::string>{})
            .def("update_atom_counts", &dynamis_state::TrajectoryWriterImpl::update_atom_counts,
                 "Update the rank offsets. Should be called after every repartition",
                 py::arg("counts"))
            .def("submit", &dynamis_state::TrajectoryWriterImpl::submit,
                 "Submits a state to be written. cell/pbc are optional and are "
                 "carried forward from the last value seen if omitted",
                 py::arg("step"), py::arg("state"), py::arg("cell") = c10::optional<at::Tensor>{},
                 py::arg("pbc") = c10::optional<at::Tensor>{})
            .def("get_cell", &dynamis_state::TrajectoryWriterImpl::get_cell,
                 "Returns the currently cached simulation cell (float32 [3, 3])")
            .def("get_pbc", &dynamis_state::TrajectoryWriterImpl::get_pbc,
                 "Returns the currently cached periodic boundary flags (uint8 [3])")
            .def("flush", &dynamis_state::TrajectoryWriterImpl::flush,
                 "Flushes the queue and closes the file");

    m.def("read_molecular_info", &dynamis_state::read_molecular_info,
          "Reads one frame from a .xyz or .dynamis file, returning "
          "(MolecularInfo, cell, pbc). frame_index defaults to -1 (last frame) "
          "and indexes real frames for both formats",
          py::arg("path"), py::arg("frame_index") = -1);

    m.def("translate_dynamis_to_xyz", &dynamis_state::translate_dynamis_to_xyz,
          "Translates every frame of a .dynamis file into a multi-frame "
          "extended-xyz file",
          py::arg("dynamis_path"), py::arg("xyz_path"),
          py::arg("opts") = std::vector<std::string>{});

    m.def("radius_graph_pbc", &radius_graph_pbc,
          "CUDA cell-list radius graph with selective PBC tiling", py::arg("pos"), py::arg("cell"),
          py::arg("pbc"), py::arg("radius"), py::arg("max_num_neighbors"),
          py::arg("enforce_max_strictly"));

    m.def("radius_graph_pbc_ex", &radius_graph_pbc_ex,
          "Extended radius graph: accepts optional target_indices subset, returns "
          "(edge_index, cell_shifts, num_neighbors_image, neighbor_counts).",
          py::arg("pos"), py::arg("cell"), py::arg("pbc"), py::arg("radius"),
          py::arg("max_num_neighbors"), py::arg("enforce_max_strictly"),
          py::arg("target_indices") = c10::optional<at::Tensor>{});

    // ===== Timing =====

    py::enum_<dynamis_state::TimerClass>(m, "TimerClass")
            .value("Compute", dynamis_state::TimerClass::Compute)
            .value("Comm", dynamis_state::TimerClass::Comm)
            // Reserved: see Level::Detail.
            .value("Wait", dynamis_state::TimerClass::Wait)
            .value("Other", dynamis_state::TimerClass::Other)
            // The denominator the others are a fraction of, not a partition member.
            .value("Total", dynamis_state::TimerClass::Total);

#ifdef DYNAMIS_TIMING
    m.attr("TIMING_ENABLED") = true;
#else
    m.attr("TIMING_ENABLED") = false;
#endif

    py::enum_<dynamis_state::Where>(m, "Where")
            .value("Host", dynamis_state::Where::Host)
            .value("Device", dynamis_state::Where::Device)
            // One host + one device record for the same region — right for a
            // Python-orchestrated region that also launches async GPU work.
            .value("Both", dynamis_state::Where::Both);

    py::enum_<dynamis_state::Level>(m, "Level")
            .value("Off", dynamis_state::Level::Off)
            .value("Basic", dynamis_state::Level::Basic)
            // Reserved for WAIT barriers, not yet implemented; behaves like Basic.
            .value("Detail", dynamis_state::Level::Detail);

    py::class_<dynamis_state::TimingRegistry>(m, "TimingRegistry")
            .def_static("instance", &dynamis_state::TimingRegistry::instance,
                        py::return_value_policy::reference)
            .def(
                    "init",
                    [](dynamis_state::TimingRegistry &self, uint32_t rank, uint32_t local_rank,
                       uint32_t world_size, py::bytes run_id, int32_t device_id,
                       const std::string &output_dir) {
                        self.init(rank, local_rank, world_size, static_cast<std::string>(run_id),
                                  device_id, output_dir);
                    },
                    py::arg("rank"), py::arg("local_rank"), py::arg("world_size"),
                    py::arg("run_id"), py::arg("device_id"), py::arg("output_dir"))
            .def("set_level", &dynamis_state::TimingRegistry::set_level, py::arg("level"))
            .def("enabled", &dynamis_state::TimingRegistry::enabled)
            .def("set_step", &dynamis_state::TimingRegistry::set_step, py::arg("step"))
            // flush() blocks on every pending device event and syncs the stream, so it
            // belongs at a quiescent point, not in the step loop — close() is the
            // normal write-once-at-exit path.
            .def("flush", &dynamis_state::TimingRegistry::flush)
            .def("close", &dynamis_state::TimingRegistry::close);

    py::class_<dynamis_state::Timer>(m, "Timer")
            .def(py::init<>())
            // Runs before the level check, so it must not call getCurrentCUDAStream
            // itself — only branch on parent_id, which is free even on Level::Off.
            // parent_id = 0 means "nest under whatever is open on this thread".
            .def(
                    "start",
                    [](dynamis_state::Timer &self, std::string_view key,
                       dynamis_state::TimerClass cls, dynamis_state::Where where,
                       uint64_t parent_id) {
                        if (parent_id)
                            self.start(key, cls, where, dynamis_state::ParentSpan{ parent_id });
                        else
                            self.start(key, cls, where);
                    },
                    py::arg("key"), py::arg("cls"), py::arg("where") = dynamis_state::Where::Host,
                    py::arg("parent_id") = 0)
            .def("stop", &dynamis_state::Timer::stop)
            // Capture on the owning thread to hand back as start()'s parent_id.
            .def_static("current_span_id", &dynamis_state::Timer::current_span_id)
            .def_static("set_level", &dynamis_state::Timer::set_level, py::arg("level"))
            .def_static("enabled", &dynamis_state::Timer::enabled)
            .def_static("set_step", &dynamis_state::Timer::set_step, py::arg("step"));

    m.def("radius_graph_pbc_counts_ex", &radius_graph_pbc_counts_ex,
          "Count-only variant: returns just neighbor_counts[N_target] int32. "
          "Skips edge buffer + prefix-sum + B4 write kernel (~50% faster).",
          py::arg("pos"), py::arg("cell"), py::arg("pbc"), py::arg("radius"),
          py::arg("target_indices") = c10::optional<at::Tensor>{});

    // ===== Distributed build_graph hot-path ops (Mode A) =====

    m.def("filter_edges_to_radius", &filter_edges_to_radius,
          "Filter edges built at r+skin down to true cutoff r: keep edge iff "
          "||pos[src] + cell_shift @ cell - pos[dst]||^2 <= r^2. Order-preserving. "
          "Fused CUDA kernel + compaction; replaces the ATen recompute (step 8).",
          py::arg("edges"), py::arg("cell_shifts"), py::arg("pos"), py::arg("cell"),
          py::arg("radius"));

    m.def("extract_halo_info", &extract_halo_info,
          "Derive halo send/recv CSR + interior mask from per-rank edges "
          "(dst in my_outer) via graph symmetry, grouped by peer and sorted by "
          "global atom id within each peer (no NCCL). Returns "
          "(send_offsets, send_local_indices, recv_offsets, recv_ghost_indices, "
          "interior_mask, recv_atom_global). step 7.",
          py::arg("edges"), py::arg("pid_outer"), py::arg("my_outer_atoms"), py::arg("my_rank"),
          py::arg("world_size"), py::arg("N"));
}
