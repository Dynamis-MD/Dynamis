#include "distnnp_state/dist_graph_ops.hpp"
#include "distnnp_state/distnnp_state.hpp"
#include "distnnp_state/partitioner.hpp"
#include "distnnp_state/radius_graph.hpp"
#include "distnnp_state/timing.hpp"
#include "distnnp_state/trajectory_writer.hpp"
#include <torch/extension.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(distnnp_state_impl, m) {
  py::class_<distnnp_state::MolecularInfo>(m, "MolecularInfo")
      .def(py::init<>())
      .def_readwrite("num_atoms", &distnnp_state::MolecularInfo::num_atoms)
      .def_readwrite("atom_vertex_numbers",
                     &distnnp_state::MolecularInfo::atom_vertex_numbers)
      .def_readwrite("atom_types", &distnnp_state::MolecularInfo::atom_types)
      .def_readwrite("atom_positions",
                     &distnnp_state::MolecularInfo::atom_positions)
      .def_readwrite("atom_velocities",
                     &distnnp_state::MolecularInfo::atom_velocities)
      .def_readwrite("edge_counts", &distnnp_state::MolecularInfo::edge_counts);

  py::class_<distnnp_state::DistEnvironment>(m, "DistEnvironment")
      .def(py::init<>())
      .def_readwrite("global_rank",
                     &distnnp_state::DistEnvironment::global_rank)
      .def_readwrite("world_size", &distnnp_state::DistEnvironment::world_size)
      .def_readwrite("gpus_per_node",
                     &distnnp_state::DistEnvironment::gpus_per_node)
      .def_readwrite("num_nodes", &distnnp_state::DistEnvironment::num_nodes)
      .def_readwrite("local_rank", &distnnp_state::DistEnvironment::local_rank);

  py::class_<distnnp_state::SimulationConfig>(m, "SimulationConfig")
      .def(py::init<>())
      .def_readwrite("cutoff", &distnnp_state::SimulationConfig::cutoff)
      .def_readwrite("ghost_cutoff_buffer",
                     &distnnp_state::SimulationConfig::ghost_cutoff_buffer)
      .def_readwrite("cell", &distnnp_state::SimulationConfig::cell)
      .def_readwrite("pbc", &distnnp_state::SimulationConfig::pbc);

  py::class_<distnnp_state::PartitionMessagePassingInfo>(
      m, "PartitionMessagePassingInfo")
      .def(py::init<>())
      .def_readwrite("partition_id",
                     &distnnp_state::PartitionMessagePassingInfo::partition_id)
      .def_readwrite(
          "partition_indices",
          &distnnp_state::PartitionMessagePassingInfo::partition_indices)
      .def_readwrite(
          "global_send_recv_matrix",
          &distnnp_state::PartitionMessagePassingInfo::global_send_recv_matrix)
      .def_readwrite("send_offsets",
                     &distnnp_state::PartitionMessagePassingInfo::send_offsets)
      .def_readwrite(
          "send_local_indices",
          &distnnp_state::PartitionMessagePassingInfo::send_local_indices)
      .def_readwrite("recv_offsets",
                     &distnnp_state::PartitionMessagePassingInfo::recv_offsets)
      .def_readwrite(
          "recv_ghost_indices",
          &distnnp_state::PartitionMessagePassingInfo::recv_ghost_indices)
      .def_readwrite("interior_mask",
                     &distnnp_state::PartitionMessagePassingInfo::interior_mask)
      .def_readwrite("edge_index",
                     &distnnp_state::PartitionMessagePassingInfo::edge_index)
      .def_readwrite(
          "edge_index_halo",
          &distnnp_state::PartitionMessagePassingInfo::edge_index_halo)
      .def_readwrite("atom_counts",
                     &distnnp_state::PartitionMessagePassingInfo::atom_counts);

  py::class_<distnnp_state::DistNNPState>(m, "DistNNPState")
      .def(py::init<>())
      .def_readwrite("dist_env", &distnnp_state::DistNNPState::dist_env)
      .def_readwrite("sim_config", &distnnp_state::DistNNPState::sim_config)
      .def_readwrite("global_molecular_info",
                     &distnnp_state::DistNNPState::global_molecular_info)
      .def_readwrite("local_molecular_info",
                     &distnnp_state::DistNNPState::local_molecular_info)
      .def_readwrite("local_partition_info",
                     &distnnp_state::DistNNPState::local_partition_info);

  py::class_<distnnp_state::KDTreePartitioner>(m, "KDTreePartitioner")
      .def(py::init<>())
      .def("partition",
           py::overload_cast<const distnnp_state::MolecularInfo &,
                             const distnnp_state::DistEnvironment &>(
               &distnnp_state::KDTreePartitioner::partition),
           "Partition: unweighted (default) or user-weighted (if edge_counts "
           "set)")
      .def("partition",
           py::overload_cast<const distnnp_state::MolecularInfo &,
                             const distnnp_state::DistEnvironment &,
                             const distnnp_state::SimulationConfig &>(
               &distnnp_state::KDTreePartitioner::partition),
           "Partition: auto-compute edge weights via radius_graph_pbc (CUDA)")
      .def("build_halo_maps",
           &distnnp_state::KDTreePartitioner::build_halo_maps);

  py::class_<distnnp_state::SimpleSpatialPartitioner>(
      m, "SimpleSpatialPartitioner")
      .def(py::init<>())
      .def("partition", &distnnp_state::SimpleSpatialPartitioner::partition)
      .def("build_halo_maps",
           &distnnp_state::SimpleSpatialPartitioner::build_halo_maps);

  py::enum_<distnnp_state::SFCCurveType>(m, "SFCCurveType")
      .value("Morton", distnnp_state::SFCCurveType::Morton)
      .value("Hilbert", distnnp_state::SFCCurveType::Hilbert);

  py::class_<distnnp_state::SFCPartitioner>(m, "SFCPartitioner")
      .def(py::init<distnnp_state::SFCCurveType>(),
           py::arg("curve_type") = distnnp_state::SFCCurveType::Morton)
      .def("partition",
           py::overload_cast<const distnnp_state::MolecularInfo &,
                             const distnnp_state::DistEnvironment &>(
               &distnnp_state::SFCPartitioner::partition),
           "Partition: unweighted (default) or user-weighted (if edge_counts "
           "set)")
      .def("partition",
           py::overload_cast<const distnnp_state::MolecularInfo &,
                             const distnnp_state::DistEnvironment &,
                             const distnnp_state::SimulationConfig &>(
               &distnnp_state::SFCPartitioner::partition),
           "Partition: auto-compute edge weights via radius_graph_pbc (CUDA)")
      .def("build_halo_maps", &distnnp_state::SFCPartitioner::build_halo_maps);

  py::class_<distnnp_state::TrajectoryWriter>(m, "TrajectoryWriter")
      .def(py::init<uint32_t, py::bytes, const std::string &,
                    const std::string &, const std::string &>(),
           py::arg("global_rank"), py::arg("run_id"), py::arg("author"),
           py::arg("output_dir") = "./trajectory",
           py::arg("output_file") = "trajectory.distnnp")
      .def("initialize",
           py::overload_cast<const std::string &, const float, const uint32_t,
                              std::vector<std::string>>(
               &distnnp_state::TrajectoryWriter::initialize),
           "Writes front matter and initializes environment from an xyz file",
           py::arg("input_file_name"), py::arg("time_per_step"),
           py::arg("steps_per_write"),
           py::arg("opts") = std::vector<std::string>{})
      .def("initialize",
           py::overload_cast<const distnnp_state::MolecularInfo &, const float,
                              const uint32_t, std::vector<std::string>>(
               &distnnp_state::TrajectoryWriter::initialize),
           "Writes front matter and initializes environment from MolecularInfo struct",
           py::arg("input_system"), py::arg("time_per_step"),
           py::arg("steps_per_write"),
           py::arg("opts") = std::vector<std::string>{})
      .def("write_to_xyz", &distnnp_state::TrajectoryWriter::write_to_xyz,
           "Writes state to an xyz file", py::arg("state"),
           py::arg("file_name"), py::arg("opts") = std::vector<std::string>{})
      .def("update_atom_counts",
           &distnnp_state::TrajectoryWriter::update_atom_counts,
           "Update the rank offsets. Should be called after every repartition",
           py::arg("counts"))
      .def("submit", &distnnp_state::TrajectoryWriter::submit,
           "Submits a state to be written. cell/pbc are optional and are "
           "carried forward from the last value seen if omitted",
           py::arg("step"), py::arg("state"),
           py::arg("cell") = c10::optional<at::Tensor>{},
           py::arg("pbc") = c10::optional<at::Tensor>{})
      .def("get_cell", &distnnp_state::TrajectoryWriter::get_cell,
           "Returns the currently cached simulation cell (float32 [3, 3])")
      .def("get_pbc", &distnnp_state::TrajectoryWriter::get_pbc,
           "Returns the currently cached periodic boundary flags (uint8 [3])")
      .def("flush", &distnnp_state::TrajectoryWriter::flush,
           "Flushes the queue and closes the file")
      .def("__enter__",
           [](distnnp_state::TrajectoryWriter &self) { return &self; })
      .def("__exit__",
           [](distnnp_state::TrajectoryWriter &self, py::args args) {});

  m.def("radius_graph_pbc", &radius_graph_pbc,
        "CUDA cell-list radius graph with selective PBC tiling", py::arg("pos"),
        py::arg("cell"), py::arg("pbc"), py::arg("radius"),
        py::arg("max_num_neighbors"), py::arg("enforce_max_strictly"));

  m.def(
      "radius_graph_pbc_ex", &radius_graph_pbc_ex,
      "Extended radius graph: accepts optional target_indices subset, returns "
      "(edge_index, cell_shifts, num_neighbors_image, neighbor_counts).",
      py::arg("pos"), py::arg("cell"), py::arg("pbc"), py::arg("radius"),
      py::arg("max_num_neighbors"), py::arg("enforce_max_strictly"),
      py::arg("target_indices") = c10::optional<at::Tensor>{});

  // ===== Timing =====

  py::enum_<distnnp_state::TimerClass>(m, "TimerClass")
      .value("Compute", distnnp_state::TimerClass::Compute)
      .value("Comm",    distnnp_state::TimerClass::Comm)
      .value("Wait",    distnnp_state::TimerClass::Wait)
      .value("Other",   distnnp_state::TimerClass::Other);

  py::enum_<distnnp_state::Where>(m, "Where")
      .value("Host",   distnnp_state::Where::Host)
      .value("Device", distnnp_state::Where::Device);

  py::enum_<distnnp_state::Level>(m, "Level")
      .value("Off",    distnnp_state::Level::Off)
      .value("Basic",  distnnp_state::Level::Basic)
      .value("Detail", distnnp_state::Level::Detail);

  py::class_<distnnp_state::TimingRegistry>(m, "TimingRegistry")
      .def_static("instance", &distnnp_state::TimingRegistry::instance,
                  py::return_value_policy::reference)
      .def("init", [](distnnp_state::TimingRegistry& self,
                      uint32_t rank, uint32_t local_rank, uint32_t world_size,
                      py::bytes run_id, int32_t device_id,
                      const std::string& output_dir) {
               self.init(rank, local_rank, world_size,
                         static_cast<std::string>(run_id), device_id, output_dir);
           },
           py::arg("rank"), py::arg("local_rank"), py::arg("world_size"),
           py::arg("run_id"), py::arg("device_id"), py::arg("output_dir"))
      .def("set_level", &distnnp_state::TimingRegistry::set_level, py::arg("level"))
      .def("enabled",   &distnnp_state::TimingRegistry::enabled)
      .def("set_step",  &distnnp_state::TimingRegistry::set_step,  py::arg("step"))
      .def("flush",     &distnnp_state::TimingRegistry::flush);

  py::class_<distnnp_state::Timer>(m, "Timer")
      .def(py::init<>())
      .def("start", &distnnp_state::Timer::start,
           py::arg("key"), py::arg("cls"),
           py::arg("where") = distnnp_state::Where::Host,
           py::arg("stream") = nullptr)
      .def("stop", &distnnp_state::Timer::stop)
      .def_static("set_level", &distnnp_state::Timer::set_level, py::arg("level"))
      .def_static("enabled",   &distnnp_state::Timer::enabled)
      .def_static("set_step",  &distnnp_state::Timer::set_step,  py::arg("step"));

  m.def("radius_graph_pbc_counts_ex", &radius_graph_pbc_counts_ex,
        "Count-only variant: returns just neighbor_counts[N_target] int32. "
        "Skips edge buffer + prefix-sum + B4 write kernel (~50% faster).",
        py::arg("pos"), py::arg("cell"), py::arg("pbc"), py::arg("radius"),
        py::arg("target_indices") = c10::optional<at::Tensor>{});

  // ===== Distributed build_graph hot-path ops (Mode A) =====

  m.def(
      "filter_edges_to_radius", &filter_edges_to_radius,
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
        py::arg("edges"), py::arg("pid_outer"), py::arg("my_outer_atoms"),
        py::arg("my_rank"), py::arg("world_size"), py::arg("N"));
}
