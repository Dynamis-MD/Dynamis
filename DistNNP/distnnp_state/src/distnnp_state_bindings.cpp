#include "distnnp_state/distnnp_state.hpp"
#include "distnnp_state/partitioner.hpp"
#include "distnnp_state/exchanger.hpp"
#include "distnnp_state/radius_pbc.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(distnnp_state_impl, m)
{
  py::class_<distnnp_state::MolecularInfo>(m, "MolecularInfo")
      .def(py::init<>())
      .def_readwrite("num_atoms", &distnnp_state::MolecularInfo::num_atoms)
      .def_readwrite("atom_indices", &distnnp_state::MolecularInfo::atom_indices)
      .def_readwrite("atom_positions", &distnnp_state::MolecularInfo::atom_positions)
      .def_readwrite("atom_velocities", &distnnp_state::MolecularInfo::atom_velocities)
      .def_readwrite("edge_counts", &distnnp_state::MolecularInfo::edge_counts);

  py::class_<distnnp_state::DistEnvironment>(m, "DistEnvironment")
      .def(py::init<>())
      .def_readwrite("global_rank", &distnnp_state::DistEnvironment::global_rank)
      .def_readwrite("world_size", &distnnp_state::DistEnvironment::world_size)
      .def_readwrite("gpus_per_node", &distnnp_state::DistEnvironment::gpus_per_node)
      .def_readwrite("num_nodes", &distnnp_state::DistEnvironment::num_nodes)
      .def_readwrite("local_rank", &distnnp_state::DistEnvironment::local_rank);

  py::class_<distnnp_state::SimulationConfig>(m, "SimulationConfig")
      .def(py::init<>())
      .def_readwrite("cutoff", &distnnp_state::SimulationConfig::cutoff)
      .def_readwrite("ghost_cutoff_buffer", &distnnp_state::SimulationConfig::ghost_cutoff_buffer)
      .def_readwrite("cell", &distnnp_state::SimulationConfig::cell)
      .def_readwrite("pbc", &distnnp_state::SimulationConfig::pbc);

  py::class_<distnnp_state::PartitionMessagePassingInfo>(m, "PartitionMessagePassingInfo")
      .def(py::init<>())
      .def_readwrite("partition_id", &distnnp_state::PartitionMessagePassingInfo::partition_id)
      .def_readwrite("partition_indices", &distnnp_state::PartitionMessagePassingInfo::partition_indices)
      .def_readwrite("local_atom_positions", &distnnp_state::PartitionMessagePassingInfo::local_atom_positions)
      .def_readwrite("local_atom_velocities", &distnnp_state::PartitionMessagePassingInfo::local_atom_velocities)
      .def_readwrite("global_send_recv_matrix", &distnnp_state::PartitionMessagePassingInfo::global_send_recv_matrix)
      .def_readwrite("send_offsets", &distnnp_state::PartitionMessagePassingInfo::send_offsets)
      .def_readwrite("send_local_indices", &distnnp_state::PartitionMessagePassingInfo::send_local_indices)
      .def_readwrite("recv_offsets", &distnnp_state::PartitionMessagePassingInfo::recv_offsets)
      .def_readwrite("recv_ghost_indices", &distnnp_state::PartitionMessagePassingInfo::recv_ghost_indices)
      .def_readwrite("interior_mask", &distnnp_state::PartitionMessagePassingInfo::interior_mask)
      .def_readwrite("edge_index", &distnnp_state::PartitionMessagePassingInfo::edge_index)
      .def_readwrite("edge_index_halo", &distnnp_state::PartitionMessagePassingInfo::edge_index_halo);

  py::class_<distnnp_state::DistNNPState>(m, "DistNNPState")
      .def(py::init<>())
      .def_readwrite("dist_env", &distnnp_state::DistNNPState::dist_env)
      .def_readwrite("global_molecular_info", &distnnp_state::DistNNPState::global_molecular_info)
      .def_readwrite("local_partition_info", &distnnp_state::DistNNPState::local_partition_info);

  py::class_<distnnp_state::KDTreePartitioner>(m, "KDTreePartitioner")
      .def(py::init<>())
      .def("partition",
           py::overload_cast<const distnnp_state::MolecularInfo &,
                             const distnnp_state::DistEnvironment &>(
               &distnnp_state::KDTreePartitioner::partition),
           "Partition: unweighted (default) or user-weighted (if edge_counts set)")
      .def("partition",
           py::overload_cast<const distnnp_state::MolecularInfo &,
                             const distnnp_state::DistEnvironment &,
                             const distnnp_state::SimulationConfig &>(
               &distnnp_state::KDTreePartitioner::partition),
           "Partition: auto-compute edge weights via radius_graph_pbc (CUDA)")
      .def("build_halo_maps", &distnnp_state::KDTreePartitioner::build_halo_maps);

  py::class_<distnnp_state::SimpleSpatialPartitioner>(m, "SimpleSpatialPartitioner")
      .def(py::init<>())
      .def("partition", &distnnp_state::SimpleSpatialPartitioner::partition)
      .def("build_halo_maps", &distnnp_state::SimpleSpatialPartitioner::build_halo_maps);

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
           "Partition: unweighted (default) or user-weighted (if edge_counts set)")
      .def("partition",
           py::overload_cast<const distnnp_state::MolecularInfo &,
                             const distnnp_state::DistEnvironment &,
                             const distnnp_state::SimulationConfig &>(
               &distnnp_state::SFCPartitioner::partition),
           "Partition: auto-compute edge weights via radius_graph_pbc (CUDA)")
      .def("build_halo_maps", &distnnp_state::SFCPartitioner::build_halo_maps);

  py::class_<distnnp_state::IPCExchanger>(m, "IPCExchanger")
      .def(py::init<>())
      .def_static("allocate_ipc_buffer", &distnnp_state::IPCExchanger::allocate_ipc_buffer,
           "Allocate a tensor via raw cudaMalloc (required for IPC)",
           py::arg("rows"), py::arg("cols"), py::arg("device_index"))
      .def("export_ipc_handle", &distnnp_state::IPCExchanger::export_ipc_handle,
           "Export CUDA IPC handle for a buffer as bytes",
           py::arg("buffer"))
      .def("setup", &distnnp_state::IPCExchanger::setup,
           "Open remote IPC handles and build push targets",
           py::arg("x_buffer"),
           py::arg("grad_recv_buffer"),
           py::arg("all_x_handles"),
           py::arg("all_grad_handles"),
           py::arg("send_offsets"),
           py::arg("send_local_indices"),
           py::arg("recv_offsets"),
           py::arg("all_recv_offsets"),
           py::arg("all_num_local"),
           py::arg("num_local"),
           py::arg("feature_dim"),
           py::arg("rank"),
           py::arg("world_size"))
      .def("push_forward", [](distnnp_state::IPCExchanger &self, torch::Tensor x_buffer,
                              int64_t ce_threshold_bytes) {
           self.push_forward(x_buffer, 0, ce_threshold_bytes);
      }, "Forward push: gather + write to peers' ghost regions.\n"
         "ce_threshold_bytes: per-peer data >= this uses Copy Engine (0 = always SM kernel)",
           py::arg("x_buffer"), py::arg("ce_threshold_bytes") = 0)
      .def("push_backward", [](distnnp_state::IPCExchanger &self, torch::Tensor grad) {
           self.push_backward(grad, 0);
      }, "Backward push: gather ghost grads + write to peers' grad buffers",
           py::arg("grad"))
      .def("create_and_export_event", &distnnp_state::IPCExchanger::create_and_export_event,
           "Create interprocess event and return IPC handle (64 bytes)")
      .def("setup_events", &distnnp_state::IPCExchanger::setup_events,
           "Open remote IPC events for stream-level sync",
           py::arg("all_event_handles"))
      .def("record_push_complete", [](distnnp_state::IPCExchanger &self) {
           self.record_push_complete(0);
      }, "Record completion event on default stream")
      .def("wait_for_pushes", [](distnnp_state::IPCExchanger &self) {
           self.wait_for_pushes(0);
      }, "Wait for all remote pushes to complete on default stream")
      .def("teardown", &distnnp_state::IPCExchanger::teardown)
      .def("is_setup", &distnnp_state::IPCExchanger::is_setup)
      .def("feature_dim", &distnnp_state::IPCExchanger::feature_dim)
      .def("num_local", &distnnp_state::IPCExchanger::num_local)
      .def("total_send", &distnnp_state::IPCExchanger::total_send)
      .def("total_recv", &distnnp_state::IPCExchanger::total_recv)
      .def("send_local_indices", &distnnp_state::IPCExchanger::send_local_indices);

  m.def("radius_graph_pbc", &radius_graph_pbc,
        "CUDA cell-list radius graph with selective PBC tiling",
        py::arg("pos"),
        py::arg("cell"),
        py::arg("pbc"),
        py::arg("radius"),
        py::arg("max_num_neighbors"),
        py::arg("enforce_max_strictly"));
}
