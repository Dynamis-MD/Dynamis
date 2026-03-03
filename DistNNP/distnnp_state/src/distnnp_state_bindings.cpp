#include "distnnp_state/distnnp_state.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(distnnp_state_impl, m)
{
  py::class_<distnnp_state::MolecularInfo>(m, "MolecularInfo")
      .def(py::init<>())
      .def_readwrite("num_atoms", &distnnp_state::MolecularInfo::num_atoms)
      .def_readwrite("atom_indices", &distnnp_state::MolecularInfo::atom_indices)
      .def_readwrite("atom_positions", &distnnp_state::MolecularInfo::atom_positions)
      .def_readwrite("atom_velocities", &distnnp_state::MolecularInfo::atom_velocities);

  py::class_<distnnp_state::DistEnvironment>(m, "DistEnvironment")
      .def(py::init<>())
      .def_readwrite("global_rank", &distnnp_state::DistEnvironment::global_rank)
      .def_readwrite("world_size", &distnnp_state::DistEnvironment::world_size)
      .def_readwrite("gpus_per_node", &distnnp_state::DistEnvironment::gpus_per_node)
      .def_readwrite("num_nodes", &distnnp_state::DistEnvironment::num_nodes)
      .def_readwrite("local_rank", &distnnp_state::DistEnvironment::local_rank);

  py::class_<distnnp_state::PartitionMessagePassingInfo>(m, "PartitionMessagePassingInfo")
      .def(py::init<>())
      .def_readwrite("partition_indices", &distnnp_state::PartitionMessagePassingInfo::partition_indices)
      .def_readwrite("local_atom_positions", &distnnp_state::PartitionMessagePassingInfo::local_atom_positions)
      .def_readwrite("local_atom_velocities", &distnnp_state::PartitionMessagePassingInfo::local_atom_velocities)
      .def_readwrite("global_send_recv_matrix", &distnnp_state::PartitionMessagePassingInfo::global_send_recv_matrix)
      .def_readwrite("halo_map", &distnnp_state::PartitionMessagePassingInfo::halo_map);

  py::class_<distnnp_state::DistNNPState>(m, "DistNNPState")
      .def(py::init<>())
      .def_readwrite("dist_env", &distnnp_state::DistNNPState::dist_env)
      .def_readwrite("global_molecular_info", &distnnp_state::DistNNPState::global_molecular_info)
      .def_readwrite("local_partition_info", &distnnp_state::DistNNPState::local_partition_info);
}